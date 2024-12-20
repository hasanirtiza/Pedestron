import os
import os.path as osp
import json

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import Hook, obj_from_dict
from mmcv.parallel import scatter, collate
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset

from .coco_utils import results2json, fast_eval_recall
from .mean_ap import eval_map
from .eval_mr import COCOeval as COCOMReval
from tools.ECPB.eval import eval
from mmdet import datasets


class DistEvalHook(Hook):

    def __init__(self, dataset, interval=1, kitti=False):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = obj_from_dict(dataset, datasets,
                                         {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval
        self.kitti = kitti

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            results[idx] = result

            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()

        if runner.rank == 0:
            print('\n')
            dist.barrier()
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            dist.barrier()
        dist.barrier()

    def evaluate(self):
        raise NotImplementedError


class DistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        gt_bboxes = []
        gt_labels = []
        gt_ignore = [] if self.dataset.with_crowd else None
        for i in range(len(self.dataset)):
            ann = self.dataset.get_ann_info(i)
            bboxes = ann['bboxes']
            labels = ann['labels']
            if gt_ignore is not None:
                ignore = np.concatenate([
                    np.zeros(bboxes.shape[0], dtype=np.bool),
                    np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
                ])
                gt_ignore.append(ignore)
                bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
                labels = np.concatenate([labels, ann['labels_ignore']])
            gt_bboxes.append(bboxes)
            gt_labels.append(labels)
        # If the dataset is VOC2007, then use 11 points mAP evaluation.
        if hasattr(self.dataset, 'year') and self.dataset.year == 2007:
            ds_name = 'voc07'
        else:
            ds_name = self.dataset.CLASSES
        mean_ap, eval_results = eval_map(
            results,
            gt_bboxes,
            gt_labels,
            gt_ignore=gt_ignore,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=ds_name,
            print_summary=True)
        runner.log_buffer.output['mAP'] = mean_ap
        runner.log_buffer.ready = True


class CocoDistEvalRecallHook(DistEvalHook):

    def __init__(self,
                 dataset,
                 interval=1,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        super(CocoDistEvalRecallHook, self).__init__(
            dataset, interval=interval)
        self.proposal_nums = np.array(proposal_nums, dtype=np.int32)
        self.iou_thrs = np.array(iou_thrs, dtype=np.float32)

    def evaluate(self, runner, results):
        # the official coco evaluation is too slow, here we use our own
        # implementation instead, which may get slightly different results
        ar = fast_eval_recall(results, self.dataset.coco, self.proposal_nums,
                              self.iou_thrs)
        for i, num in enumerate(self.proposal_nums):
            runner.log_buffer.output['AR@{}'.format(num)] = ar[i]
        runner.log_buffer.ready = True


class CocoDistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        tmp_file = osp.join(runner.work_dir, 'temp_0')
        result_files = results2json(self.dataset, results, tmp_file)

        res_types = ['bbox', 'segm'
                     ] if runner.model.module.with_mask else ['bbox']
        cocoGt = self.dataset.coco
        imgIds = cocoGt.getImgIds()
        for res_type in res_types:
            cocoDt = cocoGt.loadRes(result_files[res_type])
            iou_type = res_type
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = imgIds
            if self.kitti:
                cocoEval.params.recThrs = np.linspace(.0, 1.00, 40, endpoint=True)
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            metrics = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
            for i in range(len(metrics)):
                key = '{}_{}'.format(res_type, metrics[i])
                val = float('{:.3f}'.format(cocoEval.stats[i]))
                runner.log_buffer.output[key] = val
            runner.log_buffer.output['{}_mAP_copypaste'.format(res_type)] = (
                '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6])
        runner.log_buffer.ready = True
        for res_type in res_types:
            os.remove(result_files[res_type])


class CocoDistEvalECPMRHook(DistEvalHook):
    def __init__(self, dataset, interval=1, res_types=['bbox'], day='day', mode='val'):
        super().__init__(dataset, interval)
        self.res_types = res_types
        self.day = day
        self.mode = mode

    def convert_results(self, results):
        mock_detections = []
        if len(results) > 1:
            results = results[:][0]
        if len(results) == 0:
            return mock_detections

        for box in results[0]:
            box = {'x0': float(box[0]),
                   'x1': float(box[2]),
                   'y0': float(box[1]),
                   'y1': float(box[3]),
                   'score': float(box[4]),
                   'identity': 'pedestrian',
                   'orient': 0.0}
            mock_detections.append(box)
        return mock_detections

    def evaluate(self, runner, results):
        tmp_file = f'/netscratch/hkhan/results/mock_detections/{self.day}/{self.mode}/'

        i = 0
        for id, im in self.dataset.coco.imgs.items():
            dest = osp.join(tmp_file, os.path.basename(im['file_name']).replace('.png', '.json'))
            detections = self.convert_results(results[i])
            frame = dict(identity='frame', children=detections)
            json.dump(frame, open(dest, 'w'), indent=1)
            i += 1

        res = eval(self.day, self.mode)
        for key, value in res.items():
            runner.log_buffer.output[key] = value

        runner.log_buffer.ready = True


class CocoDistEvalMRHook(DistEvalHook):
    """ EvalHook for MR evaluation.

    Args:
        res_types(list): detection type, currently support 'bbox'
            and 'vis_bbox'.
    """
    def __init__(self, dataset, interval=1, res_types=['bbox']):
        super().__init__(dataset, interval)
        self.res_types = res_types

    def evaluate(self, runner, results):
        tmp_file = osp.join(runner.work_dir, 'temp_0')
        result_files = results2json(self.dataset, results, tmp_file)

        cocoGt = self.dataset.coco
        imgIds = cocoGt.getImgIds()
        for res_type in self.res_types:
            assert res_type in ['bbox', 'vis_bbox']
            try:
                cocoDt = cocoGt.loadRes(result_files['bbox'])
            except IndexError:
                print('No prediction found.')
                break
            metrics = ['MR_Reasonable', 'MR_Small', 'MR_Middle', 'MR_Large',
                       'MR_Bare', 'MR_Partial', 'MR_Heavy', 'MR_R+HO']
            cocoEval = COCOMReval(cocoGt, cocoDt, res_type)
            cocoEval.params.imgIds = imgIds
            for id_setup in range(0,8):
                cocoEval.evaluate(id_setup)
                cocoEval.accumulate()
                cocoEval.summarize(id_setup)
                
                key = '{}'.format(metrics[id_setup])
                val = float('{:.3f}'.format(cocoEval.stats[id_setup]))
                runner.log_buffer.output[key] = val
            runner.log_buffer.output['{}_MR_copypaste'.format(res_type)] = (
                '{mr[0]:.3f} {mr[1]:.3f} {mr[2]:.3f} {mr[3]:.3f} '
                '{mr[4]:.3f} {mr[5]:.3f} {mr[6]:.3f} {mr[7]:.3f} ').format(
                    mr=cocoEval.stats[:8])
        runner.log_buffer.ready = True
        os.remove(result_files['bbox'])
