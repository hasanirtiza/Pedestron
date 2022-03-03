
import argparse
import os
import os.path as osp
import shutil
import tempfile
import json
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from tools.cityPerson.eval_demo import validate
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np


def single_gpu_test(model, data_loader, show=False, save_img=False, save_img_dir=''):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result, id = model(return_loss=False, rescale=not show, return_id=True, **data)
        results.append((result, id))

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg, save_result=save_img,
                                     result_name=save_img_dir + '/' + str(i) + '.jpg')

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result, id = model(return_loss=False, rescale=True, return_id=True, **data)
        results.append((result, id))

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('checkpoint_start', type=int, default=1)
    parser.add_argument('checkpoint_end', type=int, default=100)
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--save_img', action='store_true', help='save result image')
    parser.add_argument('--save_img_dir', type=str, help='the dir for result image', default='')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mean_teacher', action='store_true', help='test the mean teacher pth')
    parser.add_argument('--ecp', action='store_true', help='use ECP params')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    work_dir = "/".join(args.checkpoint.split("/")[:-1] + ['runs'])
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    
    cfg = mmcv.Config.fromfile(args.config)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    tval_writer = SummaryWriter(work_dir + '/lamr_val')
    ttrain_writer = SummaryWriter(work_dir + '/lamr_train')

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        rank = 0
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        rank, world_size = get_dist_info()
    dataset = build_dataset(cfg.data.test)

    if args.out is not None and not args.out.endswith(('.json', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    for i in range(args.checkpoint_start, args.checkpoint_end):
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=1,
            dist=distributed,
            shuffle=False)

        # build the model and load checkpoint
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        fp16_cfg = cfg.get('fp16', None)

        test_json = cfg.data.test.get("ann_file", "")

        if fp16_cfg is not None:
            wrap_fp16_model(model)
        if not args.mean_teacher:
            while not osp.exists(args.checkpoint + str(i) + '.pth'):
                time.sleep(5)
            while i + 1 != args.checkpoint_end and not osp.exists(args.checkpoint + str(i + 1) + '.pth'):
                time.sleep(5)
            checkpoint = load_checkpoint(model, args.checkpoint + str(i) + '.pth', map_location='cpu')
            model.CLASSES = dataset.CLASSES
        else:
            while not osp.exists(args.checkpoint + str(i) + '.pth.stu'):
                time.sleep(5)
            while i + 1 != args.checkpoint_end and not osp.exists(args.checkpoint + str(i + 1) + '.pth.stu'):
                time.sleep(5)
            checkpoint = load_checkpoint(model, args.checkpoint + str(i) + '.pth.stu', map_location='cpu')
            checkpoint['meta'] = dict()
            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                model.CLASSES = dataset.CLASSES  # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility

        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader, args.show, args.save_img, args.save_img_dir)
        else:
            model = MMDistributedDataParallel(model.cuda())
            outputs = multi_gpu_test(model, data_loader, args.tmpdir)

        if rank == 0:
            check = []
            for j in range(len(outputs)):
                out, img_id = outputs[j]
                if len(out) > 0:
                    boxes = out
                    if type(boxes) == list:
                        boxes = boxes[0]
                    boxes[:, [2, 3]] -= boxes[:, [0, 1]]
                    if boxes is None:
                        boxes = []
                    if len(boxes) > 0:
                        for box in boxes:
                            temp = dict()
                            temp['image_id'] = img_id
                            temp['category_id'] = 1
                            temp['bbox'] = box[:4].tolist()
                            temp['score'] = float(box[4])
                            check.append(temp)
            with open(args.out, 'w') as f:
                json.dump(check, f)
            stats = validate(test_json, args.out, ecp=args.ecp)
            MRs = stats
            
            print("Checkpoint %d: [VR: %.2f], [VS: %.2f], [VH: %.2f], [VA: %.2f]" % (i, MRs[0], MRs[1], MRs[2], MRs[3]))
            tval_writer.add_scalar('Reasonable', MRs[0], i)
            tval_writer.add_scalar('Small', MRs[1], i)
            tval_writer.add_scalar('Heavy', MRs[2], i)
            tval_writer.add_scalar('All', MRs[3], i)
            tval_writer.flush()
            print("Checkpoint %d: " % i)
            print(stats)


if __name__ == '__main__':
    main()
