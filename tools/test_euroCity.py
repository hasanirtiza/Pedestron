import argparse

import os
import os.path as osp
import time
import cv2
import torch
import glob
import json
import mmcv

from mmdet.apis import inference_detector, init_detector, show_result
from tools.ECPB.eval import eval


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('checkpoint_start', type=int, default=1)
    parser.add_argument('checkpoint_end', type=int, default=100)
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
    args = parser.parse_args()
    return args



def mock_detector(model, image):
    image = cv2.imread(image)
    results = inference_detector(model, image)
    mock_detections = []
    if len(results) > 1:
        results = results[:][0]
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

def create_base_dir(dest):
    basedir = os.path.dirname(dest)
    if not os.path.exists(basedir):
        os.makedirs(basedir)

def run_detector_on_dataset(day='day', mode='val'):
    assert mode in ['val', 'test']
    assert day in ['day', 'night']

    args = parse_args()

    eval_imgs = glob.glob('datasets/EuroCity/ECP/{}/img/{}/*/*'.format(day, mode))
    destdir = './results/mock_detections/{}/{}/'.format(day, mode)
    create_base_dir(destdir)
    for i in range(args.checkpoint_start, args.checkpoint_end):
        if not args.mean_teacher:
            while not osp.exists(args.checkpoint + str(i) + '.pth'):
                time.sleep(5)
            while i+1 != args.checkpoint_end and not osp.exists(args.checkpoint + str(i+1) + '.pth'):
                time.sleep(5)
            checkpoint = args.checkpoint + str(i) + '.pth'
        else:
            while not osp.exists(args.checkpoint + str(i) + '.pth.stu'):
                time.sleep(5)
            while i+1 != args.checkpoint_end and not osp.exists(args.checkpoint + str(i+1) + '.pth.stu'):
                time.sleep(5)
            checkpoint = args.checkpoint + str(i) + '.pth.stu'

        model = init_detector(
            args.config, checkpoint, device=torch.device('cuda'))

        prog_bar = mmcv.ProgressBar(len(eval_imgs))
        for im in eval_imgs:
            detections = mock_detector(model, im)
            destfile = os.path.join(destdir, os.path.basename(im).replace('.png', '.json'))
            frame = {'identity': 'frame'}
            frame['children'] = detections
            json.dump(frame, open(destfile, 'w'), indent=1)
            prog_bar.update()
        eval(day, mode)

if __name__ == '__main__':
    run_detector_on_dataset()
    # run_detector_on_dataset(mode='test')
