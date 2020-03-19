"""
NOTE: python 3 only, because of module "concurrent.futures"

Qualitative evaluation (visualize failure modes):
Script to visualize false positives and false negatives of an evaluation run.

False positives are drawn in yellow.
False negatives are drawn in red.

Usage: Look for "# edit" comments.
"""

import logging
import os
import time

import cv2
import numpy as np
from PIL import Image

import match

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s, pid: %(process)d, %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    )

from concurrent.futures import ProcessPoolExecutor  # python3 only


def draw_boxes(img, boxes, color, thickness):
    # TODO maybe add class to drawn bounding box
    for box in boxes:
        cv2.rectangle(img, (int(box['x0']), int(box['y0'])), (int(box['x1']), int(box['y1'])), color, thickness)


def get_img(det, data_path):
    fn = det['filename']
    name = os.path.splitext(os.path.basename(fn))[0] + '.png'

    city = name.split('_')[0]
    img = None

    search_paths = [
        'day/img/train', 'day/img/val', 'day/img/test',
        'night/img/train', 'night/img/val', 'night/img/test',
    ]  # edit: only if you changed the structure of the ecp dataset
    for img_path in search_paths:
        try:
            path = os.path.join(data_path, img_path, city, name)
            img = Image.open(path)
        except FileNotFoundError:
            pass

    if img is None:
        raise FileNotFoundError('Could not find image: {}'.format(name))
    return np.array(img), name


def process_frame(frame, data_path, out_path):
    dets = frame['det']
    gts = frame['gt']
    img, name = get_img(dets, data_path)

    fn = [gt for gt in gts['children'] if gt['matched'] == 0 and gt['ignore'] == 0]
    fp = [det for det in dets['children'] if det.get('ignore', 0) == 0 and det['matched'] == 0]

    draw_boxes(img, fn, color=(255, 0, 0), thickness=2)
    draw_boxes(img, fp, color=(255, 255, 0), thickness=1)

    result = Image.fromarray(img)
    result.save(os.path.join(out_path, name))


def process_batch(args):
    batch_nr, batch, total_batches, data_path, out_path = args

    logging.info('Starting batch {:2d}/{}'.format(batch_nr, total_batches))
    start = time.time()

    for cnt, frame in enumerate(batch, start=1):
        process_frame(frame, data_path, out_path)
        if cnt % 10 == 0:
            logging.info('Processed {:3d} images for batch {:2d}/{}'.format(cnt, batch_nr, total_batches))

    end = time.time()
    elapsed = int(end - start)
    logging.info('Finished batch {:2d}/{} in {:02d}:{:02d}:{:02d}'.format(batch_nr, total_batches,
                                                                          elapsed // 3600, (elapsed // 60) % 60,
                                                                          elapsed % 60))


def run(out_path, pkl_file, data_path, cpu_thread_cnt):
    result = match.Result.load_from_disc(pkl_file).raw_result

    name = os.path.splitext(os.path.basename(pkl_file))[0]
    out_path = os.path.join(out_path, name)
    os.makedirs(out_path)  # will throw an exception if path already exists (ensures that things don't get overridden)

    logging.info('Saving results to {}'.format(out_path))

    k, m = divmod(len(result), cpu_thread_cnt)
    batches = [result[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(cpu_thread_cnt)]

    # sanity check
    total_length = 0
    for batch in batches:
        total_length += len(batch)
    assert total_length == len(result)

    # create and run jobs
    jobs = [(batch_nr, batch, cpu_thread_cnt, data_path, out_path) for batch_nr, batch in enumerate(batches, start=1)]

    with ProcessPoolExecutor(cpu_thread_cnt) as executor:
        executor.map(process_batch, jobs, chunksize=1)  # chunksize=1 is important, since our jobs are long running

    return out_path


def main():
    # The pkl_file is the result of the eval script. Multiple pkl files are written by the eval script.
    # Select the one for which you want a qualitative analysis.
    pkl_file = './results/ignore=True_difficulty=reasonable.pkl'  # edit

    # The folders which contains the ecp dataset (especially the "day" and "night" folders).
    data_path = './data'  # edit

    # Place to start looking for results.
    out_path = './qualitative_eval/'  # edit

    # The number of cores (or threads, if hyperthreading) your cpu has.
    cpu_thread_cnt = 24  # edit

    logging.info('----- START -----')
    start = time.time()

    out_path = run(out_path, pkl_file, data_path, cpu_thread_cnt)

    end = time.time()
    elapsed = int(end - start)
    logging.info('----- FINISHED in {:02d}:{:02d}:{:02d} -----'.format(elapsed // 3600,
                                                                       (elapsed // 60) % 60,
                                                                       elapsed % 60))
    logging.info('Results are located in {}'.format(out_path))


if __name__ == '__main__':
    main()
