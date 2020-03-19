import glob
import json
import os
import re
import zipfile

import numpy as np


def load_data_ecp(gt_path, det_path, gt_ext='.json', det_ext='.json'):
    def get_gt_frame(gt_dict):
        if gt_dict['identity'] == 'frame':
            pass
        elif '@converter' in gt_dict:
            gt_dict = gt_dict['children'][0]['children'][0]
        elif gt_dict['identity'] == 'seqlist':
            gt_dict = gt_dict['children']['children']

        # check if json layout is corrupt
        assert gt_dict['identity'] == 'frame'
        return gt_dict

    def get_det_frame(det_dict):
        if '@converter' in det_dict:
            det_dict = det_dict['children'][0]['children'][0]
        elif 'objects' in det_dict:
            det_dict['children'] = det_dict['objects']
        return det_dict

    def dets_from_zip(det_zip):
        zfile = zipfile.ZipFile(det_zip)
        file_list = zfile.namelist()
        file_list.sort()
        for det_file in file_list:
            if det_file.endswith(det_ext):
                det_dict = json.loads(zfile.read(det_file))
                det_frame_ = get_det_frame(det_dict)
                det_frame_['filename'] = det_file
                yield det_frame_

    if not os.path.isdir(gt_path) and not gt_path.endswith('.dataset'):
        raise IOError('{} is not a directory and not a dataset file.'.format(gt_path))

    if not os.path.isdir(det_path):
        raise IOError('{} is not a directory.'.format(det_path))

    if gt_path.endswith('.dataset'):
        with open(gt_path) as datasetf:
            gt_files = datasetf.readlines()
            gt_files = [f.strip() for f in gt_files if len(f) > 0]
    else:
        gt_files = glob.glob(gt_path + '/*' + gt_ext)
        if not gt_files:
            gt_files = glob.glob(gt_path + '/*/*' + gt_ext)

    gt_files.sort()

    if not gt_files:
        raise ValueError('ERROR: No ground truth files found at given location! ABORT.'
                         'Given path was: {} and gt ext looked for was {}'.format(gt_path, gt_ext))

    det_files = glob.glob(det_path + '/*' + det_ext)
    det_files.sort()

    if not det_files:
        raise ValueError('ERROR: No ground truth files found at given location! ABORT.'
                         'Given path was: {} and gt ext looked for was {}'.format(det_path,
                                                                                  det_ext))

    if len(gt_files) != len(det_files):
        raise ValueError('Number of detection json files {} does not match the number of '
                         'ground truth json files {}.\n'
                         'Please provide for each image in the ground truth set one detection file.'
                         .format(len(det_files), len(gt_files)))

    for gt_file, det_file in zip(gt_files, det_files):
        gt_fn = os.path.basename(gt_file)
        det_fn = os.path.basename(det_file)

        gt_frame_id = re.search('(.*?)' + gt_ext, gt_fn).group(1)
        det_frame_id = re.search('(.*?)' + det_ext, det_fn).group(1)
        if gt_frame_id != det_frame_id:
            raise ValueError('Error: Frame identifiers do not match: "{}" vs. "{}".'
                             'Check number and order of files in'
                             ' ground truth and detection folder. ABORT.'.format(gt_frame_id,
                                                                                 det_frame_id))

        with open(gt_file, 'rb') as f:
            gt = json.load(f)
        gt_frame = get_gt_frame(gt)
        for gt in gt_frame['children']:
            _prepare_ecp_gt(gt)

        with open(det_file, 'rb') as f:
            det = json.load(f)
        det_frame = get_det_frame(det)
        det_frame['filename'] = det_file
        for det in det_frame['children']:
            _prepare_det(det)

        yield {'gt': gt_frame, 'det': det_frame}


def _prepare_ecp_gt(gt):
    def translate_ecp_pose_to_image_coordinates(angle):
        angle = angle + 90.0

        # map to interval [0, 360)
        angle = angle % 360

        if angle > 180:
            # map to interval (-180, 180]
            angle -= 360.0

        return np.deg2rad(angle)

    orient = None
    if gt['identity'] == 'rider':
        if len(gt['children']) > 0:  # vehicle is annotated
            for cgt in gt['children']:
                if cgt['identity'] in ['bicycle', 'buggy', 'motorbike', 'scooter', 'tricycle',
                                       'wheelchair']:
                    orient = cgt.get('Orient', None) or cgt.get('orient', None)
    else:
        orient = gt.get('Orient', None) or gt.get('orient', None)

    if orient:
        gt['orient'] = translate_ecp_pose_to_image_coordinates(orient)
        gt.pop('Orient', None)


def _prepare_det(det):
    if 'score' not in det.keys():
        score = det.get('confidencevalues', [None])[0]
        if score is None:
            raise ValueError('Missing key "score" in detection {}'.format(det))
        det['score'] = score

    orient = det.get('orient', None) or det.get('Orient', None)
    if orient:
        det['orient'] = orient
        det.pop('Orient', None)
