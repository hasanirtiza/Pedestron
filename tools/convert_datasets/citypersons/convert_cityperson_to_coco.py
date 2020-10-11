import os
import datetime
import json
import numpy as np
import glob
from scipy.io import loadmat
from PIL import Image
import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import pycococreatortools

INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2019,
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'person',
        'supercategory': 'person',
    },
]

def parse_cityperson_mat(file_path, phase):
    '''
    Citypersons .mat annotation operations
    return: a dict, key: imagename, value: an array of n* <lbl, x1 y1 x2 y2>
    '''
    tv = phase
    k = 'anno_{}_aligned'.format(phase)
    bbox_counter = 0
    rawmat = loadmat(file_path, mat_dtype=True)
    mat = rawmat[k][0]  # uint8 overflow fix
    name_bbs_dict = {}
    for img_idx in range(len(mat)):
        # each image
        img_anno = mat[img_idx][0, 0]

        city_name = img_anno[0][0]
        img_name_with_ext = img_anno[1][0]
        bbs = img_anno[2]  # n x 10 matrix
        # 10-D: n* [class_label, x1, y1, w, h, instance_id, x1_vis, y1_vis, w_vis, h_vis]

        name_bbs_dict[img_name_with_ext] = bbs
        bbox_counter += bbs.shape[0]

    img_num = len(mat)  # - noins_img_counter
    print('Parsed {}: {} bboxes in {} images remained ({:.2f} boxes/img) '.format(tv,
                                                                                 bbox_counter,
                                                                                 img_num,
                                                                                 bbox_counter / img_num))
    return name_bbs_dict

def convert(phase, data_path):
    assert phase in ['train', 'val']
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    annotation_id = 1

    bbs_dict = parse_cityperson_mat('%s/anno_%s.mat' % (data_path, phase), phase)

    fn_lst = glob.glob('%s/leftImg8bit/%s/*/*.png' % (data_path, phase))
    positive_box_num = 0
    ignore_box_num = 0
    for image_filename in fn_lst:
        base_name = os.path.basename(image_filename)
	
        if base_name in bbs_dict:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, image_filename[image_filename.index('leftImg8bit'):], image.size)
            coco_output["images"].append(image_info)

            boxes = bbs_dict[base_name]
            # go through each associated annotation
            slt_msk = np.logical_and(boxes[:, 0] == 1, boxes[:, 4] >= 50)
            boxes_gt = boxes[slt_msk, 1:5]
            positive_box_num += boxes_gt.shape[0]
            for annotation in boxes_gt:
                annotation = annotation.tolist()
                class_id = 1
                category_info = {'id': class_id, 'is_crowd': False}
                annotation_info = pycococreatortools.create_annotation_info(
                    annotation_id, image_id, category_info, annotation,
                    image.size)
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
                annotation_id += 1

            slt_msk = np.logical_or(boxes[:, 0] != 1, boxes[:, 4] < 50)
            boxes_ig = boxes[slt_msk, 1:5]
            ignore_box_num += boxes_ig.shape[0]
            for annotation in boxes_ig:
                annotation = annotation.tolist()
                category_info = {'id': 1, 'is_crowd': True}
                annotation_info = pycococreatortools.create_annotation_info(
                    annotation_id, image_id, category_info, annotation, image.size)
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
                annotation_id += 1

        image_id = image_id + 1
    print('positive_box_num: ', positive_box_num)
    print('ignore_box_num: ', ignore_box_num)
    with open(data_path + phase + '.json', 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

if __name__ == '__main__':
    data_path = 'datasets/CityPersons/leftImg8bit_trainvaltest/'
    convert(phase = 'train', data_path=data_path)
    convert(phase='val', data_path=data_path)
