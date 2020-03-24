import os
import datetime
import json
import numpy as np
import glob
import cv2
from scipy.io import loadmat
from PIL import Image
import pycococreatortools

INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2019,
    "contributor": "ljp",
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
        'name': 'pedestrain',
        'supercategory': 'pedestrain',
    },
]

def convert():
    identity_lst = ['pedestrian', 'bicycle-group', 'person-group-far-away', 'scooter-group', 'co-rider', 'scooter', \
                    'motorbike', 'bicycle', 'rider', 'motorbike-group', 'rider+vehicle-group-far-away', None, \
                    'buggy-group', 'wheelchair-group', 'tricycle-group', 'buggy', 'wheelchair', 'tricycle']
    identity2index = dict(zip(identity_lst, range(0, len(identity_lst))))

    image_id = 1
    annotation_id = 1
    data_path = 'datasets/EuroCity/'  #check the path
    for day in ['day']:
        for phase in ['train', 'val', 'test']:
            coco_output = {
                "info": INFO,
                "licenses": LICENSES,
                "categories": CATEGORIES,
                "images": [],
                "annotations": []
            }
            box_num = 0
            ig_box_num = 0

            fold_path = '%s/ECP/%s/img/%s' % (data_path, day, phase)
            print(fold_path)
            fn_lst = glob.glob('%s/*/*.png' % fold_path)

            for img_name in fn_lst:
                image = Image.open(img_name)
                image_info = pycococreatortools.create_image_info(
                    image_id, img_name[img_name.index('ECP'):], image.size)
                coco_output["images"].append(image_info)

                if phase != 'test':
                    anno_fn = img_name.replace('img', 'labels').replace('png', 'json')

                    anno = json.load(open(anno_fn))

                    boxes = []
                    for each in anno['children']:
                        if len(each["children"]) > 0:
                            # print(each)
                            for each2 in each["children"]:
                                boxes.append([identity2index[each2["identity"]], float(each2['x0']), float(each2['y0']), \
                                              float(each2['x1']) - float(each2['x0']), float(each2['y1']) - float(each2['y0'])])
                                if "occluded>80" in each2['tags']:
                                    boxes[-1].append(1)
                                    # print('heavy occluded')
                                else:
                                    # print('normal')
                                    boxes[-1].append(0)
                        
                        boxes.append([identity2index[each["identity"]], float(each['x0']), float(each['y0']), \
                                        float(each['x1']) - float(each['x0']), float(each['y1']) - float(each['y0'])])
                        if "occluded>80" in each['tags']:
                            boxes[-1].append(1)
                            # print('heavy occluded')
                        else:
                            boxes[-1].append(0)

                    boxes = np.array(boxes)
                    if len(boxes) > 0:
                        # slt = np.logical_and(boxes[:, 0]==0, boxes[:, 4]>20, boxes[:, 5]<1)
                        slt = (boxes[:, 0]==0)
                        boxes_gt = boxes[slt, 1:5].tolist()
                        boxes_ig = boxes[~slt, 1:5].tolist()
                    else:
                        boxes_gt = []
                        boxes_ig = []
                    box_num += len(boxes_gt)
                    ig_box_num += len(boxes_ig)


                    for annotation in boxes_gt:
                        class_id = 1
                        category_info = {'id': class_id, 'is_crowd': False}
                        annotation_info = pycococreatortools.create_annotation_info(
                            annotation_id, image_id, category_info, annotation,
                            image.size)
                        if annotation_info is not None:
                            coco_output["annotations"].append(annotation_info)
                        annotation_id += 1

                    for annotation in boxes_ig:
                        category_info = {'id': 1, 'is_crowd': True}
                        annotation_info = pycococreatortools.create_annotation_info(
                            annotation_id, image_id, category_info, annotation, image.size)
                        if annotation_info is not None:
                            coco_output["annotations"].append(annotation_info)
                        annotation_id += 1

                image_id = image_id + 1

            with open(data_path + day + '_' + phase + '_all.json', 'w') as output_json_file:
                json.dump(coco_output, output_json_file)
            print('box num: ', box_num)
            print('ignore box num: ', ig_box_num)

if __name__ == '__main__':
    convert()
