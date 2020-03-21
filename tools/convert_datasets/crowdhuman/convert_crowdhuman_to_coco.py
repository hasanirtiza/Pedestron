import os
import json
from PIL import Image


def load_file(fpath):
    assert os.path.exists(fpath)  # assert() raise-if-not
    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]  # str to list
    return records



def crowdhuman2coco(odgt_path, json_path):
    records = load_file(odgt_path)
    json_dict = {"images": [], "annotations": [], "categories": []}  
    START_B_BOX_ID = 1  
    image_id = 1  
    bbox_id = START_B_BOX_ID
    image = {}  
    annotation = {}  
    categories = {}  
    record_list = len(records)  
    print(record_list)
    
    for i in range(record_list):
        file_name = records[i]['ID'] + '.jpg'  
        
        im = Image.open("../Images/" + file_name)
        image = {'file_name': file_name, 'height': im.size[1], 'width': im.size[0],
                 'id': image_id}  
        json_dict['images'].append(image)  

        gt_box = records[i]['gtboxes']
        gt_box_len = len(gt_box)  
        for j in range(gt_box_len):
            category = gt_box[j]['tag']
            if category not in categories:  
                new_id = len(categories) + 1  
                categories[category] = new_id
            category_id = categories[category]  
            fbox = gt_box[j]['fbox']  
            ignore = 0  
            if "ignore" in gt_box[j]['head_attr']:
                ignore = gt_box[j]['head_attr']['ignore']
            if "ignore" in gt_box[j]['extra']:
                ignore = gt_box[j]['extra']['ignore']
            
            annotation = {'area': fbox[2] * fbox[3], 'iscrowd': ignore, 'image_id':  
                image_id, 'bbox': fbox, 'hbox': gt_box[j]['hbox'], 'vbox': gt_box[j]['vbox'],
                          'category_id': category_id, 'id': bbox_id, 'ignore': ignore}
            json_dict['annotations'].append(annotation)

            bbox_id += 1  
        image_id += 1  
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == '__main__':
    odgt_path = 'datasets/CrowdHuman/annotations/annotation_train.odgt'
    json_path = 'datasets/CrowdHuman/annotations/train.json'

    crowdhuman2coco(odgt_path, json_path)
