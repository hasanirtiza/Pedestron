import json
import os
import glob
import numpy as np
import pathos.multiprocessing as multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool

gt_json = 'datasets/Caltech/test.json'
result_dir = 'work_dirs/cascade_ch_wider_plus_sur_ecp_day_caltech/jsons/'
result_jsons = glob.glob(result_dir + '/*.json')
print(result_jsons)
matlab_results_dir = 'tools/caltech/eval_caltech/Pedestron_Result/'
with open(gt_json, 'r') as f:
    gt_json = json.load(f)
image_names = []
image_name_to_id = {}
for image in gt_json['images']:
    image_names.append(image['file_name'])
    image_name_to_id[image['file_name']] = image['id']
image_names.sort()
videos = {}
for image_name in image_names:
    name_items = image_name.split('_')
    if not name_items[0] in videos:
        videos[name_items[0]] = {}
    if not name_items[1] in videos[name_items[0]]:
        videos[name_items[0]][name_items[1]] = []
    videos[name_items[0]][name_items[1]].append(image_name)
for video in videos:
    for video_item in videos[video]:
        videos[video][video_item].sort()

def convert_result(result_json):
    basename = os.path.basename(result_json)
    # basename = basename.split('.')[0]
    # basename = basename.replace('result', '')
    basename = basename.split('.')[0]
    #basename = basename.replace('json', '')
    matlab_result_dir = matlab_results_dir + basename
    if not os.path.exists(matlab_result_dir):
        os.makedirs(matlab_result_dir)
    with open(result_json, 'r') as f:
        result_json = json.load(f)
    image_id_bboxes ={}
    for box in result_json:
        image_id = box['image_id']
        if not image_id in image_id_bboxes:
            image_id_bboxes[image_id] = []
        image_id_bboxes[image_id].append(box)
    for video in videos:
        video_path = os.path.join(matlab_result_dir, video)
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        for video_item in videos[video]:
            video_item_txt = os.path.join(video_path, video_item + '.txt')
            results = []
            for image_name in videos[video][video_item]:
                image_id = image_name_to_id[image_name]
                original_id = int(image_name.split('.')[0].split('_')[-1][1:]) + 1
                if image_id in image_id_bboxes:
                    for box in image_id_bboxes[image_id]:
                        if box['image_id'] != image_id:
                            print('wrong')
                            continue
                        bbox = box['bbox']  # .copy()
                        if len(bbox) != 4:
                            print(box)
                        bbox.insert(0, float(original_id))
                        bbox.append(box['score'])
                        bbox = np.asarray(bbox).reshape(1, -1)
                        if bbox.shape[1] != 6:
                            print(bbox.shape)
                        if len(results) == 0:
                            results = bbox
                        else:
                            results = np.concatenate((results, bbox), axis=0)
            results = np.array(results)
            np.savetxt(video_item_txt, results, fmt='%6f')

p = Pool(multiprocessing.cpu_count())
p.map(convert_result, result_jsons)
