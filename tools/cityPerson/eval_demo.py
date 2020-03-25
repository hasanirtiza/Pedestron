import os
from tools.cityPerson.coco import COCO
from tools.cityPerson.eval_MR_multisetup import COCOeval


def validate(annFile, dt_path):
    mean_MR = []
    my_id_setup = []
    for id_setup in range(0, 4):
        cocoGt = COCO(annFile)
        cocoDt = cocoGt.loadRes(dt_path)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()
        mean_MR.append(cocoEval.summarize_nofile(id_setup))
        my_id_setup.append(id_setup)
    return mean_MR
