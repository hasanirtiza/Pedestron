import os
from tools.cityPerson.coco import COCO
from tools.cityPerson.eval_MR_multisetup import COCOeval


def validate(annFile, dt_path):
    mean_MR = []
    for id_setup in range(0, 4):
        cocoGt = COCO(annFile)
        cocoDt = cocoGt.loadRes(dt_path)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate(id_setup)
        if id_setup==0:
            fpps, scores = cocoEval.accumulate(id_setup=id_setup)
        else:
            cocoEval.accumulate(id_setup=id_setup)
        mean_MR.append(cocoEval.summarize_nofile(id_setup))
    return mean_MR, fpps, scores

if __name__ == "__main__":
    MRs = validate('/home/ljp/code/citypersons/evaluation/val_gt.json', '/home/ljp/code/mmdetection/result_ori_csp.json')
    # MRs = validate('/media/ljp/Data/data/cityscapes/leftImg8bit_trainvaltest/train_evaluation.json', '/home/ljp/code/mmdetection/train_result.json')
    print('Checkpoint %d: [Reasonable: %.2f%%], [Bare: %.2f%%], [Partial: %.2f%%], [Heavy: %.2f%%]'
          % (0, MRs[0] * 100, MRs[1] * 100, MRs[2] * 100, MRs[3] * 100))

