import torch.nn as nn

from .bbox_head import BBoxHead
from ..registry import HEADS
from ..utils import ConvModule
from ..bbox_heads.convfc_bbox_head import ConvFCBBoxHead

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import (delta2bbox, multiclass_nms, bbox_target, force_fp32,
                        auto_fp16)
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS



@HEADS.register_module
class CascadePedFCBBoxHead(ConvFCBBoxHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(CascadePedFCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            bboxes /= scale_factor

        if cfg is None:
            return bboxes, scores
        else:
            values, indices = torch.max(scores, dim=1)
            bboxes[:, 8 + 3] = bboxes[:, 8+1] + (bboxes[:, 8+3] - bboxes[:, 8+1])/0.4
            # print(bboxes[indices==3, 1])
            # print(bboxes[indices==3, 3] - bboxes[indices==3, 1])
            bboxes[:, 12 + 1] = bboxes[:, 12 + 3] - (bboxes[:, 12 + 3] - bboxes[:, 12 + 1])/0.6
            # print(bboxes[indices==3, 1])
            bboxes[indices==2, 4:8] = bboxes[indices==2, 8:12]
            bboxes[indices==3, 4:8] = bboxes[indices==3, 12:16]
            scores[:, 1] = torch.max(scores[:, 1:], dim=1)[0]
            scores[:, 2] = 0
            scores[:, 3] = 0


            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels
