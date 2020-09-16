import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import multi_apply, multiclass_nms, csp_height2bbox, csp_heightwidth2bbox, force_fp32
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob, Scale, ConvModule

import cv2
import numpy as np

INF = 1e8


@HEADS.register_module
class CSPHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_offset=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 predict_width=False):
        super(CSPHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = cls_pos()
        if not predict_width:
            self.loss_bbox = reg_pos()
        else:
            self.loss_bbox = reg_hw_pos()
        self.loss_offset = offset_pos()
        self.loss_cls_weight = loss_cls.loss_weight
        self.loss_bbox_weight = loss_bbox.loss_weight
        self.loss_offset_weight = loss_offset.loss_weight
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.predict_width = predict_width

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.offset_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.offset_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None)
            )
        self.csp_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        if not self.predict_width:
            self.csp_reg = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        else:
            self.csp_reg = nn.Conv2d(self.feat_channels, 2, 3, padding=1)
        self.csp_offset = nn.Conv2d(self.feat_channels, 2, 3, padding=1)

        self.reg_scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.offset_scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.offset_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.csp_cls, std=0.01, bias=bias_cls)
        normal_init(self.csp_reg, std=0.01)
        normal_init(self.csp_offset, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.reg_scales, self.offset_scales)

    def forward_single(self, x, reg_scale, offset_scale):
        cls_feat = x
        reg_feat = x
        offset_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.csp_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = reg_scale(self.csp_reg(reg_feat)).float()

        for offset_layer in self.offset_convs:
            offset_feat = offset_layer(offset_feat)
        offset_pred = offset_scale(self.csp_offset(offset_feat).float())
        return cls_score, bbox_pred, offset_pred

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'offset_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             offset_preds,
             gt_bboxes,
             gt_labels,
             classification_maps,
             scale_maps,
             offset_maps,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(offset_preds)
        cls_maps = self.concat_batch_gts(classification_maps)
        bbox_gts = self.concat_batch_gts(scale_maps)
        offset_gts = self.concat_batch_gts(offset_maps)

        loss_cls = []
        for cls_score, cls_gt in zip(cls_scores, cls_maps):
            loss_cls.append(self.loss_cls(cls_score, cls_gt))
        # loss_cls = torch.stack(loss_cls)
        # loss_cls = loss_cls.mean() * self.loss_cls_weight
        loss_cls = loss_cls[0] * self.loss_cls_weight

        loss_bbox = []
        for bbox_pred, bbox_gt in zip(bbox_preds, bbox_gts):
            loss_bbox.append(self.loss_bbox(bbox_pred, bbox_gt))
        # loss_bbox = torch.stack(loss_bbox).mean() * self.loss_bbox_weight
        loss_bbox = loss_bbox[0] * self.loss_bbox_weight

        loss_offset = []
        for offset_pred, offset_map in zip(offset_preds, offset_gts):
            loss_offset.append(self.loss_offset(offset_pred, offset_map))
        # loss_offset = torch.stack(loss_offset).mean() * self.loss_offset_weight
        loss_offset = loss_offset[0] * self.loss_offset_weight
        # print(loss_cls, loss_bbox, loss_offset)
        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_offset=loss_offset)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'offset_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   offset_preds,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            offset_pred_list = [
                offset_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                                offset_pred_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          offset_preds,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, offset_pred, points, stride in zip(
                cls_scores, bbox_preds, offset_preds, mlvl_points, self.strides):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            # self.show_debug_info(cls_score, bbox_pred, offset_pred, stride)
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            if not self.predict_width:
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 1).exp()
            else:
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 2).exp()
            offset_pred = offset_pred.permute(1,2,0).reshape(-1, 2)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                offset_pred = offset_pred[topk_inds, :]
            # bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            if not self.predict_width:
                bboxes = csp_height2bbox(points, bbox_pred, offset_pred, stride=stride, max_shape=img_shape)
            else:
                bboxes = csp_heightwidth2bbox(points, bbox_pred, offset_pred, stride=stride, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img)
        # center_x = (det_bboxes[:, 0] + det_bboxes[:, 2])/2
        # center_y = (det_bboxes[:, 1] + det_bboxes[:, 3])/2
        # print(torch.stack([center_y, center_x], -1), 'center ')
        return det_bboxes, det_labels

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def concat_batch_gts(self, scale_maps):
        bbox_gts = []
        for i in range(len(scale_maps[0])):
            bbox_gts.append([])
            for j in range(len(scale_maps)):
                bbox_gts[-1].append(scale_maps[j][i])
        for i in range(len(bbox_gts)):
            bbox_gts[i] = torch.cat(bbox_gts[i], 0)

        return bbox_gts

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

class cls_pos(nn.Module):
    def __init__(self):
        super(cls_pos, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pos_pred, pos_label):  # 0-gauss 1-mask 2-center
        log_loss = self.bce(pos_pred[:, 0, :, :], pos_label[:, 2, :, :])
        pos_pred = pos_pred.sigmoid()
        # pred_numpy = pos_pred[0, 0, :, :].cpu().detach().numpy()
        # label_numpy = pos_label[0, 2, :, :].cpu().detach().numpy()
        # cv2.imshow('pred', pred_numpy)
        # cv2.imshow('label', label_numpy)

        positives = pos_label[:, 2, :, :]
        negatives = pos_label[:, 1, :, :] - pos_label[:, 2, :, :]

        fore_weight = positives * (1.0-pos_pred[:, 0, :, :]) ** 2
        back_weight = negatives * ((1.0-pos_label[:, 0, :, :])**4.0) * (pos_pred[:, 0, :, :]**2.0)
        focal_weight = fore_weight + back_weight

        # weight_numpy = focal_weight.cpu().detach().numpy()[0]
        # cv2.imshow('weight', weight_numpy)
        # pos_numpy = positives.cpu().detach().numpy()[0]
        # cv2.imshow('pos', pos_numpy)
        # neg_numpy = negatives.cpu().detach().numpy()[0]
        # cv2.imshow('neg', neg_numpy)
        # cv2.waitKey(0)

        assigned_box = torch.sum(pos_label[:, 2, :, :])

        cls_loss = torch.sum(focal_weight*log_loss) / max(1.0, assigned_box)

        return cls_loss


class reg_pos(nn.Module):
    def __init__(self):
        super(reg_pos, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, h_pred, h_label):
        l1_loss = h_label[:, 1, :, :]*self.smoothl1(h_pred[:, 0, :, :]/(h_label[:, 0, :, :]+1e-10),
                                                    h_label[:, 0, :, :]/(h_label[:, 0, :, :]+1e-10))
        # pos_points = h_label[:,1,:,:].reshape(-1).nonzero()
        # if pos_points.shape[0] != 0:
        #     print(h_pred[:, 0,:,:].reshape(-1)[pos_points])
        #     print(h_label[:,0,:,:].reshape(-1)[pos_points])
        reg_loss = torch.sum(l1_loss) / max(1.0, torch.sum(h_label[:, 1, :, :]))
        return reg_loss

class reg_hw_pos(nn.Module):
    def __init__(self):
        super(reg_hw_pos, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, h_pred, h_label):
        l1_loss = h_label[:, 2, :, :]*self.smoothl1(h_pred[:, 0, :, :]/(h_label[:, 0, :, :]+1e-10),
                                                    h_label[:, 0, :, :]/(h_label[:, 0, :, :]+1e-10))
        l1_loss = l1_loss + h_label[:, 2, :, :]*self.smoothl1(h_pred[:, 1, :, :]/(h_label[:, 1, :, :]+1e-10),
                                                    h_label[:, 1, :, :]/(h_label[:, 1, :, :]+1e-10))
        # pos_points = h_label[:,1,:,:].reshape(-1).nonzero()
        # if pos_points.shape[0] != 0:
        #     print(h_pred[:, 0,:,:].reshape(-1)[pos_points])
        #     print(h_label[:,0,:,:].reshape(-1)[pos_points])
        reg_loss = torch.sum(l1_loss) / max(1.0, torch.sum(h_label[:, 2, :, :])*2)
        return reg_loss


class offset_pos(nn.Module):
    def __init__(self):
        super(offset_pos, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, offset_pred, offset_label):
        l1_loss = offset_label[:, 2, :, :].unsqueeze(dim=1)*self.smoothl1(offset_pred, offset_label[:, :2, :, :])

        # print(offset_label[:, :2, :, :].mean())
        # pos_points = offset_label[:, 2, :, :].reshape([-1]).nonzero()
        # if pos_points.shape[0] != 0:
        #     # print(offset_pred.permute([0, 2, 3, 1]).reshape([-1,2]).shape)
        #     print(offset_pred.permute([0,2,3,1]).reshape([-1,2])[pos_points])
        #     print(offset_label[:,:2,:,:].permute([0,2,3,1]).reshape([-1,2])[pos_points])

        off_loss = torch.sum(l1_loss) / max(1.0, torch.sum(offset_label[:, 2, :, :]))
        return off_loss
