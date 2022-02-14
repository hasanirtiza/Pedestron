import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from ..utils import bias_init_with_prob, Scale, ConvModule
from .csp_head import CSPHead
from .transformers import *

INF = 1e8


@HEADS.register_module
class CSPTransHead(CSPHead):

    def __init__(self, *args, **kwargs):
        super(CSPTransHead, self).__init__(*args, **kwargs)

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

        self.csp_cls = SwinTransformer(patch_size=1, in_chans=self.feat_channels, embed_dim=1,
                                       depths=[1], num_heads=[1], out_indices=(0,))
        self.csp_reg = SwinTransformer(patch_size=1, in_chans=self.feat_channels, embed_dim=1,
                                       depths=[1], num_heads=[1], out_indices=(0,))

        self.csp_offset = SwinTransformer(patch_size=1, in_chans=self.feat_channels, embed_dim=2,
                                          depths=[1], num_heads=[1], out_indices=(0,))

        self.reg_scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.offset_scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.offset_convs:
            normal_init(m.conv, std=0.01)

        self.csp_cls.init_weights()
        self.csp_reg.init_weights()
        self.csp_offset.init_weights()

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
