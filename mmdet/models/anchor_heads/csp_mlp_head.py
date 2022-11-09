import torch
import torch.nn as nn

from ..registry import HEADS
from ..utils import Scale, window_reverse, window_partition, MixerBlock
from .csp_head import CSPHead
import numpy as np

INF = 1e8


@HEADS.register_module
class CSPMLPHead(CSPHead):

    def __init__(self, *args, patch_dim=4, width=2048, height=1024, windowed_input=True, predict_aspect_ratio=False, predict_width=False, **kwargs):
        self.patch_dim = patch_dim

        super(CSPMLPHead, self).__init__(*args, **kwargs, predict_width=predict_width, predict_aspect_ratio=predict_aspect_ratio)
        self.windowed_input = windowed_input
        self.width = width/4
        self.height = height/4
        self.predict_width = predict_width
        self.predict_aspect_ratio = predict_aspect_ratio
        print("Predict Width: ", self.predict_width, " or Aspect Ratio: ", self.predict_aspect_ratio)

    def _init_layers(self):
        self.mlp_with_feat_reduced = nn.Sequential(
            MixerBlock(self.patch_dim**2, self.in_channels),
            nn.Linear(self.in_channels, self.feat_channels)
        )

        self.pos_mlp = nn.Sequential(
            MixerBlock(self.patch_dim**2, self.feat_channels),
            nn.Linear(self.feat_channels, 1),
        )

        if self.predict_width or self.predict_aspect_ratio:
            self.reg_mlp = nn.Sequential(
                MixerBlock(self.patch_dim**2, self.feat_channels),
                nn.Linear(self.feat_channels, 2) #Predict width and height
            )
        else:
            self.reg_mlp = nn.Sequential(
                MixerBlock(self.patch_dim**2, self.feat_channels),
                nn.Linear(self.feat_channels, 1) #Predict only height
            )

        self.off_mlp = nn.Sequential(
            MixerBlock(self.patch_dim**2, self.feat_channels),
            nn.Linear(self.feat_channels, 2)
        )

    def init_weights(self):
        self.reg_scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.offset_scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward_single(self, x, reg_scale, offset_scale):
        if not self.windowed_input:
            windows = window_partition(x, self.patch_dim, channel_last=False)
        else:
            windows = x
        feat = self.mlp_with_feat_reduced(windows)

        x_cls = self.pos_mlp(feat)
        x_reg = self.reg_mlp(feat)
        x_off = self.off_mlp(feat)

        h = int(self.height)
        w = int(self.width)

        x_cls = window_reverse(x_cls, self.patch_dim, w, h)
        x_reg = window_reverse(x_reg, self.patch_dim, w, h)
        x_off = window_reverse(x_off, self.patch_dim, w, h)

        return x_cls, reg_scale(x_reg).float(), offset_scale(x_off).float()
