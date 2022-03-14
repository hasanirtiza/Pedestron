import torch
import torch.nn as nn

from ..registry import HEADS
from ..utils import Scale, window_reverse, window_partition, MixerBlock
from .csp_head import CSPHead
import numpy as np

INF = 1e8


@HEADS.register_module
class CSPMLPHead(CSPHead):

    def __init__(self, *args, patch_dim=4, windowed_input=True, **kwargs):
        self.patch_dim = patch_dim
        super(CSPMLPHead, self).__init__(*args, **kwargs)
        self.windowed_input = windowed_input

    def _init_layers(self):
        self.mlp_with_feat_reduced = nn.Sequential(
            MixerBlock(self.patch_dim**2, self.in_channels),
            nn.Linear(self.in_channels, self.feat_channels)
        )

        self.pos_mlp = nn.Sequential(
            MixerBlock(self.patch_dim**2, self.feat_channels),
            nn.Linear(self.feat_channels, 1),
        )

        self.reg_mlp = nn.Sequential(
            MixerBlock(self.patch_dim**2, self.feat_channels),
            nn.Linear(self.feat_channels, 1)
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

        h = int(2**((np.log2(feat.shape[1])-1)/2)) * self.patch_dim
        w = int(h*2)

        x_cls = window_reverse(x_cls, self.patch_dim, h, w)
        x_reg = window_reverse(x_reg, self.patch_dim, h, w)
        x_off = window_reverse(x_off, self.patch_dim, h, w)

        return x_cls, reg_scale(x_reg).float(), offset_scale(x_off).float()
