import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import NECKS
import numpy as np
from ..utils import window_partition, MixerBlock,  ConvModule


@NECKS.register_module
class MLPFPN(nn.Module):
    """MLPFPN

    Args:
        in_channels (list): number of channels for each branch.
        out_channels (int): output channels of feature pyramids.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 patch_dim=8,
                 start_index=1,
                 start_stage=0,
                 end_stage=4,
                 feat_channels=[8, 16, 128],
                 mixer_count=1,
                 linear_reduction=True):
        super(MLPFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_index = start_index
        self.num_ins = len(in_channels)
        self.mixer_count = mixer_count
        self.patch_dim = patch_dim
        self.start_stage = start_stage
        self.end_stage = end_stage
        self.feat_channels = feat_channels
        self.linear_reduction = linear_reduction
        self.backlinks=[]

        pc = int(np.sum([self.feat_channels[i] * 2**(2*(self.num_ins-1 - i)) for i in range(len(feat_channels))]))
        self.intprL = nn.Linear(pc, (self.patch_dim**2)*self.out_channels)

        self.intpr = nn.ModuleList()
        for i in range(len(self.feat_channels)):
            if self.linear_reduction:
                tokens = 2**(2*(self.num_ins-1 - i))
                self.intpr.append(nn.Linear(self.in_channels[i] * tokens, self.feat_channels[i] * tokens))
            else:
                self.intpr.append(ConvModule(self.in_channels[i], self.feat_channels[i], 1))

        self.mixers = None
        if self.mixer_count > 0:
            self.mixers = nn.Sequential(*[
                MixerBlock(self.patch_dim**2, self.out_channels) for i in range(self.mixer_count)
            ])

    def init_weights(self):
        pass

    def forward(self, inputs):

        B, _, H4, W4 = inputs[0].shape
        self.backlinks[0].bbox_head.width = H4
        self.backlinks[0].bbox_head.height = W4
        parts = []

        for i in range(len(self.feat_channels)):
            if self.linear_reduction:
                part = window_partition(inputs[i], 2**(self.num_ins-1 - i), channel_last=False)
                part = torch.flatten(part, -2)
                part = self.intpr[i](part)
            else:
                part = self.intpr[i](inputs[i])
                part = window_partition(part, 2 ** (self.num_ins - 1 - i), channel_last=False)
                part = torch.flatten(part, -2)
            parts.append(part)

        out = torch.cat(parts, dim=-1)
        out = self.intprL(out)

        B, T, _ = out.shape
        outputs = out.view(B, T, self.patch_dim**2, self.out_channels)

        if self.mixers is not None:
            outputs = self.mixers(outputs)
        return tuple([outputs])
