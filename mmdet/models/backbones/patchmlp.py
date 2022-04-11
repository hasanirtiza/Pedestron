import logging

import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from mmdet.ops import DeformConv, ModulatedDeformConv, ContextBlock
from mmdet.models.plugins import GeneralizedAttention

from ..registry import BACKBONES
from ..utils import MlpBlock, MixerBlock, window_partition, window_reverse
from ..utils import build_conv_layer, build_norm_layer


class PatchMLPBlock(nn.Module):

    def __init__(self, in_patch, out_patch, in_channel, out_channel, half_mixer_count=2, in_patch_size=32,
                 out_patch_size=32):
        super(PatchMLPBlock, self).__init__()
        self.in_patch = in_patch
        self.out_patch = out_patch
        self.in_feat = in_channel
        self.out_feat = out_channel
        self.half_mixer_count = half_mixer_count
        self.patch_size = in_patch_size
        self.out_patch_size = out_patch_size

        self.channel_stage = nn.Linear(in_channel, out_channel)
        self.patch_stage = nn.Linear(in_patch, out_patch)

        self.mixer_in = nn.Sequential(
            *[MixerBlock(out_patch, out_channel) for i in range(self.half_mixer_count)]
        )

        self.mixer_out = nn.Sequential(
            *[MixerBlock(out_patch, out_channel) for i in range(self.half_mixer_count)]
        )

        self.pad = nn.ZeroPad2d(int(out_patch_size/2))

    def _stage(self, x):

        x = x.transpose(-2, -1)
        x = self.patch_stage(x)
        x = x.transpose(-2, -1)
        x = self.channel_stage(x)

        return x

    def un_pad(self, x):
        padding = int(self.patch_size/2)
        x = x[:, :, padding:-padding, padding:-padding]
        return x

    def forward(self, x):

        H, W = x.shape[2:]
        x = window_partition(x, self.patch_size, channel_last=False)
        x = self._stage(x)
        x = self.mixer_in(x)
        x = window_reverse(x, self.out_patch_size, H, W)

        x = self.pad(x)
        H, W = x.shape[2:]
        x = window_partition(x, self.out_patch_size, channel_last=False)
        x = self.mixer_out(x)
        x = window_reverse(x, self.out_patch_size, H, W)
        out = self.un_pad(x)

        return out


class PatchMLPStage(nn.Module):

    def __init__(self, block_count, patch_size, downscale, in_channel, out_channel):
        super(PatchMLPStage, self).__init__()

        self.block_count = block_count
        self.patch_size = patch_size
        self.downscale = downscale

        pz = int(patch_size / downscale)

        self.in_patch = int(self.patch_size**2)
        self.out_patch = int(pz**2)
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.blocks = nn.Sequential(
            PatchMLPBlock(
                self.in_patch, self.out_patch, self.in_channel, self.out_channel, in_patch_size=patch_size,
                out_patch_size=pz
            ),
            *[
                PatchMLPBlock(self.out_patch, self.out_patch, self.out_channel, self.out_channel,
                              in_patch_size=patch_size, out_patch_size=pz)
                for i in range(self.block_count - 1)
            ]
        )

    def forward(self, x):

        out = self.blocks(x)

        return out


@BACKBONES.register_module
class PatchMLP(nn.Module):

    def __init__(self,
                 num_stages=3,
                 out_indices=(0, 1, 2),
                 blocks=[2, 2, 4],
                 channels=[32, 64, 128],
                 patch_size=32,
                 downscales=[4, 2, 2],
                 style='pytorch',
                 frozen_stages=-1,
                 ):
        super(PatchMLP, self).__init__()
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self._freeze_stages()
        self.patch_size = patch_size
        self.downscales = downscales
        self.blocks = blocks
        self.channels = channels

        self.stages = nn.ModuleList()

        _in_channels = 3
        _in_patch_dim = self.patch_size
        for i in range(num_stages):
            block_count = self.blocks[i]
            out_channel = self.channels[i]
            downscale = self.downscales[i]
            _stage = PatchMLPStage(block_count, _in_patch_dim, downscale, _in_channels, out_channel)
            _in_channels = out_channel
            _in_patch_dim = int(_in_patch_dim/downscale)
            self.stages.append(_stage)

    def _freeze_stages(self):
        # TODO:// Implement freezing stages
        pass
        # if self.frozen_stages >= 0:
        #     self.norm1.eval()
        #     for m in [self.conv1, self.norm1]:
        #         for param in m.parameters():
        #             param.requires_grad = False
        #
        # for i in range(1, self.frozen_stages + 1):
        #     m = getattr(self, 'layer{}'.format(i))
        #     m.eval()
        #     for param in m.parameters():
        #         param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super(PatchMLP, self).train(mode)
        self._freeze_stages()
