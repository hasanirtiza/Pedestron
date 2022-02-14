import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mmcv.cnn.weight_init import caffe2_xavier_init

from .csp_neck import L2Norm
from ..utils import ConvModule
from ..registry import NECKS


@NECKS.register_module
class HRCSPFPN(nn.Module):
    """HRFPN (High Resolution Feature Pyrmamids)

    arXiv: https://arxiv.org/abs/1904.04514

    Args:
        in_channels (list): number of channels for each branch.
        out_channels (int): output channels of feature pyramids.
        num_outs (int): number of output stages.
        pooling_type (str): pooling for generating feature pyramids
            from {MAX, AVG}.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        with_cp  (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=5,
                 pooling_type='AVG',
                 upscale_factor=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 with_cp=False):
        super(HRCSPFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        if upscale_factor is not None:
            self.upscale_factor = upscale_factor
        else:
            self.upscale_factor = [2**i for i in range(1, self.num_ins)]

        self.l2_norms = nn.ModuleList()
        self.reduction_convs = nn.ModuleList()
        for j in range(len(in_channels)):
            i = in_channels[j]
            self.l2_norms.append(L2Norm(i, 10))

        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_ins):
            self.fpn_convs.append(
                ConvModule(
                    in_channels[i],
                    in_channels[i],
                    kernel_size=3,
                    dilation=([1] + self.upscale_factor)[i],
                    padding=([1] + self.upscale_factor)[i],
                    conv_cfg=self.conv_cfg,
                    activation=None))

        if pooling_type == 'MAX':
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    def forward(self, inputs):
        assert len(inputs) == self.num_ins
        outs = []
        for i in range(self.num_ins):
            feat = inputs[i]
            if i != 0:
                feat = F.interpolate(inputs[i], scale_factor=self.upscale_factor[i-1], mode='bilinear')
            feat = self.l2_norms[i](feat)
            feat = self.fpn_convs[i](feat)
            outs.append(feat)
        out = torch.cat(outs, dim=1)
        outs = [out]
        return tuple(outs)
