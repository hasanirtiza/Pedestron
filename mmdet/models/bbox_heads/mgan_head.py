import torch.nn as nn

from ..registry import HEADS
from ..utils import ConvModule
from mmdet.core import auto_fp16


@HEADS.register_module
class MGANHead(nn.Module):

    def __init__(self,
                 num_convs=2,
                 roi_feat_size=7,
                 in_channels=512,
                 conv_out_channels=512,
                 conv_cfg=None,
                 norm_cfg=None):
        super(MGANHead, self).__init__()
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        logits_in_channel = self.conv_out_channels
        self.conv_logits = nn.Conv2d(logits_in_channel, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    @auto_fp16()
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.conv_logits(x).sigmoid() * x
        return x


