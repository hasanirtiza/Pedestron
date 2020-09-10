import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule
import cv2

@NECKS.register_module
class CSPNeck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(CSPNeck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        assert self.num_ins == 3
        self.num_outs = num_outs
        self.activation = activation
        self.fp16_enabled = False

        self.p3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.p4 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=4, padding=0)
        self.p5 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=4, padding=0)

        self.p3_l2 = L2Norm(256, 10)
        self.p4_l2 = L2Norm(256, 10)
        self.p5_l2 = L2Norm(256, 10)



    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def normalize(self, x):
        x = x - x.min()
        x = x/x.max()
        return x

    def feature_map_visualization(self, x, y):
        x = x[0].detach().cpu().numpy()
        y = y[0].detach().cpu().numpy()
        first = self.normalize(x[0])
        second = self.normalize(y[0])
        cv2.imshow('1', first)
        cv2.imshow('2', second)
        cv2.waitKey(0)

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        p3 = self.p3(inputs[0])
        # self.feature_map_visualization(inputs[0], p3)
        p3 = self.p3_l2(p3)

        p4 = self.p4(inputs[1])
        p4 = self.p4_l2(p4)

        p5 = self.p5(inputs[2])
        p5 = self.p5_l2(p5)

        cat = torch.cat([p3, p4, p5], dim=1)
        outs=[cat]
        return tuple(outs)

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out