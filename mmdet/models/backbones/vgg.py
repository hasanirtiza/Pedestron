import logging

import torch.nn as nn
from mmcv.cnn import (VGG, constant_init, kaiming_init,
                      normal_init)

from mmcv.runner import load_checkpoint
from ..registry import BACKBONES


@BACKBONES.register_module
class VGG(VGG):
    def __init__(self,
                 depth=16,
                 with_last_pool=False,
                 ceil_mode=True,
                 frozen_stages=-1,
                 ):
        super(VGG, self).__init__(
            depth,
            with_last_pool=with_last_pool,
            ceil_mode=ceil_mode,
            frozen_stages=frozen_stages,
            )

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.features.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        #  remove the pool4
        for layer in self.features[:23]:
            x = layer(x)
        for layer in self.features[24:]:
            x = layer(x)
        return tuple([x])


