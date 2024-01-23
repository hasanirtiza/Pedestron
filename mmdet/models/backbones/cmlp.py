from .convmlp import ConvMLP
from ..registry import BACKBONES
from mmcv.runner import load_checkpoint
import logging


class DetConvMLP(ConvMLP):
    def __init__(self,
                 blocks,
                 dims,
                 mlp_ratios,
                 channels=64,
                 n_conv_blocks=3,
                 *args, **kwargs):
        super(DetConvMLP, self).__init__(
            blocks=blocks,
            dims=dims,
            mlp_ratios=mlp_ratios,
            channels=channels,
            n_conv_blocks=n_conv_blocks,
            classifier_head=False,
            *args, **kwargs)

    def forward(self, x):
        outs = []
        x = self.tokenizer(x)
        outs.append(x)
        x = self.conv_stages(x)
        outs.append(x)
        x = x.permute(0, 2, 3, 1)
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            if i != 1 and len(self.stages) == 3:
                outs.append(x.permute(0, 3, 1, 2).contiguous())
        return tuple(outs)

    def init_weights(self, pretrained):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        #else:
        #    raise TypeError('Invalid backbone pretrained URL/Filename: ', pretrained)


@BACKBONES.register_module
class DetConvMLPSmall(DetConvMLP):
    def __init__(self, *args, **kwargs):
        super(DetConvMLPSmall, self).__init__(
            blocks=[2, 4, 2],
            dims=[128, 256, 512],
            mlp_ratios=[2, 2, 2],
            channels=64,
            n_conv_blocks=2)


@BACKBONES.register_module
class DetConvMLPMedium(DetConvMLP):
    def __init__(self, *args, **kwargs):
        super(DetConvMLPMedium, self).__init__(
            blocks=[3, 6, 3],
            dims=[128, 256, 512],
            mlp_ratios=[3, 3, 3],
            channels=64,
            n_conv_blocks=3)


@BACKBONES.register_module
class DetConvMLPLarge(DetConvMLP):
    def __init__(self, *args, **kwargs):
        super(DetConvMLPLarge, self).__init__(
            blocks=[4, 8, 3],
            dims=[192, 384, 768],
            mlp_ratios=[3, 3, 3],
            channels=96,
            n_conv_blocks=3)

@BACKBONES.register_module
class DetConvMLPHR(DetConvMLP):
    def __init__(self, *args, **kwargs):
        super(DetConvMLPHR, self).__init__(
            blocks=[4, 8, 4],
            dims=[128, 256, 512],
            mlp_ratios=[2, 2, 2],
            channels=64,
            n_conv_blocks=3)
