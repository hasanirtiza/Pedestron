import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import NECKS
import numpy as np
from ..utils import window_partition, MixerBlock


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
                 start_index=1,
                 mixer_count=1):
        super(MLPFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_index = start_index
        self.num_ins = len(in_channels)
        self.mixer_count = mixer_count

        pc = int(np.sum([self.in_channels[i] * 2**(2*(self.num_ins-1 - i)) for i in range(self.num_ins)]))
        self.intpr = nn.Linear(pc, self.out_channels)

        self.mixers = None
        if self.mixer_count > 0:
            self.mixers = nn.Sequential([
                MixerBlock(2 ** (2 * (self.num_ins - 1)), self.out_channels) for i in range(self.mixer_count)
            ])

    def forward(self, inputs):
        assert len(inputs) == self.num_ins

        B, H4, W4, _ = inputs[0].shape
        parts = []

        for i in range(self.num_ins):
            part = window_partition(inputs[i], 2**(self.num_ins-1 - i), channel_last=False)
            parts.append(part)

        out = torch.cat(parts, dim=-1)
        out = torch.flatten(out, -2)
        out = self.intpr(out)

        B, T, _ = out.shape
        outputs = out.view(B, T, -1, self.out_channels)

        if self.mixers is not None:
            outputs = self.mixers(outputs)

        return tuple(outputs)
