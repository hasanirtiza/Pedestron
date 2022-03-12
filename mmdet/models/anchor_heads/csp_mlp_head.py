import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from ..utils import bias_init_with_prob, Scale, ConvModule
from .csp_head import CSPHead
from .transformers import *
from .swin_transformer import SwinTransformer as ST

INF = 1e8

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size* window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B,-1, H, W)
    return x
    
class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, num_channels, use_ln=True):
        super(MixerBlock, self).__init__()
        self.use_ln = use_ln
        if use_ln:
            self.ln_token = nn.LayerNorm(num_channels)
            self.ln_channel = nn.LayerNorm(num_channels)
        self.token_mix = MlpBlock(num_tokens, num_tokens*2)
        self.channel_mix = MlpBlock(num_channels, num_channels*2)

    def forward(self, x):
        if self.use_ln:
            out = self.ln_token(x)
        else:
            out = x
        out = out.transpose(1, 2)
        x = x + self.token_mix(out).transpose(1, 2)
        if self.use_ln:
            out = self.ln_channel(x)
        else:
            out = x
        x = x + self.channel_mix(out)
        return x


@HEADS.register_module
class CSPMLPHead(CSPHead):

    def __init__(self, *args, **kwargs):
        super(CSPMLPHead, self).__init__(*args, **kwargs)

    def _init_layers(self):
        self.mlp_with_feat_reduced = nn.Sequential(
            MixerBlock(16, 64),
            nn.Linear(64, 32)
        )

        self.pos_mlp = nn.Sequential(
            MixerBlock(16,32),
            nn.Linear(32,1),
        )

        self.reg_mlp = nn.Sequential(
            MixerBlock(16,32),
            nn.Linear(32,1)
        )

        self.off_mlp = nn.Sequential(
            MixerBlock(16,32),
            nn.Linear(32,2)
        )

    def init_weights(self):
        #for m in self.cls_convs:
        #    normal_init(m.conv, std=0.01)
        #for m in self.reg_convs:
        #    normal_init(m.conv, std=0.01)
        #for m in self.offset_convs:
        #    normal_init(m.conv, std=0.01)

        #self.csp_cls.init_weights()
        #self.csp_reg.init_weights()
        #self.csp_offset.init_weights()
        #bias_cls = bias_init_with_prob(0.01)
        #normal_init(self.csp_cls, std=0.01, bias=bias_cls)
        #normal_init(self.csp_reg, std=0.01)
        #normal_init(self.csp_offset, std=0.01)
        self.reg_scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.offset_scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward_single(self, x, reg_scale, offset_scale):
        cat_permuted = x.permute(0,2,3,1) #b,h,w,c
        b,h,w,c =cat_permuted.shape

        windows = window_partition(cat_permuted,4) # num_windows*B, window_size, window_size, C
        # windows = windows.permute(0,1,3) #h*w* b, p_s*p_s, c
        
        feat = self.mlp_with_feat_reduced(windows) #768 to 256

        x_cls = self.pos_mlp(feat).transpose(1,2) #b,h*w,1  # We are transposing to match the initial dimensions (n,c,h*w)
        x_reg = self.reg_mlp(feat).transpose(1,2) #b,h*w,1
        x_off = self.off_mlp(feat).transpose(1,2) #b,h*w,2

        x_cls = window_reverse(x_cls,4,h,w)
        x_reg = window_reverse(x_reg,4,h,w)
        x_off = window_reverse(x_off,4,h,w)

        return x_cls, reg_scale(x_reg).float(), offset_scale(x_off).float()
