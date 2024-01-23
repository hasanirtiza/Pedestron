from torch import nn


def window_partition(x, window_size, channel_last=True):
    """
    Args:
        x: (B, W, H, C)
        window_size (int): window size
    Returns:
        windows: (B, num_windows, window_size * window_size, C)
        :param channel_last: if channel is last dim
    """
    if not channel_last:
        x = x.permute(0, 2, 3, 1)
    B, W, H, C = x.shape
    x = x.view(B, W // window_size, window_size, H // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, W, H):
    """
    Args:
        windows: (B, num_windows, window_size*window_size, C)
        window_size (int): Window size
        W (int): Width of image
        H (int): Height of image
    Returns:
        x: (B, C, W, H)
    """
    B = windows.shape[0]
    x = windows.view(B, W // window_size, H // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, W, H)
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
        self.token_mix = MlpBlock(num_tokens, num_tokens * 2)
        self.channel_mix = MlpBlock(num_channels, num_channels * 2)

    def forward(self, x):
        if self.use_ln:
            out = self.ln_token(x)
        else:
            out = x
        out = out.transpose(-1, -2)
        x = x + self.token_mix(out).transpose(-1, -2)
        if self.use_ln:
            out = self.ln_channel(x)
        else:
            out = x
        x = x + self.channel_mix(out)
        return x

class MLP(nn.Module):
    def __init__(self,
                 embedding_dim_in,
                 hidden_dim=None,
                 embedding_dim_out=None,
                 activation=nn.GELU):
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim_in
        embedding_dim_out = embedding_dim_out or embedding_dim_in
        self.fc1 = nn.Linear(embedding_dim_in, hidden_dim)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim, embedding_dim_out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
