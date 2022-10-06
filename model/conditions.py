import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils import weight_norm
from util import WNConv2d, concat_elu

resnet18 = models.resnet18()


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))

        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ConditionalAugmentation(nn.Module):
    def __init__(self, cond_size):
        super(ConditionalAugmentation, self).__init__()
        self.cond_size = cond_size
        self.conv1 = nn.Sequential(
            nn.MaxPool2d(3, 2), ConvBNReLU(3, self.cond_size, 3, 2))
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(3, 2), ConvBNReLU(self.cond_size, self.cond_size, 3, 2))
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(3, 2), ConvBNReLU(self.cond_size, self.cond_size, 3, 2))
        self.fc = nn.Linear(64*19, 19)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.conv3(x)
        x = self.fc(x.reshape(-1, self.cond_size*64))

        return x


class FlowppCond(nn.Module):
    def __init__(self, in_channels, num_channels, num_blocks, cond_size, drop_prob, use_attn=True, aux_channels=None):
        super(FlowppCond, self).__init__()
        self.cond_size = cond_size
        self.in_conv = WNConv2d(in_channels, num_channels, kernel_size=3, padding=1)
        self.mid_convs = nn.ModuleList([ConvAttnBlock(num_channels, drop_prob, use_attn, aux_channels)
                                        for _ in range(num_blocks)])
        self.out_conv = WNConv2d(num_channels, in_channels,
                                 kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*32*3, 1024*3)
        self.fc2 = nn.Linear(1024*3, 2*cond_size)

    def forward(self, x, aux=None):
        x = self.in_conv(x)
        for conv in self.mid_convs:
            x = conv(x, aux)
        x = self.out_conv(x)
        x = self.fc1(x.reshape(-1, 32*32*3))
        x = self.fc2(x)
        self.mu = x[:, :self.cond_size]
        self.logvar = x[:, self.cond_size:]
        eps = torch.normal(0., 1., size=self.logvar.shape).to(self.mu.device)

        return self.mu + torch.exp(self.logvar) * eps, self.mu, self.logvar


class ConvAttnBlock(nn.Module):
    def __init__(self, num_channels, drop_prob, use_attn, aux_channels):
        super(ConvAttnBlock, self).__init__()
        self.conv = GatedConv(num_channels, drop_prob, aux_channels)
        self.norm_1 = nn.LayerNorm(num_channels)
        if use_attn:
            self.attn = GatedAttn(num_channels, drop_prob=drop_prob)
            self.norm_2 = nn.LayerNorm(num_channels)
        else:
            self.attn = None

    def forward(self, x, aux=None):
        x = self.conv(x, aux) + x
        x = x.permute(0, 2, 3, 1)  # (b, h, w, c)
        x = self.norm_1(x)

        if self.attn:
            x = self.attn(x) + x
            x = self.norm_2(x)
        x = x.permute(0, 3, 1, 2)  # (b, c, h, w)

        return x


class GatedAttn(nn.Module):
    """Gated Multi-Head Self-Attention Block

    Based on the paper:
    "Attention Is All You Need"
    by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
        Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
    (https://arxiv.org/abs/1706.03762).

    Args:
        d_model (int): Number of channels in the input.
        num_heads (int): Number of attention heads.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, d_model, num_heads=4, drop_prob=0.):
        super(GatedAttn, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.in_proj = weight_norm(nn.Linear(d_model, 3 * d_model, bias=False))
        self.gate = weight_norm(nn.Linear(d_model, 2 * d_model))

    def forward(self, x):
        # Flatten and encode position
        b, h, w, c = x.size()
        x = x.view(b, h * w, c)
        _, seq_len, num_channels = x.size()
        pos_encoding = self.get_pos_enc(seq_len, num_channels, x.device)
        x = x + pos_encoding

        # Compute q, k, v
        memory, query = torch.split(self.in_proj(x), (2 * c, c), dim=-1)
        q = self.split_last_dim(query, self.num_heads)
        k, v = [self.split_last_dim(tensor, self.num_heads)
                for tensor in torch.split(memory, self.d_model, dim=2)]

        # Compute attention and reshape
        key_depth_per_head = self.d_model // self.num_heads
        q *= key_depth_per_head ** -0.5
        x = self.dot_product_attention(q, k, v)
        x = self.combine_last_two_dim(x.permute(0, 2, 1, 3))
        x = x.transpose(1, 2).view(b, c, h, w).permute(0, 2, 3, 1)  # (b, h, w, c)

        x = self.gate(x)
        a, b = x.chunk(2, dim=-1)
        x = a * torch.sigmoid(b)

        return x

    def dot_product_attention(self, q, k, v, bias=False):
        """Dot-product attention.

        Args:
            q (torch.Tensor): Queries of shape (batch, heads, length_q, depth_k)
            k (torch.Tensor): Keys of shape (batch, heads, length_kv, depth_k)
            v (torch.Tensor): Values of shape (batch, heads, length_kv, depth_v)
            bias (bool): Use bias for attention.

        Returns:
            attn (torch.Tensor): Output of attention mechanism.
        """
        weights = torch.matmul(q, k.permute(0, 1, 3, 2))
        if bias:
            weights += self.bias
        weights = F.softmax(weights, dim=-1)
        weights = F.dropout(weights, self.drop_prob, self.training)
        attn = torch.matmul(weights, v)

        return attn

    @staticmethod
    def split_last_dim(x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
            x (torch.Tensor): Tensor with shape (..., m)
            n (int): Size of second-to-last dimension.
        Returns:
            ret (torch.Tensor): Resulting tensor with shape (..., n, m/n)
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)

        return ret.permute(0, 2, 1, 3)

    @staticmethod
    def combine_last_two_dim(x):
        """Merge the last two dimensions of `x`.

        Args:
            x (torch.Tensor): Tensor with shape (..., m, n)

        Returns:
            ret (torch.Tensor): Resulting tensor with shape (..., m * n)
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)

        return ret

    @staticmethod
    def get_pos_enc(seq_len, num_channels, device):
        position = torch.arange(seq_len, dtype=torch.float32, device=device)
        num_timescales = num_channels // 2
        log_timescale_increment = math.log(10000.) / (num_timescales - 1)
        inv_timescales = torch.arange(num_timescales,
                                      dtype=torch.float32,
                                      device=device)
        inv_timescales *= -log_timescale_increment
        inv_timescales = inv_timescales.exp_()
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        encoding = torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)
        encoding = F.pad(encoding, [0, num_channels % 2, 0, 0])
        encoding = encoding.view(1, seq_len, num_channels)

        return encoding


class GatedConv(nn.Module):
    """Gated Convolution Block

    Originally used by PixelCNN++ (https://arxiv.org/pdf/1701.05517).

    Args:
        num_channels (int): Number of channels in hidden activations.
        drop_prob (float): Dropout probability.
        aux_channels (int): Number of channels in optional auxiliary input.
    """
    def __init__(self, num_channels, drop_prob=0., aux_channels=None):
        super(GatedConv, self).__init__()
        self.nlin = concat_elu
        self.conv = WNConv2d(2 * num_channels, num_channels, kernel_size=3, padding=1)
        self.drop = nn.Dropout2d(drop_prob)
        self.gate = WNConv2d(2 * num_channels, 2 * num_channels, kernel_size=1, padding=0)
        if aux_channels is not None:
            self.aux_conv = WNConv2d(2 * aux_channels, num_channels, kernel_size=1, padding=0)
        else:
            self.aux_conv = None

    def forward(self, x, aux=None):
        x = self.nlin(x)
        x = self.conv(x)
        if aux is not None:
            aux = self.nlin(aux)
            x = x + self.aux_conv(aux)
        x = self.nlin(x)
        x = self.drop(x)
        x = self.gate(x)
        a, b = x.chunk(2, dim=1)
        x = a * torch.sigmoid(b)

        return x


class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x


