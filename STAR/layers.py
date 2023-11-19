import torch
import torch.nn as nn
import torch.nn.functional as fn
from einops import rearrange, reduce
from torch import Tensor
from torch.nn import Parameter, Linear

from attentions import AddNorm, FeedForward, SparseAttention, LinearAttention
from linalg import batched_spmm, batched_transpose, softmax_


class SpatialEncoderLayer(nn.Module):
    def __init__(self,
                 in_channels=6,
                 mdl_channels=64,
                 heads=8,
                 beta=True,
                 dropout=None):
        super(SpatialEncoderLayer, self).__init__()
        self.in_channels = in_channels
        self.mdl_channels = mdl_channels
        self.heads = heads
        self.beta = beta
        if dropout is None:
            self.dropout = [0.5, 0.5, 0.5, 0.5]  # temp_conv, sparse_attention, add_norm, ffn
        else:
            self.dropout = dropout

        self.lin_qkv = Linear(in_channels, mdl_channels * 3, bias=False)  # 获得qkv

        self.multi_head_attn = SparseAttention(in_channels=mdl_channels // heads)  # 计算多头

        self.add_norm_att = AddNorm(self.mdl_channels, False, self.dropout[2], self.heads)
        self.add_norm_ffn = AddNorm(self.mdl_channels, False, self.dropout[2], self.heads)
        self.ffn = FeedForward(self.mdl_channels, self.mdl_channels, self.dropout[3])

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_qkv.reset_parameters()
        self.add_norm_att.reset_parameters()
        self.add_norm_ffn.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, x, adj=None):
        f, n, c = x.shape
        query, key, value = self.lin_qkv(x).chunk(3, dim=-1)

        query = rearrange(query, 'f n (h c) -> f n h c', h=self.heads)
        key = rearrange(key, 'f n(h c) -> f n h c', h=self.heads)
        value = rearrange(value, 'f n (h c) -> f n h c', h=self.heads)

        t = self.multi_head_attn(query, key, value, adj)
        t = rearrange(t, 'f n h c -> f n (h c)', h=self.heads)

        x = self.add_norm_att(x, t)
        x = self.add_norm_ffn(x, self.ffn(x))

        return x


class TemporalEncoderLayer(nn.Module):
    def __init__(self,
                 in_channels=6,
                 mdl_channels=64,
                 heads=8,
                 beta=False,
                 dropout=0.1):
        super(TemporalEncoderLayer, self).__init__()
        self.in_channels = in_channels
        self.mdl_channels = mdl_channels
        self.heads = heads
        self.beta = beta
        if dropout is None:
            self.dropout = [0.5, 0.5, 0.5, 0.5]  # temp_conv, sparse_attention, add_norm, ffn
        else:
            self.dropout = dropout

        self.lin_qkv = Linear(in_channels, mdl_channels * 3, bias=False)

        self.multi_head_attn = LinearAttention(in_channels=mdl_channels // heads,
                                               attention_dropout=self.dropout[0])

        self.add_norm_att = AddNorm(self.mdl_channels, self.beta, self.dropout[2], self.heads)
        self.add_norm_ffn = AddNorm(self.mdl_channels, False, self.dropout[2], self.heads)
        self.ffn = FeedForward(self.mdl_channels, self.mdl_channels, self.dropout[3])

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_qkv.reset_parameters()
        self.add_norm_att.reset_parameters()
        self.add_norm_ffn.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, x, bi=None):
        f, n, c = x.shape

        query, key, value = self.lin_qkv(x).chunk(3, dim=-1)

        query = rearrange(query, 'n f (h c) -> n f h c', h=self.heads)
        key = rearrange(key, 'n f (h c) -> n f h c', h=self.heads)
        value = rearrange(value, 'n f (h c) -> n f h c', h=self.heads)

        t = self.multi_head_attn(query, key, value, bi)
        t = rearrange(t, 'n f h c -> n f (h c)', h=self.heads)

        x = self.add_norm_att(x, t)
        x = self.add_norm_ffn(x, self.ffn(x))

        return x
