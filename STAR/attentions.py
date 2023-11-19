import math

import torch
import torch.nn as nn
import torch.nn.functional as fn
from einops import rearrange
# from fast_transformers.feature_maps import elu_feature_map
from torch.nn import Linear
from torch_scatter import scatter_sum, scatter_mean

from linalg import softmax_, spmm_


class SparseAttention(nn.Module):

    def __init__(self,
                 in_channels,
                 softmax_temp=None,
                 # num_adj=1,
                 attention_dropout=0.1):
        super(SparseAttention, self).__init__()
        self.in_channels = in_channels
        self.softmax_temp = softmax_temp
        self.dropout = attention_dropout

    def forward(self, queries, keys, values, adj, pe=None):
        # Extract some shapes and compute the temperature
        _, l, _, e = queries.shape  # batch, n_heads, length, depth

        softmax_temp = self.softmax_temp or 1. / math.sqrt(e)

        # Compute the un-normalized sparse attention according to adjacency matrix indices
        """
        这里有一个需要理解的地方
        为什么q与k矩阵要取对应的下标元素 -》 为了减少计算
        一般情况下，通过input得到qkv，这种qkv是所有节点对所有节点的所有的注意力，如果这句话放到骨架系统中就会出现这样的一个情况，头连着脚，这肯定不科学的。
        在注意力网络的骨架系统中，我们让注意力集中关注骨骼的变化，所以adj就是需要的图，表示起始节点与终止节点

        在ST—TR中这方面的注意力网络计算是直接把权重分成了三个部分（沿用了ST-GCN的思想），但是直接用了一个NxN的矩阵来表示
        这里是做到了极致，直接用了adj中有两行，一行是起始节点，一行是终止节点，一一对应
        """
        if isinstance(adj, torch.Tensor):
            qk = torch.sum(queries.index_select(dim=-3, index=adj[0]) *
                           keys.index_select(dim=-3, index=adj[1]), dim=-1)
        else:
            raise NotImplemented("not implemented yet for non-tensor.")

        # Compute the attention and the weighted average, adj[0] is cols idx in the same row
        alpha = fn.dropout(softmax_(softmax_temp * qk, adj[0], kernel=pe),
                           p=self.dropout,
                           training=self.training)
        # sparse matmul, adj as indices and qk as nonzero
        v = spmm_(adj, alpha, l, l, values)
        # Make sure that what we return is contiguous
        return v.contiguous()


class LinearAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 softmax_temp=None,
                 feature_map=None,
                 eps=1e-6,
                 attention_dropout=0.1):
        super(LinearAttention, self).__init__()
        self.in_channels = in_channels
        self.softmax_temp = softmax_temp
        self.dropout = attention_dropout
        self.eps = eps

    def forward(self, queries, keys, values, bi=None):
        n, l, h, e = queries.shape  # batch, n_heads, length, depth
        softmax_temp = self.softmax_temp or (e ** -0.25)  # TODO: how to use this?
        (queries, keys) = map(lambda x: x * softmax_temp, (queries, keys))
        q = fn.elu(queries, inplace=True) + 1.  # self.feature_map.forward_queries(queries)
        k = fn.elu(keys, inplace=True) + 1.  # self.feature_map.forward_keys(keys)
        """
        线性注意力是一个近似softmax的注意力，在性能上做了优化
        这里的分段主要是利用frame indices in a batch来实现，所以如果bi为None则使用的是线性注意力
        """
        if bi is None: #线性注意力
            kv = torch.einsum("nshd, nshm -> nhmd", k, values)
            z = 1 / (torch.einsum("nlhd, nhd -> nlh", q, k.sum(dim=1)) + self.eps)
            v = torch.einsum("nlhd, nhmd, nlh -> nlhm", q, kv, z)
        else: # 分段线性注意力
            # change the dimensions of values to (N, H, L, 1, D) and keys to (N, H, L, D, 1)
            q = rearrange(q, 'n l h d -> n h l d')
            k = rearrange(k, 'n l h d -> n h l d')
            kv = torch.matmul(rearrange(k, 'n h l d -> n h l d 1'),
                              rearrange(values, 'n l h d -> n h l 1 d'))  # N H L D1 D2
            kv = scatter_sum(kv, bi, dim=-3).index_select(dim=-3, index=bi)  # N H (L) D1 D2
            k_ = scatter_sum(k, bi, dim=-2).index_select(dim=-2, index=bi)
            z = 1 / torch.sum(q * k_, dim=-1)
            v = torch.matmul(rearrange(q, 'n h l d -> n h l 1 d'),
                             kv).squeeze(dim=-2) * z.unsqueeze(-1)
        return rearrange(v, 'n h l d -> n l h d').contiguous()


class GlobalContextAttention(nn.Module):
    def __init__(self, in_channels):
        super(GlobalContextAttention, self).__init__()
        self.in_channels = in_channels
        self.weights = nn.Parameter(torch.FloatTensor(in_channels, in_channels))
        nn.init.xavier_normal_(self.weights)

    def forward(self, x, batch_index):
        # Global context
        gc = torch.matmul(scatter_mean(x, batch_index, dim=1), self.weights)
        gc = torch.tanh(gc)[..., batch_index, :]  # extended according to batch index
        gc_ = torch.sigmoid(torch.sum(torch.mul(x, gc), dim=-1, keepdim=True))
        return scatter_mean(gc_ * x, index=batch_index, dim=1)


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, beta, dropout, heads, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape, elementwise_affine=True)
        # self.ln = MaskPowerNorm(normalized_shape, group_num=heads, warmup_iters=1671 * 3)
        self.beta = beta
        if self.beta:
            self.lin_beta = Linear(3 * normalized_shape, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        # self.ln.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x, y):
        if self.beta:
            b = self.lin_beta(torch.cat([y, x, y - x], dim=-1))
            b = b.sigmoid()
            return self.ln(b * x + (1 - b) * self.dropout(y))

        return self.ln(self.dropout(y) + x)


class FeedForward(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, in_channels),
            nn.Dropout(dropout)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x):
        return self.net(x)

