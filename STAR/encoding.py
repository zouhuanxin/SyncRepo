import torch
from torch import nn
from torch_geometric.utils import get_laplacian
from torch_geometric.utils.num_nodes import maybe_num_nodes



class SeqPosEncoding(nn.Module):
    def __init__(self,
                 model_dim: int,
                 use_weight=False):
        """ Sequential Positional Encoding
            This kind of encoding uses the trigonometric functions to
            incorporate the relative position information into the input
            sequence
        :param model_dim (int): the dimension of the token (feature channel length)
        """
        super(SeqPosEncoding, self).__init__()
        self.model_dim = model_dim
        scale = model_dim ** -0.5
        if use_weight:
            self.weight = nn.Parameter(torch.randn(model_dim, model_dim) * scale)
        else:
            self.weight = None

    @staticmethod
    def segment(pos, bi, device):
        offset = torch.zeros(int(max(bi)) + 1).to(device)
        diff = bi[1:] - bi[:-1]
        offset[1:] = torch.nonzero((diff == 1), as_tuple=True)[0]
        return pos - offset[bi]

    def forward(self, x, bi=None) -> torch.Tensor:
        d = self.model_dim
        sequence_length = x.shape[-2]
        pos = torch.arange(sequence_length, dtype=torch.float).to(x.device)
        if bi is not None:
            pos = self.segment(pos, bi, x.device)
        pos = pos.reshape(1, -1, 1).to(x.device)
        dim = torch.arange(d, dtype=torch.float).reshape(1, 1, -1).to(x.device)
        phase = (pos / 1e4) ** (dim / d)
        assert x.shape[-2] == sequence_length and x.shape[-1] == self.model_dim
        if self.weight is None:
            return x + torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))
        return x + torch.matmul(torch.where(dim.long() % 2 == 0,
                                            torch.sin(phase),
                                            torch.cos(phase)),
                                self.weight)


class PositionalEncoding(object):
    def __init__(self, zero_diagonal=False) -> None:
        super(PositionalEncoding, self).__init__()
        self.zero_diagonal = zero_diagonal
        self.cached_pos_enc = None

    def eval(self, edge_index, edge_attr, **kwargs):
        pass

    def apply_to(self, tensor):
        return


class DiffusionEncoding(PositionalEncoding):
    def __init__(self,
                 beta=1.,
                 use_edge_attr=False,
                 normalization=None,
                 zero_diagonal=False) -> None:
        super().__init__(zero_diagonal=zero_diagonal)
        self.beta = beta
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def eval(self, edge_index, edge_attr, num_nodes=None):
        edge_attr = edge_attr if self.use_edge_attr else None
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        edge_index, edge_attr = get_laplacian(edge_index, edge_attr,
                                              normalization=self.normalization,
                                              num_nodes=num_nodes)
        # TODO the second term below seems not correct
        return edge_index, torch.exp(-self.beta * edge_attr)
