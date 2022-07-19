import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, LSTM, GRU

from torch_geometric.nn import GCNConv, SAGEConv, JumpingKnowledge,SAGPooling
from torch_geometric.nn import GINConv
from pyg_gnn_layer import GeoLayer
# from gin_conv import GINConv2
# from gcn_conv import GCNConv2
from geniepath import GeniePathLayer
from conv import GCNConv as GCNConv_mol
from conv import GINConv as GINConv_mol
from conv import TransformerConv, GINEConv, GATConv, GATv2Conv, MFConv, GENConv, ExpC

NA_OPS = {
    'sage': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'sage'),
    'sage_sum': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'sum'),
    'sage_max': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'max'),
    'gcn': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gcn'),
    'gat': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gat'),
    'gin': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gin'),
    'gat_sym': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gat_sym'),
    'gat_linear': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'linear'),
    'gat_cos': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'cos'),
    'gat_generalized_linear': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'generalized_linear'),
    'geniepath': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'geniepath'),

    # with Edge
    'Transformerconv': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'Transformerconv'),
    'gine': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gine'),
    'gcn_mol': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gcn_mol'),
    'gin_mol': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gin_mol'),
    'gen': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gen'),
    'EXPC': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'EXPC'),
}

SC_OPS={
    'zero': lambda: Zero(),
    'identity': lambda: Identity(),
}

FF_OPS = {
    'sum': lambda hidden_size, num_layers: LaAggregator('sum', hidden_size, num_layers),
    'mean': lambda hidden_size, num_layers: LaAggregator('mean', hidden_size, num_layers),
    'max': lambda hidden_size, num_layers: LaAggregator('max', hidden_size, num_layers),
    'concat': lambda hidden_size, num_layers: LaAggregator('cat', hidden_size, num_layers),
    'lstm': lambda hidden_size, num_layers: LaAggregator('lstm', hidden_size, num_layers),
    'gate': lambda hidden_size, num_layers: LaAggregator('gate', hidden_size, num_layers),
    'att': lambda hidden_size, num_layers: LaAggregator('att', hidden_size, num_layers),
}

class NaAggregator(nn.Module):

    def __init__(self, in_dim, out_dim, aggregator):
        super(NaAggregator, self).__init__()
        #aggregator, K = agg_str.split('_')
        self.aggregator = aggregator
        if 'sage' == aggregator:
            self._op = SAGEConv(in_dim, out_dim, normalize=True)
        if 'gcn' == aggregator:
            self._op = GCNConv(in_dim, out_dim)
        if 'gat' == aggregator:
            heads = 4
            out_dim /= heads
            self._op = GATConv(in_dim, int(out_dim), heads=heads, dropout=0.5, edge_dim=in_dim)
        if aggregator == 'Transformerconv':
            heads = 4
            out_dim /= heads
            self._op = TransformerConv(in_dim, int(out_dim), heads=heads, dropout=0.5, edge_dim=in_dim)
        if 'gin' == aggregator:
            nn1 = Sequential(Linear(in_dim, out_dim), ReLU(), Linear(out_dim, out_dim))
            self._op = GINConv(nn1)
        if aggregator == 'gcn_mol':
            self._op = GCNConv_mol(in_dim)
        if aggregator == 'gen':
            self._op = GENConv(in_dim, out_dim, t=1.0, learn_t=True, encode_edge=True, bond_encoder=True, mlp_layers=1)
        if aggregator == 'mf':
            self._op = MFConv(in_dim, out_dim)
        if aggregator == 'EXPC':
            self._op = ExpC(in_dim, 1, {"BN": "Y"})
        if aggregator == 'gin_mol':
            self._op = GINConv_mol(in_dim)
        if aggregator in ['gat_sym', 'cos', 'linear', 'generalized_linear']:
            heads = 8
            out_dim /= heads
            self._op = GeoLayer(in_dim, int(out_dim), heads=heads, att_type=aggregator, dropout=0.5)
        if aggregator in ['sum', 'max']:
            self._op = GeoLayer(in_dim, out_dim, att_type='const', agg_type=aggregator, dropout=0.5)
        if aggregator in ['geniepath']:
            self._op = GeniePathLayer(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None):
        if self.aggregator == 'mlp':
            return self._op(x)
        if self.aggregator == 'Transformerconv' or self.aggregator == 'gcn_mol' or self.aggregator == 'gin_mol'\
            or self.aggregator == 'gat' or self.aggregator == 'gine' or self.aggregator == 'gatv2' or self.aggregator == 'mf' or self.aggregator == 'gen' or self.aggregator == 'EXPC':
            return self._op(x, edge_index, edge_attr)
        else:
            return self._op(x, edge_index, edge_weight=edge_weight)


class LaAggregator(nn.Module):

    def __init__(self, mode, hidden_size, num_layers=3):
        super(LaAggregator, self).__init__()
        self.mode = mode
        if mode in ['lstm', 'cat', 'max']:
            self.jump = JumpingKnowledge(mode, hidden_size, num_layers=num_layers)
        elif mode == 'att':
            self.att = Linear(hidden_size, 1)

        if mode == 'cat':
            self.lin = Linear(hidden_size * num_layers, hidden_size)
        else:
            self.lin = Linear(hidden_size, hidden_size)

    def forward(self, xs):
        if self.mode in ['lstm', 'cat', 'max']:
            output = self.jump(xs)
        elif self.mode == 'sum':
            output = torch.stack(xs, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            output = torch.stack(xs, dim=-1).mean(dim=-1)
        # elif self.mode == 'att':
        #     input = torch.stack(xs, dim=-1).transpose(1, 2)
        #     weight = self.att(input)
        #     weight = F.softmax(weight, dim=1)# cal the weightes of each layers and each node
        #     output = torch.mul(input, weight).transpose(1, 2).sum(dim=-1) #weighte sum
        return self.lin(F.elu(output))

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Zero(nn.Module):

    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x.mul(0.)
