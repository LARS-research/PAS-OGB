import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Conv1d, ELU, PReLU

from torch_geometric.nn import SAGEConv, JumpingKnowledge
from torch_geometric.nn import GCNConv, GINConv,GraphConv,LEConv,SGConv,DenseSAGEConv,DenseGCNConv,DenseGINConv,DenseGraphConv
from torch_geometric.nn import global_add_pool,global_mean_pool,global_max_pool,global_sort_pool,GlobalAttention,Set2Set
from torch_geometric.nn import SAGPooling,TopKPooling,EdgePooling,ASAPooling,dense_diff_pool
from math import ceil
# from pyg_gnn_layer import GeoLayer
# from gin_conv import GINConv2
# from geniepath import GeniePathLayer
from genotypes import NA_MLP_PRIMITIVES
from torch_geometric.utils import to_dense_adj, to_dense_batch
from pooling_zoo import SAGPool_mix, ASAPooling_mix, TOPKpooling_mix, Hoppooling_mix, Gappool_Mixed
from agg_zoo import GAT_mix,SAGE_mix,Geolayer_mix,GIN_mix
from torch_geometric.nn.inits import reset
from conv import GCNConv as GCNConv_mol
from conv import GINConv as GINConv_mol
from conv import TransformerConv, GINEConv, GATConv, GATv2Conv, MFConv, GENConv, GINEPLUS

NA_OPS = {
    #SANE
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
  'mlp': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'mlp'),
  'gatv2': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gatv2'),
  'mf': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'mf'),
  'gineplus': lambda in_dim, out_dim, k: NaAggregator(in_dim, out_dim, 'gineplus', k=k),

  # with Edge
  'Transformerconv': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'Transformerconv'),
  'gine': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gine'),
  'gcn_mol': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gcn_mol'),
  'gin_mol': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gin_mol'),
  'gen': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gen'),


  #graph classification:
  'graphconv_add': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'graphconv_add'),
  'graphconv_mean': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'graphconv_mean'),
  'graphconv_max': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'graphconv_max'),
  'sgc': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'sgc'),
  'leconv': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'leconv'),

}
POOL_OPS = {

  'hoppool_1': lambda hidden,ratio,num_nodes:Pooling_func(hidden,ratio,'hoppool_1',num_nodes=num_nodes),
  'hoppool_2': lambda hidden,ratio,num_nodes:Pooling_func(hidden,ratio,'hoppool_2',num_nodes=num_nodes),
  'hoppool_3': lambda hidden,ratio,num_nodes:Pooling_func(hidden,ratio,'hoppool_3',num_nodes=num_nodes),

  'mlppool': lambda hidden, ratio, num_nodes: Pooling_func(hidden, ratio, 'mlppool', num_nodes=num_nodes),
  'topkpool': lambda hidden, ratio, num_nodes: Pooling_func(hidden, ratio, 'topkpool', num_nodes=num_nodes),

  'gappool': lambda hidden, ratio, num_nodes: Pooling_func(hidden, ratio, 'gappool', num_nodes=num_nodes),

  'asappool': lambda hidden, ratio, num_nodes: Pooling_func(hidden, ratio, 'asappool', num_nodes=num_nodes),
  'sagpool': lambda hidden, ratio, num_nodes: Pooling_func(hidden, ratio, 'sagpool', num_nodes=num_nodes),
  'sag_graphconv': lambda hidden, ratio, num_nodes: Pooling_func(hidden, ratio, 'graphconv', num_nodes=num_nodes),

  'none': lambda hidden,ratio,num_nodes:Pooling_func(hidden,ratio, 'none', num_nodes=num_nodes),
  'none2': lambda hidden,ratio,num_nodes:Pooling_func(hidden,ratio, 'none', num_nodes=num_nodes),
  'none3': lambda hidden,ratio,num_nodes:Pooling_func(hidden,ratio, 'none', num_nodes=num_nodes),
  'diffgcn': lambda hidden,ratio,num_nodes:Pooling_func(hidden,ratio, 'diffgcn',num_nodes=num_nodes),
  'diffgraphconv': lambda hidden,ratio,num_nodes:Pooling_func(hidden,ratio, 'diffgraphconv',num_nodes=num_nodes),
  'diffsage': lambda hidden,ratio,num_nodes:Pooling_func(hidden,ratio, 'diffsage',num_nodes=num_nodes),
  'edgepool': lambda hidden, ratio, num_nodes: Pooling_func(hidden, ratio, 'edgepool', num_nodes=num_nodes),

}
READOUT_OPS = {
    "global_mean": lambda hidden :Readout_func('mean', hidden),
    "global_sum": lambda hidden  :Readout_func('add', hidden),
    "global_max": lambda hidden  :Readout_func('max', hidden),
    "none":lambda hidden  :Readout_func('none', hidden),
    'global_att': lambda hidden  :Readout_func('att', hidden),
    'global_sort': lambda hidden  :Readout_func('sort',hidden),
    'set2set': lambda hidden  :Readout_func('set2set',hidden),
    'mean_max': lambda hidden  :Readout_func('mema', hidden),
    # 'MLP': lambda hidden  :Readout_func('mlp', hidden)
}

NA_MLP_OPS = {}
for op in NA_MLP_PRIMITIVES:
    parts = op.split('_')
    w, d = int(parts[1]), int(parts[2])
    NA_MLP_OPS[op] = lambda in_dim, out_dim : NaMLPAggregator(in_dim, out_dim, w, d)

NA_OPS2 = {
  'sage': lambda in_dim, out_dim: NaAggregator2(in_dim, out_dim, 'sage'),
  'sage_sum': lambda in_dim, out_dim: NaAggregator2(in_dim, out_dim, 'sum'),
  'sage_max': lambda in_dim, out_dim: NaAggregator2(in_dim, out_dim, 'max'),
  'gcn': lambda in_dim, out_dim: NaAggregator2(in_dim, out_dim, 'gcn'),
  'gat': lambda in_dim, out_dim: NaAggregator2(in_dim, out_dim, 'gat'),
  'gin': lambda in_dim, out_dim: NaAggregator2(in_dim, out_dim, 'gin'),
  'gat_sym': lambda in_dim, out_dim: NaAggregator2(in_dim, out_dim, 'gat_sym'),
  'gat_linear': lambda in_dim, out_dim: NaAggregator2(in_dim, out_dim, 'linear'),
  'gat_cos': lambda in_dim, out_dim: NaAggregator2(in_dim, out_dim, 'cos'),
  'gat_generalized_linear': lambda in_dim, out_dim: NaAggregator2(in_dim, out_dim, 'generalized_linear'),
#  'geniepath': lambda in_dim, out_dim: NaAggregator2(in_dim, out_dim, 'geniepath'),
}

SC_OPS={
  'none': lambda: Zero(),
  'skip': lambda: Identity(),
  }

LA_OPS={
  'l_max': lambda hidden_size, num_layers: LaAggregator('max', hidden_size, num_layers),
  'l_concat': lambda hidden_size, num_layers: LaAggregator('cat', hidden_size, num_layers),
  'l_mean': lambda hidden_size, num_layers: LaAggregator('mean', hidden_size, num_layers),
  'l_sum':  lambda hidden_size, num_layers: LaAggregator('sum', hidden_size, num_layers),
  'l_lstm': lambda hidden_size, num_layers: LaAggregator('lstm', hidden_size, num_layers)
  #min/max
}

class NaAggregator(nn.Module):

  def __init__(self, in_dim, out_dim, aggregator, k=0):
    super(NaAggregator, self).__init__()
    #aggregator, K = agg_str.split('_')
    self.aggregator = aggregator
    if 'sage' == aggregator:
      # self._op = SAGEConv(in_dim, out_dim, normalize=True)
      self._op = SAGE_mix(in_dim, out_dim)
    if 'gcn' == aggregator:
      self._op = GCNConv(in_dim, out_dim)
    if 'gat' == aggregator:
      heads = 4
      out_dim /= heads
      self._op = GATConv(in_dim, int(out_dim), heads=heads, dropout=0.5, edge_dim=in_dim)
    if 'gatv2' == aggregator:
      heads = 4
      out_dim /= heads
      self._op = GATv2Conv(in_dim, int(out_dim), heads=heads, dropout=0.5, edge_dim=in_dim)
    if 'gin' == aggregator:
      nn1 = Sequential(Linear(in_dim, out_dim), ELU(), Linear(out_dim, out_dim))
      self._op = GIN_mix(nn1)
    if aggregator in ['gat_sym', 'cos', 'linear', 'generalized_linear']:
      heads = 2
      out_dim /= heads
      self._op = Geolayer_mix(in_dim, int(out_dim), heads=heads, att_type=aggregator, dropout=0.5)
    if aggregator in ['sum', 'max']:
      self._op = Geolayer_mix(in_dim, out_dim, att_type='const', agg_type=aggregator, dropout=0.5)
    if aggregator in ['geniepath']:
      self._op = GeniePathLayer(in_dim, out_dim)
    if aggregator =='sgc':
        self._op = SGConv(in_dim, out_dim)
    if 'graphconv'in aggregator:
      aggr = aggregator.split('_')[-1]
      self._op = GraphConv(in_dim, out_dim, aggr=aggr)
    if aggregator == 'leconv':
        self._op = LEConv(in_dim, out_dim)
    if aggregator == 'mlp':
      self._op = Sequential(Linear(in_dim, out_dim), ELU(), Linear(out_dim, out_dim))
    if aggregator == 'Transformerconv':
      heads = 4
      out_dim /= heads
      self._op = TransformerConv(in_dim, int(out_dim), heads=heads, dropout=0.5, edge_dim=in_dim)
    if aggregator == 'gine':
      self._op = GINEConv(in_dim, int(out_dim), train_eps=True)
    if aggregator == 'gineplus':
      self._op = GINEPLUS(MLP(in_dim, out_dim), out_dim, k=k)
    if aggregator == 'gcn_mol':
      self._op = GCNConv_mol(in_dim)
    if aggregator == 'gen':
      self._op = GENConv(in_dim, out_dim, t=1.0, learn_t=True, p=1.0, encode_edge=True, bond_encoder=True, mlp_layers=1)
    if aggregator == 'mf':
      self._op = MFConv(in_dim, out_dim)
    if aggregator == 'gin_mol':
      self._op = GINConv_mol(in_dim)
  def reset_params(self):
    if self.aggregator == 'mlp':
      reset(self._op)
    elif self.aggregator == 'gin_mol' or self.aggregator == 'gcn_mol' or self.aggregator == 'gen':
      a = 0
    else:
      self._op.reset_parameters()

  def forward(self, x, edge_index, edge_weight=None, edge_attr=None, multihop_edge_index = None, distance = None):

    if self.aggregator == 'mlp':
      return self._op(x)
    if self.aggregator == 'Transformerconv' or self.aggregator == 'gcn_mol' or self.aggregator == 'gin_mol'\
            or self.aggregator == 'gat' or self.aggregator == 'gine' or self.aggregator == 'gatv2' or self.aggregator == 'mf' or self.aggregator == 'gen':
      return self._op(x, edge_index, edge_attr)
    elif self.aggregator == 'gineplus':
      return self._op(x, edge_index, distance, edge_attr)
    else:
      return self._op(x, edge_index, edge_weight=edge_weight)

class NaMLPAggregator(nn.Module):

  def __init__(self, in_dim, out_dim, w, d):
    super(NaMLPAggregator, self).__init__()
    if d == 1:
      nn1 = Sequential(Linear(in_dim, out_dim))
    elif d == 2:
      nn1 = Sequential(Linear(in_dim, w), ELU(), Linear(w, out_dim))
    elif d == 3:
      nn1 = Sequential(Linear(in_dim, w), ELU(), Linear(w, w), ELU(), Linear(w, out_dim))
    self._op = GINConv(nn1)

  def forward(self, x, edge_index):
    return self._op(x, edge_index)


class NaAggregator2(nn.Module):

  def __init__(self, in_dim, out_dim, aggregator):
    super(NaAggregator2, self).__init__()
    self.aggregator = aggregator
    if 'sage' == aggregator:
      self._op = SAGEConv(in_dim, out_dim)
    if 'gcn' == aggregator:
      self._op = GCNConv(in_dim, out_dim)
    if 'gat' == aggregator:
      heads = 2
      out_dim /= heads
      self._op = GATConv(in_dim, int(out_dim), heads=heads, dropout=0.5)
    if 'gin' == aggregator:
      nn1 = Sequential(Linear(in_dim, out_dim), ELU(), Linear(out_dim, out_dim))
      self._op = GINConv(nn1)
    if aggregator in ['gat_sym', 'cos', 'linear', 'generalized_linear']:
      heads = 2
      out_dim /= heads
      self._op = GeoLayer(in_dim, int(out_dim), heads=heads, att_type=aggregator, dropout=0.5)
    if aggregator in ['sum', 'max']:
      self._op = GeoLayer(in_dim, out_dim, att_type='const', agg_type=aggregator, dropout=0.5)
    if aggregator in ['geniepath']:
      self._op = GeniePathLayer(in_dim, out_dim)

  def forward(self, x, edge_index, size):
      if self.aggregator in ['gin']:
        return self._op(x[0], edge_index, size=size)
      else:
        return self._op(x, edge_index, size=size)

class LaAggregator(nn.Module):

  def __init__(self, mode, hidden_size, num_layers=3):
    super(LaAggregator, self).__init__()
    self.mode = mode
    if self.mode in ['lstm', 'max', 'cat']:
      self.jump = JumpingKnowledge(mode, channels=hidden_size, num_layers=num_layers)
    if mode == 'cat':
        self.lin = Linear(hidden_size * num_layers, hidden_size)
    else:
        self.lin = Linear(hidden_size, hidden_size)
  def reset_params(self):
    self.lin.reset_parameters()
  def forward(self, xs):
    if self.mode in ['lstm', 'max', 'cat']:
      return self.lin(F.elu(self.jump(xs)))
    elif self.mode =='sum':
      return self.lin(F.elu(torch.stack(xs, dim=-1).sum(dim=-1)))
    elif self.mode =='mean':
      return self.lin(F.elu(torch.stack(xs, dim=-1).mean(dim=-1)))
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

class Readout_func(nn.Module):
  def __init__(self, readout_op, hidden):

    super(Readout_func, self).__init__()
    self.readout_op = readout_op

    if readout_op == 'mean':
      self.readout = global_mean_pool

    elif readout_op == 'max':
      self.readout = global_max_pool

    elif readout_op == 'add':
      self.readout = global_add_pool

    elif readout_op == 'att':
      self.readout = GlobalAttention(Linear(hidden, 1))

    elif readout_op == 'set2set':
      processing_steps = 2
      self.readout = Set2Set(hidden, processing_steps=processing_steps)
      self.s2s_lin = Linear(hidden*processing_steps, hidden)


    elif readout_op == 'sort':
      self.readout = global_sort_pool
      self.k = 10
      self.sort_conv = Conv1d(hidden, hidden, 5)#kernel size 3, output size: hidden,
      self.sort_lin = Linear(hidden*(self.k-5 + 1), hidden)
    elif readout_op =='mema':
      self.readout = global_mean_pool
      self.lin = Linear(hidden*2, hidden)
    elif readout_op == 'none':
      self.readout = global_mean_pool
    # elif self.readout_op == 'mlp':

  def reset_params(self):
    if self.readout_op =='sort':
      self.sort_conv.reset_parameters()
      self.sort_lin.reset_parameters()
    if self.readout_op in ['set2set', 'att']:
      self.readout.reset_parameters()
    if self.readout_op =='set2set':
      self.s2s_lin.reset_parameters()
    if self.readout_op == 'mema':
      self.lin.reset_parameters()
  def forward(self, x, batch, mask):
    #sparse data
    if self.readout_op == 'none':
      x = self.readout(x, batch)
      return x.mul(0.)
      # return None
    elif self.readout_op == 'sort':
      x = self.readout(x, batch, self.k)
      x = x.view(len(x), self.k, -1).permute(0, 2, 1)
      x = F.elu(self.sort_conv(x))
      x = x.view(len(x), -1)
      x = self.sort_lin(x)
      return x
    elif self.readout_op == 'mema':
      x1 = global_mean_pool(x, batch)
      x2 = global_max_pool(x, batch)
      x = torch.cat([x1, x2], dim=-1)
      x = self.lin(x)
      return x
    else:
      x = self.readout(x, batch)
      if self.readout_op == 'set2set':
        x = self.s2s_lin(x)
      return x

# if len(list(x.size())) == 3:
    #   print('dense data:', self.readout_op)
    #   #dense data
    #   if self.readout_op == 'mean':
    #     x = x.mean(dim=1)
    #     return x
    #
    #   elif self.readout_op == 'add':
    #     x = x.sum(dim=1)
    #     return x
    #
    #   elif self.readout_op == 'max':
    #     x = x.max(dim=1)
    #     return x
    #   elif self.readout_op == 'none':
    #     x = x.mean(dim=1)
    #     return x.mul(0.)
    #
    # else:
    #   print('sparse data:', self.readout_op)
    #
    #   #sparse data
    #   if self.readout_op == 'none':
    #     x = self.readout(x, batch)
    #     return torch.mul(x, 0.)
    #   else:
    #     return self.readout(x, batch)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
      super().__init__()
      self.main = [
        nn.Linear(in_dim, 2 * in_dim),
        nn.BatchNorm1d(2 * in_dim),
        nn.ReLU()
      ]
      self.main.append(nn.Linear(2 * in_dim, out_dim))
      self.main = nn.Sequential(*self.main)

    def forward(self, x):
      return self.main(x)


class Pooling_func(nn.Module):
  def __init__(self, hidden, ratio, op, dropout=0.6, num_nodes=0):
    super(Pooling_func, self).__init__()
    self.op = op
    self.max_num_nodes = num_nodes
    if op =='sagpool':
      self._op = SAGPool_mix(hidden, ratio=ratio, gnn_type='gcn')
    elif op =='mlppool':
      self._op = SAGPool_mix(hidden, ratio=ratio, gnn_type='mlp')
    elif op =='graphconv':
      self._op = SAGPool_mix(hidden, ratio=ratio, gnn_type='graphconv')

    elif 'hop' in op:
      hop_num = int(op.split('_')[-1])
      self._op = Hoppooling_mix(hidden, ratio=ratio, walk_length=hop_num)
    elif op == 'gappool':
      self._op = Gappool_Mixed(hidden, ratio=ratio)
    elif op == 'topkpool':
      # self._op = TopKPooling(hidden, ratio)
      self._op = TOPKpooling_mix(hidden, ratio=ratio)

    elif op == 'edgepool':
      self._op = EdgePooling(hidden)
    elif op == 'asappool':
      # self._op = ASAPooling(hidden, ratio, dropout=dropout)
      self._op = ASAPooling_mix(hidden, ratio=ratio, dropout=dropout)
    elif op == 'diffgcn':
      self._op = DenseGCNConv(hidden, ceil(ratio * num_nodes))

    elif op == 'diffgraphconv':
      self._op = DenseGraphConv(hidden, ceil(ratio * num_nodes))

    elif op == 'diffsage':
      self._op = DenseSAGEConv(hidden, ceil(ratio * num_nodes))
  def reset_params(self):
    if self.op != 'none':
      self._op.reset_parameters()

  def forward(self, x, edge_index, edge_weights, data, batch, mask, ft=False):
    if self.op == 'none':
      perm = torch.ones(x.size(0), dtype=torch.float64, device=x.device)
      return x, edge_index, edge_weights, batch, perm

    elif self.op in ['asappool', 'topkpool', 'sagpool', 'mlppool', 'hoppool_1', 'hoppool_2', 'hoppool_3', 'gappool', 'graphconv']:
      # print('operations:', self.op)
      x, edge_index, edge_weight, batch, perm = self._op(x=x, edge_index=edge_index, edge_weight=edge_weights, batch=batch, ft=ft)
      return x, edge_index, edge_weight, batch, perm

    # elif self.op =='edgepool':
    #   x, edge_index, batch, _ = self._op(x=x, edge_index=edge_index, batch=batch)
    #   return x,edge_index,batch,None

    # elif self.op in ['sagpool','topkpool','edgepool']:
    #   x, edge_index, _, batch, _, _ = self._op(x, edge_index, batch=batch)
    #   return x, edge_index, batch, None

    elif 'diff' in self.op:
      x_new, mask_new = to_dense_batch(x, batch=batch, max_num_nodes=self.max_num_nodes)
      adj_new = to_dense_adj(edge_index, batch=batch, max_num_nodes=self.max_num_nodes)
      x_new, mask_new, adj_new = x_new.cuda(), mask_new.cuda(), adj_new.cuda()

      assign_gnn = self._op(x_new, adj_new)
      x, adj, _, _ = dense_diff_pool(x_new, adj_new, assign_gnn, mask=mask_new)
      # x, edge_index, _, batch = to_sparse_batch(x, adj, assign_gnn)
      x, edge_index, _, batch = to_sparse_batch_mine(x, adj, th=0.3)
      print('size:x/edge_index/batch', x.size(), edge_index.size(), batch.size())
      return x, edge_index, batch, None

#
# def dense2sparse(x, adj, th):
#   num_nodes = x.size()[1]
#   adj = adj[:num_nodes, :num_nodes]
#   mask = adj >= th
#   edge_index = mask.nonzero().t()
#   edge_index = edge_index[0:2, :]
#
#   return edge_index
# def to_sparse_batch_mine(x,adj,th):
#   B, N_max, D = x.shape
#   x_new = x.reshape(-1, D)
#
#   mask = adj >= th
#   sparse_edge = mask.nonzero().t()
#   edge_index = sparse_edge[0:2, :]
#   edge_weight = sparse_edge[-1, :]
#   batch = torch.zeros_like(x[:, 0], dtype=torch.int64)
#
#   start=0
#   for i in range(B):
#       num_nodes = x[i, :, :].size()[1]
#       batch[start:start+num_nodes] = i
#       start += num_nodes
#
#   return x_new, edge_index, edge_weight, batch
# def to_sparse_batch(x, adj, mask):
#   # transform x (B x N x D), adj (B x N x N), mask (B x N), here N is N_max
#   # to x, edge_index, edge_attr/weight, batch
#   B, N_max, D = x.shape
#   # mask out adj
#   adj = (adj * mask.unsqueeze(2)).transpose(1, 2)
#   adj = (adj * mask.unsqueeze(2)).transpose(1, 2)
#
#   # get number of graphs
#   num_nodes_graphs = mask.sum(dim=1)  # B
#
#   # get offset with size B
#   offset_graphs = torch.cumsum(num_nodes_graphs, dim=0)  # B
#
#   # get x
#   x = x.reshape(-1, D)[mask.reshape(-1)]  # total_nodes * D
#
#   # get weight and index
#   edge_weight = adj[adj.nonzero(as_tuple=True)]
#   nnz_index = adj.nonzero().t()
#   graph_idx, edge_index = nnz_index[0], nnz_index[1:]
#
#   # init batch
#   batch = torch.zeros_like(x[:, 0], dtype=torch.int64).fill_(B - 1)
#
#   # add offset to edge_index, and setup batch
#   start = 0
#   for i, offset in enumerate(offset_graphs[:-1]):
#     edge_index[:, graph_idx == i + 1] += offset
#     batch[start:offset] = i
#     start = offset
#
#   return x, edge_index, edge_weight, batch
