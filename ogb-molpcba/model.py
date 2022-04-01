import torch
import torch.nn as nn
# from operations import *
from op_graph_classification import *
from torch.autograd import Variable
from torch_geometric.nn import global_mean_pool, global_add_pool
import torch_geometric.nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import BatchNorm1d
from  torch_geometric.utils import add_self_loops,remove_self_loops,remove_isolated_nodes, degree
import pyximport
from copy import copy
# import algos
from operations import make_degree, make_multihop_edges
from torch_scatter import scatter_add

def act_map(act):
  if act == "linear":
    return lambda x: x
  elif act == "elu":
    return torch.nn.functional.elu
  elif act == "sigmoid":
    return torch.sigmoid
  elif act == "tanh":
    return torch.tanh
  elif act == "relu":
    return torch.nn.functional.relu
  elif act == "relu6":
    return torch.nn.functional.relu6
  elif act == "softplus":
    return torch.nn.functional.softplus
  elif act == "leaky_relu":
    return torch.nn.functional.leaky_relu
  elif act == "prelu":
    return torch.nn.PReLU
  else:
    raise Exception("wrong activate function")

class GlobalPool(nn.Module):
    def __init__(self, fun, cat_size=False, cat_candidates=False, hidden=0):
        super().__init__()
        self.cat_size = cat_size
        if fun in ['mean', 'max', 'add']:
            self.fun = getattr(torch_geometric.nn, "global_{}_pool".format(fun.lower()))
        else:
            self.fun = torch_geometric.nn.GlobalAttention(gate_nn =
                    torch.nn.Sequential(
                        torch.nn.Linear(hidden, hidden)
                        ,torch.nn.BatchNorm1d(hidden)
                        ,torch.nn.ReLU()
                        ,torch.nn.Linear(hidden, 1)))

        self.cat_candidates = cat_candidates

    def forward(self, batch):
        x, b = batch.x, batch.batch
        pooled = self.fun(x, b, size=batch.num_graphs)
        if self.cat_size:
            sizes = torch_geometric.nn.global_add_pool(torch.ones(x.size(0), 1).type_as(x), b, size=batch.num_graphs)
            pooled = torch.cat([pooled, sizes], dim=1)
        if self.cat_candidates:
            ei = batch.edge_index
            mask = batch.edge_attr == 3
            candidates = scatter_add(x[ei[0, mask]], b[ei[0, mask]], dim=0, dim_size=batch.num_graphs)
            pooled = torch.cat([pooled, candidates], dim=1)
        return pooled


class APPNP(nn.Module):
    def __init__(self, K=5, alpha=0.8):
        super().__init__()
        self.appnp = torch_geometric.nn.conv.APPNP(K=K, alpha=alpha)

    def forward(self, data):
        data = new(data)
        # print(data.x)
        # print(data.x.shape)
        # print(data.edge_index)
        # print(data.edge_index.shape)
        x, ei, ea, b = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.appnp(x, ei)
        data.x = h
        return data

def new(data):

    return copy(data)


class NaOp(nn.Module):
  def __init__(self, primitive, in_dim, out_dim, act, with_linear=False, with_act=True, is_gineplus=0):
    super(NaOp, self).__init__()
    print(primitive)
    # self.bond_encoder = BondEncoder(emb_dim=in_dim)
    self.is_gineplus = is_gineplus
    if self.is_gineplus == 0:
      self._op = NA_OPS[primitive](in_dim, out_dim)
    else:
      self._op = NA_OPS[primitive](in_dim, out_dim, k=is_gineplus)


    if with_linear:
      self.op_linear = nn.Linear(in_dim, out_dim)
    if not with_act:
      act = 'linear'
    self.act = act_map(act)
    self.with_linear = with_linear

  def reset_params(self):
    self._op.reset_params()
    # self.op_linear.reset_parameters()

  def forward(self, x, edge_index, edge_weights, edge_attr, distance = None):
    if self.with_linear:
      return self.act(self._op(x, edge_index, edge_weight=edge_weights, edge_attr=edge_attr) + self.op_linear(x))
    else:
      if self.is_gineplus == 0:
        return self.act(self._op(x, edge_index, edge_weight=edge_weights, edge_attr=edge_attr))
      else:
        return self.act(self._op(x, edge_index, edge_weight=edge_weights, edge_attr=edge_attr, distance = distance))


    # mixed_res = []
    # for w, op in zip(weights, self._ops):
    #  mixed_res.append(w * F.relu(op(x, edge_index)))
    # return sum(mixed_res)


class ScOp(nn.Module):
  def __init__(self, primitive):
    super(ScOp, self).__init__()
    # self._ops = nn.ModuleList()
    # for primitive in SC_PRIMITIVES:
    #  op = SC_OPS[primitive]()
    #  self._ops.append(op)
    self._op = SC_OPS[primitive]()

  def forward(self, x):
    # mixed_res = []
    # for w, op in zip(weights, self._ops):
    #  mixed_res.append(w * F.relu(op(x)))
    # return sum(mixed_res)
    return self._op(x)


class LaOp(nn.Module):
  def __init__(self, primitive, hidden_size, act, num_layers=None):
    super(LaOp, self).__init__()
    self._op = LA_OPS[primitive](hidden_size, num_layers)
    self.act = act_map(act)

  def reset_params(self):
    self._op.reset_params()

  def forward(self, x):
    # return self.act(self._op(x))
    return self._op(x)


class NaMLPOp(nn.Module):
  def __init__(self, primitive, in_dim, out_dim, act):
    super(NaMLPOp, self).__init__()
    self._op = NA_MLP_OPS[primitive](in_dim, out_dim)
    self.act = act_map(act)

  def forward(self, x, edge_index):
    return self.act(self._op(x, edge_index))



class PoolingOp(nn.Module):
  def __init__(self, primitive, hidden, ratio, num_nodes=0):
    super(PoolingOp, self).__init__()
    self._op = POOL_OPS[primitive](hidden, ratio, num_nodes)
    self.primitive = primitive
  def reset_params(self):
    self._op.reset_params()
  def forward(self, x, edge_index,edge_weights, data, batch, mask):
    new_x, new_edge_index, _, new_batch, _ = self._op(x, edge_index, edge_weights, data, batch, mask, ft=True)
    return new_x, new_edge_index, new_batch, None


class ReadoutOp(nn.Module):
  def __init__(self, primitive, hidden):
    super(ReadoutOp, self).__init__()
    self._op = READOUT_OPS[primitive](hidden)
  def reset_params(self):
    self._op.reset_params()
  def forward(self, x, batch, mask):

    return self._op(x, batch, mask)



class NetworkGNN(nn.Module):
  '''
      implement this for sane.
      Actually, sane can be seen as the combination of three cells, node aggregator, skip connection, and layer aggregator
      for sane, we dont need cell, since the DAG is the whole search space, and what we need to do is implement the DAG.
  '''
  def __init__(self, genotype, criterion, in_dim, out_dim, hidden_size, num_layers=3, in_dropout=0.2, out_dropout=0.2, act='relu', args=None,is_mlp=False, num_nodes=0):
    super(NetworkGNN, self).__init__()
    hidden_size = hidden_size
    self.genotype = genotype
    # self.beta = torch.nn.Parameter(torch.Tensor([1.]), requires_grad=True)
    self.in_dim = in_dim
    self.atom_encoder = AtomEncoder(hidden_size)
    # self.bond_encoder = BondEncoder(hidden_size)
    self.out_dim = out_dim
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.in_dropout = in_dropout
    self.out_dropout = out_dropout
    self.dropout = in_dropout
    self._criterion = criterion
    ops = genotype.split('||')
    # self.in_degree_encoder = nn.Embedding(64, hidden_size, padding_idx=0)
    # self.out_degree_encoder = nn.Embedding(64, hidden_size, padding_idx=0)

    ### set the initial virtual node embedding to 0.
    # self.virtualnode_embedding = torch.nn.Embedding(1, hidden_size)
    # torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

    ### List of MLPs to transform virtual node at every layer
    # self.mlp_virtualnode_list = torch.nn.ModuleList()

    # for layer in range(num_layers - 1):
    #   self.mlp_virtualnode_list.append(
    #     torch.nn.Sequential(torch.nn.Linear(hidden_size, 2 * hidden_size), torch.nn.BatchNorm1d(2 * hidden_size),
    #                         torch.nn.ReLU(), \
    #                         torch.nn.Linear(2 * hidden_size, hidden_size), torch.nn.BatchNorm1d(hidden_size),
    #                         torch.nn.ReLU()))

    # self.outdeg_emb = nn.Linear(1, hidden_size)
    self.args = args
    self.pool = global_mean_pool

    self.norms = torch.nn.ModuleList()
    for layer in range(self.num_layers):
      self.norms.append(torch.nn.BatchNorm1d(hidden_size))

    #node aggregator op
    # self.lin1 = nn.Linear(hidden_size, hidden_size)
    if is_mlp:
      self.gnn_layers = nn.ModuleList([NaMLPOp(ops[i], hidden_size, hidden_size, act) for i in range(num_layers)])
    else:
      # acts from train_search or fine_tune
      if self.args.search_act:
        act = ops[num_layers: num_layers*2]
        print(act)
      else:
        act = [act for i in range(num_layers)]
        print(act)
        print(args.with_linear)
      if ops[0] == 'gineplus':
        self.gnn_layers = nn.ModuleList(
        [NaOp(ops[i], hidden_size, hidden_size, act, with_linear=args.with_linear, with_act=False, is_gineplus=min(i + 1, 3)) for i in range(num_layers)])
        self.conv_type = 'gineplus'
      else:
        self.gnn_layers = nn.ModuleList(
          [NaOp(ops[i], hidden_size, hidden_size, act, with_linear=args.with_linear, with_act=False) for i in
           range(num_layers)])
        self.conv_type = 'none'

      self.act = act

    self.in_degree_encoder = nn.Embedding(64, hidden_size, padding_idx=0)
    # self.out_degree_encoder = nn.Embedding(64, hidden_size, padding_idx=0)

    self.aggregate = nn.Sequential(
        APPNP(K=5, alpha=0.8),
        GlobalPool('mean', hidden=hidden_size),
        nn.Linear(hidden_size, out_dim))

    ### set the initial virtual node embedding to 0.
    self.virtualnode_embedding = torch.nn.Embedding(1, hidden_size)
    torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

    ### List of MLPs to transform virtual node at every layer
    self.mlp_virtualnode_list = torch.nn.ModuleList()
    for layer in range(num_layers - 1):
        self.mlp_virtualnode_list.append(
            torch.nn.Sequential(torch.nn.Linear(hidden_size, 2 * hidden_size), torch.nn.BatchNorm1d(2 * hidden_size),
                                torch.nn.ReLU(), \
                                torch.nn.Linear(2 * hidden_size, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU()))


    self.graph_pred_linear = nn.Linear(hidden_size, out_dim)


  def reset_params(self):

    # self.lin1.reset_parameters()

    for i in range(self.num_layers):
      self.gnn_layers[i].reset_params()
      # self.pooling_layers[i].reset_params()

    # for i in range(self.num_layers+1):
    #     self.readout_layers[i].reset_params()

    # self.layer6.reset_params()
    # self.lin_output.reset_parameters()
    self.classifier.reset_parameters()

  def forward(self, data, perturb=None):
    # data = self.preprocess_item(data)
    # degree = data.in_degree
    data = make_degree(data)
    data = make_multihop_edges(data, 3)
    x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
    multihop, distance = data.multihop_edge_index, data.distance

    ### virtual node embeddings for graphs
    virtualnode_embedding = self.virtualnode_embedding(
        torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))


    if self.args.data == 'ogbg-molhiv' or self.args.data == 'ogbg-molpcba':

        #flag
        # x = self.atom_encoder(x) + perturb if perturb is not None else self.atom_encoder(x)
        x = self.atom_encoder(x) + perturb if perturb is not None else self.atom_encoder(x)
        x = x + self.in_degree_encoder(data.in_degree)
        # x = self.atom_encoder(x)

    h = [x]

    # x = self.gnn_layers[0](x, edge_index, edge_weights=None, edge_attr=edge_attr)
    for i in range(0, self.num_layers - 1):
        if self.conv_type == 'gineplus':
            x = self.gnn_layers[i](h, multihop, distance = distance, edge_weights=None, edge_attr=edge_attr)
            h = x[0]
        else:
            x = self.gnn_layers[i](h[0], edge_index, distance=distance, edge_weights=None, edge_attr=edge_attr)
            h = x

        h = self.norms[i](h)
        if not self.args.search_act:
            h = F.relu(h)
        else:
            if self.act[i] == 'relu':
                # x2 = self.prelu(x1)
                h = F.relu(h)
            elif self.act[i] == 'sigmoid':
                h= torch.sigmoid(h)
            elif self.act[i] == 'tanh':
                h = torch.tanh(h)
            elif self.act[i] == 'softplus':
                h = F.softplus(h)
            elif self.act[i] == 'leaky_relu':
                h = F.leaky_relu(h)
            elif self.act[i] == 'relu6':
                h = F.relu6(h)
            elif self.act[i] == 'elu':
                h = F.elu(h)

        h = F.dropout(h, p=self.dropout, training=self.training) + virtualnode_embedding[batch]
        # graph_representations[i] += virtualnode_embedding[batch]

        if i < self.num_layers:
            ### add message from graph nodes to virtual nodes
            virtualnode_embedding_temp = global_add_pool(h, batch) + virtualnode_embedding
            ### transform virtual nodes using MLP

            virtualnode_embedding = virtualnode_embedding + F.dropout(
                  self.mlp_virtualnode_list[i - 1](virtualnode_embedding_temp), self.dropout, training=self.training)

        x[0] = h
        h = x

    i += 1
    if self.conv_type == 'gineplus':
      x = self.gnn_layers[i](h, multihop, distance=distance, edge_weights=None, edge_attr=edge_attr)
      h = x[0]
      # print(h[0].shape)
    else:
      x = self.gnn_layers[i](h[0], edge_index, distance=distance, edge_weights=None, edge_attr=edge_attr)
      h = x
    h = self.norms[i](h)
    h = F.dropout(h, p=self.dropout, training=self.training)
    x[0] = h
    h = x

    # print(h[0].shape)

    # h = F.relu(h)
    data.x = h[0]
    return self.aggregate(data)

    x = self.pool(h, batch)
    x = self.graph_pred_linear(x)


    return x
    # if self.out_dim == 1:
    #   return x
    #
    # return F.log_softmax(x, dim=-1)

  def _loss(self, logits, target):
    return self._criterion(logits, target)

  # def _initialize_alphas(self):
  #
  #   num_na_ops = len(NA_PRIMITIVES)
  #   num_sc_ops = len(SC_PRIMITIVES)
  #   num_la_ops = len(LA_PRIMITIVES)
  #
  #
  #   self.log_na_alphas = Variable(torch.zeros(self.num_layers,num_na_ops).normal_(self._loc_mean, self._loc_std).cuda(), requires_grad=True)
  #   if self.num_layers>1:
  #     self.log_sc_alphas = Variable(torch.zeros(self.num_layers - 1, num_sc_ops).normal_(self._loc_mean, self._loc_std).cuda(), requires_grad=True)
  #   else:
  #     self.log_sc_alphas = Variable(torch.zeros(1, num_sc_ops).normal_(self._loc_mean, self._loc_std).cuda(), requires_grad=True)
  #
  #   self.log_la_alphas = Variable(torch.zeros(1, num_la_ops).normal_(self._loc_mean, self._loc_std).cuda(), requires_grad=True)
  #
  #   self._arch_parameters = [
  #     self.log_na_alphas,
  #     self.log_sc_alphas,
  #     self.log_la_alphas
  #   ]

  def arch_parameters(self):
    return self._arch_parameters

  # def genotype(self):
  #
  #   def _parse(na_weights, sc_weights, la_weights):
  #     gene = []
  #     na_indices = torch.argmax(na_weights, dim=-1)
  #     for k in na_indices:
  #         gene.append(NA_PRIMITIVES[k])
  #     #sc_indices = sc_weights.argmax(dim=-1)
  #     sc_indices = torch.argmax(sc_weights, dim=-1)
  #     for k in sc_indices:
  #         gene.append(SC_PRIMITIVES[k])
  #     #la_indices = la_weights.argmax(dim=-1)
  #     la_indices = torch.argmax(la_weights, dim=-1)
  #     for k in la_indices:
  #         gene.append(LA_PRIMITIVES[k])
  #     return '||'.join(gene)

    #gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    #gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
    # gene = _parse(F.softmax(self.log_na_alphas, dim=-1).data.cpu(), F.softmax(self.log_sc_alphas, dim=-1).data.cpu(),
    #               F.softmax(self.log_la_alphas, dim=-1).data.cpu())

    #concat = range(2+self._steps-self._multiplier, self._steps+2)
    #genotype = Genotype(
    #  normal=gene_normal, normal_concat=concat,
    #  reduce=gene_reduce, reduce_concat=concat
    #)
    # return gene





