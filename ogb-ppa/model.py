import torch
import torch.nn as nn
# from operations import *
from op_graph_classification import *
from torch.autograd import Variable
from torch_geometric.nn import global_mean_pool, global_add_pool
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import BatchNorm1d
from  torch_geometric.utils import add_self_loops,remove_self_loops,remove_isolated_nodes, degree
import pyximport
# import algos
from operations import make_degree, make_multihop_edges

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

class NaOp(nn.Module):
  def __init__(self, primitive, in_dim, out_dim, act, with_linear=False, with_act=True):
    super(NaOp, self).__init__()
    print(primitive)
    # self.bond_encoder = BondEncoder(emb_dim=in_dim)
    self._op = NA_OPS[primitive](in_dim, out_dim)
    if with_linear:
      self.op_linear = nn.Linear(in_dim, out_dim)
    if not with_act:
      act = 'linear'
    self.act = act_map(act)
    self.with_linear = with_linear

  def reset_params(self):
    self._op.reset_params()
    # self.op_linear.reset_parameters()

  def forward(self, x, edge_index, edge_weights, edge_attr):
    if self.with_linear:
      return self.act(self._op(x, edge_index, edge_weight=edge_weights, edge_attr=edge_attr) + self.op_linear(x))
    else:
      return self.act(self._op(x, edge_index, edge_weight=edge_weights, edge_attr=edge_attr))
    # mixed_res = []
    # for w, op in zip(weights, self._ops):
    #  mixed_res.append(w * F.relu(op(x, edge_index)))
    # return sum(mixed_res)

class SeOp(nn.Module):

  def __init__(self, primitives):
    super(SeOp, self).__init__()
    self._ops = []
    for primitive in primitives:
      self._ops.append(primitive)


  def forward(self, h_list):
    res = []
    for op, h in zip(self._ops, h_list):
      if op == 'true':
        res.append(h)

    return res

class FuOp(nn.Module):

  def __init__(self, primitive, hidden_size, num_block=None):
    super(FuOp, self).__init__()
    self._op = FU_OPS[primitive](hidden_size, num_block)


  def forward(self, x_list):
    if len(x_list) == 0:
      return 0
    res = self._op(x_list)

    return res

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
    # self.atom_encoder = AtomEncoder(hidden_size)


    # self.bond_encoder = BondEncoder(hidden_size)
    self.out_dim = out_dim
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.in_dropout = in_dropout
    self.out_dropout = out_dropout
    self.dropout = in_dropout
    self._criterion = criterion
    ops = genotype.split('||')
    print(ops)
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
    self.pool = global_add_pool

    self.norms = torch.nn.ModuleList()
    self.Se_layers = torch.nn.ModuleList()
    self.Fu_layers = torch.nn.ModuleList()

    self.node_encoder = torch.nn.Embedding(1, self.hidden_size)
    self.init_mlp = torch.nn.Linear(self.hidden_size, self.hidden_size)
    self.last_mlp = torch.nn.Linear(self.hidden_size, self.hidden_size)

    for layer in range(self.num_layers):
      self.norms.append(torch.nn.BatchNorm1d(hidden_size))

    #node aggregator op
    # self.lin1 = nn.Linear(hidden_size, hidden_size)
    index = 0
    selection_list = []
    for i in range(num_layers + 1):

        self.Se_layers.append(SeOp(ops[index:index + i + 1]))
        selection_list.append(ops[index:index + i + 1])
        # print(ops[index:index + i + 1])
        index = index + i + 1

    # print(selection_list)
    num_block = []
    for selection in selection_list:
        num = 0
        for select in selection:
            if select == 'true':
              num = num + 1

        num_block.append(num)
    #
    # print(num_block)
    # print(index)
    # print(ops[index])

    for i in range(num_layers + 1):
      self.Fu_layers.append(FuOp(ops[index], self.hidden_size, num_block[i]))
      index = index + 1

    if is_mlp:
      self.gnn_layers = nn.ModuleList([NaMLPOp(ops[index + i], hidden_size, hidden_size, act) for i in range(num_layers)])
    else:
      # acts from train_search or fine_tune
      act = [act for i in range(num_layers)]
      print(act)
      print(args.with_linear)
    self.gnn_layers = nn.ModuleList(
        [NaOp(ops[index + i], hidden_size, hidden_size, act, with_linear=args.with_linear, with_act=False) for i in range(num_layers)])
    self.act = act

    # self.in_degree_encoder = nn.Embedding(64, hidden_size, padding_idx=0)
    # self.out_degree_encoder = nn.Embedding(64, hidden_size, padding_idx=0)

    ### set the initial virtual node embedding to 0.
    # self.virtualnode_embedding = torch.nn.Embedding(1, hidden_size)
    # torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
    #
    # ### List of MLPs to transform virtual node at every layer
    # self.mlp_virtualnode_list = torch.nn.ModuleList()
    # for layer in range(num_layers - 1):
    #     self.mlp_virtualnode_list.append(
    #         torch.nn.Sequential(torch.nn.Linear(hidden_size, 2 * hidden_size), torch.nn.BatchNorm1d(2 * hidden_size),
    #                             torch.nn.ReLU(), \
    #                             torch.nn.Linear(2 * hidden_size, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU()))


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
    # data = make_degree(data)
    # data = make_multihop_edges(data, 3)
    x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
    # print(edge_attr.shape)
    ### virtual node embeddings for graphs
    # virtualnode_embedding = self.virtualnode_embedding(
    #     torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

    x = self.node_encoder(x)
    h_list = []

    # x = F.elu(self.init_mlp(x))
    # x = F.dropout(x, p=self.dropout, training=self.training)
    h_list.append(x)

    for i in range(0, self.num_layers):
        x_list = self.Se_layers[i](h_list)
        # print(x_list)
        if len(x_list) == 0:
          x2 = torch.zeros(x.shape).to(x.device)
          h_list.append(x2)
        else:
          x2 = self.Fu_layers[i](x_list)

          x = self.gnn_layers[i](x2, edge_index, edge_weights=None, edge_attr=edge_attr)
          x = F.elu(x)

          # x = F.dropout(x, p=self.dropout, training=self.training)

          h_list.append(x)

        # x2 = F.dropout(x2, p=self.dropout, training=self.training) + virtualnode_embedding[batch]
        # graph_representations[i] += virtualnode_embedding[batch]

        # if i < self.num_layers:
        #     ### add message from graph nodes to virtual nodes
        #     virtualnode_embedding_temp = global_add_pool(x, batch) + virtualnode_embedding
        #     ### transform virtual nodes using MLP
        #
        #     virtualnode_embedding = virtualnode_embedding + F.dropout(
        #           self.mlp_virtualnode_list[i - 1](virtualnode_embedding_temp), self.dropout, training=self.training)

    x_list = self.Se_layers[i + 1](h_list)
    x2 = self.Fu_layers[i + 1](x_list)
    x = F.elu(self.last_mlp(x2))


    x = F.dropout(x, p=self.dropout, training=self.training)

    x = self.pool(x, batch)

    x = self.graph_pred_linear(x)

    return x / 10
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





