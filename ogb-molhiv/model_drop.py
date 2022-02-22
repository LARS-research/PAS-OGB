import torch
import torch.nn as nn
# from operations import *
from op_graph_classification import *
from torch.autograd import Variable
from torch_geometric.nn import global_mean_pool, global_add_pool
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import BatchNorm1d
from torch_geometric.utils import add_self_loops, remove_self_loops, remove_isolated_nodes, degree
import pyximport
import numpy as np
import numpy, scipy.sparse
import scipy.sparse as sp
from torch_geometric.utils import dropout_adj, subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import dropout_adj


# import algos

def get_adj_matrix(edge_index_fea, N):
    adj = torch.zeros([N, N])
    adj[edge_index_fea[0, :], edge_index_fea[1, :]] = 1
    Asp = scipy.sparse.csr_matrix(adj)
    Asp = Asp + Asp.T.multiply(Asp.T > Asp) - Asp.multiply(Asp.T > Asp)
    Asp = Asp + sp.eye(Asp.shape[0])

    D1_ = np.array(Asp.sum(axis=1))**(-0.5)
    D2_ = np.array(Asp.sum(axis=0))**(-0.5)
    D1_ = sp.diags(D1_[:,0], format='csr')
    D2_ = sp.diags(D2_[0,:], format='csr')
    A_ = Asp.dot(D1_)
    A_ = D2_.dot(A_)
    A_ = sparse_mx_to_torch_sparse_tensor(A_)
    return A_

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def propagate(feature, A, order):
    x = feature
    y = feature
    for i in range(order):
        # print('x')
        # print(x)
        x = torch.spmm(A, x).detach_()
        # print(x)
        y.add_(x)
        # print(y)
        # print(y.div_(order+1.0).detach_())
    return y.div_(order+1.0).detach_()

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

class DropNode(nn.Module):
    """
    DropNode: Sampling node using a uniform distribution.
    """

    def __init__(self, drop_rate):
        super(DropNode, self).__init__()
        self.drop_rate = drop_rate

    def forward(self, edge_index, edge_attr=None, edge_weight=None, num_nodes=None):
        if not self.training:
            return edge_index, edge_attr

        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        nodes = torch.arange(num_nodes, dtype=torch.int64)
        mask = torch.full_like(nodes, 1 - self.drop_rate, dtype=torch.float32)
        mask = torch.bernoulli(mask).to(torch.bool)
        subnodes = nodes[mask]

        return subgraph(subnodes, edge_index, edge_attr=edge_attr, num_nodes=num_nodes)


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

    def forward(self, x, edge_index, edge_weights, data, batch, mask):
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

    def __init__(self, genotype, criterion, in_dim, out_dim, hidden_size, num_layers=3, in_dropout=0, out_dropout=0.5,
                 act='elu', args=None, is_mlp=False, num_nodes=0):
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
        self.prelu = nn.PReLU()
        self.out_dropout = out_dropout
        self.dropout = in_dropout
        self._criterion = criterion
        self.beta = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.dropnode = DropNode(drop_rate=0.1)
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
        self.pooling_ratios = [[0.1], [0.25, 0.25], [0.5, 0.5, 0.5], [0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7],
                               [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                               [1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7],
                               [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8],
                               [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9],
                               [1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10],
                               [1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11],
                               [1 / 12, 1 / 12, 1 / 12, 1 / 12, 1 / 12, 1 / 12, 1 / 12, 1 / 12, 1 / 12, 1 / 12, 1 / 12,
                                1 / 12],
                               [1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13,
                                1 / 13, 1 / 13],
                               [1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13,
                                1 / 13, 1 / 13]]
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers + 1):
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_size))
        if self.args.data in ['NCI1', 'NCI109']:
            self.pooling_ratios = [[0.1], [0.5, 0.5], [0.5, 0.5, 0.5], [0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7],
                                   [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]]

        # if num_layers == 1:
        #     self.pooling_ratio = [0.1]
        # elif num_layers == 2:
        #     self.pooling_ratio = [0.25, 0.25]
        # elif num_layers == 3:
        #     self.pooling_ratio = [0.5, 0.5, 0.5]
        # elif num_layers == 4:
        #     self.pooling_ratio = [0.6, 0.6, 0.6, 0.6]
        # elif num_layers == 5:
        #     self.pooling_ratio = [0.7, 0.7, 0.7, 0.7, 0.7]
        # elif num_layers == 6:
        # it shoule be [num_layers-1]
        self.pooling_ratio = self.pooling_ratios[num_layers - 1]
        # print('genotype:', genotype)

        # node aggregator op
        # self.lin1 = nn.Linear(hidden_size, hidden_size)
        if is_mlp:
            self.gnn_layers = nn.ModuleList([NaMLPOp(ops[i], hidden_size, hidden_size, act) for i in range(num_layers)])
        else:
            # acts from train_search or fine_tune
            if self.args.search_act:
                act = ops[num_layers: num_layers * 2]
                print(act)
            else:
                act = [act for i in range(num_layers)]
                print(act)
                print(args.with_linear)
            self.gnn_layers = nn.ModuleList(
                [NaOp(ops[i], hidden_size, hidden_size, act, with_linear=args.with_linear, with_act=False) for i in
                 range(num_layers)])

        # if self.args.one_pooling:
        #   poolops = [ops[num_layers*2+i] if i in [1, 3] else 'none' for i in range(num_layers)]
        #   num_pool_ops = num_layers//2
        #   # it should be self.pooling_ratios[num_pool_ops-1] but i forget [-1]. it doesn't matter.
        #   self.pooling_ratio = [self.pooling_ratios[num_pool_ops-1][0] for i in range(num_layers)]
        # elif self.args.remove_pooling:
        #   poolops = ['none' for i in range(num_layers)]
        # else:
        #   poolops = [ops[num_layers*2+i] for i in range(num_layers)]
        #
        # if self.args.fixpooling != 'null':
        #   # use a fix pooling
        #   if self.args.one_pooling:
        #     poolops = [self.args.fixpooling if i in [1, 3] else 'none' for i in range(num_layers)]
        #   else:
        #     poolops = [self.args.fixpooling for i in range(num_layers)]

        # self.pooling_layers = nn.ModuleList(
        #   [PoolingOp(poolops[i], hidden_size, self.pooling_ratio[i]) for i in range(num_layers)])

        # nonop = [ops[num_layers*3 + i] != 'none' for i in range(num_layers+1)]
        # print('____________________________________nonop:', nonop)
        # nonop = sum(nonop)
        # if nonop == 0:
        #   # ops[num_layers*4 + 1] ='global_sum'
        #   ops[-2] ='global_sum'
        #   nonop=1
        # if self.args.remove_jk:
        #   ops[-2] = 'global_sum'
        # if self.args.remove_readout:
        #   if ops[-2] == 'none':
        #     ops[-2] = 'global_sum'
        # self.readout_layers = nn.ModuleList(
        #   [ReadoutOp(ops[num_layers*3 + i], hidden_size) for i in range(num_layers+1)])

        # learnable_LN
        if self.args.with_layernorm_learnable:
            self.lns_learnable = torch.nn.ModuleList()
            for i in range(self.num_layers):
                self.lns_learnable.append(torch.nn.BatchNorm1d(hidden_size))

        # layer aggregator op
        # if self.args.fixjk:
        #   self.layer6 = LaOp('l_concat', hidden_size, 'linear', num_layers+1)
        # else:
        #   self.layer6 = LaOp(ops[-1], hidden_size, 'linear', num_layers+1)

        # self.lin_output = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, out_dim)

    # other feature computation
    # def convert_to_single_emb(self, x, offset=512):
    #     feature_num = x.size(1) if len(x.size()) > 1 else 1
    #     feature_offset = 1 + \
    #                      torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    #     x = x + feature_offset.to(x.device)
    #     return x

    # def preprocess_item(self, item):
    #     edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    #     N = x.size(0)
    #     # x = self.convert_to_single_emb(x)
    #
    #     # node adj matrix [N, N] bool
    #     adj = torch.zeros([N, N], dtype=torch.bool)
    #     adj[edge_index[0, :], edge_index[1, :]] = True
    #
    #     # edge feature here
    #     # if len(edge_attr.size()) == 1:
    #     #     edge_attr = edge_attr[:, None]
    #     # attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    #     # attn_edge_type[edge_index[0, :], edge_index[1, :]
    #     # ] = self.convert_to_single_emb(edge_attr) + 1
    #     # print('edge feature done')
    #
    #     # shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    #     # print('shortest path done')
    #     # max_dist = np.amax(shortest_path_result)
    #     # edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    #     # spatial_pos = torch.from_numpy((shortest_path_result)).long()
    #     # print('edge input done')
    #     # attn_bias = torch.zeros(
    #     #     [N + 1, N + 1], dtype=torch.float)  # with graph token
    #
    #     # combine
    #     item.x = x
    #     # item.adj = adj
    #     # item.attn_bias = attn_bias
    #     # item.attn_edge_type = attn_edge_type
    #     # item.spatial_pos = spatial_pos
    #     item.in_degree = adj.long().sum(dim=1).view(-1)
    #     item.out_degree = adj.long().sum(dim=0).view(-1)
    #     # item.edge_input = torch.from_numpy(edge_input).long()
    #
    #     return item

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
        edge_index, batch, edge_attr = data.edge_index, data.batch, data.edge_attr

        new_fea_list = []
        # for every graph
        # for i in range(data.y.shape[0]):
        #     # 每张图的所有节点特征进入atom_encoder
        #     new_fea = self.atom_encoder(data[i].x)
        #     # get edge_index
        #     edge_index_fea = data[i].edge_index
        #     # for adjacency matrix
        #     N = new_fea.size(0)
        #     drop_rate = 0.2
        #
        #     if self.training:
        #         # 利用伯努利分布，得到随机率为dropout的节点Mask
        #         drop_rates = torch.FloatTensor(np.ones(N) * drop_rate)
        #         masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
        #         # 利用mask，随机将一些节点feature置0
        #         new_fea = masks.to(new_fea.device) * new_fea
        #     else:
        #         # 否则，直接让所有feature降低，为了保持图的能量一致
        #         new_fea = new_fea * (1. - drop_rate)
        #     ori_fea = new_fea
        #     # 得到邻接矩阵,本质上是根据节点的度，设置ratio，比如节点度为3，每个邻居得到的信息的ratio就是0.33
        #     adj = get_adj_matrix(edge_index_fea, N).to(edge_index.device)
        #     order = 1
        #     new_fea = propagate(new_fea, adj, order)
        #     new_fea_list.append(new_fea)
        #
        # x = torch.cat(new_fea_list, dim =0)

        x = self.atom_encoder(data.x)
        # edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=0.025, training = self.training)


        # in_degree, out_degree = data.in_degree.to(x.device), data.out_degree.to(x.device)

        ### virtual node embeddings for graphs
        # virtualnode_embedding = self.virtualnode_embedding(
        #   torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        # row, col = edge_index
        # deg = degree(row, x.size(0), dtype=x.dtype) + 1

        # mgf_maccs_pred = data.y[:, 2]

        # if self.args.data == 'ogbg-molhiv':
        #     # flag
        #     # x = self.atom_encoder(x) + perturb if perturb is not None else self.atom_encoder(x)
        #     x = self.atom_encoder(x)

            # x = x + self.in_degree_encoder(deg) + self.out_degree_encoder(deg)
            # edge_attr = self.bond_encoder(edge_attr)

        # degree
        # x = self.deg_BN(deg) + self.atom_BN(x)

        # x = F.elu(self.conv1(x, edge_index, edge_attr))

        # add self_loop
        # edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size()[0])

        # print('init shape', x.size(), batch.size())
        # generate weights by softmax
        # graph_representations = []

        # input linear
        # x = F.elu(self.lin1(x.float()))

        # tmp_res = self.readout_layers[0](x, batch, None)
        # if tmp_res != None:
        #   graph_representations.append(tmp_res)

        # readout change
        # graph_representations.append(self.readout_layers[0](x, batch, None))
        # graph_representations.append(x)

        # x = F.dropout(x, p=self.in_dropout, training=self.training)
        # edge_weights = torch.ones(edge_index.size()[1], device=edge_index.device).float()
        x = self.gnn_layers[0](x, edge_index, edge_weights=None, edge_attr=edge_attr)
        for i in range(1, self.num_layers):
            x1 = self.batch_norms[i - 1](x)
            x2 = F.relu(x1)
            # x2 = self.prelu(x1)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
            # graph_representations[i] += virtualnode_embedding[batch]
            x = self.gnn_layers[i](x2, edge_index, edge_weights=None, edge_attr=edge_attr) + x
            # print('evaluate data {}-th gnn:'.format(i), x.size(), batch.size())
            # if self.args.with_layernorm_learnable:
            #   x = self.lns_learnable[i](x)
            # elif self.args.with_layernorm:
            #   layer_norm = nn.LayerNorm(normalized_shape=x.size(), elementwise_affine=False)
            #   x = layer_norm(x)

            # print()

            # x, edge_index, _, _ = self.pooling_layers[i](x, edge_index, edge_weights, data, batch, None)
            # x, edge_index, batch, _ = self.pooling_layers[i](x, edge_index, None, data, batch, None)
            # print('evaluate data {}-th pooling:'.format(i), x.size(), batch.size())

            # residual
            # x += graph_representations[i]

            # graph_representations.append(self.readout_layers[i+1](x, batch, None))
            # graph_representations.append(x)

            # if i < self.num_layers - 1:
            #   ### add message from graph nodes to virtual nodes
            #   virtualnode_embedding_temp = global_add_pool(graph_representations[i], batch) + virtualnode_embedding
            #   virtualnode_embedding = virtualnode_embedding + F.dropout(
            #     self.mlp_virtualnode_list[i](virtualnode_embedding_temp), self.in_dropout,
            #     training=self.training)
            ## transform virtual nodes using MLP

        # x = self.conv1(x, edge_index, edge_attr)
        # x = self.batch_norms[i + 1](x)
        # x = F.dropout(x, p=self.in_dropout, training=self.training)
        # graph_representations.append(x)
        x = self.batch_norms[self.num_layers - 1](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.dropout(x, p=0.2, training=self.training)
        # if self.args.remove_jk or self.args.remove_readout:
        #   x = graph_representations[-1]
        # else:
        #   x = self.layer6(graph_representations)

        # read_out:
        # x = self.readout_layers[i + 1](x, batch, None)

        # x = self.readout_layers[0](x, batch, None)
        x = self.pool(x, batch)
        # x = self.last_pool(x, batch)

        # out linear
        # x = F.elu(self.lin_output(x), inplace=True)
        # x = F.dropout(x, p=self.out_dropout, training=self.training)

        gnn_pred = self.classifier(x)
        # return gnn_pred
        rf_pred = data.y[:, 2]

        # mol_fingerprint
        # h_graph_final = torch.cat((logits, mgf_maccs_pred.reshape(-1, 1)), 1)
        # att = torch.nn.functional.softmax(h_graph_final * self.beta, -1)

        # return torch.sum(h_graph_final * att, -1).reshape(-1, 1)
        # return torch.sigmoid(gnn_pred)
        # return rf_pred.reshape(-1, 1)

        if self.out_dim == 1:
            return (1-self.beta)*torch.sigmoid(gnn_pred).reshape(-1, 1) + (self.beta) * rf_pred.reshape(-1, 1)

        return F.log_softmax(x, dim=-1)

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

    # gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    # gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
    # gene = _parse(F.softmax(self.log_na_alphas, dim=-1).data.cpu(), F.softmax(self.log_sc_alphas, dim=-1).data.cpu(),
    #               F.softmax(self.log_la_alphas, dim=-1).data.cpu())

    # concat = range(2+self._steps-self._multiplier, self._steps+2)
    # genotype = Genotype(
    #  normal=gene_normal, normal_concat=concat,
    #  reduce=gene_reduce, reduce_concat=concat
    # )
    # return gene