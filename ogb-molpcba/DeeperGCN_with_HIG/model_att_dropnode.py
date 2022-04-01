import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from gcn_lib.sparse.torch_vertex import GENConv
from gcn_lib.sparse.torch_nn import norm_layer, MLP
import logging
import numpy as np
import numpy, scipy.sparse
import scipy.sparse as sp


def get_adj_matrix(edge_index_fea, N):
    adj = torch.zeros([N, N])
    adj[edge_index_fea[0, :], edge_index_fea[1, :]] = 1
    Asp = scipy.sparse.csr_matrix(adj)
    Asp = Asp + Asp.T.multiply(Asp.T > Asp) - Asp.multiply(Asp.T > Asp)
    Asp = Asp + sp.eye(Asp.shape[0])

    D1_ = np.array(Asp.sum(axis=1)) ** (-0.5)
    D2_ = np.array(Asp.sum(axis=0)) ** (-0.5)
    D1_ = sp.diags(D1_[:, 0], format='csr')
    D2_ = sp.diags(D2_[0, :], format='csr')
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
        x = torch.spmm(A, x).detach_()
        y.add_(x)
    return y.div_(order + 1.0).detach_()


class DeeperGCN(torch.nn.Module):
    def __init__(self, args):
        super(DeeperGCN, self).__init__()

        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.block = args.block
        self.conv_encode_edge = args.conv_encode_edge
        self.add_virtual_node = args.add_virtual_node

        hidden_channels = args.hidden_channels
        num_tasks = args.num_tasks
        conv = args.conv
        aggr = args.gcn_aggr
        t = args.t
        self.learn_t = args.learn_t
        p = args.p
        self.learn_p = args.learn_p
        y = args.y
        self.learn_y = args.learn_y

        self.beta = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

        self.msg_norm = args.msg_norm
        learn_msg_scale = args.learn_msg_scale
        self.activation_func = F.relu if args.activations == 'relu' else F.elu

        norm = args.norm
        mlp_layers = args.mlp_layers

        graph_pooling = args.graph_pooling

        print('The number of layers {}'.format(self.num_layers),
              'Aggr aggregation method {}'.format(aggr),
              'block: {}'.format(self.block))
        if self.block == 'res+':
            print('LN/BN->ReLU->GraphConv->Res')
        elif self.block == 'res':
            print('GraphConv->LN/BN->ReLU->Res')
        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')
        elif self.block == "plain":
            print('GraphConv->LN/BN->ReLU')
        else:
            raise Exception('Unknown block Type')

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        if self.add_virtual_node:
            self.virtualnode_embedding = torch.nn.Embedding(1, hidden_channels)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

            self.mlp_virtualnode_list = torch.nn.ModuleList()

            for layer in range(self.num_layers - 1):
                self.mlp_virtualnode_list.append(MLP([hidden_channels] * 3,
                                                     norm=norm))

        for layer in range(self.num_layers):
            if conv == 'gen':
                gcn = GENConv(hidden_channels, hidden_channels,
                              aggr=aggr,
                              t=t, learn_t=self.learn_t,
                              p=p, learn_p=self.learn_p,
                              y=y, learn_y=self.learn_p,
                              msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                              encode_edge=self.conv_encode_edge, bond_encoder=True,
                              norm=norm, mlp_layers=mlp_layers)
            else:
                raise Exception('Unknown Conv Type')
            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)

        if not self.conv_encode_edge:
            self.bond_encoder = BondEncoder(emb_dim=hidden_channels)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise Exception('Unknown Pool Type')

        self.graph_pred_linear = torch.nn.Linear(hidden_channels, num_tasks)

    def forward(self, input_batch, mode='train'):
        x = input_batch.x

        edge_index = input_batch.edge_index
        edge_attr = input_batch.edge_attr
        batch = input_batch.batch

        new_fea_list = []
        for i in range(input_batch.y.shape[0]):
            new_fea = self.atom_encoder(input_batch[i].x)

            edge_index_fea = input_batch[i].edge_index
            N = new_fea.size(0)
            drop_rate = 0.2

            if self.training:
                drop_rates = torch.FloatTensor(np.ones(N) * drop_rate)
                masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
                new_fea = masks.cuda() * new_fea
            else:
                new_fea = new_fea * (1. - drop_rate)
            ori_fea = new_fea
            adj = get_adj_matrix(edge_index_fea, N).to(edge_index.device)
            order = 1
            new_fea = propagate(new_fea, adj, order)
            new_fea_list.append(new_fea)
        h = torch.cat(new_fea_list, dim=0)

        # h = self.atom_encoder(x)

        if self.add_virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
            h = h + virtualnode_embedding[batch]

        if self.conv_encode_edge:
            edge_emb = edge_attr
        else:
            edge_emb = self.bond_encoder(edge_attr)

        if self.block == 'res+':

            h = self.gcns[0](h, edge_index, edge_emb)

            for layer in range(1, self.num_layers):
                h1 = self.norms[layer - 1](h)
                # h2 = self.activation_func(h1)
                if layer == 1:
                    h2 = torch.sigmoid(h1)
                elif layer == 2:
                    h2 = F.relu6(h1)
                elif layer == 2:
                    h2 = F.relu6(h1)
                elif layer == 3:
                    h2 = F.relu(h1)
                elif layer == 4:
                    h2 = F.tanh(h1)
                elif layer == 5:
                    h2 = torch.sigmoid(h1)
                elif layer == 6:
                    h2 = F.relu6(h1)
                elif layer == 7:
                    h2 = F.relu(h1)
                elif layer == 8:
                    h2 = F.relu6(h1)
                elif layer == 9:
                    h2 = F.leaky_relu(h1)
                elif layer == 10:
                    h2 = torch.sigmoid(h1)
                elif layer == 11:
                    h2 = F.relu6(h1)
                elif layer == 12:
                    h2 = F.leaky_relu(h1)
                elif layer == 13:
                    h2 = F.relu(h1)
                elif layer == 14:
                    h2 = F.relu(h1)
                h2 = F.dropout(h2, p=self.dropout, training=self.training)

                if self.add_virtual_node:
                    virtualnode_embedding_temp = global_add_pool(h2, batch) + virtualnode_embedding
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer - 1](virtualnode_embedding_temp),
                        self.dropout, training=self.training)

                    h2 = h2 + virtualnode_embedding[batch]

                h = self.gcns[layer](h2, edge_index, edge_emb) + h

            h = self.norms[self.num_layers - 1](h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'res':

            h = self.activation_func(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                h = self.activation_func(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')

        elif self.block == 'plain':

            h = self.activation_func(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                if layer != (self.num_layers - 1):
                    h = self.activation_func(h2)
                else:
                    h = h2
                h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            raise Exception('Unknown block Type')

        h_graph = self.pool(h, batch)  # N, 256

        dcn_pred = self.graph_pred_linear(h_graph)
        rf_pred = input_batch.y[:, 2]
        # return torch.sigmoid(dcn_pred)
        return (1 - self.beta) * torch.sigmoid(dcn_pred).reshape(-1, 1) + (self.beta) * rf_pred.reshape(-1, 1)

    def print_params(self, epoch=None, final=False):

        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                print('Final t {}'.format(ts))
            else:
                logging.info('Epoch {}, t {}'.format(epoch, ts))
        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                print('Final p {}'.format(ps))
            else:
                logging.info('Epoch {}, p {}'.format(epoch, ps))
        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                print('Final s {}'.format(ss))
            else:
                logging.info('Epoch {}, s {}'.format(epoch, ss))
