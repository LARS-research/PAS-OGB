import os.path as osp
from torch import cat
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T
import torch.nn.functional as F
from ogb.graphproppred import PygGraphPropPredDataset
from wrapper import MyGraphPropPredDataset, MyZINCDataset
import ogb
import numpy as np

import torch
from torch import tensor
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from functools import partial
from collator import collator

class HandleNodeAttention(object):
    def __call__(self, data):
        data.attn = torch.softmax(data.x[:, 0], dim=0)
        data.x = data.x[:, 1:]
        return data
def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.uint8, device=index.device)
    mask[index] = 1
    return mask
class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

def derive_colors(dataset, ratio=0.1):
    labels = []
    for data in dataset:
        labels.append(data.y.item())

    from collections import Counter
    count_labels = Counter(labels)
    print(count_labels)
    data_mask = torch.zeros(len(labels), dtype=torch.uint8, device=data.y.device)

    labels = torch.tensor(labels)
    for i in range(len(count_labels)):
        idx = torch.where(labels == i)[0]
        sampled_idx = int(count_labels[i]*ratio)
        print(i, sampled_idx, len(idx))
        data_mask[idx[:sampled_idx]] = 1
    print(data_mask.sum())
    return dataset[data_mask]

def get_dataset(dataset_name='abaaba'):
    global dataset
    # if dataset is not None:
    #     return dataset

    # max_node is set to max(max(num_val_graph_nodes), max(num_test_graph_nodes))
    if dataset_name == 'ogbg-molpcba':
        dataset = {
            'num_class': 128,
            'loss_fn': F.binary_cross_entropy_with_logits,
            'metric': 'ap',
            'metric_mode': 'max',
            'evaluator': ogb.graphproppred.Evaluator('ogbg-molpcba'),
            'dataset': MyGraphPropPredDataset('ogbg-molpcba', root='../../dataset'),
            'max_node': 128,
        }
    elif dataset_name == 'ogbg-molhiv':
        dataset = {
            'num_class': 1,
            'loss_fn': F.binary_cross_entropy_with_logits,
            'metric': 'rocauc',
            'metric_mode': 'max',
            'evaluator': ogb.graphproppred.Evaluator('ogbg-molhiv'),
            'dataset': MyGraphPropPredDataset(name="ogbg-molhiv"),
            'max_node': 128,
        }
    elif dataset_name == 'PCQM4M-LSC':
        dataset = {
            'num_class': 1,
            'loss_fn': F.l1_loss,
            'metric': 'mae',
            'metric_mode': 'min',
            'evaluator': ogb.lsc.PCQM4MEvaluator(),
            'dataset': MyPygPCQM4MDataset(root='../../dataset'),
            'max_node': 128,
        }
    elif dataset_name == 'ZINC':
        dataset = {
            'num_class': 1,
            'loss_fn': F.l1_loss,
            'metric': 'mae',
            'metric_mode': 'min',
            'evaluator': ogb.lsc.PCQM4MEvaluator(),  # same objective function, so reuse it
            'train_dataset': MyZINCDataset(subset=True, root='../../dataset/pyg_zinc', split='train'),
            'valid_dataset': MyZINCDataset(subset=True, root='../../dataset/pyg_zinc', split='val'),
            'test_dataset': MyZINCDataset(subset=True, root='../../dataset/pyg_zinc', split='test'),
            'max_node': 128,
        }
    else:
        raise NotImplementedError

    print(f' > {dataset_name} loaded!')
    print(dataset)
    print(f' > dataset info ends')
    return dataset

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def load_data(dataset_name='DD', cleaned=False, split_seed=12345, batch_size=32, remove_large_graph=True):
    # torch.cuda.manual_seed_all(split_seed)
    # torch.manual_seed(split_seed)
    #
    # torch.backends.cudnn.deterministic = True

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')

    if dataset_name == 'COLORS-3':
        dataset = TUDataset(path, 'COLORS-3', use_node_attr=True,
                            transform=HandleNodeAttention())
        dataset=derive_colors(dataset)

    elif dataset_name == 'ogbg-molhiv':
        # dataset_1 = get_dataset(dataset_name="ogbg-molhiv")['dataset']
        dataset = PygGraphPropPredDataset(name="ogbg-molhiv")
        rf_pred = np.load('rf_preds/rf_final_pred.npy')
        dataset.data.y = torch.cat((dataset.data.y, torch.from_numpy(rf_pred)), 1)
    elif dataset_name == 'ogbg-molpcba':
        dataset = PygGraphPropPredDataset(name="ogbg-molpcba", root='/data/wangxu/dataset')
    elif dataset_name == 'ogbg-ppa':
        dataset = PygGraphPropPredDataset(name="ogbg-ppa", root='/data/wangxu/dataset', transform = add_zeros)
    else:
        dataset = TUDataset(path, dataset_name, cleaned=cleaned)
    # dataset.data.edge_attr = None
    #load and process

    # if dataset.data.x is None:
    #     max_degree = 0
    #     degs = []
    #     for data in dataset:
    #         degs += [degree(data.edge_index[0], dtype=torch.long)]
    #         max_degree = max(max_degree, degs[-1].max().item())
    #
    #     if max_degree < 1000:
    #         dataset.transform = T.OneHotDegree(max_degree)
    #     else:
    #         deg = torch.cat(degs, dim=0).to(torch.float)
    #         mean, std = deg.mean().item(), deg.std().item()
    #         dataset.transform = NormalizedDegree(mean, std)

    #for diffpool method: remove latge graphs
    num_nodes = max_num_nodes = 0

    num_dataset = PygGraphPropPredDataset(name="ogbg-molhiv").data,
    for data in num_dataset:
        num_nodes += data.num_nodes
        max_num_nodes = max(data.num_nodes, max_num_nodes)

    # # Filter out a few really large graphs in order to apply DiffPool.
    if dataset_name == 'REDDIT-BINARY':
        num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
    else:
        num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)
    #
    #
    # if remove_large_graph:
    #     indices = []
    #     for i, data in enumerate(dataset):
    #         if data.num_nodes <= num_nodes:
    #             indices.append(i)
    #     dataset = dataset[torch.tensor(indices)]


    #split 811
    if dataset_name == 'ogbg-molpcba' or dataset_name == 'ogbg-ppa':
        split = dataset.get_idx_split()
        train_dataset = dataset[split['train']]
        val_dataset = dataset[split['valid']]
        test_dataset = dataset[split['test']]
        print(dataset[0])

    elif dataset_name == 'ogbg-molhiv':
        split = dataset.get_idx_split()
        train_dataset = dataset[split['train']]
        val_dataset = dataset[split['valid']]
        test_dataset = dataset[split['test']]
    else:
        skf = StratifiedKFold(10, shuffle=True, random_state=split_seed)
        idx = [torch.from_numpy(i) for _, i in skf.split(torch.zeros(len(dataset)), dataset.data.y[:len(dataset)])]
        split = [cat(idx[:8], 0), cat(idx[8:9], 0), cat(idx[9:], 0)]

        train_dataset = dataset[split[0]]
        val_dataset = dataset[split[1]]
        test_dataset = dataset[split[2]]

    # train_dataset = dataset[index_to_mask(split[0], len(dataset))]
    # val_dataset = dataset[index_to_mask(split[1], len(dataset))]
    # test_dataset = dataset[index_to_mask(split[2], len(dataset))]

    # print('train:{}, val:{}, test:{}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))

    num_workers = 0
    multi_hop_max_dist = 5
    spatial_pos_max = 1024

    num_features = dataset[0].num_features

    train_loader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)


    # train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    # Graphs g_i has n_i nodes. In each batch, reorder all nodes (\sum_{i=1}^{batch\_size}n_i) and assign new id,
    # resort edge_index according the new ids.
    # record the graph id in data.batch.

    # print('batch_size:{}, num_batch:{}'.format(batch_size, train_loader.__len__()))
    # for test_batch in test_loader:
    #     print('x.size:{}, edge_index_size:{}, batch.size:{}'.format(test_batch.x.size(), test_batch.edge_index.size(), test_batch.batch.size()))
    #     import numpy as np
    #     set_batch = np.array(test_batch.batch)
    #     print('batch.set:{}'.format(set(set_batch)))
    # from pooling_zoo import ASAPooling_mix
    # pooling = ASAPooling_mix(in_channels=data.x.size()[1])
    # for test_batch in test_loader:
    #     print('x.size:{}, edge_index_size:{}, batch.size:{}'.format(test_batch.x.size(), test_batch.edge_index.size(), test_batch.batch.size()))
    #     x, edge_index, edge_weight, batch, perm = pooling(test_batch.x, test_batch.edge_index, batch=test_batch.batch)
    #     print('x.size:{}, edge_index_size:{}, edge_weight:{}, batch.size:{}, '.format(x.size(), edge_index.size(), edge_weight.size(), batch.size()))

    # return [train_loader, val_loader, test_loader], num_nodes, num_features
    return [dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader], num_nodes

def load_k_fold(dataset, folds, batch_size):
    print('10fold split')
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y[:len(dataset)]):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    data_10fold = []
    for i in range(folds):
        data_ith = [0, 0, 0, 0] #align with 811 split process.
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        # train_indices.append(train_mask.nonzero().view(-1))
        # print('start_idx:', torch.where(train_mask == 1)[0][0], torch.where(val_indices[i]==1)[0][0], torch.where(test_indices[i]==1)[0][0])

        train_mask = train_mask.nonzero().view(-1)

        data_ith.append(DataLoader(dataset[train_mask], batch_size, shuffle=True))
        data_ith.append(DataLoader(dataset[val_indices[i]], batch_size, shuffle=True))
        data_ith.append(DataLoader(dataset[test_indices[i]], batch_size, shuffle=True))
        data_10fold.append(data_ith)

    return data_10fold









