import os
from datetime import datetime
import time
import argparse
import json
import pickle
import logging
import numpy as np
from torch_geometric.data import DataLoader

import hyperopt
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK
import random

import torch
import statistics
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from model_dropnode import NetworkGNN as Network
from DeeperGCN.utils.ckpt_util import save_ckpt

graph_classification_dataset=['DD', 'MUTAG', 'PROTEINS', 'NCI1', 'NCI109','IMDB-BINARY', 'REDDIT-BINARY', 'BZR', 'COX2', 'IMDB-MULTI','COLORS-3', 'COLLAB', 'REDDIT-MULTI-5K', 'ogbg-molhiv', 'ogbg-molpcba']
node_classification_dataset = ['Cora', 'CiteSeer', 'PubMed', 'Amazon_Computers', 'Coauthor_CS', 'Coauthor_Physics', 'Amazon_Photo']

def get_args():
    parser = argparse.ArgumentParser("sane")
    parser.add_argument('--gpu', type=int, default=7, help='gpu device id')
    parser.add_argument('--checkpoint_path', type=str, default="./model_0206_gamma_500/")
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers (default: 0)')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--gamma', type=float, default=300)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--loss', type=str, default='auroc', help='')
    parser.add_argument('--data', type=str, default='ogbg-molhiv', help='location of the data corpus')
    parser.add_argument('--model_save_path', type=str, default='model_drop',
                        help='the directory used to save models')
    parser.add_argument('--add_virtual_node', action='store_true')
    parser.add_argument('--arch_filename', type=str, default='', help='given the location of searched res')
    parser.add_argument('--arch', type=str, default='', help='given the specific of searched res')
    parser.add_argument('--num_layers', type=int, default=14, help='num of GNN layers in SANE')
    parser.add_argument('--tune_topK', action='store_true', default=False, help='whether to tune topK archs')
    parser.add_argument('--use_hyperopt', action='store_true', default=False, help='whether to tune topK archs')
    parser.add_argument('--record_time', action='store_true', default=False, help='whether to tune topK archs')
    parser.add_argument('--with_linear', action='store_true', default=False, help='whether to use linear in NaOp')
    parser.add_argument('--with_layernorm', action='store_true', default=False, help='whether to use layer norm')
    parser.add_argument('--with_layernorm_learnable', action='store_true', default=False, help='use the learnable layer norm')
    parser.add_argument('--BN',  action='store_true', default=True,  help='use BN.')
    parser.add_argument('--flag', action='store_true', default=False,  help='use flag.')
    parser.add_argument('--feature', type=str, default='full',
                        help='two options: full or simple')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--optimizer', type=str, default='pesg', help='')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate set for optimizer.')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='the dimension of embeddings of nodes and edges')

    parser.add_argument('--batch_size', type=int, default=512, help='batch size of data.')
    parser.add_argument('--model', type=str, default='SANE')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--is_mlp', action='store_true', default=False, help='is_mlp')
    parser.add_argument('--ft_weight_decay', action='store_true', default=False, help='with weight decay in finetune stage.')
    parser.add_argument('--ft_dropout', action='store_true', default=False, help='with dropout in finetune stage')
    parser.add_argument('--ft_mode', type=str, default='811', choices=['811', '622', '10fold'], help='data split function.')
    parser.add_argument('--hyper_epoch', type=int, default=1, help='hyper epoch in hyperopt.')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs for each model')
    parser.add_argument('--cos_lr',  action='store_true', default=False,  help='use cos lr.')
    parser.add_argument('--lr_min',  type=float, default=0.0,  help='use cos lr.')
    parser.add_argument('--show_info',  action='store_true', default=True,  help='print training info in each epoch')
    parser.add_argument('--withoutjk', action='store_true', default=False, help='remove la aggregtor')
    parser.add_argument('--search_act', action='store_true', default=False, help='search act in supernet.')
    parser.add_argument('--one_pooling', action='store_true', default=False, help='only one pooling layers after 2th layer.')
    parser.add_argument('--seed', type=int, default=0, help='seed for finetune')
    parser.add_argument('--remove_pooling', action='store_true', default=True,
                        help='remove pooling block.')
    parser.add_argument('--remove_readout', action='store_true', default=True,
                        help='remove readout block. Only search the last readout block.')
    parser.add_argument('--remove_jk', action='store_true', default=False,
                        help='remove ensemble block. In the last readout block,use global sum. Graph representation = Z3')
    parser.add_argument('--fixpooling', type=str, default='null',
                        help='use fixed pooling functions')
    parser.add_argument('--fixjk',action='store_true', default=False,
                        help='use concat,rather than search from 3 ops.')

    # flag
    parser.add_argument('--step_size', type=float, default=1e-3)
    parser.add_argument('-m', type=int, default=3)
    parser.add_argument('--test_freq', type=int, default=1)
    parser.add_argument('--attack', type=str, default='none')
    parser.add_argument('--save', type=str, default='EXP', help='experiment nam ')

    global args
    args = parser.parse_args()
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("HYPEROPT_FMIN_SEED", str(args.seed))

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, device, loader, optimizer, task_type, grad_clip=0.):
    loss_list = []
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            # print(model.device)
            # print(batch.device)
            optimizer.zero_grad()
            pred = model(batch)
            # pred = torch.sigmoid(pred)
            is_labeled = batch.y[:,0] == batch.y[:,0]
            loss = aucm_criterion(pred.to(torch.float32)[is_labeled].reshape(-1, 1), batch.y[:,0:1].to(torch.float32)[is_labeled].reshape(-1, 1))
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(),
                    grad_clip)

            optimizer.step()

            loss_list.append(loss.item())
    return statistics.mean(loss_list)


@torch.no_grad()
def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            pred = model(batch)
            # pred = torch.sigmoid(pred)
            y_true.append(batch.y[:,0:1].view(pred.shape).detach().cpu()) # remove random forest pred
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true,
                  "y_pred": y_pred}

    return evaluator.eval(input_dict)

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu)

    sub_dir = 'BS_{}-NF_{}'.format(args.batch_size, args.feature)
    set_all_seeds(args.seed)
    dataset = PygGraphPropPredDataset(name=args.dataset)

    # Load RF predictions
    # npy = os.listdir('rf_preds')[args.seed]
    # rf_pred = np.load(os.path.join('rf_preds', npy))
    npy = 'rf_preds/rf_pred_auc_0.8324_0.8310_RS_5.npy'
    rf_pred = np.load(npy)
    print(npy)
    dataset.data.y = torch.cat((dataset.data.y, torch.from_numpy(rf_pred)), 1)

    args.num_tasks = dataset.num_tasks
    # logging.info('%s' % args)

    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    evaluator = Evaluator(args.dataset)
    split_idx = dataset.get_idx_split()

    set_all_seeds(args.seed)
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    set_all_seeds(args.seed)
    criterion = aucm_criterion.cuda()
    lines = open(args.arch_filename, 'r').readlines()
    suffix = args.arch_filename.split('_')[-1][:-4]
    arch_set = set()

    for ind, l in enumerate(lines):
        # with open('tuned_res/%s_res_%s_%s.pkl' % (args1.data, tune_str, suffix), 'wb+') as fw:
        #     test={'a':[1, 2, 3], 'b':('string','abc'),'c':'hello'}
        #     pickle.dump(test, fw)
        try:
            print('**********process {}-th/{}'.format(ind+1, len(lines)))
            logging.info('**********process {}-th/{}**************8'.format(ind+1, len(lines)))
            res = {}
            #iterate each searched architecture
            parts = l.strip().split(',')
            arch = parts[1].split('=')[1]
            args.arch = arch
            if arch in arch_set:
                logging.info('the %s-th arch %s already searched....info=%s', ind+1, arch, l.strip())
                continue
            else:
                arch_set.add(arch)
        except Exception as e:
            logging.info('errror occured for %s-th, arch_info=%s, error=%s', ind + 1, l.strip(), e)
            import traceback
            traceback.print_exc()
    genotype = args.arch
    # print(genotype)
    model = Network(genotype, aucm_criterion, args.hidden_size, 1, args.hidden_size,
                    num_layers=args.num_layers, in_dropout=args.dropout,
                    out_dropout=args.dropout,
                    act=args.activation, args=args, is_mlp=args.is_mlp)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')
    model = model.to(device)
    if True:
        # checkpoint_path = './model_ckpt_seed2/'
        checkpoint_path = args.checkpoint_path
        best_pth = sorted(os.listdir(checkpoint_path))[-1]
        # best_pth = 'BS_256-NF_full_valid_best_AUC-FP_E_48_R1.pth'
        args.model_load_path = os.path.join(checkpoint_path, best_pth)
        print(args.model_load_path)
        trained_stat_dict = torch.load(args.model_load_path)['model_state_dict']
        # trained_stat_dict.pop('graph_pred_linear.weight', None)
        ##trained_stat_dict.pop('graph_pred_linear.bias', None)
        model.load_state_dict(trained_stat_dict, strict=False)

        # fr = open("./tuned_res/ogbg-molhiv_res_20220117-014247_res-20220111-161817-eps0.0-reg1e-05.pkl", 'rb')
        # inf = pickle.load(fr)
        # print(inf)

    optimizer = PESG(model,
                     a=aucm_criterion.a,
                     b=aucm_criterion.b,
                     alpha=aucm_criterion.alpha,
                     lr=args.lr,
                     gamma=args.gamma,
                     margin=args.margin,
                     weight_decay=args.weight_decay)

    # get imbalance ratio from train set
    args.imratio = float((train_loader.dataset.data.y[:, 0].sum() / train_loader.dataset.data.y[:, 0].shape[0]).numpy())
    aucm_criterion.p = args.imratio
    print(aucm_criterion.p)

    # save
    datetime_now = '2022-01-17'
    pretrained_prefix = 'pre_' if args.pretrained else ''
    virtual_node_prefilx = '-vt' if args.add_virtual_node else ''
    args.configs = '[%s]Train_%s_im_%.4f_rd_%s_%s%s-FP_%s_%s_wd_%s_lr_%s_B_%s_E_%s_%s_%s_g_%s_m_%s' % (
    datetime_now, args.dataset, args.imratio, args.seed, pretrained_prefix, args.arch,
    virtual_node_prefilx, args.activation, args.weight_decay, args.lr, args.batch_size, args.epochs, args.loss,
    args.optimizer, args.gamma, args.margin)
    logging.info(args.save)
    logging.info(args.configs)

    results = {'highest_valid': 0,
               'final_train': 0,
               'final_test': 0,
               'highest_train': 0}

    start_time = time.time()
    start_time_local = time.time()
    for epoch in range(1, args.epochs + 1):

        if epoch in [int(args.epochs * 0.33), int(args.epochs * 0.66)] and args.loss != 'ce':
            optimizer.update_regularizer(decay_factor=5)

        epoch_loss = train(model, device, train_loader, optimizer, dataset.task_type, grad_clip=0.)
        # epoch_loss = 1

        # logging.info('Evaluating...')
        train_result = eval(model, device, train_loader, evaluator)[dataset.eval_metric]
        # train_result = 0.99
        valid_result = eval(model, device, valid_loader, evaluator)[dataset.eval_metric]
        test_result = eval(model, device, test_loader, evaluator)[dataset.eval_metric]

        print("Epoch:%s, train_auc:%.4f, valid_auc:%.4f, test_auc:%.4f, lr:%.4f, time:%.4f" % (
        epoch, train_result, valid_result, test_result, optimizer.lr, time.time() - start_time_local))
        start_time_local = time.time()
        # model.print_params(epoch=epoch)

        if train_result > results['highest_train']:
            results['highest_train'] = train_result

        if valid_result > results['highest_valid']:
            results['highest_valid'] = valid_result
            results['final_train'] = train_result
            results['final_test'] = test_result

            save_ckpt(model, optimizer,
                      round(epoch_loss, 4), epoch,
                      args.model_save_path,
                      sub_dir, name_post='valid_best_AUC-FP_E_%s_R%s' % (epoch, args.seed))

    logging.info("%s" % results)

    end_time = time.time()
    total_time = end_time - start_time
    logging.info('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))


if __name__ == "__main__":
    get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu)
    cls_criterion = torch.nn.BCEWithLogitsLoss()
    reg_criterion = torch.nn.MSELoss()

    # https://github.com/Optimization-AI/LibAUC
    aucm_criterion = AUCMLoss().to(device)
    main()
