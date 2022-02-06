import os
import os.path as osp
import sys
import time
import glob
import numpy as np
import torch
import utils_1 as utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch import cat
import pickle
from torch_geometric.loader import DataLoader
from torch.autograd import Variable
from model_search import Network
from architect import Architect
# from utils import gen_uniform_60_20_20_split, save_load_split
from ogb.nodeproppred import PygNodePropPredDataset
from dataset import load_data
from genotypes import NA_PRIMITIVES, NA_PRIMITIVES2, SC_PRIMITIVES, LA_PRIMITIVES
# from parallel_util import MyDataParallel
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull, Reddit, PPI
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import StratifiedKFold
from logging_util import init_logger
from sklearn.metrics import f1_score
graph_classification_dataset=['DD','MUTAG','PROTEINS','NCI1','NCI109','IMDB-BINARY','REDDIT-BINARY', 'BZR', 'COX2', 'IMDB-MULTI', 'COLORS-3','COLLAB', 'REDDIT-MULTI-5K', 'ogbg-molhiv', 'ogbg-molpcba']
node_classification_dataset = ['Cora','CiteSeer','PubMed','Amazon_Computers','Coauthor_CS','Coauthor_Physics','Amazon_Photo',
                               'small_Reddit','small_arxiv','Reddit','ogbn-arxiv']
parser = argparse.ArgumentParser("sane-train-search")
parser.add_argument('--dataset', type=str, default='ogbg-molpcba', help='location of the data corpus')
parser.add_argument('--record_time', action='store_true', default=False, help='used for run_with_record_time func')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers (default: 0)')
parser.add_argument('--learning_rate', type=float, default=0.0025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--epsilon', type=float, default=0.0, help='the explore rate in the gradient descent process')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=0.08, help='learning rate for arch encoding')
parser.add_argument('--arch_learning_rate_min', type=float, default=0.005, help='minimum learning rate for arch encoding')
# parser.add_argument('--cos_arch_lr', action='store_true', default=False, help='lr decay for learning rate')
parser.add_argument('--arch_weight_decay', type=float, default=1e-5, help='weight decay for arch encoding')
parser.add_argument('--with_conv_linear', type=bool, default=False, help=' in NAMixOp with linear op')
parser.add_argument('--num_layers', type=int, default=14, help='num of layers of GNN method.')
parser.add_argument('--withoutjk', action='store_true', default=False, help='remove la aggregtor')
parser.add_argument('--alpha_mode', type=str, default='train_loss', help='how to update alpha', choices=['train_loss', 'valid_loss', 'valid_acc'])
parser.add_argument('--search_act', action='store_true', default=True, help='search act in supernet.')
parser.add_argument('--hidden_size',  type=int, default=256, help='default hidden_size in supernet')
parser.add_argument('--BN',  type=int, default=256, help='default hidden_size in supernet')
parser.add_argument('--num_sampled_archs',  type=int, default=1, help='sample archs from supernet')

###for ablation stuty
parser.add_argument('--remove_pooling', action='store_true', default=True, help='remove pooling block.')
parser.add_argument('--remove_readout', action='store_true', default=True, help='exp5, only search the last readout block.')
parser.add_argument('--remove_jk', action='store_true', default=False, help='remove ensemble block, Graph representation = Z3')

#in the stage of update theta.
parser.add_argument('--temp', type=float, default=0.2, help=' temperature in gumble softmax.')
parser.add_argument('--loc_mean', type=float, default=10, help='initial mean value to generate the location')
parser.add_argument('--loc_std', type=float, default=0.01, help='initial std to generate the location')
parser.add_argument('--lamda', type=int, default=2, help='sample lamda architectures in calculate natural policy gradient.')
parser.add_argument('--adapt_delta', action='store_true', default=False, help='adaptive delta in update theta.')
parser.add_argument('--delta', type=float, default=1.0, help='a fixed delta in update theta.')
parser.add_argument('--w_update_epoch', type=int, default=1, help='epoches in update W')
parser.add_argument('--model_type', type=str, default='snas', help='how to update alpha', choices=['darts', 'snas'])

args = parser.parse_args()
args.graph_classification_dataset = graph_classification_dataset
args.node_classification_dataset = node_classification_dataset
torch.set_printoptions(precision=4)
def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    global device
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    args.save = 'logs/search-{}'.format(args.save)
    if not os.path.exists(args.save):
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_filename = os.path.join(args.save, 'log.txt')
    init_logger('', log_filename, logging.INFO, False)
    print('*************log_filename=%s************' % log_filename)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args.__dict__)
    num_features = num_classes = 0

    dataset = PygGraphPropPredDataset(name=args.dataset)
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # data, num_nodes = load_data(args.data, batch_size=args.batch_size, split_seed=args.seed)
    # num_features = data[0].num_features

    hidden_size = args.hidden_size

    if args.dataset =='PPI' or args.dataset == 'ogbg-molhiv':
        criterion = nn.BCEWithLogitsLoss()
        num_classes = 1
    elif args.dataset == 'ogbg-molpcba':
        criterion = nn.BCEWithLogitsLoss()
        num_classes = dataset.num_tasks
    else:
        criterion = F.nll_loss

    model = Network(args.dataset, criterion, num_features, num_classes, hidden_size, epsilon=args.epsilon,
                    args=args, with_conv_linear=args.with_conv_linear, num_layers=args.num_layers)
    print('with_conv_linear: ', args.with_conv_linear)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.Adam(
        model.parameters(),
        args.learning_rate,
        # momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    arch_optimizer = torch.optim.Adam(
        model.arch_parameters(),
        lr=args.arch_learning_rate,
        # betas=(0.5, 0.999),
        weight_decay=args.arch_weight_decay) #fix lr in arch_optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    scheduler_arch = torch.optim.lr_scheduler.CosineAnnealingLR(arch_optimizer, float(args.epochs), eta_min=args.arch_learning_rate_min)
    # scheduler_arch = torch.optim.lr_scheduler.ExponentialLR(arch_optimizer, 0.98)
    # architect = Architect(model, args) # arch_parameter.
    test_acc_with_time = []
    cur_t = 0
    for epoch in range(args.epochs):
        print(epoch)
        t1 = time.time()
        lr = scheduler.get_lr()[0]
        arch_lr = scheduler_arch.get_lr()[0]
        if epoch % 1 == 0:
            logging.info('epoch %d lr %e', epoch, lr)
            genotype = model.genotype()
            logging.info('genotype = %s', genotype)

        train_acc, train_obj = train_graph(train_loader, model, criterion, optimizer, arch_optimizer, lr, arch_lr)
        scheduler.step()

        # valid_acc, valid_obj = infer(data, model, criterion, test=False)
        # test_acc, test_obj = infer(data, model, criterion, test=True)
        valid_obj, valid_acc = infer_graph(valid_loader, model, criterion)
        test_obj, test_acc = infer_graph(test_loader, model, criterion)
        s_valid_obj, s_valid_acc = infer_graph(valid_loader, model, criterion, mode='evaluate_single_path')
        s_test_obj, s_test_acc = infer_graph(test_loader, model, criterion, mode='evaluate_single_path')
        scheduler_arch.step()

        if epoch % 1 == 0:
            logging.info('epoch=%s, train_acc=%f, train_loss=%f, valid_acc=%f, valid_loss=%f, test_acc=%f, test_loss=%f, explore_num=%s', epoch, train_acc, train_obj, valid_acc, valid_obj, test_acc, test_obj, model.explore_num)
            print('epoch={}, train_acc={:.04f}, train_loss={:.04f},valid_acc={:.04f}, valid_loss={:.04f},test_acc={:.04f},test_loss={:.04f}, explore_num={}'.
                  format(epoch, train_acc, train_obj, valid_acc, valid_obj, test_acc, test_obj, model.explore_num))

            logging.info('single path evaluate. epoch=%s, valid_acc=%f, valid_loss=%f, test_acc=%f, test_loss=%f, explore_num=%s', epoch, s_valid_acc, s_valid_obj, s_test_acc, s_test_obj, model.explore_num)
            print('single_path evaluation.  epoch={}, valid_acc={:.04f}, valid_loss={:.04f},test_acc={:.04f},test_loss={:.04f}, explore_num={}'.
                  format(epoch, s_valid_acc, s_valid_obj, s_test_acc, s_test_obj, model.explore_num))
        utils.save(model, os.path.join(args.save, 'weights.pt'))
        c_val_acc, c_test_acc = 0, 0
        if args.record_time:
            from fine_tune import tune_arch
            t2 = time.time()
            cur_t += (t2 - t1)
            genotype = model.genotype()
            val_acc, test_acc = tune_arch(genotype)
            c_val_acc = val_acc if val_acc > c_val_acc else c_val_acc
            c_test_acc = test_acc if test_acc > c_test_acc else c_test_acc
            test_acc_with_time.append('%.4f, %.4f, %.4f' % (cur_t, c_val_acc, c_test_acc))
    if args.record_time:
        wfilename = os.path.join(args.save, '%s_sane_acc_with_record_time.txt' % args.dataset)
        fw = open(wfilename, 'w+')
        fw.write('\n'.join(test_acc_with_time))
        print('test_acc_with_time saved in %s' % wfilename)
        logging.info('test_acc_with_time saved in %s' % wfilename)

    logging.shutdown()
    return genotype

def train_graph(train_loader, model, criterion, model_optimizer, arch_optimizer, lr, arch_lr):
    model.train()
    total_loss = 0
    y_true = []
    y_pred = []

    # data:[dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
    train_iters = train_loader.__len__()//args.w_update_epoch + 1
    print('train_iters:{},train_data_num:{}'.format(train_iters, range(train_iters * args.w_update_epoch)))
    from itertools import cycle

    zip_train_data = list(zip(range(train_iters * args.w_update_epoch), cycle(train_loader)))

    # for train_data, valid_data in zip_list:
    # print(count_parameters(model))
    for iter in range(train_iters):
        print('iter' + str(iter))
        arch_optimizer.zero_grad()
        for i in range(args.w_update_epoch):
            model_optimizer.zero_grad()
            train_data = zip_train_data[iter*args.w_update_epoch+i][1].to(device)
            print(train_data)
            if train_data.x.shape[0] == 1 or train_data.batch[-1] == 0:
                pass

            output, sample_z = model(train_data)
            # print('size:', output.size(), train_data.y.reshape([-1, 1]).size())
            # print('output:', torch.cat([output, train_data.y.reshape([-1, 1])], dim=-1))
            # print('prediction:{}'.format(output[0:5,:]))
            # print('label:{}'.format(train_data.y.view(-1)[0:5]))
            output = output.to(device)
            is_labeled = train_data.y == train_data.y

            #error loss and resource loss
            error_loss = criterion(output.to(torch.float32)[is_labeled], train_data.y.to(torch.float32)[is_labeled])
            y_true.append(train_data.y.view(output.shape).detach().cpu())
            y_pred.append(output.detach().cpu())

            print('loss:{:.08f}'.format(error_loss.item()))

            total_loss += error_loss.item()
            # arch_optimizer.zero_grad()
            error_loss.backward(retain_graph=True)
            model_optimizer.step()
        # model_optimizer.zero_grad()
        # print('################# z:')
        # for i in [0,2,3,4]:
        #     # print('z:{}, z.grad:{},alphas:{},grad:{}'.format(sample_z[i], sample_z[i].grad, model.arch_parameters()[i], model.arch_parameters()[i].grad))
        #     print('z:{}'.format(sample_z[i]))
        if args.alpha_mode =='train_loss':
            arch_optimizer.step()

        del train_data, error_loss, output
        torch.cuda.empty_cache()
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    print(y_true.shape)
    print(y_pred.shape)
    input_dict = {'y_true':y_true, 'y_pred':y_pred}
    evaluator = Evaluator(args.dataset)
        # print('#################alphas updated:')
        # for i in range(5):
        #     print('alphas:{}'.format(model.arch_parameters()[i]))
    return evaluator.eval(input_dict)['ap'], total_loss / len(train_loader)

def infer_graph(loader, model, criterion, mode='none'):

    model.eval()
    # valid_loss,test_loss = 0, 0
    # valid_acc, test_acc = 0, 0
    total_loss, acc = 0, 0
    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            logits, _ = model(data, mode=mode)
            logits = logits.to(device)
        target = data.y
        is_labeled = target[:, 0] == target[:, 0]


        loss = criterion(logits[is_labeled].float(), target.to(torch.float32)[is_labeled])
        total_loss += loss.item()
        y_true.append(target.view(logits.shape).detach().cpu())
        y_pred.append(logits.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    evaluator = Evaluator(args.dataset)

    return total_loss/len(loader), evaluator.eval(input_dict)['ap']

def run_by_seed():
    res = []
    for i in range(args.num_sampled_archs):
        print('searched {}-th for {}...'.format(i+1, args.dataset))
        args.save = '{}-{}'.format(args.dataset, time.strftime("%Y%m%d-%H%M%S"))
        genotype = main()
        res.append('seed={},genotype={},saved_dir={}'.format(args.seed, genotype, args.save))
    filename = './exp_res/%s-searched_res-%s-eps%s-reg%s.txt' % (args.dataset, time.strftime('%Y%m%d-%H%M%S'), args.epsilon, args.weight_decay)
    fw = open(filename, 'w+')
    fw.write('\n'.join(res))
    fw.close()
    print('searched res for {} saved in {}'.format(args.dataset, filename))


if __name__ == '__main__':
        cls_criterion = torch.nn.BCEWithLogitsLoss()
        reg_criterion = torch.nn.MSELoss()
        run_by_seed()

