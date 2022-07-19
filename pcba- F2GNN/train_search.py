import os
import sys
import time
import glob
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model_search import Network
import numpy as np
from logging_util import init_logger
import hyperopt
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK, rand
from genotypes import SC_PRIMITIVES, NA_PRIMITIVES, FF_PRIMITIVES
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from tqdm import trange, tqdm
from torch_geometric.loader import DataLoader

parser = argparse.ArgumentParser("sane-train-search")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--record_time', action='store_true', default=False, help='used for run_with_record_time func')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=4, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--epsilon', type=float, default=0.0, help='the explore rate in the gradient descent process')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
# parser.add_argument('--transductive', action='store_true', help='use transductive settings in train_search.')
parser.add_argument('--sample_num', type=int, default=1, help='sample numbers of the supernet')
# parser.add_argument('--cal_grad', type=bool, default=False, help='calculate the path gradient in each epoch.')
parser.add_argument('--hidden_size',  type=int, default=256, help='default hidden_size in supernet')
parser.add_argument('--train_subset', action='store_true')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers (default: 0)')

#search space
parser.add_argument('--alpha_step', type=int, default=1, help='alpha update step comparing with w.')
parser.add_argument('--num_layers', type=int, default=4, help='framework layers')
parser.add_argument('--agg', type=str, default='sage', help='aggregations used in this framework')
parser.add_argument('--search_agg', type=bool, default=False, help='search aggregators')

#search algo
parser.add_argument('--algo', type=str, default='darts', help='search algorithm', choices=['darts', 'snas','random','bayes'])
parser.add_argument('--alpha_mode', type=str, default='valid', help='update alpha, with train/valid data', choices=['train', 'valid'])
parser.add_argument('--loc_mean', type=float, default=1, help='initial mean value to generate the location')
parser.add_argument('--loc_std', type=float, default=0.01, help='initial std to generate the location')
parser.add_argument('--temp', type=float, default=0.5, help='temp in softmax')
parser.add_argument('--temp_min', type=float, default=0, help='min temp in softmax')
parser.add_argument('--cos_temp', type=bool, default=False, help='temp decay')
parser.add_argument('--w_update_epoch', type=int, default=1, help='epoches in update W')

#random baseline
parser.add_argument('--random_epoch', type=int, default=0, help='for ramdom baseline,  the num of sampled architectures.')

#training trick
parser.add_argument('--layer_norm', type=bool, default=False, help='use layer norm in trainging supernet.')
parser.add_argument('--batch_norm', type=bool, default=False, help='use batch norm in trainging supernet.')
parser.add_argument('--with_conv_linear', type=bool, default=False, help='add extra linear in convs.')

args = parser.parse_args()

def main(log_filename):
    global device
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    print('*************log_filename=%s************' % log_filename)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args.__dict__)

    # load dataset
    dataset = PygGraphPropPredDataset(name="ogbg-molpcba", root='/data/wangxu/dataset')
    metric = dataset.eval_metric
    split_idx = dataset.get_idx_split()

    if args.train_subset:
        subset_ratio = 0.1
        subset_idx = torch.randperm(len(split_idx["train"]))[:int(subset_ratio*len(split_idx["train"]))]
        train_dataset = dataset[split_idx["train"][subset_idx]]
        train_loader = DataLoader(dataset[split_idx["train"][subset_idx]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    else:
        train_dataset = dataset[split_idx["train"]]
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    valid_dataset = dataset[split_idx["valid"]]
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    hidden_size = args.hidden_size
    num_features = dataset.num_features
    num_classes = 128

    # criterion = nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion = criterion.cuda()

    model = Network(criterion, num_features, num_classes, hidden_size, args=args)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    model_optimizer = torch.optim.Adam(
        model.parameters(),
        args.learning_rate,
        # momentum=args.momentum,
        # weight_decay=args.weight_decay
    )

    arch_optimizer = torch.optim.Adam(
        model.arch_parameters(),
        lr=args.arch_learning_rate,
        )
    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    temp_scheduler = utils.Temp_Scheduler(int(args.epochs/2), args.temp, args.temp, temp_min=args.temp_min)

    search_cost = 0
    global epoch

    for epoch in range(args.epochs):
        t1 = time.time()
        lr = model_scheduler.get_last_lr()[0]
        if args.cos_temp and epoch >= int(args.epochs/2):
            model.temp = temp_scheduler.step()
        else:
            model.temp = args.temp
        train_loss, train_ap = train(train_loader, model, criterion, model_optimizer, arch_optimizer, metric)
        model_scheduler.step()
        t2 = time.time()
        search_cost += (t2 - t1)

        # if (epoch + 1) % 10 == 0:
        if epoch % 1 == 0:
            valid_loss, valid_ap = infer(valid_loader, model, criterion, metric)
            print(
                'epoch={}, train_loss={:.04f}, train_ap={:.04f}, val_loss={:.04f}, valid_ap={:.04f}, explore_num={}'.format(
                    epoch, train_loss, train_ap, valid_loss, valid_ap, model.explore_num))

        if epoch % 1 == 0:
            logging.info('epoch %d lr %e', epoch, lr)
            if args.algo in ['random', 'bayes']:
                genotype = model.genotype(sample=True)
            else:
                genotype = model.genotype()
            logging.info('genotype = %s', genotype)

    logging.info('The search process costs %.2fs', search_cost)
    return genotype, valid_ap


def train(data, model, criterion, model_optimizer, arch_optimizer, metric, dataset_name = 'none'):
    return train_trans(data, model, criterion, model_optimizer, arch_optimizer, metric, alpha_mode=args.alpha_mode)
def infer(data, model, criterion, metric, single_path=False, dataset_name = 'none'):
    val_loss, val_mae = infer_trans(data, model, criterion, metric, test=False, single_path=single_path)
    # test_loss, test_acc = infer_trans(data, model, criterion, test=True, single_path=single_path)
    return val_loss, val_mae

def train_trans(train_loader, model, criterion, model_optimizer, arch_optimizer, metric, alpha_mode='train'):
    model.train()
    total_loss = 0
    y_true = []
    y_pred = []

    for step, loader in enumerate(tqdm(train_loader, desc="iteration")):
        #zero grad
        loader = loader.to(device)
        if loader.x.shape[0] == 1 or loader.batch[-1] == 0:
            pass
        else:
            if args.algo in ['random', 'bayes']:
                logits = model.forward_rb(loader, gnn)
            else:
                logits = model(loader)

            model_optimizer.zero_grad()
            arch_optimizer.zero_grad()

            is_labeled = loader.y == loader.y
            train_loss = criterion(logits.to(torch.float32)[is_labeled], loader.y.to(torch.float32)[is_labeled])
            train_loss.backward()
            model_optimizer.step()

            y_true.append(loader.y.detach().cpu())
            y_pred.append(logits.detach().cpu())

            #update alpha
            if alpha_mode == 'train':
                arch_optimizer.step()
                total_loss += train_loss.detach().cpu().item()
            else:
                model_optimizer.zero_grad()
                arch_optimizer.zero_grad()
                val_loss = criterion(model(loader), loader.y)
                total_loss += val_loss.detach().cpu().item()
                val_loss.backward()
                arch_optimizer.step()
    
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    evaluator = Evaluator('ogbg-molpcba')

    return train_loss.item(), evaluator.eval(input_dict)[metric]

def infer_trans(data, model, criterion, metric, test=False, single_path=False):
    model.eval()
    valid_loss = 0

    y_val_true = []
    y_val_pred = []

    for step, val_data in enumerate(tqdm(data, desc='iteration')):
        val_data = val_data.to(device)
        with torch.no_grad():
            if args.algo in ['random', 'bayes']:
                logits = model.forward_rb(val_data, gnn)
            else:
                logits = model(val_data, single_path=single_path)
        loss = criterion(logits, val_data.y.view(logits.shape))
        valid_loss += loss.detach().cpu().item()

        y_val_true.append(val_data.y.view(logits.shape).detach().cpu())
        y_val_pred.append(logits.detach().cpu())

    y_val_true = torch.cat(y_val_true, dim=0)
    y_val_pred = torch.cat(y_val_pred, dim=0)


    val_input_dict = {'y_true': y_val_true, 'y_pred': y_val_pred}
    evaluator = Evaluator('ogbg-molpcba')

    return valid_loss, evaluator.eval(val_input_dict)[metric]

def run_by_seed():
    res = []
    print('searched archs for pcba...')
    args.save = 'pcba-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
    args.save = 'logs/search-{}'.format(args.save)
    if not os.path.exists(args.save):
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    log_filename = os.path.join(args.save, 'log.txt')
    if not os.path.exists(log_filename):
        init_logger('', log_filename, logging.INFO, False)

    for i in range(args.sample_num):
        seed = np.random.randint(0, 10000)
        args.seed = seed
        genotype, val_ap = main(log_filename)
        res.append('seed={},genotype={},saved_dir={},val_ap={}'.format(seed, genotype, args.save, val_ap))
    filename = 'exp_res/pcba-searched_res-%s-eps%s-reg%s.txt' % (time.strftime('%Y%m%d-%H%M%S'), args.epsilon, args.weight_decay)
    fw = open(filename, 'w+')
    fw.write('\n'.join(res))
    fw.close()
    print('searched res for pcba saved in {}'.format(filename))

def generate_args(arch_args):
    arch = []
    num_edges = (args.num_layers + 2) * (args.num_layers + 1) / 2
    for i in range(int(num_edges)):
        arch.append(arch_args['edge'+str(i)])
    for i in range(args.num_layers + 1):
        arch.append(arch_args['fuse'+str(i)])
    if args.search_agg:
        for i in range(args.num_layers):
            arch.append(arch_args['agg'+str(i)])
    return torch.tensor(arch)

def rb_objective(args):
    print('current_hyper:', args)
    global gnn
    gnn = generate_args(args)
    genotype, valid_acc, test_acc = main(log_filename)
    return {
        'loss': -valid_acc,
        'archs': genotype,
        'test_acc': test_acc,
        'status': STATUS_OK,
        'eval_time': round(time.time(), 2),
    }
def rb_search():
    res = []
    print('searched archs for {}... with {}'.format(args.data, args.algo))
    args.save = '{}-{}'.format(args.data, time.strftime("%Y%m%d-%H%M%S"))
    args.save = 'logs/search-{}'.format(args.save)
    if not os.path.exists(args.save):
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    global log_filename
    log_filename = os.path.join(args.save, 'log.txt')
    if not os.path.exists(log_filename):
        init_logger('', log_filename, logging.INFO, False)

    rb_search_space = {}
    num_edges = (args.num_layers + 2) * (args.num_layers + 1) / 2
    for i in range(int(num_edges)):
        rb_search_space['edge'+str(i)] = hp.choice('edge'+str(i), [i for i in range(len(SC_PRIMITIVES))])
    for i in range(args.num_layers + 1):
        rb_search_space['fuse'+str(i)] = hp.choice('fuse'+str(i), [i for i in range(len(FF_PRIMITIVES))])
    if args.search_agg:
        for i in range(args.num_layers):
            rb_search_space['agg'+str(i)] = hp.choice('agg'+str(i), [i for i in range(len(NA_PRIMITIVES))])

    trials = Trials()
    if args.algo =='random':
        best = fmin(rb_objective, rb_search_space, algo=rand.suggest, max_evals=args.random_epoch, trials=trials)
    elif args.algo == 'bayes':
        n_startup_jobs = int(args.random_epoch/5)
        best = fmin(rb_objective, rb_search_space, algo=partial(tpe.suggest, n_startup_jobs=n_startup_jobs),
                    max_evals=args.random_epoch, trials=trials)

    space = hyperopt.space_eval(rb_search_space, best)
    print(trials.results)
    val_accs = []
    for d in trials.results:
        val_accs.append(-d['loss'])
    val_accs = np.array(val_accs)
    topk = np.argsort(val_accs)[-5:]
    for i in topk:
        d = trials.results[i]
        print(d)
        res.append('seed={},genotype={},saved_dir={},val_acc={},test_acc={}'.format(args.seed, d['archs'], args.save, -d['loss'], d['test_acc']))
    filename = 'exp_res/%s-searched_res-%s-eps%s-reg%s.txt' % (args.data, time.strftime('%Y%m%d-%H%M%S'), args.epsilon, args.weight_decay)
    fw = open(filename, 'w+')
    fw.write('\n'.join(res))
    fw.close()
    print('searched res for {} saved in {}'.format(args.data, filename))


if __name__ == '__main__':
    if args.algo in ['random', 'bayes']:
        # random/bayesian
        rb_search()
    else:
        run_by_seed() 