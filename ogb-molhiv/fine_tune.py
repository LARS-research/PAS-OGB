import os
from datetime import datetime
import time
import argparse
import json
import pickle
import logging
import numpy as np

import hyperopt
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK
import random
from logging_util import init_logger
from train4tune import main
from test4tune import main as test_main
import torch

sane_space ={'model': 'SANE',
         # 'hidden_size': hp.choice('hidden_size', [16, 32, 64, 128, 256, 512]),
         'hidden_size': hp.choice('hidden_size', [256]),
         # 'hidden_size': hp.choice('hidden_size', [128, ]),
         # 'learning_rate': hp.uniform("lr", 0.005, 0.05),
         'learning_rate': hp.uniform("lr", 0.05, 0.0500001),
         'weight_decay': hp.uniform("wr", -5, -4),
         # 'optimizer': hp.choice('opt', ['adam', 'adagrad']),
         'optimizer': hp.choice('opt', ['pseg']),
         # 'in_dropout': hp.randint('in_dropout', 10),
         'in_dropout': hp.choice('in_dropout', [0.2]),
         # 'out_dropout': hp.randint('out_dropout', 10),
         'out_dropout': hp.choice('out_dropout', [0.5]),
         # 'activation': hp.choice('act', ['relu', 'elu'])
         'activation': hp.choice('act', ['relu'])
         # 'activation': hp.choice('act', ["sigmoid", "tanh", "relu", "softplus", "leaky_relu", "relu6", "elu"])
         }
graph_classification_dataset=['DD', 'MUTAG', 'PROTEINS', 'NCI1', 'NCI109','IMDB-BINARY', 'REDDIT-BINARY', 'BZR', 'COX2', 'IMDB-MULTI','COLORS-3', 'COLLAB', 'REDDIT-MULTI-5K', 'ogbg-molhiv', 'ogbg-molpcba']
node_classification_dataset = ['Cora', 'CiteSeer', 'PubMed', 'Amazon_Computers', 'Coauthor_CS', 'Coauthor_Physics', 'Amazon_Photo']
def get_args():
    parser = argparse.ArgumentParser("sane")
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--model_save_path', type=str, default='model_ckpt',
                        help='the directory used to save models')
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

    parser.add_argument('--batch_size', type=int, default=512, help='batch size of data.')
    parser.add_argument('--is_mlp', action='store_true', default=False, help='is_mlp')
    parser.add_argument('--ft_weight_decay', action='store_true', default=False, help='with weight decay in finetune stage.')
    parser.add_argument('--ft_dropout', action='store_true', default=False, help='with dropout in finetune stage')
    parser.add_argument('--ft_mode', type=str, default='811', choices=['811', '622', '10fold'], help='data split function.')
    parser.add_argument('--hyper_epoch', type=int, default=1, help='hyper epoch in hyperopt.')
    parser.add_argument('--epochs', type=int, default=300, help='training epochs for each model')
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

    global args1
    args1 = parser.parse_args()
    random.seed(args1.seed)
    torch.cuda.manual_seed_all(args1.seed)
    torch.manual_seed(args1.seed)
    np.random.seed(args1.seed)
    os.environ.setdefault("HYPEROPT_FMIN_SEED", str(args1.seed))
class ARGS(object):

    def __init__(self):
        super(ARGS, self).__init__()

def generate_args(arg_map):
    args = ARGS()
    for k, v in arg_map.items():
        setattr(args, k, v)
    setattr(args, 'rnd_num', 1)

    args.ft_mode = args1.ft_mode
    if args1.ft_dropout:
        args.in_dropout = args.in_dropout / 10.0
        args.out_dropout = args.out_dropout / 10.0
    else:
        args.in_dropout = args.out_dropout = 0

    if args1.ft_weight_decay:
        args.weight_decay = 10**args.weight_decay
    else:
        args.weight_decay = 0


    args.data = args1.data
    #args.save = '{}_{}'.format(args.data, datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
    #args1.save = 'logs/tune-{}'.format(args.save)

    args.graph_classification_dataset = graph_classification_dataset
    args.node_classification_dataset = node_classification_dataset

    # if args.data in args.graph_classification_dataset or args.data =='PPI':
    #     args.epochs = 140
    # elif args.data =='CiteSeer':
    #     args.epochs = 300
    # else:
    #     args.epochs = 400

    args.epochs = args1.epochs
    args.is_mlp = args1.is_mlp
    args.batch_size = args1.batch_size
    args.arch = args1.arch
    args.gpu = args1.gpu
    args.num_layers = args1.num_layers
    args.seed = args1.seed
    args.grad_clip = 0.
    args.momentum = 0.9
    args.cos_lr = args1.cos_lr
    args.lr_min = args1.lr_min
    args.BN = args1.BN
    args.with_linear = args1.with_linear
    args.with_layernorm = args1.with_layernorm
    args.with_layernorm_learnable = args1.with_layernorm_learnable
    args.show_info = args1.show_info
    args.withoutjk = args1.withoutjk
    args.search_act = args1.search_act
    args.one_pooling = args1.one_pooling
    args.model_save_path = args1.model_save_path
    args.feature = args1.feature

    args.remove_pooling = args1.remove_pooling
    args.remove_jk = args1.remove_jk
    args.remove_readout = args1.remove_readout
    args.fixpooling = args1.fixpooling
    args.fixjk = args1.fixjk
    args.flag = args1.flag
    args.step_size = args1.step_size
    args.test_freq = args1.test_freq
    args.attack = args1.attack
    args.m = args1.m

    return args

def objective(args):
    args = generate_args(args)

    # try:
    #     vali_acc, test_acc, test_std, args = main(args)
    # except:
    #     vali_acc, test_acc, test_std, args = 0, 0, 0, args
    #     print('OOM')
    vali_acc, test_acc, test_std, args = main(args)
    print(args)
    # vali_acc, test_acc, test_std, args = main(args)
    return {
        'loss': -vali_acc,
        'test_acc': test_acc,
        'test_std': test_std,
        'status': STATUS_OK,
        'eval_time': round(time.time(), 2),
        }

def run_fine_tune():

    tune_str = time.strftime('%Y%m%d-%H%M%S')
    path = 'logs/tune-%s_%s' % (args1.data, tune_str)
    if not os.path.exists(path):
        os.mkdir(path)
    log_filename = os.path.join(path, 'log.txt')
    init_logger('fine-tune', log_filename, logging.INFO, False)

    lines = open(args1.arch_filename, 'r').readlines()

    suffix = args1.arch_filename.split('_')[-1][:-4]  # need to re-write the suffix?

    test_res = []
    arch_set = set()

    if args1.search_act:
        #search act in train supernet. Finetune stage remove act search.
        sane_space['activation'] = hp.choice("act", [0])

    if not args1.ft_weight_decay:
        sane_space['weight_decay'] = hp.choice("wr", [0])
    if not args1.ft_dropout:
        sane_space['in_dropout'] = hp.choice('in_dropout', [0])
        sane_space['out_dropout'] = hp.choice('out_dropout', [0])
    if args1.data in ['COLLAB', 'REDDIT-MULTI-5K', 'NCI1','NCI109','DD','COX2']:
        sane_space['learning_rate'] = hp.uniform("lr", 0.005, 0.02)
    for ind, l in enumerate(lines):
        # with open('tuned_res/%s_res_%s_%s.pkl' % (args1.data, tune_str, suffix), 'wb+') as fw:
        #     test={'a':[1, 2, 3], 'b':('string','abc'),'c':'hello'}
        #     pickle.dump(test, fw)
        try:
            print('**********process {}-th/{}, logfilename={}**************'.format(ind+1, len(lines), log_filename))
            logging.info('**********process {}-th/{}**************8'.format(ind+1, len(lines)))
            res = {}
            #iterate each searched architecture
            parts = l.strip().split(',')
            arch = parts[1].split('=')[1]
            args1.arch = arch
            if arch in arch_set:
                logging.info('the %s-th arch %s already searched....info=%s', ind+1, arch, l.strip())
                continue
            else:
                arch_set.add(arch)
            res['searched_info'] = l.strip()

            start = time.time()
            trials = Trials()
            #tune with validation acc, and report the test accuracy with the best validation acc
            best = fmin(objective, sane_space, algo=partial(tpe.suggest, n_startup_jobs=int(args1.hyper_epoch/5)), max_evals=args1.hyper_epoch, trials=trials)

            space = hyperopt.space_eval(sane_space, best)
            args = generate_args(space)
            print('best space is ', args.__dict__)
            res['tuned_args'] = args.__dict__

            record_time_res = []
            c_vali_acc, c_test_acc = 0, 0
            #report the test acc with the best vali acc
            for d in trials.results:
                if -d['loss'] > c_vali_acc:
                    c_vali_acc = -d['loss']
                    c_test_acc = d['test_acc']
                    record_time_res.append('%s,%s,%s' % (d['eval_time'] - start, c_vali_acc, c_test_acc))

            res['test_acc'] = c_test_acc
            print('test_acc={}'.format(c_test_acc))

            #cal std and record the best results.
            if args.use_hyperopt:
                test_accs = []
                test_stds=[]
                for i in range(5):
                    # args.epochs = 100
                    vali_acc, t_acc, t_std, test_args = main(args)
                    print('cal std: times:{}, valid_Acc:{}, test_acc:{:.04f}+-{:.04f}'.format(i, vali_acc, t_acc, t_std))
                    test_accs.append(t_acc)
                    test_stds.append(t_std)
                test_accs = np.array(test_accs)
                test_stds = np.array(test_stds)
                for i in range(5):
                    print('Train from scratch {}/5: Test_acc:{:.04f}+-{:.04f}'.format(i, test_accs[i], test_stds[i]))
                print('test_results_5_times:{:.04f}+-{:.04f}'.format(np.mean(test_accs), np.std(test_accs)))
                res['accs_train_from_scratch'] = test_accs
                res['stds_train_from_scratch'] = test_stds
                test_res.append(res)

                with open('tuned_res/%s_res_%s_%s.pkl' % (args1.data, tune_str, suffix), 'wb+') as fw:
                    pickle.dump(test_res, fw)
                logging.info('test_results of 5 times:{:.04f}+-{:.04f}'.format(np.mean(test_accs), np.std(test_accs)))
                logging.info('**********finish {}-th/{}**************8'.format(ind + 1, len(lines)))
        except Exception as e:
            logging.info('errror occured for %s-th, arch_info=%s, error=%s', ind + 1, l.strip(), e)
            import traceback
            traceback.print_exc()
    print('finsh tunining {} archs, saved in {}'.format(len(arch_set),
                                                        'tuned_res/%s_res_%s_%s.pkl' % (args1.data, tune_str, suffix)))


def tune_topK_arch():
    '''
        given the best arch, find the best meta-parameters with hyper-opt, then run&test for 5 times, return the average;
    '''
    tune_str = time.strftime('%Y%m%d-%H%M%S')
    path = 'logs/tune-topK-%s_%s' % (args1.data, tune_str)
    if not os.path.exists(path):
      os.mkdir(path)
    log_filename = os.path.join(path, 'log.txt')
    init_logger('tune-topK', log_filename, logging.INFO, False)

    print('**********tune given arch: {}, logfilename={}**************'.format(args1.arch, log_filename))
    logging.info('**********tune given arch: {}, logfilename={}**************8'.format(args1.arch, log_filename))

    with open(args1.arch_filename, 'rb') as f:
        archs = pickle.load(f)
    archs = sorted(archs, key=lambda d:d['test_acc'], reverse=True)
    topK_archs = set([r['tuned_args']['arch'] for r in archs[:3]])
    print('fine tune {} topK archs: {}'.format(len(topK_archs), ','.join(topK_archs)))
    top_res, max_acc, max_std, best_arch = [], -0.1, 0.0, ''
    if args1.data in ['small_Reddit']:
        sane_space['hidden_size'] = hp.choice('hidden_size', [16, 32, 64])
    for arch in list(topK_archs):
        args1.arch = arch

        trials = Trials()
        best = fmin(objective, sane_space, algo=partial(tpe.suggest, n_startup_jobs=50), max_evals=200, trials=trials)

        space = hyperopt.space_eval(sane_space, best)
        args = generate_args(space)
        res = {}
        res['tuned_args'] = args.__dict__
        res['round_test_acc'] = []
        logging.info('best args is: %s', args.__dict__)
        for rnd in range(3):
            round_args = ARGS()
            for k, v in args.__dict__.items():
                setattr(round_args, k, v)
            round_args.rnd_num = rnd + 1
            round_args.epochs = 100
            #round_args.save = '{}_{}'.format(args.data, datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
            round_res = {}
            vali_acc, t_acc, test_args = main(round_args)
            test_args.model_path = os.path.join(test_args.save, 'weights.pt')
            res['retrain_log'] = os.path.join(test_args.save, 'log.txt')
            test_acc, save_dir = test_main(test_args)
            res.setdefault('round_test_acc', []).append(test_acc / 100)
            print('re-train round={}, test_acc={:.4f}, save_dir={}'.format(rnd+1, test_acc, save_dir))
            logging.info('re-train round={}, test_acc={:.4f}, save_dir={}'.format(rnd+1, test_acc, save_dir))
        logging.info('**********finish arch={},res={}, avg_acc={:.4f}({:.4f})**************'.format(arch, res, np.mean(res['round_test_acc']), np.std(res['round_test_acc'])))
        print('**********finish arch={}, res={}, avg_acc={}({})**************'.format(arch, res, np.mean(res['round_test_acc']), np.std(res['round_test_acc'])))
        top_res.append('arch={},test_acc={:.4f}({:.4f})'.format(arch, np.mean(res['round_test_acc']), np.std(res['round_test_acc'])))
        if np.mean(res['round_test_acc']) > max_acc:
            max_acc = np.mean(res['round_test_acc'])
            max_std = np.std(res['round_test_acc'])
            best_arch = arch

    print('finish tune topK archs for {0}: best acc={1:.4f}({2:.4f}), arch={3}'.format(args1.data, max_acc, max_std, best_arch))
    print('\n'.join(top_res))
    logging.info('finish tune topK archs: %s, best acc=%.4f(%.4f), arch=%s', ','.join(topK_archs), max_acc, max_std, best_arch)

def tune_arch(arch_str):
    '''
        given the best arch, find the best meta-parameters with hyper-opt, then run&test for 5 times, return the average;
    '''
    get_args()

    args1.arch = arch_str
    # if args1.data == 'PPI' or args1.data in node_classification_dataset:
    #     sane_space['learning_rate'] = hp.uniform("lr", 0.001, 0.01)
    # if args1.data == 'CiteSeer':
    #     sane_space['hidden_size'] = hp.choice('hidden_size', [16, 32, 64, 128])
    if args1.data == 'PPI' or args1.data in node_classification_dataset:
        sane_space['learning_rate'] = hp.uniform("lr", 0.001, 0.01)
        sane_space['hidden_size'] = hp.choice('hidden_size', [64, 128, 256, 512])
        args1.cos_lr = True
        args1.with_linear = True
        args1.with_layernorm = True


    if args1.data == 'CiteSeer':
        sane_space['hidden_size'] = hp.choice('hidden_size', [16, 32, 64, 128])
    if args1.data in ['Cora', 'CiteSeer']:
        sane_space['weight_decay'] = hp.uniform("wr", -4, -2)
        sane_space['learning_rate'] = hp.uniform("lr", 0.00001, 0.005)

    if args1.data == 'DD' and args1.num_layers == 6:
        sane_space['hidden_size'] = hp.choice('hidden_size', [16, 32, 64, 128])
    args1.ft_dropout = True
    args1.ft_weight_decay = True
    #print('**********tune given arch: {}, logfilename={}**************'.format(args1.arch, log_filename))
    #logging.info('**********tune given arch: {}, logfilename={}**************9'.format(args1.arch, log_filename))

    trials = Trials()
    best = fmin(objective, sane_space, algo=partial(tpe.suggest, n_startup_jobs=5), max_evals=20, trials=trials)

    space = hyperopt.space_eval(sane_space, best)
    args = generate_args(space)
    res = {}
    res['round_test_acc'] = []
    for rnd in range(1):
        #tmp_args = {'dropout': 6, 'hidden_size': 256, 'learning_rate': 0.000633112605984229, 'model': 'SANE', 'optimizer': 'adam', 'weight_decay': 0.00011496238432878642, 'data': 'Amazon_Computers', 'save': 'Amazon_Computers_20200127-073537', 'epochs': 800, 'arch': 'gat_linear||gat_linear||gat_generalized_linear||skip||none||l_concat', 'gpu': 5, 'seed': 2, 'grad_clip': 5, 'momentum': 0.9}
        round_args = ARGS()
        for k, v in args.__dict__.items():
            setattr(round_args, k, v)
        if args1.data =='PPI':
            round_args.epochs = 100
        else:
            round_args.epochs = 600
        round_res = {}
        vali_acc, t_acc, _, test_args = main(round_args)
        test_args.model_path = os.path.join(test_args.save, 'weights.pt')
        res['retrain_log'] = os.path.join(test_args.save, 'log.txt')
        # test_acc, save_dir = test_main(test_args)
        res.setdefault('round_test_acc', []).append(t_acc)
        #print('re-train round={}, test_acc={}, save_dir={}'.format(rnd+1, test_acc, save_dir))
    #logging.info('**********finish res={}, avg_acc={}({})**************'.format(res, np.mean(res['round_test_acc']), np.std(res['round_test_acc'])))
    #print('**********finish res={}\\ avg_acc={}({})**************'.format(res, np.mean(res['round_test_acc']), np.std(res['round_test_acc'])))
    return vali_acc, t_acc

if __name__ == '__main__':
    get_args()
    if args1.arch_filename:
        if args1.tune_topK:
            tune_topK_arch()
        else:
            run_fine_tune()
    elif args1.arch:
        tune_arch()

