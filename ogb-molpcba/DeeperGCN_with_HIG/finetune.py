import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from model_att import DeeperGCN
from tqdm import tqdm
import numpy as np
import pandas as pd
from args import ArgsInit
from utils.ckpt_util import save_ckpt
import logging
import time, os
import statistics
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import torch.nn.functional as F
from utils.logger import create_exp_dir

# for AUC margin loss
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
args = ArgsInit().save_exp()


def train(model, device, loader, optimizer, task_type, grad_clip=0.):
    loss_list = []
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            pred = model(batch)
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
            pred = model(batch, mode='test')
            y_true.append(batch.y[:,0:1].view(pred.shape).detach().cpu()) # remove random forest pred
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true,
                  "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    
    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')

    sub_dir = 'BS_{}-NF_{}'.format(args.batch_size, args.feature)
    set_all_seeds(args.random_seed)
    dataset = PygGraphPropPredDataset(name=args.dataset)
    
    # Load RF predictions
    npy = os.listdir('rf_preds')[args.random_seed]
    rf_pred = np.load(os.path.join('rf_preds', npy))
    print (npy)
    dataset.data.y = torch.cat((dataset.data.y, torch.from_numpy(rf_pred)), 1)
    
    args.num_tasks = dataset.num_tasks
    #logging.info('%s' % args)

    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    evaluator = Evaluator(args.dataset)
    split_idx = dataset.get_idx_split()

    set_all_seeds(args.random_seed)
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,  
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)


    set_all_seeds(args.random_seed)
    model = DeeperGCN(args).to(device)
     
    if True:
       checkpoint_path = './saved_models/EXP-B_res+-C_gen-L_14-F_256-DP_0.2-GA_softmax-T_1.0-LT_True-P_1.0-LP_False-Y_0.0-LY_False-MN_False-LS_False-RS_%s/model_ckpt/'%(args.random_seed)
       best_pth = sorted(os.listdir(checkpoint_path))[-1]
       args.model_load_path = os.path.join(checkpoint_path, best_pth)
       trained_stat_dict = torch.load(args.model_load_path)['model_state_dict']
       #trained_stat_dict.pop('graph_pred_linear.weight', None)
       ##trained_stat_dict.pop('graph_pred_linear.bias', None)
       model.load_state_dict(trained_stat_dict, strict=False)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    optimizer = PESG(model, 
                        a=aucm_criterion.a, 
                        b=aucm_criterion.b, 
                        alpha=aucm_criterion.alpha, 
                        lr=args.lr, 
                        gamma=args.gamma, 
                        margin=args.margin, 
                        weight_decay=args.weight_decay)


    # get imbalance ratio from train set
    args.imratio = float((train_loader.dataset.data.y[:, 0].sum()/train_loader.dataset.data.y[:,0].shape[0]).numpy())
    aucm_criterion.p = args.imratio
    print (aucm_criterion.p)

    # save 
    datetime_now = '2021-10-09'
    pretrained_prefix = 'pre_' if args.pretrained else ''
    virtual_node_prefilx = '-vt' if args.add_virtual_node else ''
    args.configs = '[%s]Train_%s_im_%.4f_rd_%s_%s%s-FP_%s_%s_wd_%s_lr_%s_B_%s_E_%s_%s_%s_g_%s_m_%s'%(datetime_now, args.dataset, args.imratio,  args.random_seed, pretrained_prefix, args.model_name, virtual_node_prefilx, args.activations, args.weight_decay, args.lr, args.batch_size, args.epochs,  args.loss, args.optimizer, args.gamma, args.margin)
    logging.info(args.save)  
    logging.info(args.configs)      

    results = {'highest_valid': 0,
               'final_train': 0,
               'final_test': 0,
               'highest_train': 0}

    start_time = time.time()
    start_time_local = time.time()
    for epoch in range(1, args.epochs + 1):
        
        if epoch in [int(args.epochs*0.33),  int(args.epochs*0.66)] and args.loss!= 'ce':
            optimizer.update_regularizer(decay_factor=10)

        epoch_loss = train(model, device, train_loader, optimizer, dataset.task_type, grad_clip=args.grad_clip)
        
        #logging.info('Evaluating...')
        train_result = eval(model, device, train_loader, evaluator)[dataset.eval_metric]
        valid_result = eval(model, device, valid_loader, evaluator)[dataset.eval_metric]
        test_result = eval(model, device, test_loader, evaluator)[dataset.eval_metric]

        logging.info("Epoch:%s, train_auc:%.4f, valid_auc:%.4f, test_auc:%.4f, lr:%.4f, time:%.4f"%(epoch, train_result, valid_result, test_result, optimizer.lr, time.time()-start_time_local))
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
                      sub_dir, name_post='valid_best_AUC-FP_E_%s_R%s'%(epoch, args.random_seed))
            

    logging.info("%s" % results)

    end_time = time.time()
    total_time = end_time - start_time
    logging.info('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))


if __name__ == "__main__":
    cls_criterion = torch.nn.BCEWithLogitsLoss()
    reg_criterion = torch.nn.MSELoss()
    
    # https://github.com/Optimization-AI/LibAUC
    aucm_criterion = AUCMLoss()
    main()
