import sys
import numpy as np
import torch
import utils
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score
from model import NetworkGNN as Network

import logging
from sklearn.metrics import pairwise_distances
from torch_scatter import scatter_mean,scatter_sum
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def main(exp_args, run=0):
    global train_args
    train_args = exp_args

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    #np.random.seed(train_args.seed)
    torch.cuda.set_device(train_args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(train_args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(train_args.seed)

    # load dataset
    dataset = PygGraphPropPredDataset(name="ogbg-molpcba", root='/data/wangxu/dataset')
    metric = dataset.eval_metric
    split_idx = dataset.get_idx_split()

    if train_args.train_subset:
        subset_ratio = 0.1
        subset_idx = torch.randperm(len(split_idx["train"]))[:int(subset_ratio*len(split_idx["train"]))]
        train_dataset = dataset[split_idx["train"][subset_idx]]
        train_loader = DataLoader(dataset[split_idx["train"][subset_idx]], batch_size=train_args.batch_size, shuffle=True, num_workers = train_args.num_workers)
    else:
        train_dataset = dataset[split_idx["train"]]
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=train_args.batch_size, shuffle=True, num_workers = train_args.num_workers)

    valid_dataset = dataset[split_idx["valid"]]
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=train_args.batch_size, shuffle=False, num_workers = train_args.num_workers)
    test_dataset = dataset[split_idx["test"]]
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=train_args.batch_size, shuffle=False, num_workers = train_args.num_workers)

    hidden_size = train_args.hidden_size
    num_features = dataset.num_features
    num_classes = 128

    criterion = torch.nn.BCEWithLogitsLoss()
    criterion = criterion.cuda()
    genotype = train_args.arch
    hidden_size = train_args.hidden_size

    model = Network(genotype, criterion, num_features, num_classes, hidden_size,
                    num_layers=train_args.num_layers, dropout=train_args.dropout,
                    act=train_args.activation, args=train_args)

    model = model.cuda()
    print(model)

    num_parameters = np.sum(np.prod(v.size()) for name, v in model.named_parameters())
    print('params size:', num_parameters)
    logging.info("genotype=%s, param size = %fMB, args=%s", genotype, utils.count_parameters_in_MB(model), train_args.__dict__)

    if train_args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            train_args.lr,
            weight_decay = 0,
        )
    elif train_args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            train_args.learning_rate,
            momentum=train_args.momentum,
            weight_decay=train_args.weight_decay
        )
    elif train_args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            train_args.learning_rate,
            weight_decay=train_args.weight_decay
        )

    best_valid_ap = 0

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(train_args.epochs), eta_min=train_args.min_lr)
    best_val_acc = best_test_acc = 0
    results = []
    best_line = 0

    for epoch in range(train_args.epochs):
        print(optimizer)
        train_loss = train_trans(train_loader, model, criterion, optimizer, metric)
        valid_ap = infer_trans(valid_loader, model, criterion, metric)
        print({'Train_loss': train_loss, 'Validation': valid_ap})
        # if train_args.cos_lr:
        #     scheduler.step()
        if valid_ap > best_valid_ap:
            best_valid_ap = valid_ap
            # if train_args.checkpoint_dir != '':
            #     print('Saving checkpoint...')
            #     checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val_ap': best_valid_ap, 'num_params': num_parameters}
            #     torch.save(checkpoint, os.path.join(train_args.checkpoint_dir, 'checkpoint.pt'))

        print(f'Best validation AP so far: {best_valid_ap}')
        
        results.append([valid_ap])

    return best_val_ap, train_args

def train_trans(train_loader, model, criterion, optimizer, metric):
    model.train()
    total_loss = 0
    total_loss = 0

    # output, loss, accuracy

    for step, loader in enumerate(tqdm(train_loader, desc="iteration")):
        loader = loader.to(device)
        if loader.x.shape[0] == 1 or loader.batch[-1] == 0:
            pass
        else:
            logits = model(loader)
            optimizer.zero_grad()
            is_labeled = loader.y == loader.y

            train_loss = criterion(logits.to(torch.float32)[is_labeled], loader.y.to(torch.float32)[is_labeled])
            train_loss.backward()
            optimizer.step()
         
            total_loss += train_loss.detach().cpu().item()

    return total_loss

def infer_trans(valid_loader, model, criterion, metric):
    model.eval()

    y_val_true = []
    y_val_pred = []

    for step, val_data in enumerate(tqdm(valid_loader, desc='iteration')):
        val_data = val_data.to(device)
        with torch.no_grad():
            logits = model(val_data)

        y_val_true.append(val_data.y.view(logits.shape).detach().cpu())
        y_val_pred.append(logits.detach().cpu())

    y_val_true = torch.cat(y_val_true, dim=0)
    y_val_pred = torch.cat(y_val_pred, dim=0)

    val_input_dict = {'y_true': y_val_true, 'y_pred': y_val_pred}
    evaluator = Evaluator('ogbg-molpcba')

    return evaluator.eval(val_input_dict)[metric]


if __name__ == '__main__':
    main()