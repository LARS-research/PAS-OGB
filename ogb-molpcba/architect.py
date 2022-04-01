import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

def check_paras(xs, L):
    filename = 'tmp/grad.txt'
    fw = open(filename, 'w+')
    w_res = []
    cnt = 0
    for x in xs:
        w_res.append('%s' % x)
        if x is None:
            print('empty, cnt=%s, Line=%s' % (cnt, L))
            #import pdb;pdb.set_trace()
            #return
        else:

            pass
        cnt += 1
    fw.write('\n'.join(w_res))
    fw.close()

def check_unrolled_models(model, L):
    filename = './tmp/unrolled_model.txt'
    fw = open(filename, 'w+')
    xs = model.named_parameters()
    w_res = []
    cnt = 0
    for name, x in xs:
        w_res.append('%s,%s' % (name, x))
        if x.grad is None:
            print('grad empty, is_leaf=%s, cnt=%s, name=%s, Line=%s' % (x.is_leaf, cnt, name, L))
            #import pdb;pdb.set_trace()
            #return
        else:

            pass
        cnt += 1
    fw.write('\n'.join(w_res))
    fw.close()
    return

class Architect(object):

    def __init__(self, model, args):
        self.args = args
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, data, eta, network_optimizer):
        total_loss = torch.tensor(0.0, requires_grad=True).to(device)
        for train_data in data[0]:
            train_data = train_data.to(device)
            loss = self.model._loss(train_data, is_valid=False) #train loss
            total_loss += loss
        theta = _concat(self.model.parameters()).data# w
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        #import pdb;pdb.set_trace()
        #check_paras(torch.autograd.grad(loss, self.model.parameters(), create_graph=True, allow_unused=True), 75)
        dtheta = _concat(torch.autograd.grad(total_loss, self.model.parameters(), allow_unused=True)).data + self.network_weight_decay*theta #gradient, L2 norm
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta)) # one-step update, get w' for Eq.7 in the paper
        return unrolled_model

    def step(self,input_train, input_valid, eta, network_optimizer, unrolled):

        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, input_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid, is_valid=True)
        self.optimizer.step()

    def _backward_step(self, data, is_valid=True):

        loss = self.model._loss(data, is_valid)
        loss.backward()

    def _backward_step_unrolled(self, data, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(data, eta, network_optimizer)
        for valid_data in data[1]:
            valid_data = valid_data.to(device)
            unrolled_loss = unrolled_model._loss(valid_data, is_valid=True) # validation loss
            unrolled_loss.backward() # one-step update for w?
        dalpha = [v.grad for v in unrolled_model.arch_parameters()] #L_vali w.r.t alpha
        check_unrolled_models(unrolled_model, 122)
        vector = [v.grad.data for v in unrolled_model.parameters()] # gradient, L_train w.r.t w, double check the model construction
        implicit_grads = self._hessian_vector_product(vector, data)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        #update alpha, which is the ultimate goal of this func, also the goal of the second-order darts
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, data, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v) # R * d(L_val/w', i.e., get w^+
        total_loss_p = torch.tensor(0.0,requires_grad=True).to(device)
        for train_data in data[0]:
            train_data = train_data.to(device)
            loss = self.model._loss(train_data, is_valid=False) # train loss
            total_loss_p +=loss
        grads_p = torch.autograd.grad(total_loss_p, self.model.arch_parameters()) # d(L_train)/d_alpha, w^+

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2*R, v) # get w^-, need to subtract 2 * R since it has add R

        total_loss_n = torch.tensor(0.0, requires_grad=True).to(device)
        for train_data in data[0]:
            train_data = train_data.to(device)
            loss = self.model._loss(train_data, is_valid=False)  # train loss
            total_loss_n += loss
        grads_n = torch.autograd.grad(total_loss_n, self.model.arch_parameters())# d(L_train)/d_alpha, w^-

        #reset to the orignial w, always using the self.model, i.e., the original model
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]
