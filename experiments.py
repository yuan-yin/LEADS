import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics

from utils import set_requires_grad, fix_seed, make_basedir, convert_tensor
from utils import CalculateNorm, Logger

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
matplotlib.rcParams["figure.dpi"] = 100

_EPSILON = 1e-5

import os, time, io, logging, sys
from functools import partial

from torchvision.utils import make_grid

import collections

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class BaseExperiment(object):
    def __init__(self, device, path='./exp', seed=None):
        self.device = device
        self.path = make_basedir(path)
            
        if seed is not None:
            fix_seed(seed)

    def training(self, mode=True):
        for m in self.modules():
            m.train(mode)

    def evaluating(self):
        self.training(mode=False)

    def zero_grad(self):
        for optimizer in self.optimizers():
            optimizer.zero_grad()        

    def to(self, device):
        for m in self.modules():
            m.to(device)
        return self

    def modules(self):
        for name, module in self.named_modules():
            yield module

    def named_modules(self):
        for name, module in self._modules.items():
            yield name, module

    def datasets(self):
        for name, dataset in self.named_datasets():
            yield dataset

    def named_datasets(self):
        for name, dataset in self._datasets.items():
            yield name, dataset

    def optimizers(self):
        for name, optimizer in self.named_optimizers():
            yield optimizer

    def named_optimizers(self):
        for name, optimizer in self._optimizers.items():
            yield name, optimizer

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            if not hasattr(self, '_modules'):
                self._modules = collections.OrderedDict()
            self._modules[name] = value
        elif isinstance(value, DataLoader):
            if not hasattr(self, '_datasets'):
                self._datasets = collections.OrderedDict()
            self._datasets[name] = value
        elif isinstance(value, Optimizer):
            if not hasattr(self, '_optimizers'):
                self._optimizers = collections.OrderedDict()
            self._optimizers[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        if '_datasets' in self.__dict__:
            datasets = self.__dict__['_datasets']
            if name in datasets:
                return datasets[name]
        if '_optimizers' in self.__dict__:
            optimizers = self.__dict__['_optimizers']
            if name in optimizers:
                return optimizers[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __delattr__(self, name):
        if name in self._modules:
            del self._modules[name]
        elif name in self._datasets:
            del self._datasets[name]
        elif name in self._optimizers:
            del self._optimizers[name]
        else:
            object.__delattr__(self, name)

def show(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

class LoopExperiment(BaseExperiment):
    def __init__(
        self, train, test=None, root=None, nepoch=10, niter=300, ndisplay=1, **kwargs):
        super().__init__(**kwargs)
        self.train = train
        self.test = test
        self.nepoch = nepoch
        self.ndisplay = ndisplay
        self.niter = niter
        self.logger = Logger(filename=os.path.join(self.path, 'log.txt'))
        print(' '.join(sys.argv))

    def train_step(self, batch, val=False):
        self.training()
        batch = convert_tensor(batch, self.device)
        loss, output = self.step(batch)
        return batch, output, loss

    def val_step(self, batch, val=False):
        self.evaluating()
        with torch.no_grad():
            batch = convert_tensor(batch, self.device)
            loss, output = self.step(batch, backward=False)
            metric = self.metric(**output, **batch)

        return batch, output, loss, metric

    def log(self, epoch, iteration, metrics):
        message = '[{step}][{epoch}/{max_epoch}][{i}/{max_i}]'.format(
            step=epoch *len(self.train)+ iteration+1,
            epoch=epoch+1,
            max_epoch=self.nepoch,
            i=iteration+1,
            max_i=len(self.train)
        )
        for name, value in metrics.items():
            message += ' | {name}: {value:.3e}'.format(name=name, value=value)
            
        print(message)

    def step(self, **kwargs):
        raise NotImplementedError


class MultiEnvExperiment(LoopExperiment):
    def __init__(self, net, optimizer, k, min_op, n_env, decomp_type, calculate_net_norm, lambda_inv, factor_lip, loss='mse', nupdate=10, nlog=50,
                load_pretrained_model=None, **kwargs):
        super().__init__(**kwargs)

        if loss == 'mse':
            self.traj_loss = nn.MSELoss()
        elif loss == 'l1':
            self.traj_loss = nn.L1Loss()

        self.net = net.to(self.device)
        if calculate_net_norm is True:
            self.cal_norm = CalculateNorm(self.net.right_model) 

        if load_pretrained_model is not None:
            assert len(self.net.left_model) == 1
            print("Load pretrained model")
            pretrained_dict = torch.load(load_pretrained_model)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.find('left_model') != -1}
            model_dict = self.net.state_dict()
            model_dict.update(pretrained_dict) 
            self.net.load_state_dict(model_dict)
            set_requires_grad(self.net.left_model, False)

        self.optimizer = optimizer

        self.min_op = min_op
        self.factor_lip = factor_lip
        self.lambda_inv = lambda_inv

        self._i = 1.
        self._epsilon = k
        self.k = k

        self.decomp_type = decomp_type

        self.n_env = int(n_env)
        self.mini_batch_size = None

        self.nlog = nlog
        self.nupdate = nupdate
    
    def epsilon_update(self):
        self._i += 1
        self._epsilon = self.k ** self._i
        logging.info(f'espilon: {self._epsilon}')

    def set_subbatch_size(self, x):
        self.mini_batch_size = int(x // self.n_env)
    
    def _inference(self, state, t, backward, enable_deriv_min):
        state = state.clone()
        state.requires_grad=True

        target = state
        if backward:
            mini_batch_states = torch.split(state, self.mini_batch_size)
            preds_train, preds, derivs = (list() for _ in range(3))

            for env, mini_batch_state in enumerate(mini_batch_states):
                pred_train = self.net(mini_batch_state, t, env=env, epsilon=self._epsilon)
                with torch.no_grad():
                    pred_env = self.net(mini_batch_state, t, env=env)
                preds_train.append(pred_train)
                preds.append(pred_env)
                if enable_deriv_min:
                    deriv_env = self.net.right_model[env](mini_batch_state)
                    derivs.append(deriv_env)
            
            pred_train  = torch.cat(preds_train)
            pred        = torch.cat(preds)
            if enable_deriv_min:
                deriv       = torch.cat(derivs      , dim=0)

            loss_train = self.traj_loss(pred_train, target)
            with torch.no_grad():
                loss_mse = F.mse_loss(pred, target)

            if enable_deriv_min:
                if self.min_op == 'sum_spectral':
                    loss_op_a = self.cal_norm.calculate_spectral_norm().sum()
                    derivs = torch.split(deriv, self.mini_batch_size)
                    loss_ops = [((deriv_e.norm(p=2, dim=1) / (state_e.norm(p=2, dim=1) + _EPSILON)) ** 2).mean() for deriv_e, state_e in zip(derivs, mini_batch_states)]
                    loss_op_b = torch.stack(loss_ops).sum()
                    loss_op = loss_op_a * self.factor_lip + loss_op_b
                elif self.min_op == 'f_norm':
                    loss_op = (self.cal_norm.calculate_frobenius_norm() ** 2).sum()
                if self.lambda_inv > 0:
                    loss_total = loss_train + loss_op * self.lambda_inv
                else:
                    loss_total = loss_train
            else:
                loss_total = loss_train 

            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
        else:
            mini_batch_states = torch.split(state, self.mini_batch_size)
            preds = list()

            for env, mini_batch_state in enumerate(mini_batch_states):
                pred_env = self.net(mini_batch_state, t, env=env)
                preds.append(pred_env)
            
            pred = torch.cat(preds)

            loss_mse = F.mse_loss(pred, target)
            loss_train = loss_mse

        loss = {
            'loss': loss_mse,
            'loss_train': loss_train,
        }

        mini_batch_states = torch.split(state, self.mini_batch_size)
        mini_batch_states_pred = torch.split(pred, self.mini_batch_size)
        for env, (mini_batch_state, mini_batch_state_pred) in enumerate(zip(mini_batch_states, mini_batch_states_pred)):
            loss[f'loss_e{env}']  = F.mse_loss(mini_batch_state, mini_batch_state_pred)

        output = {
            'state_pred'     : pred,
        }
        return loss, output

    def step(self, batch, backward=True):
        state = batch['state']
        batch_size = state.size(0)
        self.set_subbatch_size(batch_size)
        t = batch['t'][0]
        dt = torch.abs(t[0] - t[1])

        if self.decomp_type == 'leads_decomp':
            loss, output = self._inference(state, t, backward, enable_deriv_min=True)
        elif self.decomp_type in ['one_per_env', 'one_for_all']:
            loss, output = self._inference(state, t, backward, enable_deriv_min=False)
        return loss, output

    def _reduction(self, score, per_env=True, temporal=True):
        mini_batch_scores = torch.split(score, self.mini_batch_size)
        dim = score.dim()
        dims = [0,1]
        if not temporal:
            dims = dims + [2]
        dims = dims + list(range(dim))[3:]
        
        scores_list = [mini_batch_score.mean(dim=dims) for mini_batch_score in mini_batch_scores]

        score_e = torch.stack(scores_list, dim=0) # ne x t ou ne

        if per_env: 
            out = score_e
        else:
            out = score_e.mean(dim=0)
        
        return out

    def metric(self, state, state_pred, **kwargs):
        mse = F.mse_loss(state, state_pred, reduction='none')
        mse_env = self._reduction(mse, per_env=True, temporal=False)
        metrics = {}
        metrics['mse'] = mse.mean()
        
        for env, l in enumerate(torch.split(mse_env, 1)):
            metrics[f'mse_e{env}']  = l

        return metrics

    def run(self):
        loss_test_min = None
        for epoch in range(self.nepoch): 
            for iteration, data in enumerate(self.train, 0):
                batch, output, metric = self.train_step(data)
                self.log(epoch, iteration, metric)
                
                if (epoch * (len(self.train)) + (iteration + 1)) % self.nupdate == 0:
                    self.epsilon_update()
                    
                if (epoch * (len(self.train)) + (iteration + 1)) % self.nlog == 0:
                    loss_test = []
                    
                    with torch.no_grad():
                        for j, data_test in enumerate(self.test, 0):
                            batch, output, loss, metric = self.val_step(data_test)
                            loss_test.append(loss['loss'].item())
                            
                        loss_test_mean = statistics.mean(loss_test)
                        loss_test_std = statistics.stdev(loss_test)

                        if loss_test_min == None or loss_test_min > loss_test_mean:
                            loss_test_min = loss_test_mean
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.net.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': loss_test_min, 
                            }, self.path + f'/model_{loss_test_min:.3e}.pt')
                        metric_test = {
                            'loss_test_mean': loss_test_mean,
                            'loss_test_std': loss_test_std,
                        }
                        self.log(epoch, iteration, metric_test)