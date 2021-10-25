import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
from torchdiffeq import odeint
from functools import partial
from torch import optim
import math
import numbers
import functools

from networks import *

class DerivativeEstimatorMultiEnv(nn.Module):

    def __init__(self, left_model, right_model, n_env, decomp_type, ignore_right=False):
        super().__init__()

        assert isinstance(  left_model, nn.ModuleList)
        assert isinstance( right_model, nn.ModuleList)

        self.left_model   = left_model
        self.right_model  = right_model
        self.decomp_type  = decomp_type
        self.env = None
        self.enable_right = None
        self.ignore_right = ignore_right
        self.n_env = n_env

    def set_env(self, env):
        self.env = env
    
    def set_enable_right(self, right: bool):
        self.enable_right = right

    def forward(self, t, u):
        left_res, right_res = None, None
        if self.decomp_type == 'leads_decomp':
            # 1 model on the left, n_env models on the right
            assert len(self.left_model) == 1
            assert self.env is not None

            left_res = self.left_model[0](u)
            right_res = self.right_model[self.env](u)
        elif self.decomp_type == 'one_for_all':
            # 1 model on the left, 1 model on the right
            # This case is adapted to experments with changing environments
            assert len(self.left_model) == len(self.right_model)
            change_every = self.n_env // len(self.left_model)
            left_res  = self.left_model[self.env // change_every](u)
            right_res = self.right_model[self.env // change_every](u) 
        elif self.decomp_type == 'one_per_env':
            # n_env models on the left, n_env models on the right
            assert len(self.left_model) == len(self.right_model)
            assert self.env is not None

            left_res = self.left_model[self.env](u)
            right_res = self.right_model[self.env](u)
        else:
            change_every = len(self.right_model) // len(self.left_model)
            left_res = self.left_model[self.env // change_every](u)
            right_res = self.right_model[self.env](u)

        if right_res is not None and (self.enable_right and not self.ignore_right):
            return left_res + right_res
        else:
            return left_res

class Forecaster(nn.Module):
    def __init__(self, in_c, out_c, n_env, hidden, net_type, n_left=None, n_right=None, options=None, factor=1., method=None, decomp_type=None, ignore_right=False):
        super().__init__()

        code_c = 2
        if decomp_type == 'leads_decomp':
            n_left = 1
            n_right = n_env
        elif decomp_type == 'one_for_all':
            if n_left is None and n_right is None:
                n_left = n_right = 1
            else:
                n_left = n_right
        elif decomp_type == 'one_per_env':
            n_left = n_right = n_env
        else:
            n_left = n_left
            n_right = n_right

        if net_type == 'mlp':
            self.left_model  = nn.ModuleList([MLPEstimator(in_c=in_c, out_c=out_c, hidden=hidden, factor=factor) for _ in range(n_left)])
            self.right_model = nn.ModuleList([MLPEstimator(in_c=in_c, out_c=out_c, hidden=hidden, factor=factor) for _ in range(n_right)])
        elif net_type == 'linear':
            self.left_model  = nn.ModuleList([Linear(in_c=in_c, out_c=out_c, factor=factor) for _ in range(n_left)])
            self.right_model = nn.ModuleList([Linear(in_c=in_c, out_c=out_c, factor=factor) for _ in range(n_right)])
        elif net_type in ['conv', 'fno']:
            self.left_model  = nn.ModuleList([ConvNetEstimator(in_c=in_c, out_c=out_c, hidden=hidden, factor=factor, net_type=net_type) for _ in range(n_left)])
            self.right_model = nn.ModuleList([ConvNetEstimator(in_c=in_c, out_c=out_c, hidden=hidden, factor=factor, net_type=net_type) for _ in range(n_right)])
        else:
            raise NotImplementedError

        self.derivative_estimator = DerivativeEstimatorMultiEnv(self.left_model, self.right_model, n_env=n_env, decomp_type=decomp_type, ignore_right=ignore_right)
        self.method = method
        self.options = options
        self.int_ = odeint 
        self.net_type = net_type
        
    def forward(self, y, t, env, enable_right=True, epsilon=None):
        self.derivative_estimator.set_enable_right(enable_right)
        self.derivative_estimator.set_env(env)
        if epsilon is None:
            y0 = y[:,:,0]
            ret = self.int_(self.derivative_estimator, y0=y0, t=t, method=self.method, options=self.options)
        else:
            eval_points = np.random.random(len(t)) < epsilon
            eval_points[-1] = False
            start_i, end_i = 0, None
            res = []
            eval_points = eval_points[1:]
            for i, eval_point in enumerate(eval_points):
                if eval_point == True:
                    end_i = i+1
                    y0 = y[:,:,start_i]
                    t_seg = t[start_i:end_i+1]
                    res_seg = self.int_(self.derivative_estimator, y0=y0, t=t_seg, method=self.method, options=self.options)
                    if len(res) == 0:
                        res.append(res_seg)
                    else:
                        res.append(res_seg[1:])
                    start_i = end_i
                    end_i = None

            y0 = y[:,:,start_i]
            t_seg = t[start_i:]

            res_seg = self.int_(self.derivative_estimator, y0=y0, t=t_seg, method=self.method, options=self.options)
            if len(res) == 0:
                res.append(res_seg)
            else:
                res.append(res_seg[1:])
            ret = torch.cat(res, dim=0)
        
        dim = y.dim()
        dims = [1, 2, 0] + list(range(dim))[3:]
        return ret.permute(*dims)
