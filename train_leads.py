from torchdiffeq import odeint
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from scipy.integrate import solve_ivp
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler
from torch.nn import init
import numpy as np
import copy, random, os, sys, math, argparse
from functools import partial

from experiments import MultiEnvExperiment
from forecasters import Forecaster
from utils import init_weights
from datasets import init_dataloaders

__doc__ = '''Training LEADS.'''

def cmdline_args():
        # Make parser object
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    
    p.add_argument("dataset", type=str,
                   help='''choose dataset: 
    'lv' - Lotka-Volterra 
    'gs' - Gray-Scott
    'ns' - Navier-Stokes''')
    p.add_argument("-p", "--path", type=str, default='./exp',
                   help='''Root path for the experiments.''')
    p.add_argument("-e", "--exp_type", type=str, default='leads',
                   help='''choose decomposition type: 
    'leads' - LEADS (default)
    'leads_no_min' - LEADS no min.
    'one_for_all' - One-For-All
    'one_per_env' - One-Per-Env.''')
    p.add_argument('-d', '--device', type=str, default='cpu',
                   help='''choose device:
    'cpu' - CPU only (default, recommended for Lotka-Volterra)
    'cuda:X' - CUDA device.''')
    return p.parse_args()

def train_leads(dataset_name, exp_type, path, device):
    if exp_type in ['leads', 'leads_no_min']:
        decomp_type = 'leads_decomp'
    else:
        decomp_type = exp_type

    if dataset_name == 'lv':
        n_env = 10
        net = Forecaster(in_c=2, out_c=2, n_env=n_env, hidden=64, net_type='mlp', factor=1., method='rk4', decomp_type=decomp_type)
        init_weights(net, init_type='normal', init_gain=0.05)
        train, test = init_dataloaders('lv')
        optimizer = optim.Adam(net.parameters(), lr=1.e-3, betas=(0.9, 0.999))
        lambda_inv = 1 / 5e3
        factor_lip = 1.e-2
    elif dataset_name == 'gs':
        n_env = 3
        net = Forecaster(in_c=2, out_c=2, n_env=n_env, hidden=64, net_type='conv', factor=1.e-3, method='rk4', decomp_type=decomp_type)
        init_weights(net, init_type='normal', init_gain=0.1)
        train, test = init_dataloaders('gs')
        optimizer = optim.Adam(net.parameters(), lr=1.e-3, betas=(0.9, 0.999))
        lambda_inv = 1 / 1e3
        factor_lip = 1.e-2
    elif dataset_name == 'ns':
        n_env = 4
        net = Forecaster(in_c=1, out_c=1, n_env=n_env, hidden=64, net_type='fno', factor=1., method='euler', decomp_type=decomp_type)
        init_weights(net, init_type='normal', init_gain=0.1)
        train, test = init_dataloaders('ns', buffer_filepath=os.path.join(path,'ns_buffer'))
        optimizer = optim.Adam(net.parameters(), lr=1.e-3, betas=(0.9, 0.999))
        lambda_inv = 1 / 1e5
        factor_lip = 1.e-4
    
    if exp_type == 'leads_no_min':
        lambda_inv = 0.

    experiment = MultiEnvExperiment(
            train=train, test=test, net=net, optimizer=optimizer, 
            min_op='sum_spectral', n_env=n_env, calculate_net_norm=True, 
            k=0.99, lambda_inv=lambda_inv, factor_lip=factor_lip,
            nupdate=10, nepoch=120000, decomp_type=decomp_type,
            path=path, device=device
        )
    experiment.run()

if __name__ == '__main__':
    
    if sys.version_info<(3,7,0):
        sys.stderr.write("You need python 3.7 or later to run this script.\n")
        sys.exit(1)
        
    args = cmdline_args()
    os.makedirs(args.path, exist_ok=True)
    train_leads(args.dataset, exp_type=args.exp_type, path=args.path, device=args.device)