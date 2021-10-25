from .lv import LotkaVolterraDataset
from .ns import NavierStokesDataset
from .gs import GrayScottReactionDataset 
from .linear import LinearDataset
from .samplers import *

import torch
import math
from torch.utils.data import DataLoader

def param_lv():    
    params = [
        {'alpha': 0.5 , 'beta': 0.5, 'gamma': 0.5 , 'delta': 0.5},
        {'alpha': 0.25, 'beta': 0.5, 'gamma': 0.5 , 'delta': 0.5},
        {'alpha': 0.75, 'beta': 0.5, 'gamma': 0.5 , 'delta': 0.5},
        {'alpha': 1.0 , 'beta': 0.5, 'gamma': 0.5 , 'delta': 0.5},
        {'alpha': 0.5 , 'beta': 0.5, 'gamma': 0.25, 'delta': 0.5},
        {'alpha': 0.5 , 'beta': 0.5, 'gamma': 0.75, 'delta': 0.5},
        {'alpha': 0.5 , 'beta': 0.5, 'gamma': 1.0 , 'delta': 0.5},
        {'alpha': 0.25, 'beta': 0.5, 'gamma': 0.25, 'delta': 0.5},
        {'alpha': 0.75, 'beta': 0.5, 'gamma': 0.75, 'delta': 0.5},
        {'alpha': 1.0 , 'beta': 0.5, 'gamma': 1.0 , 'delta': 0.5},
    ]

    n_env = len(params)
    mini_batch_size = 1

    dataset_train_params = {
        'num_traj_per_env': 1,
        'time_horizon': 10, 
        'params': params,
        'dt': 0.5, 
        'method': 'RK45',
        'group': 'train',
    }

    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params['num_traj_per_env'] = 32
    dataset_test_params['group'] = 'test'

    dataset_train = LotkaVolterraDataset(**dataset_train_params)
    dataset_test  = LotkaVolterraDataset(**dataset_test_params)
    sampler_train = SubsetRamdomSampler(indices=dataset_train.indices, mini_batch_size=mini_batch_size)
    sampler_test  = SubsetSequentialSampler(indices=dataset_test.indices , mini_batch_size=1)

    dataloader_train_params = {
        'dataset'    : dataset_train,
        'batch_size' : mini_batch_size * n_env,
        'num_workers': 0,
        'sampler'    : sampler_train,
        'pin_memory' : True,
        'drop_last'  : False,
    }

    dataloader_test_params = {
        'dataset'    : dataset_test,
        'batch_size' : n_env,
        'num_workers': 0,
        'sampler'    : sampler_test,
        'pin_memory' : True,
        'drop_last'  : False,
    }

    dataloader_train = DataLoader(**dataloader_train_params)
    dataloader_test  = DataLoader(**dataloader_test_params)

    return dataloader_train, dataloader_test      

def param_gs():    
    params = [
        {'D_u': 0.2097 , 'D_v': 0.105 , 'F': 0.037 , 'k': 0.060},
        {'D_u': 0.2097 , 'D_v': 0.105 , 'F': 0.030 , 'k': 0.062},
        {'D_u': 0.2097 , 'D_v': 0.105 , 'F': 0.039 , 'k': 0.058},
    ]

    dataset_train_params = {
        'num_traj_per_env': 1,
        'time_horizon': 400, 
        'params': params,
        'dt_eval': 40, 
        'method': 'RK45',
        'group': 'train',
        'size': 32,
        'dx': 1.,
        'n_block': 3,
    }

    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params['num_traj_per_env'] = 32
    dataset_test_params['group'] = 'test'

    n_env = len(params)
    mini_batch_size = 1

    dataset_train = GrayScottReactionDataset(**dataset_train_params)
    dataset_test  = GrayScottReactionDataset(**dataset_test_params)
    sampler_train = SubsetRamdomSampler(indices=dataset_train.indices, mini_batch_size=mini_batch_size)
    sampler_test  = SubsetSequentialSampler(indices=dataset_test.indices , mini_batch_size=1)

    dataloader_train_params = {
        'dataset'    : dataset_train,
        'batch_size' : mini_batch_size * n_env,
        'num_workers': 0,
        'sampler'    : sampler_train,
        'pin_memory' : True,
        'drop_last'  : False,
        'shuffle'    : False,
    }

    dataloader_test_params = {
        'dataset'    : dataset_test,
        'batch_size' : n_env,
        'num_workers': 0,
        'sampler'    : sampler_test,
        'pin_memory' : True,
        'drop_last'  : False,
        'shuffle'    : False,
    }
    dataloader_train = DataLoader(**dataloader_train_params)
    dataloader_test  = DataLoader(**dataloader_test_params)

    return dataloader_train, dataloader_test        

def param_ns(buffer_filepath):
    size = 32
    tt = torch.linspace(0, 1, size+1)[0:-1]
    X,Y = torch.meshgrid(tt, tt)
    params = [
        {'f': 0.1 * (torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y))), 'visc': 1e-3},
        {'f': 0.1 * (torch.sin(2*math.pi*(1 * X + 1 * Y)) + torch.cos(2*math.pi*(1 * X + 2 * Y))), 'visc': 1e-3},
        {'f': 0.1 * (torch.sin(2*math.pi*(1 * X + 1 * Y)) + torch.cos(2*math.pi*(2 * X + 1 * Y))), 'visc': 1e-3},
        {'f': 0.1 * (torch.sin(2*math.pi*(1 * X + 2 * Y)) + torch.cos(2*math.pi*(2 * X + 1 * Y))), 'visc': 1e-3},
    ]

    n_env = len(params)
    mini_batch_size = 2

    dataset_train_params = {
        'num_traj_per_env': 8,
        'time_horizon': 10, 
        'size': size,
        'params': params,
        'dt_eval': 1, 
        'group': 'train',
        'buffer_filepath': buffer_filepath+'_train',
    }

    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params['num_traj_per_env'] = 32
    dataset_test_params['group'] = 'test'
    dataset_test_params['buffer_filepath'] = buffer_filepath+'_test'

    dataset_train = NavierStokesDataset(**dataset_train_params)
    dataset_test  = NavierStokesDataset(**dataset_test_params)
    sampler_train = SubsetRamdomSampler(indices=dataset_train.indices, mini_batch_size=mini_batch_size)
    sampler_test  = SubsetSequentialSampler(indices=dataset_test.indices , mini_batch_size=1)

    dataloader_train_params = {
        'dataset'    : dataset_train,
        'batch_size' : mini_batch_size * n_env,
        'num_workers': 0,
        'sampler'    : sampler_train,
        'pin_memory' : True,
        'drop_last'  : False,
        'shuffle'    : False,
    }

    dataloader_test_params = {
        'dataset'    : dataset_test,
        'batch_size' : n_env,
        'num_workers': 0,
        'sampler'    : sampler_test,
        'pin_memory' : True,
        'drop_last'  : False,
        'shuffle'    : False,
    }
    dataloader_train = DataLoader(**dataloader_train_params)
    dataloader_test  = DataLoader(**dataloader_test_params)

    return dataloader_train, dataloader_test         


def init_dataloaders(dataset, buffer_filepath=None):
    if dataset == 'lv':
        return param_lv()
    elif dataset == 'gs':
        return param_gs()
    elif dataset == 'ns':
        assert buffer_filepath is not None
        return param_ns(buffer_filepath)