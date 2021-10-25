#####################################################################################
# Credit for the original Navier-Stokes dataset                                     #
#                                                                                   #
# Projet: https://github.com/zongyi-li/fourier_neural_operator                      #
# Copyright (c) 2020 Zongyi Li                                                      #
# Licence: https://github.com/zongyi-li/fourier_neural_operator/blob/master/LICENSE #
#####################################################################################

import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.integrate import solve_ivp
from functools import partial
import math
import matplotlib.pyplot as plt
import matplotlib
from timeit import default_timer
import scipy.io
import shelve

class GaussianRF(object):

    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic"):

        self.dim = dim

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1), \
                           torch.arange(start=-k_max, end=0, step=1)), 0)

            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1), \
                                    torch.arange(start=-k_max, end=0, step=1)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1), \
                                    torch.arange(start=-k_max, end=0, step=1)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self):

        coeff = torch.randn(*self.size, dtype=torch.cfloat)
        coeff = self.sqrt_eig * coeff
        u = torch.fft.ifftn(coeff)
        u = u.real
        
        return u

class NavierStokesDataset(Dataset):

    def __init__(self, num_traj_per_env, size, time_horizon, dt_eval, params, buffer_filepath=None, group='train'):
        super().__init__()
        self.size = int(size)
        tt = torch.linspace(0, 1, self.size+1)[0:-1]
        X,Y = torch.meshgrid(tt, tt)
        self.params_eq = params
        self.forcing_zero = self.params_eq[0]['f']
        self.num_traj_per_env = num_traj_per_env
        self.num_env = len(params)
        self.len = num_traj_per_env * self.num_env
        self.time_horizon = float(time_horizon)
        self.n = int(time_horizon / dt_eval)

        self.sampler = GaussianRF(2, self.size, alpha=2.5, tau=7)
        self.dt_eval = dt_eval
        self.dt = 1e-3
        self.buffer = shelve.open(buffer_filepath)
        self.test = group == 'test'
        self.max = np.iinfo(np.int32).max
        self.indices = [list(range(env * num_traj_per_env, (env + 1) * num_traj_per_env)) for env in range(self.num_env)]

    def navier_stokes_2d(self, w0, f, visc, T, delta_t, record_steps):

        #Grid size - must be power of 2
        N = w0.size()[-1]

        #Maximum frequency
        k_max = math.floor(N/2.0)

        #Number of steps to final time
        steps = math.ceil(T/delta_t)

        #Initial vorticity to Fourier space
        w_h = torch.fft.fftn(w0, (N,N))

        #Forcing to Fourier space
        f_h = torch.fft.fftn(f, (N,N))

        #If same forcing for the whole batch
        if len(f_h.size()) < len(w_h.size()):
            f_h = torch.unsqueeze(f_h, 0)

        #Record solution every this number of steps
        record_time = math.floor(steps/record_steps)

        #Wavenumbers in y-direction
        k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device), torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N,1)
        #Wavenumbers in x-direction
        k_x = k_y.transpose(0,1)
        #Negative Laplacian in Fourier space
        lap = 4*(math.pi**2)*(k_x**2 + k_y**2)
        lap[0,0] = 1.0
        #Dealiasing mask
        dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0/3.0)*k_max, torch.abs(k_x) <= (2.0/3.0)*k_max).float(), 0)

        #Saving solution and time
        sol = torch.zeros(*w0.size(), record_steps, 1, device=w0.device, dtype=torch.float)
        sol_t = torch.zeros(record_steps, device=w0.device)

        #Record counter
        c = 0
        #Physical time
        t = 0.0
        for j in range(steps):
            if j % record_time == 0:
                #Solution in physical space
                w = torch.fft.ifftn(w_h, (N,N))
                #Record solution and time
                sol[...,c,0] = w.real
                # sol[...,c,1] = w.imag
                sol_t[c] = t

                c += 1

            #Stream function in Fourier space: solve Poisson equation
            psi_h = w_h.clone()
            psi_h = psi_h / lap

            #Velocity field in x-direction = psi_y
            q = psi_h.clone()
            temp = q.real.clone()
            q.real = -2*math.pi * k_y * q.imag
            q.imag =  2*math.pi * k_y * temp
            q = torch.fft.ifftn(q, (N,N))

            #Velocity field in y-direction = -psi_x
            v = psi_h.clone()
            temp = v.real.clone()
            v.real = 2*math.pi*k_x*v.imag
            v.imag = -2*math.pi*k_x*temp
            v = torch.fft.ifftn(v, (N,N))

            #Partial x of vorticity
            w_x = w_h.clone()
            temp = w_x.real.clone()
            w_x.real = -2*math.pi*k_x*w_x.imag
            w_x.imag = 2*math.pi*k_x*temp
            w_x = torch.fft.ifftn(w_x, (N,N))

            #Partial y of vorticity
            w_y = w_h.clone()
            temp = w_y.real.clone()
            w_y.real = -2*math.pi*k_y*w_y.imag
            w_y.imag = 2*math.pi*k_y*temp
            w_y = torch.fft.ifftn(w_y, (N,N))

            #Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
            F_h = torch.fft.fftn(q*w_x + v*w_y, (N,N))

            #Dealias
            F_h = dealias* F_h

            #Cranck-Nicholson update
            w_h = (-delta_t*F_h + delta_t*f_h + (1.0 - 0.5*delta_t*visc*lap)*w_h)/(1.0 + 0.5*delta_t*visc*lap)

            #Update real time (used only for recording)
            t += delta_t

        return sol, sol_t
    
    def _get_init_cond(self, index):
        torch.manual_seed(index if not self.test else self.max-index)
        if self.buffer.get(f'init_cond_{index}') is None:
            w0 = self.sampler.sample()
            state, _ = self.navier_stokes_2d(w0, f=self.forcing_zero, visc=self.params_eq[0]['visc'], T=30.0, delta_t=self.dt, record_steps=20)
            init_cond = state[:, :, -1, 0]
            self.buffer[f'init_cond_{index}'] = init_cond.numpy()
        else:
            init_cond = torch.from_numpy(self.buffer[f'init_cond_{index}'])

        return init_cond

    def __getitem__(self, index):
        env = index // self.num_traj_per_env
        env_index = index % self.num_traj_per_env
        t = torch.arange(0, self.time_horizon, self.dt_eval).float()
        if self.buffer.get(f'{env},{env_index}') is None:
            print(f'calculating index {env_index} of env {env}')
            w0 = self._get_init_cond(env_index)
            state, _ = self.navier_stokes_2d(w0, f=self.params_eq[env]['f'], visc=self.params_eq[env]['visc'], T=self.time_horizon, delta_t=self.dt, record_steps=self.n)
            state = state.permute(3, 2, 0, 1)[:, :self.n] # nc, t, h, w
            
            self.buffer[f'{env},{env_index}'] = {
                'state': state.numpy(),
            }
            return {
                'state'  : state,
                't'      : t,
                'env'    : env,
            }
        else:
            buf = self.buffer[f'{env},{env_index}']
            return {
                'state'  : torch.from_numpy(buf['state'][:, :self.n]),
                't'      : t,
                'env'    : env,
            }

    def __len__(self):
        return self.len