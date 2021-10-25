import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.integrate import solve_ivp
from functools import partial
from statistics import mean
from scipy.stats import ortho_group

class LinearDataset(Dataset):

    def __init__(self, num_traj_per_env, time_horizon, params, dt, method='RK45', group='train'):
        super().__init__()
        self.num_traj_per_env = num_traj_per_env
        self.params_eq = params
        self.num_env = len(params)
        self.dim = len(params[0]['eig_vals'])
        self.len = num_traj_per_env * self.num_env
        self.time_horizon = float(time_horizon) 
        self.dt = dt

        np.random.seed(19700101)
        self.mat_ortho = ortho_group.rvs(self.dim) 

        self.test = group == 'test'
        self.max = np.iinfo(np.int32).max
        self.buffer = dict()
        self.method = method
        self.indices = [list(range(env * num_traj_per_env, (env + 1) * num_traj_per_env)) for env in range(self.num_env)]

    def _f(self, t, x, env=0):
        eig_vals = np.array(self.params_eq[env]['eig_vals'])
        b = self.params_eq[env].get('b')
        sigma = np.diag(eig_vals) 
        deriv = self.mat_ortho.T @ sigma @ self.mat_ortho @ x
        if b is not None:
            deriv += b
        return deriv

    def _get_init_cond(self, index):
        np.random.seed(index if not self.test else self.max-index)
        return np.random.randn(self.dim)

    def __getitem__(self, index):
        env       = index // self.num_traj_per_env
        env_index = index %  self.num_traj_per_env
        t = torch.arange(0, self.time_horizon, self.dt).float()
        if self.buffer.get(index) is None:
            u_0 = self._get_init_cond(env_index)

            res = solve_ivp(partial(self._f, env=env), (0., self.time_horizon), y0=u_0, method=self.method, t_eval=np.arange(0., self.time_horizon, self.dt))
            res_u = torch.from_numpy(res.y).float()

            self.buffer[index] = res_u.numpy()
            return {
                'state'   : res_u,
                't'       : t,
                'env'     : env,
            }
        else:
            return {
                'state'   : torch.from_numpy(self.buffer[index]),
                't'       : t,
                'env'     : env,
            }

    def __len__(self):
        return self.len