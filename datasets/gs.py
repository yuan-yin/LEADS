import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.integrate import solve_ivp
from functools import partial

class GrayScottReactionDataset(Dataset):

    def __init__(self, num_traj_per_env, size, time_horizon, dt_eval, params, n_block, dx=2., buffer=dict(), method='RK45', group='train'):
        super().__init__()
        self.num_traj_per_env = num_traj_per_env
        self.num_env = len(params)
        self.len = num_traj_per_env * self.num_env
        self.size = int(size)  
        self.dx = dx
        self.time_horizon = float(time_horizon)
        self.n = int(time_horizon / dt_eval)
        self.dt_eval = dt_eval

        self.params_eq = params
        self.test = group == 'test'
        self.max = np.iinfo(np.int32).max
        self.buffer = buffer
        self.method = method
        self.indices = [list(range(env * num_traj_per_env, (env + 1) * num_traj_per_env)) for env in range(self.num_env)]
        self.n_block = n_block

    def _laplacian2D(self, a):
        a_zz = a

        a_nz = np.roll(a_zz,(+1, 0),(0,1))
        a_pz = np.roll(a_zz,(-1, 0),(0,1))
        a_zn = np.roll(a_zz,( 0,+1),(0,1))
        a_zp = np.roll(a_zz,( 0,-1),(0,1))

        a_nn = np.roll(a_zz,(+1,+1),(0,1))
        a_np = np.roll(a_zz,(+1,-1),(0,1))
        a_pn = np.roll(a_zz,(-1,+1),(0,1))
        a_pp = np.roll(a_zz,(-1,-1),(0,1))

        return (
            - 3 * a
            + 0.5  * (a_nz + a_pz + a_zn + a_zp)
            + 0.25 * (a_nn + a_np + a_pn + a_pp)
        ) / (self.dx ** 2)

    def _vec_to_mat(self, vec_uv):
        UV = np.split(vec_uv, 2)
        U = np.reshape(UV[0], (self.size, self.size))
        V = np.reshape(UV[1], (self.size, self.size))
        return U, V

    def _mat_to_vec(self, mat_U, mat_V):
        dudt = np.reshape(mat_U, self.size * self.size)
        dvdt = np.reshape(mat_V, self.size * self.size)
        return np.concatenate((dudt, dvdt))

    def _f(self, t, uv, env):
        D_u = self.params_eq[env]['D_u']
        D_v = self.params_eq[env]['D_v']
        F = self.params_eq[env]['F']
        k = self.params_eq[env]['k']
        u, v = self._vec_to_mat(uv)

        delta_u = self._laplacian2D(u)
        delta_v = self._laplacian2D(v)

        dudt = (D_u * delta_u - u * (v ** 2) + F * (1. - u))
        dvdt = (D_v * delta_v + u * (v ** 2) - (F + k) * v)
        
        duvdt = self._mat_to_vec(dudt, dvdt)
        return duvdt
    
    def _get_init_cond(self, index):
        np.random.seed(index if not self.test else self.max-index)
        size = (self.size, self.size)
        u0 = 0.95 * np.ones(size)
        v0 = 0.05 * np.ones(size)
        for _ in range(self.n_block):
            r = int(self.size / 10)
            N2 = np.random.randint(low=0, high=self.size-r, size=2)

            u0[N2[0]:N2[0]+r, N2[1]:N2[1]+r] = 0.
            v0[N2[0]:N2[0]+r, N2[1]:N2[1]+r] = 1.
        return u0, v0

    def __getitem__(self, index):
        env = index // self.num_traj_per_env
        env_index = index % self.num_traj_per_env
        t = torch.arange(0, self.time_horizon, self.dt_eval).float()
        if self.buffer.get(index) is None:
            uv_0 = self._mat_to_vec(*self._get_init_cond(env_index))
            
            res = solve_ivp(partial(self._f, env=env), (0., self.time_horizon), y0=uv_0, method=self.method, t_eval=np.arange(0., self.time_horizon, self.dt_eval))
            res_uv = res.y

            u = []
            v = []
            for i in range(self.n):
                res_U, res_V = self._vec_to_mat(res_uv[:, i])
                u.append(torch.from_numpy(res_U).unsqueeze(0))
                v.append(torch.from_numpy(res_V).unsqueeze(0))

            u = torch.stack(u, dim=1)
            v = torch.stack(v, dim=1)

            state = torch.cat([u, v],dim=0).float()
            self.buffer[index] = state.numpy()
            return {
                'state'  : state,
                't'      : t,
                'env'    : env,
            }
        else:
            return {
                'state'  : torch.from_numpy(self.buffer[index]),
                't'      : t,
                'env'    : env,
            }

    def __len__(self):
        return self.len