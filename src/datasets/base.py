import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchdiffeq import odeint
from typing import Tuple

class HamiltonianSystem(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def input_dim(self) -> int:
        # By default 2n system with $n=1$.
        return 2

    def hamiltonian(self, t, z):
        raise NotImplementedError

    def dynamics(self, t, z):
        z = z.clone().requires_grad_(True)
        H = self.hamiltonian(t, z)
        gradH = torch.autograd.grad(H.sum(), z, create_graph=True)[0]
        n = self.input_dim // 2
        zeros = torch.zeros(n, n, dtype=z.dtype, device=z.device)
        I     = torch.eye(  n,   dtype=z.dtype, device=z.device)
        J_top    = torch.cat([zeros, I],     dim=1)   # [n×2n]
        J_bottom = torch.cat([-I,    zeros], dim=1)   # [n×2n]
        J = torch.cat([J_top, J_bottom], dim=0)    
        return gradH @ J.T

    def forward(self, t, z):
        return self.dynamics(t, z)

    def integrate(self, z0, t_span: Tuple[float, float], dt: float):
        ts = torch.arange(t_span[0], t_span[1], dt, device=z0.device)
        zt = odeint(self.dynamics, z0, ts, method='fehlberg2', atol=1e-8, rtol=1e-8)
        return zt, ts

class TrajectoryGenerator:
    def __init__(self, system, T, timescale, samples, z_sampler, sigma, config=None):
        self.system = system
        self.T = T
        self.timescale = timescale
        self.samples = samples
        self.z_sampler = z_sampler
        self.sigma = sigma
        self.config = config if config is not None else {}

    def generate_trajectories(self):
        z0s = self.z_sampler(self.samples)
        t = torch.linspace(0, self.T, int(self.T * self.timescale + 1))
        xts = odeint(self.system, z0s, t, method='dopri5', atol=1e-8, rtol=1e-8)
        noise = torch.randn_like(xts) * self.sigma
        yts = xts + noise
        config = self._sanitize_config()
        print(config)
        return {
            'yts': yts.permute(1, 0, 2).detach(),
            't': t.detach(),
            'config': config
        }

    def _sanitize_config(self):
        def convert(v):
            return v.item() if isinstance(v, torch.Tensor) else v
        return {k: convert(v) for k, v in self.config.items()}
    
class TrajectoryDataset(Dataset):
    def __init__(self, data_path, batch_time=5, expected_config=None):
        data = torch.load(data_path)
        self.yts = data['yts']
        self.t_eval = data['t']
        self.meta = data.get('config', None)
        self.batch_step = int(((len(self.t_eval)-1)/self.t_eval[-1]).item() * batch_time)

        if expected_config and self.meta:
            mismatches = []
            for key, val in expected_config.items():
                if key in self.meta and not torch.isclose(torch.tensor(self.meta[key]), torch.tensor(val), atol=1e-4).item():
                    mismatches.append(f"{key}: expected {val}, got {self.meta[key]}")
            if mismatches:
                raise ValueError("Dataset config mismatch:\n" + "\n".join(mismatches))

    def __len__(self):
        return self.yts.shape[0]

    def __getitem__(self, idx):
        n_points = self.yts.shape[1]
        p_idx = torch.randint(0, n_points - self.batch_step, (1,)).item()
        batch_x0 = self.yts[idx, p_idx].reshape(1, -1)
        batch_t = self.t_eval[:self.batch_step + 1]
        batch_x = self.yts[idx, p_idx:p_idx + self.batch_step + 1].reshape([self.batch_step + 1, 1, -1])
        return batch_x0, batch_t, batch_x
