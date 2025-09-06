from datasets.single_pendulum import SinglePendulum
from datasets.base import TrajectoryGenerator
import torch

class DampedPendulum(SinglePendulum):
    def __init__(self, gamma=0.1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def dynamics(self, t, z):
        z = z.clone().requires_grad_(True)
        q, p = z[:, 0], z[:, 1]
        H = self.hamiltonian(t, z)
        gradH = torch.autograd.grad(H.sum(), z, create_graph=True)[0]
        dq = gradH[:, 1]
        dp = -gradH[:, 0] - self.gamma * dq
        return torch.stack([dq, dp], dim=-1)

class DampedPendulumDataset:
    def __init__(self, T=10.0, timescale=10, samples=100, sigma=0.1,
                 q_lower=-torch.pi/4, q_upper=torch.pi/4, m=1.0, l=1.0, g=9.81, gamma=0.1):
        self.q_lower = q_lower
        self.q_upper = q_upper
        self.config = dict(T=T, timescale=timescale, samples=samples, sigma=sigma,
                           q_lower=q_lower, q_upper=q_upper, m=m, l=l, g=g, gamma=gamma)
        self.system = DampedPendulum(m=m, l=l, g=g, gamma=gamma)
        self.generator = TrajectoryGenerator(
            self.system, T, timescale, samples, self.sample_initial_conditions,
            sigma, config=self.config)

    def sample_initial_conditions(self, n):
        q = torch.linspace(self.q_lower, self.q_upper, n)
        p = torch.zeros_like(q)
        return torch.stack([q, p], dim=-1).requires_grad_(True)

    def generate(self):
        return self.generator.generate_trajectories()
