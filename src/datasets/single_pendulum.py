from datasets.base import HamiltonianSystem, TrajectoryGenerator
import torch

class SinglePendulum(HamiltonianSystem):
    def __init__(self, m=1.0, l=1.0, g=9.81):
        super().__init__()
        self.m, self.l, self.g = m, l, g

    def hamiltonian(self, t, z):
        q, p = z[:, 0], z[:, 1]
        T = p**2 / (2 * self.m * self.l**2)
        V = self.m * self.g * self.l * (1 - torch.cos(q))
        return T + V


class SinglePendulumDataset:
    def __init__(self, T=10.0, timescale=10, samples=100, sigma=0.1,
                 q_lower=-torch.pi/4, q_upper=torch.pi/4, m=1.0, l=1.0, g=9.81):
        self.q_lower = q_lower
        self.q_upper = q_upper
        self.config = dict(T=T, timescale=timescale, samples=samples, sigma=sigma,
                           q_lower=q_lower, q_upper=q_upper, m=m, l=l, g=g)
        self.system = SinglePendulum(m=m, l=l, g=g)
        self.generator = TrajectoryGenerator(
            self.system, T, timescale, samples, self.sample_initial_conditions,
            sigma, config=self.config)

    def sample_initial_conditions(self, n):
        q = torch.linspace(self.q_lower, self.q_upper, n)
        p = torch.zeros_like(q)
        return torch.stack([q, p], dim=-1).requires_grad_(True)

    def generate(self):
        return self.generator.generate_trajectories()
