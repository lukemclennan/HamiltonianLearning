from datasets.base import HamiltonianSystem, TrajectoryGenerator
import torch

class DoublePendulum(HamiltonianSystem):
    def __init__(self, m1=1.0, m2=1.0, L1=1.0, L2=1.0):
        super().__init__()
        self.m1, self.m2, self.L1, self.L2 = m1, m2, L1, L2

    @property
    def input_dim(self):
        return 4

    # def dynamics(self, t, z):
    #     z = z.clone().requires_grad_(True)
    #     H = self.hamiltonian(t, z)
    #     gradH = torch.autograd.grad(H.sum(), z, create_graph=True)[0]
    #     J = torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 1.], [-1., 0., 0., 0.], [0., -1., 0., 0.]], dtype=z.dtype, device=z.device)
    #     return gradH @ J.T

    def hamiltonian(self, t, z):
        q1, q2, p1, p2 = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
        delta = q2 - q1
        D = 2 * self.L1**2 * self.L2**2 * (self.m1 + self.m2 * (1 - torch.cos(delta)**2))
        num = (self.m2 * self.L2**2 * p1**2 + (self.m1 + self.m2) * self.L1**2 * p2**2 -
               2 * self.m2 * torch.cos(delta) * self.L1 * self.L2 * p1 * p2)
        V = -(self.m1 + self.m2) * 9.81 * self.L1 * torch.cos(q1) - self.m2 * 9.81 * self.L2 * torch.cos(q2)
        return num / D + V


class DoublePendulumDataset:
    def __init__(self, T=10.0, timescale=10, samples=100, sigma=0.1,
                 q1_lower=-0.5, q1_upper=0.5, q2_lower=-0.5, q2_upper=0.5,
                 m1=1.0, m2=1.0, L1=1.0, L2=1.0):
        self.q1_lower = q1_lower
        self.q1_upper = q1_upper
        self.q2_lower = q2_lower
        self.q2_upper = q2_upper
        self.config = dict(T=T, timescale=timescale, samples=samples, sigma=sigma,
                           q1_lower=q1_lower, q1_upper=q1_upper, q2_lower=q2_lower, q2_upper=q2_upper,
                           m1=m1, m2=m2, L1=L1, L2=L2)
        self.system = DoublePendulum(m1=m1, m2=m2, L1=L1, L2=L2)
        self.generator = TrajectoryGenerator(
            self.system, T, timescale, samples, self.sample_initial_conditions,
            sigma, config=self.config)

    def sample_initial_conditions(self, n):
        q1 = torch.linspace(self.q1_lower, self.q1_upper, n)
        q2 = torch.linspace(self.q2_lower, self.q2_upper, n)
        p1 = torch.zeros_like(q1)
        p2 = torch.zeros_like(q2)
        return torch.stack([q1, q2, p1, p2], dim=-1).requires_grad_(True)

    def generate(self):
        return self.generator.generate_trajectories()
