from datasets.base import HamiltonianSystem, TrajectoryGenerator
import torch

class Duffing(HamiltonianSystem):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.2, F_ext=1.0):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.F_ext = alpha, beta, gamma, F_ext

    def hamiltonian(self, t, z):
        q, p = z[:, 0], z[:, 1]
        T = 0.5 * p**2
        V = 0.5 * self.alpha * q**2 + 0.25 * self.beta * q**4
        return T + V

    def dynamics(self, t, z):
        z = z.clone().requires_grad_(True)
        H = self.hamiltonian(t, z)
        gradH = torch.autograd.grad(H.sum(), z, create_graph=True)[0]
        dq = gradH[:, 1]
        dp = -gradH[:, 0] - self.gamma * dq + self.F_ext
        return torch.stack([dq, dp], dim=-1)

class DuffingDataset:
    def __init__(self, T=10.0, timescale=10, samples=100, sigma=0.1,
                 q_lower=-1.0, q_upper=1.0, alpha=1.0, beta=1.0, gamma=0.2, F_ext=1.0):
        self.q_lower = q_lower
        self.q_upper = q_upper
        self.config = dict(T=T, timescale=timescale, samples=samples, sigma=sigma,
                           q_lower=q_lower, q_upper=q_upper, alpha=alpha, beta=beta, gamma=gamma, F_ext=F_ext)
        print(self.config)
        self.system = Duffing(alpha=alpha, beta=beta, gamma=gamma, F_ext=F_ext)
        self.generator = TrajectoryGenerator(
            self.system, T, timescale, samples, self.sample_initial_conditions,
            sigma, config=self.config)

    def sample_initial_conditions(self, n):
        q = torch.linspace(self.q_lower, self.q_upper, n)
        p = torch.zeros_like(q)
        return torch.stack([q, p], dim=-1).requires_grad_(True)

    def generate(self):
        return self.generator.generate_trajectories()
