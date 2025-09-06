from datasets.base import HamiltonianSystem, TrajectoryGenerator
import torch

class HenonHeiles(HamiltonianSystem):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha 
    def hamiltonian(self, t, z):
        x, y, px, py = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
        T = 0.5 * (px**2 + py**2)
        V = 0.5 * (x**2 + y**2) + self.alpha * (x**2 * y - y**3 / 3)
        return T + V
    
    @property
    def input_dim(self):
        return 4

class HenonHeilesDataset:
    def __init__(self, T=10.0, timescale=10, samples=100, alpha=1.0, sigma=0.1,
                 q_lower=-0.5, q_upper=0.5):
        self.q_lower = q_lower
        self.q_upper = q_upper
        self.config = dict(T=T, timescale=timescale, samples=samples, alpha=alpha, sigma=sigma,
                           q_lower=q_lower, q_upper=q_upper)
        self.system = HenonHeiles(alpha=alpha)
        self.generator = TrajectoryGenerator(
            self.system, T, timescale, samples, self.sample_initial_conditions,
            sigma, config=self.config)

    def sample_initial_conditions(self, n):
        qx = torch.linspace(self.q_lower, self.q_upper, n)
        qy = torch.linspace(self.q_lower, self.q_upper, n)
        px = torch.zeros_like(qx)
        py = torch.zeros_like(qy)
        return torch.stack([qx, qy, px, py], dim=-1).requires_grad_(True)

    def generate(self):
        return self.generator.generate_trajectories()
