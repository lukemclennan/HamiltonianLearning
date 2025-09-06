from datasets.base  import HamiltonianSystem, TrajectoryGenerator
import torch

class DampedSpring(HamiltonianSystem):
    def __init__(self, k=1.0, m=1.0, gamma=0.1):
        """
        k: spring constant
        m: mass
        gamma: damping coefficient (Rayleigh dissipation)
        """
        super().__init__()
        self.k, self.m, self.gamma = k, m, gamma

    @property
    def input_dim(self) -> int:
        # only (q,p)
        return 2

    def hamiltonian(self, t, z):
        """
        z: (batch, 2) tensor = [q, p]
        H(q,p) = ½ k q² + p²/(2m)
        """
        q, p = z[:, 0], z[:, 1]
        T = p ** 2 / (2 * self.m)
        V = 0.5 * self.k * (q ** 2)
        return T + V

class DampedSpringDataset:
    def __init__(self, T=10.0, timescale=10, samples=100, sigma=0.1,
                 q_lower=-torch.pi/4, q_upper=torch.pi/4, k=1.0, m=1.0, gamma=0.1):
        self.q_lower = q_lower
        self.q_upper = q_upper
        self.config = dict(T=T, timescale=timescale, samples=samples, sigma=sigma,
                           q_lower=q_lower, q_upper=q_upper, k=k, m=m, gamma=gamma)
        self.system = DampedSpring(k=k, m=m, gamma=gamma)
        self.generator = TrajectoryGenerator(
            self.system, T, timescale, samples, self.sample_initial_conditions,
            sigma, config=self.config)

    def sample_initial_conditions(self, n):
        q = torch.linspace(self.q_lower, self.q_upper, n)
        p = torch.zeros_like(q)
        return torch.stack([q, p], dim=-1).requires_grad_(True)

    def generate(self):
        return self.generator.generate_trajectories()
