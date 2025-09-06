from datasets.base import HamiltonianSystem, TrajectoryGenerator
import torch


class SpringPendulum(HamiltonianSystem):
    def __init__(self, k=1.0, m=1.0, g=9.81, r0=1.0):
        super().__init__()
        self.k, self.m, self.g, self.r0 = k, m, g, r0

    
    @property
    def input_dim(self):
        return 4


    def hamiltonian(self, t, z):
        """
        z shape: (batch, 4) corresponds to (q1, q2, p1, p2)
        q1 = radial extension from rest length
        q2 = pendulum angle
        p1 = radial momentum, p2 = angular momentum
        """
        q1, q2, p1, p2 = z[:, 0], z[:, 1], z[:, 2], z[:, 3]

        # Kinetic energy
        T_r     = p1.pow(2) / (2 * self.m)
        T_theta = p2.pow(2) / (2 * self.m * (q1 + self.r0).pow(2))

        # Potential energy
        V_spring = 0.5 * self.k * q1.pow(2)
        V_grav   = self.m * self.g * (q1 + self.r0) * (1 - torch.cos(q2))

        return T_r + T_theta + V_spring + V_grav


class SpringPendulumDataset:
    def __init__(self,
                 T=10.0,
                 timescale=10,
                 samples=100,
                 sigma=0.1,
                 q1_lower=-0.5,
                 q1_upper=0.5,
                 q2_lower=-torch.pi/4,
                 q2_upper=torch.pi/4,
                 k=1.0,
                 m=1.0,
                 g=9.81,
                 r0=1.0):
        """
        - T: total integration time
        - timescale: number of integration steps per unit time
        - samples: how many trajectories to generate
        - sigma: initial‐condition noise
        - q1_lower/upper: radial displacement bounds
        - q2_lower/upper: angular displacement bounds
        - k, m, g, r0: physical parameters
        """
        self.q1_lower = q1_lower
        self.q1_upper = q1_upper
        self.q2_lower = q2_lower
        self.q2_upper = q2_upper
        self.config = dict(
            T=T, timescale=timescale, samples=samples, sigma=sigma,
            q1_lower=q1_lower, q1_upper=q1_upper,
            q2_lower=q2_lower, q2_upper=q2_upper,
            k=k, m=m, g=g, r0=r0
        )

        self.system = SpringPendulum(k=k, m=m, g=g, r0=r0)
        self.generator = TrajectoryGenerator(
            self.system,
            T,
            timescale,
            samples,
            self.sample_initial_conditions,
            sigma,
            config=self.config
        )

    def sample_initial_conditions(self, n):
        # Uniformly sample q1 and q2, zero momenta
        q1 = torch.linspace(self.q1_lower, self.q1_upper, n)
        q2 = torch.linspace(self.q2_lower, self.q2_upper, n)
        p1 = torch.zeros(n)
        p2 = torch.zeros(n)
        # stack into shape (n,4) and require grad if needed
        return torch.stack([q1, q2, p1, p2], dim=-1).requires_grad_(True)

    def generate(self):
        return self.generator.generate_trajectories()