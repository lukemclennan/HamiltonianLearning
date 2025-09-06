import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from datasets.base import HamiltonianSystem, TrajectoryGenerator

class WindyPendulum(nn.Module):
    def __init__(self, mass, gravity, length, friction, wind_res, wind_speed):
        super().__init__()
        # Pendulum properties
        self.m = mass
        self.g = gravity
        self.l = length
        self.r = friction  # Damping factor
        self.k = wind_res
        self.v = wind_speed
        # Skew-symmetric matrix for Hamiltonian mechanics
        self.S = torch.tensor([[0, 1], [-1, 0]]).float()

    def hamiltonian(self, coords, t):
        # Kinetic + Potential Energy
        q, p = coords[:, 0], coords[:, 1]
        d = torch.exp((self.r - self.k * self.l**2)*t/(self.m * self.l**2))
        H = p.pow(2) / (2 * d * self.m * self.l ** 2) + d * self.m * self.g * self.l * (1 - torch.cos(q)) + d * self.k * self.v * self.l * t * q
        return H
    
    def forward(self, t, x):
        # Calculate the Hamiltonian
        H = self.hamiltonian(x,t)
        # Compute gradient of Hamiltonian
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        # Hamiltonian vector field calculation
        field = dH @ self.S.t()
        return field


class WindyPendulumDataset:
    def __init__(self, T=10.0, timescale=10, samples=100, sigma=0.1,
                 q_lower=-torch.pi/4, q_upper=torch.pi/4, wind_res=0.3, wind_speed=2.0, mass=1.0, length=1.0, gravity=9.81, friction=0.2):
        self.q_lower = q_lower
        self.q_upper = q_upper
        self.config = dict(T=T, timescale=timescale, samples=samples, sigma=sigma,
                           q_lower=q_lower, q_upper=q_upper, wind_res=wind_res, wind_speed=wind_speed, 
                           mass=mass, length=length, gravity=gravity, friction=friction)
        self.system = WindyPendulum(mass=mass, gravity=gravity, length=length, wind_res=wind_res, wind_speed=wind_speed, friction=friction)
        self.generator = TrajectoryGenerator(
            self.system, T, timescale, samples, self.sample_initial_conditions,
            sigma, config=self.config)

    def sample_initial_conditions(self, n):
        q = torch.linspace(self.q_lower, self.q_upper, n)
        p = torch.zeros_like(q)
        return torch.stack([q, p], dim=-1).requires_grad_(True)

    def generate(self):
        return self.generator.generate_trajectories()


# class WindyPendulumDataset:
#     def __init__(self, mass, gravity, length, friction, wind_res, wind_speed, T, timescale, q_lower, q_upper, samples, sigma):
#         """
#         Initializes parameters for simulating a pendulum dataset.
        
#         Parameters:
#             mass (float): Mass of the pendulum bob.
#             gravity (float): Acceleration due to gravity.
#             length (float): Length of the pendulum.
#             friction (float): Damping coefficient.
#             T (float): Total time of simulation.
#             timescale (int): Number of time steps per unit time.
#             samples (int): Number of initial conditions to simulate.
#             q_lower (float): Lower bound of the pendulum's initial angle.
#             q_upper (float): Upper bound of the pendulum's initial angle.
#             sigma (float): Standard deviation of measurement noise.
#         """
#         self.ode = WindyPendulum(mass, gravity, length, friction, wind_res, wind_speed)
#         self.samples = samples
#         self.T = T
#         self.timescale = timescale
#         self.q_lower = q_lower
#         self.q_upper = q_upper
#         self.sigma = sigma
    
#     def get_initial_conditions(self):
#         """
#         Generates initial conditions for the pendulum's angle (q) ranging from -pi to pi (excluding the endpoints) 
#         and conjugate momentum (p_theta) set to zero.
        
#         Returns:
#             torch.Tensor: Initial conditions [q, p_theta] with shape (samples, 2).
#         """
#         # Create a grid of values for q ranging from -pi+0.01 to pi-0.01
#         q_values = torch.linspace(self.q_lower, self.q_upper, self.samples)
        
#         # Set the conjugate momenta (p_theta) to zero
#         p_values = torch.zeros_like(q_values)
        
#         # Concatenate q and p_theta to form initial conditions
#         initial_conditions = torch.stack([q_values, p_values], dim=1)
#         initial_conditions.requires_grad_(True)  # Enable gradient computation for initial conditions
        
#         return initial_conditions


#     def generate_trajectories(self):
#         """
#         Simulates the pendulum dynamics and generates trajectories with added noise.
        
#         Returns:
#             dict: Dictionary containing the simulated state variables and their derivatives.
#         """
#         x0s = self.get_initial_conditions()
#         t = torch.linspace(0, self.T, int(self.T * self.timescale + 1))
#         xts = odeint(self.ode, x0s, t, method='dopri5', atol=1e-8, rtol=1e-8)
        
#         # Add Gaussian noise to each state variable to model measurement errors
#         noise = torch.randn(xts.shape) * self.sigma
#         yts = xts + noise
#         yts = yts.permute(1, 0, 2)
#         return {'yts': yts.detach(), 't': t.detach()}
    
class WindyPendulumTrajectoryDataset(Dataset):
    def __init__(self, data_path, batch_time=5, expected_config=None):
        data = torch.load(data_path)
        self.yts = data['yts']
        self.t_eval = data['t']
        self.meta = data.get('config', None)
        self.batch_step = int(((len(self.t_eval)-1)/self.t_eval[-1]).item() * batch_time)

        # Validate configuration if expected_config is given
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
