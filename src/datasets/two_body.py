from base import HamiltonianSystem
import torch

class TwoBody(HamiltonianSystem):
    def __init__(self, G=1.0, m1=1.0, m2=1.0):
        super().__init__()
        self.G, self.m1, self.m2 = G, m1, m2

    def hamiltonian(self, t, z):
        x, y, px, py = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
        r = torch.sqrt(x**2 + y**2)
        T = 0.5 * (px**2 + py**2) / (self.m1 + self.m2)
        V = -self.G * self.m1 * self.m2 / r
        return T + V
