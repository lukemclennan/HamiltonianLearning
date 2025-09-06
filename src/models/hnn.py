# Adapted from https://github.com/greydanus/hamiltonian-nn

import torch 
from models.mlp import MLP

class HNN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, hid_layers, act=torch.nn.Tanh()):
        super(HNN, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.hid_layers = hid_layers
        self.hnet = MLP(in_dim, 1, hid_dim, hid_layers, act)
        self.J = self.permutation_tensor(in_dim)
    
    def forward(self, t, x):
        Hx = self.hnet(x)
        dHx = torch.autograd.grad(Hx.sum(), x, create_graph=True)[0]
        self.J = self.J.to(x.device)
        return dHx @ self.J.t()
    
    def permutation_tensor(self, n):
        J = torch.eye(n)
        J = torch.cat([J[n // 2:], -J[:n // 2]])
        return J