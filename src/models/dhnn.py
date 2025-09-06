# Adapted from https://github.com/greydanus/dissipative_hnns

import torch 
from models.mlp import MLP
from models.hnn import HNN

class DHNN(HNN):
    def __init__(self, in_dim, hid_dim, hid_layers, act=torch.nn.Tanh()):
        super(DHNN, self).__init__(in_dim, hid_dim, hid_layers, act)
        self.dnet = MLP(in_dim, 1, hid_dim, hid_layers, act)
    
    def forward(self, t, x):
        Hx = self.hnet(x)
        dHx = torch.autograd.grad(Hx.sum(), x, create_graph=True)[0]
        self.J = self.J.to(x.device)
        JdHx = dHx @ self.J.t()
        Dx = self.dnet(x)
        dDx = torch.autograd.grad(Dx.sum(), x, create_graph=True)[0]
        return JdHx + dDx