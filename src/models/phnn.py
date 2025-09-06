# Adapted from https://github.com/shaandesai1/PortHNN/

import torch 
from models.mlp import MLP
from models.hnn import HNN

class PHNN(HNN):
    def __init__(self, in_dim, hid_dim, hid_layers, act=torch.nn.Tanh()):
        super(PHNN, self).__init__(in_dim, hid_dim, hid_layers, act)
        self.D = torch.nn.Parameter(torch.zeros(int(self.in_dim/2)))
        self.F = MLP(1, int(in_dim/2), hid_dim, hid_layers, act)
    
    def forward(self, t, x):
        Hx = self.hnet(x)
        dHx = torch.autograd.grad(Hx.sum(), x, create_graph=True)[0]
        D = torch.diag(torch.cat((torch.zeros_like(self.D), self.D)))
        Ft = self.F(t.reshape(-1,1))
        F = torch.cat((torch.zeros_like(Ft), Ft), dim=1)
        self.J = self.J.to(x.device)
        output = dHx @ (self.J.t() + D.t()) + F
        # print(output.shape, dHx.shape, self.J.shape, D.shape, F.shape)
        return output