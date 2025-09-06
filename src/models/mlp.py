import torch
from collections import OrderedDict

class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, hid_layers, act):
        super(MLP, self).__init__()
        self.layers = [
            ('linear0', torch.nn.Linear(in_dim, hid_dim)),
            ('act0', act)
        ]
        for i in range(hid_layers):
            self.layers.append(('linear'+str(i+1), torch.nn.Linear(hid_dim, hid_dim)))
            self.layers.append(('act'+str(i+1), act))
        self.layers.append(('linear'+str(hid_layers+1), torch.nn.Linear(hid_dim, out_dim)))
        self.net = torch.nn.Sequential(OrderedDict(self.layers))
        
    def forward(self, x):
        return self.net(x)