from models.ssgp import SSGP
from models.mlp import MLP
import torch


class PSSGP(SSGP):
    def __init__(self, input_dim, basis, friction, K, dropout_rate=0):
        super(PSSGP, self).__init__(input_dim, basis, friction, K, dropout_rate=dropout_rate)
        self.F = MLP(1, int(input_dim/2), basis, 0, torch.nn.ReLU())
    def forward(self, t, x):
        # Compute the sqrt(Var) of the kernel's spectral density
        s = self.epsilon @ torch.diag((1 / torch.sqrt(4 * torch.pi ** 2 * self.lam ** 2)))
        
        # Dissipation matrix
        R = torch.eye(self.d, device=self.a.device)
        R[:int(self.d / 2), :int(self.d / 2)] = 0
        
        # Compute the Symplectic Random Fourier Feature (SRFF) matrix (Psi)
        Psi = 2 * torch.pi * ((self.S - self.eta ** 2 * R) @ s.T).T
        # Reshape x to have shape [samples, input_dim]
        x = x.squeeze(dim=1)
        # print('x', x.shape)
        samples = x.shape[0]
        sim = 2 * torch.pi * s @ x.T
        basis_s = -torch.sin(sim)
        basis_c = torch.cos(sim)

        # Deterministic
        tmp = []
        for i in range(self.d):
            tmp.extend([Psi[:, i]] * samples)
        tmp = torch.stack(tmp).T

        aug_mat = torch.vstack([tmp, tmp])

        # Ensure aug_s and aug_c have consistent shapes
        aug_s = torch.hstack([basis_s] * self.d)
        aug_c = torch.hstack([basis_c] * self.d)
        
        # Ensure aug_basis has the correct dimensions
        aug_basis = torch.vstack([aug_s, aug_c]).reshape(aug_mat.shape)
        
        PHI = aug_mat * aug_basis
        aug_W = torch.stack([self.w] * samples * self.d).T
        F = PHI * aug_W
        f = torch.vstack(torch.split(F.sum(axis=0), samples)).T
        f = f.reshape([samples, 1, self.d])
        f[:,:,int(self.d/2):] = f[:,:,int(self.d/2):] + self.F(t.reshape(-1))
        return f

