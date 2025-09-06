import numpy as np
import torch
from torch import nn
from math import floor, sqrt

class SSGP(torch.nn.Module):
    def __init__(self, input_dim, basis, friction, K, dropout_rate=0):
        super(SSGP, self).__init__()
        self.sigma = nn.Parameter(torch.tensor([1e-1]))
        self.a = nn.Parameter(torch.ones(input_dim) * 1e-1)
        self.b = nn.Parameter(1e-4 * (torch.rand(basis * 2) - 0.5))
        self.SqrtC = nn.Parameter(torch.ones(basis, basis) * 1e-2 + torch.eye(basis) * 1e-2)
        self.sigma_0 = nn.Parameter(torch.tensor([1e-0]))
        self.lam = nn.Parameter(torch.ones(input_dim) * 1.5)
        self.eta = nn.Parameter(torch.tensor([1e-16])) if friction else torch.tensor([0.0])
        
        self.register_buffer('S', self.permutation_tensor(input_dim))
        # self.S = self.permutation_tensor(input_dim)
        tmp = torch.normal(0, 1, size=(basis // 2, input_dim))
        self.register_buffer('epsilon', torch.vstack([tmp, -tmp]))  # Register epsilon as a buffer
        self.d = input_dim
        self.num_basis = basis
        self.K = K
        self.drop = nn.Dropout(p=dropout_rate)

    def sampling_epsilon_f(self):
        sqrt_C = torch.block_diag(self.SqrtC, self.SqrtC)
        device = self.b.device  # get the current model parameter device

        epsilon = torch.randn((sqrt_C.shape[0], 1), device=device)
        self.w = self.b + (sqrt_C @ epsilon).squeeze()

        for _ in range(self.K):
            epsilon = torch.randn((sqrt_C.shape[0], 1), device=device)
            self.w += self.b + (sqrt_C @ epsilon).squeeze()

        self.w = self.w / self.K
        self.w = self.drop(self.w)

    def mean_w(self):
        self.w = self.b * 1

    def neg_loglike(self, batch_x, pred_x):
        n_samples, n_points, _, input_dim = batch_x.shape
        likelihood = ((-(pred_x - batch_x) ** 2 / self.sigma ** 2 / 2).nansum()
                      - torch.log(self.sigma ** 2) / 2 * n_samples * n_points * input_dim)
        return -likelihood

    def KL_x0(self, x0):
        n, d = x0.shape
        S = torch.diag(self.a ** 2)
        return 0.5 * ((x0 * x0).sum() / n + d * torch.trace(S) - d * torch.logdet(S) - d)

    def KL_w(self):
        num = self.b.shape[0]
        C = self.SqrtC @ self.SqrtC.T
        C = torch.block_diag(C, C)
        term3 = (self.b * self.b).sum() / (self.sigma_0 ** 2 / num * 2)
        term2 = torch.diag(C).sum() / (self.sigma_0 ** 2 / num * 2)
        term1_1 = torch.log(self.sigma_0 ** 2 / num * 2) * num
        term1_2 = torch.logdet(C)
        return 0.5 * (term1_1 - term1_2 + term2 + term3)

    def sampling_x0(self, x0):
        n, _, d = x0.shape
        device = self.a.device

        noise = torch.normal(0, 1, size=(n, 1, d), device=device)
        scale = torch.stack([self.a ** 2] * n, dim=0).reshape(n, 1, d)
        return x0 + torch.sqrt(scale) * noise

    def permutation_tensor(self, n):
        S = torch.eye(n)
        return torch.cat([S[n // 2:], -S[:n // 2]], dim=0)

    def forward(self, t, x):
        # Compute the sqrt(Var) of the kernel's spectral density
        s = self.epsilon @ torch.diag((1 / torch.sqrt(4 * torch.pi ** 2 * self.lam ** 2)))
        
        # Dissipation matrix
        R = torch.eye(self.d, device=self.a.device)
        R[:int(self.d / 2), :int(self.d / 2)] = 0
        
        # Compute the Symplectic Random Fourier Feature (SRFF) matrix (Psi)
        Psi = 2 * torch.pi * ((self.S - self.eta.to(self.a.device) ** 2 * R) @ s.T).T
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
        return f.reshape([samples, 1, self.d])

    def sample_hamiltonian(self, x):
        # Compute the sqrt(Var) of the kernel's spectral density
        s = self.epsilon @ torch.diag((1 / torch.sqrt(4 * torch.pi ** 2 * self.lam ** 2)))

        # Compute the Symplectic Random Fourier Feature (SRFF) matrix (Psi)
        sim = 2 * torch.pi * s @ x.squeeze(dim=1).T
        basis_c = torch.cos(sim)
        basis_s = torch.sin(sim)

        samples = x.shape[0]

        aug_basis = torch.vstack([basis_c, basis_s])

        aug_W = torch.stack([self.w] * samples).T

        # Compute the Hamiltonian
        H_aug = aug_basis * aug_W

        H = torch.vstack(torch.split(H_aug.sum(axis=0), samples)).T

        return H
    
    def cons_vector_field(self, t, x):
        # Compute the sqrt(Var) of the kernel's spectral density
        s = self.epsilon @ torch.diag((1 / torch.sqrt(4 * torch.pi ** 2 * self.lam ** 2)))
        
        # Dissipation matrix
        R = torch.eye(self.d, device=self.a.device)
        R[:int(self.d / 2), :int(self.d / 2)] = 0
        
        # Compute the Symplectic Random Fourier Feature (SRFF) matrix (Psi)
        Psi = 2 * torch.pi * ((self.S) @ s.T).T
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
        return f.reshape([samples, 1, self.d])