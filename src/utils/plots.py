import torch
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import torch

def plot_trajectories(data, n=10, filename="trajectories.png"):
    """
    Plot the first n trajectories (first dimension of q and p) from the dataset.
    data: dict containing 'yts' (B x T x D)
    """
    yts = data['yts']  # shape: (B, T, D)
    if isinstance(yts, torch.Tensor):
        yts = yts.cpu().detach().numpy()

    D = yts.shape[2]
    assert D % 2 == 0, "Expected state dimension to be even (q and p)"
    q_dim = D // 2

    plt.figure(figsize=(10, 8))
    for i in range(min(n, len(yts))):
        q_traj = yts[i, :, 0]      # q trajectory
        p_traj = yts[i, :, q_dim]  # p trajectory
        plt.plot(q_traj, p_traj, label=f'Traj {i+1}', lw=1)

    plt.title(f'First {n} Trajectories (q vs p)')
    plt.xlabel('q')
    plt.ylabel('p')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()