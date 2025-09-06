import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from torchdiffeq import odeint
from matplotlib.animation import FuncAnimation

from models.ssgp import SSGP

def load_checkpoint(filepath):
    """
    Load the model checkpoint from the given file path.
    """
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)  # Ensure it loads on CPU
    return checkpoint

def load_data(filepath):
    """Load the dataset from a file."""
    return torch.load(filepath, weights_only=False)

def plot_losses(stats):
    """
    Plot the training and validation losses from the training statistics.
    """
    train_losses = stats['train_loss']
    val_losses = stats['val_loss']
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss', linestyle='--')
    plt.title('Training and Validation Losses')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('losses.png')
    plt.close()

modelnames = ['no_reg_50', 'equal_50', 'gda_50']
model_filepaths = [('models/' + modelname + '.pth.tar') for modelname in modelnames]
checkpoints = [load_checkpoint(mf) for mf in model_filepaths]
stats = [c['stats'] for c in checkpoints]
loss_types = ['train', 'val', 'gap']

steps_per_frame = 50

for l in loss_types:
    losses = []
    for i in range(len(modelnames)):
        if l == 'gap':
            losses.append(torch.tensor(stats[i]['val_loss']) - (torch.tensor(stats[i]['train_neg_loglike']) + torch.tensor(stats[i]['train_kl_x0']) + torch.tensor(stats[i]['train_kl_w']))/400)
        else:
            losses.append(torch.tensor(stats[i][l+'_loss']))

    # === Load data ===
    # Replace these with paths to your own files
    ssgp_loss = losses[0]          # shape: (epochs,)
    equal_loss = losses[1]            # shape: (epochs,)
    gda_loss = losses[2]      # or load if precomputed

    epochs = min(len(ssgp_loss), len(equal_loss), len(gda_loss))

    # === Set up the plot ===
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, steps_per_frame)
    ymin = min(ssgp_loss.min(), equal_loss.min(), gda_loss.min()) * 0.9
    ymax = max(ssgp_loss.max(), equal_loss.max(), gda_loss.max()) * 1.1
    ax.set_ylim(ymin, ymax)

    ax.set_title(l.capitalize() + " Loss Curves During Training")
    ax.set_xlabel("steps")
    ax.set_ylabel(l+" loss")
    ax.set_yscale('symlog')
    ax.grid(True)

    # Initialize empty line objects
    train_line, = ax.plot([], [], label='Base SSGP', color='blue')
    test_line, = ax.plot([], [], label='Equal Regularization', color='green')
    gap_line, = ax.plot([], [], label='GDA Balanceed', color='red')
    ax.legend()

    # === Animation function ===
    def update(frame):
        print(frame)
        x = np.arange(steps_per_frame*(frame + 1))
        train_line.set_data(x, ssgp_loss[:steps_per_frame*(frame + 1)])
        test_line.set_data(x, equal_loss[:steps_per_frame*(frame + 1)])
        gap_line.set_data(x, gda_loss[:steps_per_frame*(frame + 1)])
        ax.set_xlim(0, steps_per_frame*(frame + 1))
        return train_line, test_line, gap_line

    # === Animate ===
    ani = FuncAnimation(fig, update, frames=int(epochs/steps_per_frame), interval=50, blit=True)

    # === Save the animation (optional) ===
    ani.save('figures/'+l+"_loss_animation_style_1.gif", writer='ffmpeg', dpi=150)

    plt.show()
