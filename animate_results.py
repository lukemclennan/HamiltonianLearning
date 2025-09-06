import yaml
import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from pathlib import Path
from src.utils.registers import build_from_config
import csv
from matplotlib.animation import FuncAnimation
from scripts.visualize.visualize import FILETYPE, CMAP, LINECOLOR
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

def animate_phase_map(
        qs=np.linspace(-1.0, 1.0, 300),
        ps=np.linspace(-1.0, 1.0, 300),
        energy_func=None,
        hamiltonian_rhs=None,  # function (t, y) -> [dq/dt, dp/dt]
        levels=None,
        contour_line_numbers=60,
        title="Phase Space Energy",
        q_spec_label="(position)",
        p_spec_label="(momentum)",
        t_vals=np.linspace(0, 10, 100),   # times for animation
        init_points=[(0.5, 0.0), (-0.5, 0.5)],  # list of (q0, p0)
        save_anim=False,
        anim_path='visualizations/phase_map_animation.mp4',
        cmap="viridis",
        linecolor="black",
        fps=15
    ):
    """
    Animate phase space energy contours over time with fixed color scale.
    energy_func must be callable: energy_func(Q, P, t).
    """
    Q, P = np.meshgrid(qs, ps)

    # Precompute global min/max across all times for fixed color scale
    H_min, H_max = np.inf, -np.inf
    for t in t_vals:
        H = energy_func(Q, P, t)
        H_min = min(H_min, np.min(H))
        H_max = max(H_max, np.max(H))

    if levels is None:
        levels = np.linspace(H_min, H_max, contour_line_numbers)

    #  # Integrate trajectories
    # trajectories = []
    # for q0, p0 in init_points:
    #     sol = odeint(
    #         hamiltonian_rhs, 
    #         t_span=(t_vals[0], t_vals[-1]),
    #         y0=[q0, p0],
    #         t_eval=t_vals,
    #         method='fehlberg2',
    #         rtol=1e-4, atol=1e-4
    #     )
    #     trajectories.append(sol.y)  # shape (2, len(t_vals))

    fig, ax = plt.subplots(figsize=(8, 6))
    H0 = energy_func(Q, P, t_vals[0])
    contourf = ax.contourf(Q, P, H0, levels=levels, cmap=cmap)
    cbar = fig.colorbar(contourf, ax=ax, label='Hamiltonian Energy $\\mathcal{H}$')
    ax.set_title(title + f"\nTime t = {t_vals[0]:.2f}")
    ax.set_xlabel('$q$ ' + q_spec_label)
    ax.set_ylabel('$p$ ' + p_spec_label)
    ax.grid(True)


    # # Line + marker for each trajectory
    # lines = []
    # points = []
    # for _ in init_points:
    #     (line,) = ax.plot([], [], lw=1.5)
    #     (point,) = ax.plot([], [], 'o', markersize=6)
    #     lines.append(line)
    #     points.append(point)

    def update(frame_idx):
        t = t_vals[frame_idx]
        H = energy_func(Q, P, t)

        # Remove previous contours
        for coll in ax.collections:
            coll.remove()

        # # Update trajectories
        # for traj, line, point in zip(trajectories, lines, points):
        #     q_traj, p_traj = traj
        #     line.set_data(q_traj[:frame_idx+1], p_traj[:frame_idx+1])
        #     point.set_data(q_traj[frame_idx], p_traj[frame_idx])

        cf = ax.contourf(Q, P, H, levels=levels, cmap=cmap)
        ax.set_title(title + f"\nTime t = {t:.2f}")
        return ax.collections #+ lines + points

    anim = FuncAnimation(fig, update, frames=len(t_vals), blit=False, interval=1000/fps)

    if save_anim:
        anim.save(anim_path, fps=fps, dpi=200)
        plt.close(fig)
    else:
        plt.show()

    return anim


def plot_phase_map(
        qs = np.linspace(-1.0, 1.0, 500),
        ps = np.linspace(-1.0, 1.0, 500),
        energy_func = None, 
        levels = None,
        countour_line_numbers = 60, 
        title="Phase Space Energy Level Sets of Two-Body Problem\n(Fixed $q_y=0$, $p_x=0$)",
        q_spec_label="(relative position)",
        p_spec_label="(momentum)",
        save_fig= False, 
        figure_path='visualizations/phase_map',
        cmap = CMAP,
        linecolor=LINECOLOR,
        true_hamiltonian=None
        ):
    plt.figure(figsize=(8, 6))
    Q, P = np.meshgrid(qs, ps)
    print(Q.shape, P.shape)
    X = torch.cat((torch.tensor(Q, dtype=torch.float).reshape(-1,1), torch.tensor(P, dtype=torch.float).reshape(-1,1)), dim=1)
    H = energy_func(X).reshape(Q.shape).detach().numpy()
    H = H - np.min(H)
    if true_hamiltonian is not None:
        H_true = true_hamiltonian(Q,P)
        H_true = H_true - np.min(H_true)
        H = np.abs(H - H_true) #/ (np.abs(H_true)+1e-6))
    if levels is None:
        levels = np.linspace(np.min(H), np.max(H), 30)
    _ = plt.contourf(Q, P, H, levels=levels, cmap=cmap)
    plt.colorbar(label='Hamiltonian Energy $\\mathcal{H}$')
    plt.title(title)
    plt.xlabel('$q$ ' + q_spec_label)
    plt.ylabel('$p$ ' + p_spec_label)
    plt.grid(True)
    plt.tight_layout()
    if save_fig:
        plt.savefig(figure_path + '.' + FILETYPE, dpi=300, format=FILETYPE)
        plt.close()
    else:
        plt.show()

def load_checkpoint(filepath):
    """
    Load the model checkpoint from the given file path.
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    return checkpoint

def load_data(filepath):
    """Load the dataset from a file."""
    return torch.load(filepath)

# roots
# CHECKPOINT_PATH = Path('checkpoints/all_port_d/windy_pendulum_gdad_adam_pssgp/gdad_best_epoch.pth.tar')
CHECKPOINT_PATH = Path('checkpoints/torchjd/damped_pendulum_1_torchjd_adam_pssgp/torchjd_best_epoch.pth.tar')
CONFIG_PATH     = Path('configs/model/pssgp.yaml')

# Load checkpoint and stats
ckpt = load_checkpoint(CHECKPOINT_PATH)
stats = ckpt.get('stats', {})

# Load and build model from config, overriding args with checkpoint hyperparams
with open(CONFIG_PATH) as f:
    model_cfg = yaml.safe_load(f)
# Merge hyperparameters from checkpoint into model args
hyperparams = ckpt.get('model_hyperparameters', {})
model_cfg_args = model_cfg.get('args', {})
for key, val in hyperparams.items():
    if key in model_cfg_args:
        model_cfg_args[key] = val
model_cfg['args'] = model_cfg_args

model = build_from_config(model_cfg)
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval()
model.mean_w()  # Use mean weights for evaluation

def energy_function(q, p, t):
    x = torch.tensor(np.stack((q.flatten(), p.flatten()), axis=1), dtype=torch.float)
    H = model.sample_hamiltonian(x).detach().numpy().reshape(q.shape)
    return torch.exp(-t * model.eta**2).detach().numpy() * (H + q * model.F(torch.tensor([t], dtype=torch.float)).detach().numpy())

anim = animate_phase_map(
    energy_func=energy_function,
    hamiltonian_rhs=None,
    t_vals=np.linspace(0, 10, 100),  # animate from t=0 to 20
    init_points=[(0.5, 0.0), (-0.5, 0.5)],
    title="Phase Space Energy for Damped Pendulum",
    save_anim=True,
    anim_path='learned_damped_pendulum.mp4',
    cmap='coolwarm'
)