import os
import yaml
import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from pathlib import Path
from src.utils.registers import build_from_config
import csv
from scripts.visualize.visualize import FILETYPE, CMAP, LINECOLOR
import numpy as np

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

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

def load_checkpoint(filepath):
    """
    Load the model checkpoint from the given file path.
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    return checkpoint

def load_data(filepath):
    """Load the dataset from a file."""
    return torch.load(filepath)

def plot_losses(stats, save_path):
    """
    Plot the training and validation losses from the training statistics.
    """
    train_losses = stats.get('train_loss', [])
    val_losses = stats.get('val_loss', [])
    
    train_losses = [l.flatten().detach().numpy() for l in train_losses]
    plt.figure(figsize=(10, 5))
    plt.plot([float(l) for l in train_losses], label='Training Loss')
    plt.plot([float(l) for l in val_losses], label='Validation Loss', linestyle='--')
    plt.yscale('log')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_trajectory_comparison(actual, predicted, t, save_path, index=0):
    """
    Plot a single trajectory: actual vs predicted for q over time.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(t.numpy(), actual[:, 0], label='True q')
    plt.plot(t.numpy(), predicted[:, 0], label='Predicted q', linestyle='--')
    plt.title(f'Trajectory Comparison (Sample {index})')
    plt.xlabel('Time')
    plt.ylabel('q')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def visualize(checkpoint_path, data_path, model_config_path, output_dir='visualizations', num_examples=5, checkpoint_name=None, true_hamiltonian=None):
    """
    High-level entrypoint to visualize training results and sample trajectories.
    
    Parameters:
        checkpoint_path (str): Path to the .pth.tar checkpoint file.
        data_path (str): Path to the test dataset .pth file.
        model_config_path (str): Path to the model YAML config defining module/class/args.
        output_dir (str): Directory to save figures.
        num_examples (int): Number of sample trajectories to plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if checkpoint_name is None:
        checkpoint_name = os.path.basename(checkpoint_path).replace('.pth.tar', '')

    # Load checkpoint and stats
    ckpt = load_checkpoint(checkpoint_path)
    stats = ckpt.get('stats', {})
    
    # Plot losses
    plot_losses(stats, os.path.join(output_dir, 'losses.png'))
    
    # Load and build model from config, overriding args with checkpoint hyperparams
    with open(model_config_path) as f:
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
    
    # Load test data
    test_data = load_data(data_path)
    actual_trajectories = test_data['yts']
    t = test_data['t']
    
    # Ensure correct shape: [N, T, D]
    if actual_trajectories.dim() == 4:
        actual_trajectories = actual_trajectories.squeeze(2)
    
    # Generate predictions
    # with torch.no_grad():
    x0 = actual_trajectories[:, 0, :].unsqueeze(1).requires_grad_(True)
    if hasattr(model, 'mean_w'):
        model.mean_w()
    pred = odeint(model, x0, t, method='fehlberg2', atol=1e-4, rtol=1e-4)
    pred = pred.squeeze(2).permute(1, 0, 2)
    
    # Plot sample trajectories
    N = actual_trajectories.shape[0]
    num_examples = min(num_examples, N)
    for i in range(num_examples):
        plot_trajectory_comparison(
            actual_trajectories[i], pred[i].detach(), t,
            save_path=os.path.join(output_dir, f'traj_compare_{i}.png'),
            index=i
        )
    
    # Print MSE
    mse = torch.mean((pred - actual_trajectories) ** 2).item()
    print(f'Mean squared error over test set: {mse:.6f}')

    std = torch.std(torch.mean((pred - actual_trajectories) ** 2, dim=1))
    print(f'Std of MSE across trajectories: {std:.6f}')



    # actual_trajectories += 0.1*torch.randn_like(actual_trajectories)
    # plt.plot(actual_trajectories[:5, :, 0].T, actual_trajectories[:5, :, 1].T)
    # plt.savefig('visualizations/true_phase_space.png')
    # plt.close()

    # plt.plot(pred[:5, :, 0].detach().T, pred[:5, :, 1].detach().T)
    # plt.savefig('visualizations/pred_phase_space.png')
    # plt.close()

    if hasattr(model, 'sample_hamiltonian'): 
        energy_func = model.sample_hamiltonian
    else: 
        energy_func = model.hnet

    plot_phase_map(
        qs = np.linspace(-1.0, 1.0, 500),
        ps = np.linspace(-1.0, 1.0, 500),
        energy_func = energy_func,
        true_hamiltonian = true_hamiltonian,
        title="",
        q_spec_label="(position)",
        p_spec_label="(momentum)",
        save_fig= True, 
        figure_path=os.path.join(output_dir, 'phase_maps', checkpoint_name),
        # cmap="jet"
    )

    return mse, std

def H_pendulum(q, p):
    m, l, g = 1.0, 1.0, 9.81
    return p**2 / (2 * m * l**2) - m * g * l * np.cos(q)

def H_spring(q, p):
    k, m = 1.0, 1.0
    return 0.5 * k * q**2 + 0.5 * p**2 / m

def H_henon_heiles(q, p):
    q1, p1 = 1.0, 1.0
    q2, p2 = q, p
    a = 1.0
    H = 0.5 * (q1 ** 2 + q2 ** 2) +  0.5 * (p1 ** 2 + p2 ** 2) + a * (q1 * q1 * q2 - q2 ** 3 / 3.0)
    return H

def H_duffing(q, p):
    alpha, beta, m = -1.0, 1.0, 1.0
    return 0.5 * alpha * q**2 + 0.25 * beta * q**4 + 0.5 * p**2 / m



# map a unique substring in each folder name to its test‐set filename
dataset_map = {
    'damped_pendulum_2':    'damped_pendulum_2_test_trajectories.pth',
    'damped_pendulum_05':   'damped_pendulum_05_test_trajectories.pth',
    'damped_pendulum_01':   'damped_pendulum_01_test_trajectories.pth',
    'damped_pendulum_0':    'damped_pendulum_0_test_trajectories.pth',
    'damped_pendulum_1':    'damped_pendulum_test_trajectories.pth',
    'damped_pendulum':      'damped_pendulum_test_trajectories.pth',
    'damped_pend':          'damped_pendulum_test_trajectories.pth',
    'damp_pend':            'damped_pendulum_test_trajectories.pth',   # some dirs use “damp_” instead of “damped_”
    'damped_spring':        'damped_spring_test_trajectories.pth',
    'spring_pend':          'spring_pendulum_test_trajectories.pth',     # if “spring_pend” really is the same as damped_spring
    'single_pend':          'single_pendulum_test_trajectories.pth',
    'sing_pend':            'single_pendulum_test_trajectories.pth',
    'windy_pend':           'windy_pendulum_test_trajectories.pth',
    'hh':                   'henon_heiles_test_trajectories.pth',      # for folders starting with “hh_…”
    'henon':                'henon_heiles_test_trajectories.pth',
    'henon_heiles':         'henon_heiles_test_trajectories.pth',
    'unforced_duffing':     'unforced_duffing_test_trajectories.pth',
    'forced_spring':        'forced_spring_test_trajectories.pth',
    'conservative_spring_2':  'conservative_spring_2_test_trajectories.pth',
    'conservative_spring_01':  'conservative_spring_01_test_trajectories.pth',
    'conservative_spring_05':  'conservative_spring_05_test_trajectories.pth',
    'conservative_spring_0':  'conservative_spring_0_test_trajectories.pth',
    'conservative_spring':  'conservative_spring_test_trajectories.pth',
    'chaotic_duffing_2':      'chaotic_duffing_2_test_trajectories.pth',
    'chaotic_duffing_05':      'chaotic_duffing_05_test_trajectories.pth',
    'chaotic_duffing_01':      'chaotic_duffing_01_test_trajectories.pth',
    'chaotic_duffing_0':      'chaotic_duffing_0_test_trajectories.pth',
    'chaotic_duffing':      'chaotic_duffing_test_trajectories.pth'
}

true_hamiltonian_map = {
    'damped_pendulum':      H_pendulum,
    'damped_pend':          H_pendulum,
    'damp_pend':            H_pendulum,
    'damped_spring':        H_spring,
    'damp_spring':          H_spring,
    'single_pendulum':      H_pendulum,
    'single_pend':          H_pendulum,
    'sing_pend':            H_pendulum,
    'windy_pendulum':       H_pendulum,
    'windy_pend':           H_pendulum,
    'henon_heiles':         H_henon_heiles,
    'hh':                   H_henon_heiles,
    'unforced_duffing':     H_duffing,
    'chaotic_duffing':      H_duffing,
    'forced_spring':        H_spring,
    'conservative_spring':  H_spring
}

# roots
CHECKPOINT_ROOT = Path('checkpoints/noise_informed')
DATA_ROOT       = Path('data')
# default config (override or extend with a config_map if needed)
CONFIG_PATH     = Path('configs/model/ssgp.yaml')

def run_all_models(filepath="results.csv"):
    with open(filepath, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['name', 'mse', 'std'])
        for ckpt_dir in CHECKPOINT_ROOT.iterdir():
            if not ckpt_dir.is_dir():
                continue

            # find the “best_epoch” checkpoint
            best_ckpt = next(ckpt_dir.glob('*_best_epoch.pth.tar'), None)
            if best_ckpt is None:
                best_ckpt = next(ckpt_dir.glob('*_epoch5000.pth.tar'), None)
                if best_ckpt is None:
                    # e.g. skip tacc_checkpoints.tar.gz or any non-model folder
                    continue

            # pick the right dataset by checking which key appears in the folder name
            ds_file = None
            for key, fname in dataset_map.items():
                if key in ckpt_dir.name:
                    ds_file = DATA_ROOT / fname
                    break

            if ds_file is None or not ds_file.exists():
                print(f'No dataset found for “{ckpt_dir.name}” - skipping')
                continue

            # load the true Hamiltonian function
            true_hamiltonian = None
            for key, func in true_hamiltonian_map.items():
                if key in ckpt_dir.name:
                    true_hamiltonian = func
                    break

            if true_hamiltonian is None:
                print(f'No true Hamiltonian function found for “{ckpt_dir.name}” - skipping')
                continue

            print(f'Running visualize on {ckpt_dir.name}')
            mse,std = visualize(
                str(best_ckpt),
                str(ds_file),
                str(CONFIG_PATH),
                checkpoint_name=ckpt_dir.name,
                true_hamiltonian=true_hamiltonian,
            )

            writer.writerow([ckpt_dir.name, mse, std])


if __name__ == '__main__':
    run_all_models("results_noise_informed_damped_pendulum.csv")

