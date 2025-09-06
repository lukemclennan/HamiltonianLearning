import torch

dataset_name = 'forced_spring'

clean_dataset_path = f'data/{dataset_name}_train_trajectories.pth'
noise_levels = [0.01, 0.05, 0.1, 0.2]

clean_dataset = torch.load(clean_dataset_path)
yts_clean = clean_dataset['yts']
t_eval = clean_dataset['t']
config = clean_dataset['config']

for sigma in noise_levels:
    noisy_yts = yts_clean + torch.randn_like(yts_clean) * sigma
    new_config = config.copy()
    new_config['sigma'] = sigma
    noisy_dataset = {
        'yts': noisy_yts,
        't': t_eval,
        'config': new_config
    }
    noisy_dataset_path = f'data/{dataset_name}_{sigma}_train_trajectories.pth'
    torch.save(noisy_dataset, noisy_dataset_path)
    print(f"Saved noisy dataset with sigma={sigma} to {noisy_dataset_path}")
