import torch
import sys

# Path to the dataset
dataset_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/processed/lorenz/lorenz_train_small_trajectories.pt"

# Load the dataset
data = torch.load(dataset_path)

# Print the keys in the dataset
print("Dataset keys:", data.keys())

# Print the shape and type of each key
for key in data:
    if isinstance(data[key], torch.Tensor):
        print(f"{key} shape: {data[key].shape}")
    else:
        print(f"{key} type: {type(data[key])}")

# Check y dimensions in detail
if 'y' in data:
    y = data['y']
    print(f"y dimensions: {y.dim()}")
    print(f"y shape: {y.shape}")
    # Print the first element shape
    if y.dim() >= 3:
        print(f"First element shape: {y[0, 0].shape}")

# Check batch calculation
t_eval = data['t']
batch_time = 5
print(f"t_eval length: {len(t_eval)}")
print(f"t_eval[-1]: {t_eval[-1]}")
batch_step = int(((len(t_eval)-1) / t_eval[-1]).item() * batch_time)
print(f"Calculated batch_step: {batch_step}")
print(f"n_points - batch_step: {len(t_eval) - batch_step}")