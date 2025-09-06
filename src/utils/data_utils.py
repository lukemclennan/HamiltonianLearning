import torch

def load_data(filepath):
    """Load the dataset from a file."""
    return torch.load(filepath, weights_only=False)

def get_batch(x, t_eval, batch_step):
    """Get a random batch of data for training."""
    n_points, n_samples, input_dim = x.shape
    N = n_samples

    # Ensure batch_step is valid
    batch_step = min(batch_step, n_points - 2)
    
    # Using torch to generate indices
    n_ids = torch.arange(N)
    # Randomly select starting points for each trajectory
    p_ids = torch.randint(0, n_points - batch_step, (N,))

    batch_x0 = x[p_ids, torch.arange(N)].reshape([N, 1, input_dim])
    batch_step += 1
    batch_t = t_eval[:batch_step]
    batch_x = torch.stack([x[p_ids + i, torch.arange(N)] for i in range(batch_step)], dim=0).reshape([batch_step, N, 1, input_dim])

    return batch_x0, batch_t, batch_x

def arrange(x, t_eval):
    """Arrange data for validation or testing."""
    n_points, n_samples, input_dim = x.shape

    # Using torch to generate indices
    n_ids = torch.arange(n_samples)
    p_ids = torch.zeros(n_samples, dtype=torch.int64)

    batch_x0 = x[0, torch.arange(n_samples)].reshape([n_samples, 1, input_dim])
    batch_t = t_eval
    batch_x = torch.stack([x[i, torch.arange(n_samples)] for i in range(n_points)], dim=0).reshape([n_points, n_samples, 1, input_dim])

    return batch_x0, batch_t, batch_x

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Saves model and training parameters at checkpoint."""
    try:
        # Ensure the state contains all required items
        required_keys = ['state_dict', 'model_hyperparameters']
        for key in required_keys:
            if key not in state:
                print(f"Warning: Missing required key '{key}' in checkpoint state")
        
        # Save the checkpoint
        torch.save(state, filename)
        print(f"Checkpoint saved to {filename}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")