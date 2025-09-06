import csv
import os
import numpy as np
import pandas as pd
import torch
from torchdiffeq import odeint
from datetime import datetime

def evaluate_model(model, test_data, metrics=None):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained SSGP model
        test_data: Dictionary with 't' and 'y' keys containing test trajectories
        metrics: List of metric names to compute (default: ['mse', 'energy_cons', 'volume_cons'])
        
    Returns:
        Dictionary of computed metrics
    """
    if metrics is None:
        metrics = ['mse', 'energy_cons', 'volume_cons']
    
    results = {}
    
    # Ensure model is in evaluation mode and uses mean weights
    model.eval()
    model.mean_w()
    
    with torch.no_grad():
        # Get time points
        t = test_data['t']
        
        # Test data shape is [n_points, n_samples, dim]
        # We need to transpose to [n_samples, n_points, dim] for comparison
        full_trajectories = test_data['y'].permute(1, 0, 2)  # [n_samples, n_points, dim]
        
        # Sample a subset of trajectories if there are too many
        max_samples = 50  # Maximum number of samples to evaluate
        n_samples = min(full_trajectories.shape[0], max_samples)
        full_trajectories = full_trajectories[:n_samples]
        
        # Get initial conditions for each trajectory
        x0 = full_trajectories[:, 0, :].unsqueeze(1)  # [n_samples, 1, dim]
        
        # Predict trajectories
        pred_trajectories = odeint(model, x0, t, method='fehlberg2', atol=1e-4, rtol=1e-4)
        
        # Handle the permutation based on pred_trajectories dimensions
        if pred_trajectories.dim() == 4:  # If shape is [n_points, n_samples, 1, dim]
            pred_trajectories = pred_trajectories.squeeze(2)  # Remove the singleton dimension to get [n_points, n_samples, dim]
            pred_trajectories = pred_trajectories.permute(1, 0, 2)  # [n_samples, n_points, dim]
        else:  # Normal case with shape [n_points, n_samples, dim]
            pred_trajectories = pred_trajectories.permute(1, 0, 2)  # [n_samples, n_points, dim]
        
        # Calculate MSE
        if 'mse' in metrics:
            mse = torch.mean((pred_trajectories - full_trajectories)**2).item()
            results['mse'] = mse
        
        # Calculate energy conservation
        if 'energy_cons' in metrics:
            # Compute initial Hamiltonians
            initial_energy = []
            for i in range(n_samples):
                initial_energy.append(model.sample_hamiltonian(x0[i:i+1]).item())
            
            # Compute Hamiltonians at all time steps
            energy_error = 0
            for t_idx in range(1, pred_trajectories.shape[1]):
                current_energy = []
                for i in range(n_samples):
                    x_t = pred_trajectories[i, t_idx, :].unsqueeze(0).unsqueeze(1)
                    current_energy.append(model.sample_hamiltonian(x_t).item())
                
                # Compute error
                energy_error += np.mean([(e2 - e1)**2 for e1, e2 in zip(initial_energy, current_energy)])
            
            energy_error /= (pred_trajectories.shape[1] - 1)
            results['energy_cons'] = energy_error
        
        # Calculate volume conservation
        if 'volume_cons' in metrics:
            # Use a surrogate measure for volume conservation
            from src.utils.regularization import cons_vol_loss
            # Reshape pred_trajectories to match expected format for cons_vol_loss
            pred_x = pred_trajectories.permute(1, 0, 2)  # [n_points, n_samples, dim]
            
            # Ensure pred_x has the right shape with a singleton dimension if needed
            if pred_x.dim() == 3:
                pred_x = pred_x.unsqueeze(2)  # [n_points, n_samples, 1, dim]
                
            vol_error = cons_vol_loss(pred_x).item()
            results['volume_cons'] = vol_error
    
    return results

def benchmark_models(model_configs, system_types, dataset_size='small', metrics=None, results_dir='results'):
    """
    Benchmark models on all specified systems and save results to CSV and LaTeX.
    
    Args:
        model_configs: List of dictionaries with keys 'method', 'hyperparams'
        system_types: List of system types to evaluate
        dataset_size: Size of the dataset to use
        metrics: List of metrics to compute
        results_dir: Directory to save results
    
    Returns:
        DataFrame with benchmark results
    """
    if metrics is None:
        metrics = ['mse', 'energy_cons', 'volume_cons']
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a results list for the DataFrame
    results_list = []
    
    # Create a date string for the filename
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Loop through all configurations
    for model_config in model_configs:
        method = model_config['method']
        hyperparams = model_config.get('hyperparams', {})
        
        for system in system_types:
            # Define path to trained model
            model_path = os.path.join('models', f"{system.replace('_', '')}_{method}_{dataset_size}.pth.tar")
            
            # Check if model exists
            if os.path.exists(model_path):
                print(f"Evaluating {method} on {system} ({dataset_size})...")
                
                # Load the model
                try:
                    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                    model_hyperparams = checkpoint['model_hyperparameters']
                    
                    from src.models import SSGP
                    model = SSGP(**model_hyperparams)
                    try:
                        # First try loading with strict mode
                        model.load_state_dict(checkpoint['state_dict'])
                    except RuntimeError as e:
                        print(f"Warning: {e}")
                        print(f"Trying to load with strict=False")
                        # If that fails, try with strict=False to handle missing/extra keys
                        model.load_state_dict(checkpoint['state_dict'], strict=False)
                except Exception as e:
                    print(f"Error loading model {model_path}: {e}")
                    continue
                
                # Load test data
                from src.utils import load_data, get_dataset_path
                test_path = get_dataset_path(system, 'test', 'test')
                test_data = load_data(test_path)
                
                # Evaluate the model
                evaluation = evaluate_model(model, test_data, metrics)
                
                # Save the results
                result_row = {
                    'system': system,
                    'method': method,
                    'dataset_size': dataset_size,
                    'train_loss': checkpoint.get('best_train_loss', float('nan')),
                    'val_loss': checkpoint.get('min_val_loss', float('nan')),
                    'best_step': checkpoint.get('best_step', 0),
                }
                
                # Add evaluation metrics
                for metric, value in evaluation.items():
                    result_row[metric] = value
                
                results_list.append(result_row)
            else:
                print(f"Model not found: {model_path}")
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results_list)
    
    # Save to CSV
    csv_path = os.path.join(results_dir, f'benchmark_results_{date_str}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    
    # Generate LaTeX table
    latex_table = generate_latex_table(results_df, metrics)
    latex_path = os.path.join(results_dir, f'benchmark_results_{date_str}.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved LaTeX table to {latex_path}")
    
    return results_df

def generate_latex_table(results_df, metrics):
    """
    Generate a LaTeX table from the benchmark results.
    
    Args:
        results_df: DataFrame with benchmark results
        metrics: List of metrics to include in the table
        
    Returns:
        LaTeX table as a string
    """
    # Group by system
    systems = sorted(results_df['system'].unique())
    methods = sorted(results_df['method'].unique())
    
    header = "\\begin{table}[htbp]\n\\centering\n\\caption{Benchmark Results}\n\\begin{tabular}{l" + "".join(["c" for _ in methods]) + "}\n\\toprule\n"
    header += "System & " + " & ".join(methods) + " \\\\\n\\midrule\n"
    
    content = ""
    
    # For each metric, create a section
    for metric in metrics:
        # Fix for the f-string syntax error
        metric_title = metric.replace('_', ' ').title()
        content += f"\\multicolumn{{{len(methods) + 1}}}{{c}}{{\\textbf{{{metric_title}}}}} \\\\\n"
        
        for system in systems:
            system_display = system.replace('_', ' ').title()
            row = f"{system_display}"
            
            for method in methods:
                mask = (results_df['system'] == system) & (results_df['method'] == method)
                if mask.any():
                    value = results_df.loc[mask, metric].values[0]
                    # Format based on the metric
                    if metric == 'mse':
                        value_str = f"{value:.2e}"
                    else:
                        value_str = f"{value:.4f}"
                else:
                    value_str = "N/A"
                
                row += f" & {value_str}"
            
            row += " \\\\\n"
            content += row
        
        if metric != metrics[-1]:
            content += "\\midrule\n"
    
    footer = "\\bottomrule\n\\end{tabular}\n\\label{tab:benchmark-results}\n\\end{table}"
    
    return header + content + footer