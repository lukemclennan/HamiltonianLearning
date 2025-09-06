import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import yaml
import torch
import importlib
from glob import glob
from utils.plots import plot_trajectories
from datasets.base import TrajectoryDataset

def instantiate_from_config(config):
    module_path = config['module']
    class_name = config['class']
    params = config.get('params', {})
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**params)

def validate_or_generate_dataset(loader_config, generator_config, split='train', overwrite=False):
    data_path = loader_config['args']['data_path']
    expected_config = loader_config['args'].get('expected_config', {})
    if not os.path.exists(data_path) or overwrite:
        parent_dir = os.path.dirname(data_path)
        os.makedirs(parent_dir, exist_ok=True)
        print(f"[{split.upper()}] '{data_path}' not found. Generating...")
        generator = instantiate_from_config(generator_config)
        print(generator_config)
        data = generator.generate()
        print(data['config'])
        torch.save(data, data_path)
        return

    try:
        _ = TrajectoryDataset(data_path, expected_config=expected_config)
        print(f"[{split.upper()}] '{data_path}' is valid.")
    except Exception as e:
        print(f"[{split.upper()}] Validation failed: {e}. Regenerating...")
        generator = instantiate_from_config(generator_config)
        data = generator.generate()
        torch.save(data, data_path)

def plot_if_possible(loader_config, label):
    path = loader_config['args']['data_path']
    if os.path.exists(path):
        data = torch.load(path)
        filename = os.path.splitext(os.path.basename(path))[0] + f"_{label}_plot.png"
        print(f"[PLOT] Generating plot: {filename}")
        plot_trajectories(data, n=10, filename=filename)
    else:
        print(f"[PLOT] Cannot find file: {path}")

def main(wrapper_config_dir="configs/dataset_generator", overwrite=False):   
    configs = sorted(glob(os.path.join(wrapper_config_dir, "*.yaml")))
    for path in configs:
        print(f"Processing config: {os.path.basename(path)}")
        with open(path, "r") as f:
            wrapper_cfg = yaml.safe_load(f)
        #TODO: generate dataset parent folder
        validate_or_generate_dataset(wrapper_cfg["train_loader"], wrapper_cfg["train_generator"], split="train", overwrite=overwrite)
        validate_or_generate_dataset(wrapper_cfg["test_loader"], wrapper_cfg["test_generator"], split="test", overwrite=overwrite)
        plot_if_possible(wrapper_cfg["test_loader"], label="test")

if __name__ == "__main__":
    main(overwrite=False)
