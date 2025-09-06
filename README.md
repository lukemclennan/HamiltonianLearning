# Project Overview

Code repo of paper "Learning Generalized Hamiltonian Dynamics with Stability from Noisy Trajectory Data".

This repository provides a flexible, YAML‑driven framework for:
- **Generating** physics datasets (e.g. pendulum trajectories)
- **Training** a variety of models (SSGP, HNN, etc.) with different trainers (GDA, Equal, No‑Reg)
- **Validating** and **visualizing** results
- **Benchmarking** multiple methods over multiple systems
- **Sweeping** hyperparameters in parallel on HPC clusters

Everything is configured via YAML files.


---

## Prerequisites & Setup

1. **Python 3.8+**  
2. Clone the repo:
   ```bash
   git clone <repo_url>
   cd <repo>
   ```
3. (Optional) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install numpy torch torchdiffeq matplotlib tensorboard PyYAML torchjd
   ```

---

## Configuration

- **configs/default.yaml** selects one variant from each subgroup:
  ```yaml
  dataset:   windy_pendulum.yaml
  model:     ssgp.yaml
  optimizer: adam.yaml
  trainer:   gda.yaml

  data_root: "data/"
  mode:      train            # or "visualize"
  save_dir:  "checkpoints/{trainer}/{dataset}"
  ```
- **Subgroup YAMLs** define module, class, and `args:` for each component.
- Support some **CLI overrides** let you tweak any field without editing files:
  ```bash
  python main.py     --config configs/default.yaml     -O optimizer.args.lr=0.0005     -O trainer.args.epochs=200     -O save_dir="checkpoints/gda/custom_run"
  ```



---

## Running Training or Visualization

### Train
```bash
python main.py --config configs/default.yaml
```

Dataset are listed in ```data/``` folder, one can verify and regenerate it via ```scripts/generate_all_datasets.py```


### Parallel  Training

1. Define `experiments.csv` with columns: `dataset, lr, optimizer, model, trainer, epochs, save_dir`  
2. Run:
   ```bash
   sbatch scrips/run_experiment_array.slurm
   ```

### Visualize
```bash
python main.py   --config configs/default.yaml   -O mode=visualize   -O checkpoint="checkpoints/gda/windy_pendulum_ssgp_adam.pth.tar"
```
We also provide some standalone visualization scripts (for phase space) in ```scripts/visualize``` and ```visualize_results.py```.



## TensorBoard

TensorBoard logs are by default written to `runs/checkpoints/`. To inspect:
```bash
tensorboard --logdir runs/checkpoints
```

---
