#!/bin/bash

#SBATCH -J test_rgh_net
#SBATCH --array=1-12
#SBATCH -o runs/job.o%j
#SBATCH -e runs/job.e%j
#SBATCH -p queue
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH --mail-type=all
#SBATCH --mail-user=example@email.com
#SBATCH -A ALLOCATION

# load envs
module load gcc cuda python3
source ./.env/bin/activate

# read the line corresponding to $SLURM_ARRAY_TASK_ID
params=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" experiments.csv)
# parse it into Bash variables
IFS=',' read -r dataset lr optimizer trainer model epochs save_dir <<< "$params"

python main.py \
  --config configs/default.yaml \
  -O dataset=${dataset}.yaml \
  -O optimizer=${optimizer}.yaml \
  -O trainer=${trainer}.yaml \
  -O model=${model}.yaml \
  -O optimizer.args.lr=${lr} \
  -O trainer.args.epochs=${epochs} \
  -O trainer.args.save_dir="${save_dir}" \