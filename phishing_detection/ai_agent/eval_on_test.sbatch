#!/bin/bash
#SBATCH --account=ec12
#SBATCH --partition=accel         # GPU nodes
#SBATCH --gpus=1                  # or: --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

module load Python/3.12.3-GCCcore-13.3.0
source ~/venvs/myenv/bin/activate


# Run your code

python eval_on_test.py \
  --model_dir models/deberta-phishy/best \
  --test_csv data/ephish/test.csv \
  --batch_size 32

