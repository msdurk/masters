#!/bin/bash
#SBATCH --account=ec12
#SBATCH --partition=accel         # GPU nodes
#SBATCH --gpus=6                  # or: --gpus=a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out

module load Python/3.12.3-GCCcore-13.3.0
source ~/venvs/myenv/bin/activate

# Run your code
python lora.py
