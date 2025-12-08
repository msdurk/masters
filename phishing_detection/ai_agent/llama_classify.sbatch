#!/bin/bash
#SBATCH --account=ec12
#SBATCH --job-name=llama_classify
#SBATCH --partition=accel
#SBATCH --gpus=1                  # Changed from 2 to 1 (Efficient for 8B)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                 # Bumped slightly to handle data + model safely
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out

# Strict mode (stops execution if a command fails)
set -e 

module purge
module load Python/3.12.3-GCCcore-13.3.0
source ~/venvs/myenv/bin/activate

echo "Starting Inference on $(hostname)"
python llama_classify.py