#!/bin/bash
#SBATCH --job-name=gemma-finetune
#SBATCH --account=ec12
#SBATCH --partition=accel    # GPU nodes
#SBATCH --gpus=a100:1        # Request 1 A100
#SBATCH --cpus-per-task=8    # 8 CPU cores is sufficient for data loading
#SBATCH --mem=64G            # 64GB RAM is plenty for loading 8B model
#SBATCH --time=08:00:00      # 4 hours (likely finishes in <1 hour for 1 epoch)
#SBATCH --output=logs/%x_%j.out

# Clean environment setup
module purge
module load Python/3.12.3-GCCcore-13.3.0

# Activate your environment
source ~/venvs/myenv/bin/activate

# Print debug info
echo "Job started on $(hostname)"
echo "GPU Info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Run the training
python train_gemma.py