#!/bin/bash
#SBATCH --job-name=gemma-finetune
#SBATCH --account=ec12
#SBATCH --partition=accel    # GPU nodes
#SBATCH --gpus=a100:1        # Request 1 A100
#SBATCH --cpus-per-task=8    # 8 CPU cores is sufficient for data loading
#SBATCH --mem=64G            # 64GB RAM is plenty for loading 8B model
#SBATCH --time=08:00:00      # 4 hours (likely finishes in <1 hour for 1 epoch)
#SBATCH --output=logs/%x_%j.out


# 1. Load Modules
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1



# 2. Activate Environment
# Update this path if your env is named differently
source /fp/projects01/ec12/mathisdu/llm_env/bin/activate
# Print debug info
echo "Job started on $(hostname)"
echo "GPU Info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# 1. Create a cache folder in your project area (if it doesn't exist)
mkdir -p /fp/projects01/ec12/mathisdu/hf_cache

# 2. Tell Hugging Face to use this new folder
export HF_HOME="/fp/projects01/ec12/mathisdu/hf_cache"
export HF_DATASETS_CACHE="/fp/projects01/ec12/mathisdu/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Run the training
python train_gemma.py
