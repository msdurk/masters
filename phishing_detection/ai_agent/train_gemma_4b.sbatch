#!/bin/bash
#SBATCH --job-name=gemma-4b-ft   # Updated job name
#SBATCH --account=ec12
#SBATCH --partition=accel
#SBATCH --gpus=a100:1            # A100 is overkill for 4B, but guarantees fast training
#SBATCH --cpus-per-task=4        # Reduced: 4 CPU cores is plenty for 4B data loading
#SBATCH --mem=32G                # Reduced: 32GB RAM is more than enough for 4B (saves resources)
#SBATCH --time=04:00:00          # Reduced: 4B trains much faster
#SBATCH --output=logs/%x_%j.out

# 1. Load Modules
module purge
module load Python/3.11.3-GCCcore-12.3.0 
module load CUDA/12.1.1 

# 2. Activate Environment
# Update this path if your env is named differently
source /fp/projects01/ec12/mathisdu/llm_env/bin/activate [cite: 2]

# Print debug info
echo "Job started on $(hostname)" [cite: 2]
echo "GPU Info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)" [cite: 2]

# 3. Cache Setup (Crucial for offline/cluster nodes)
mkdir -p /fp/projects01/ec12/mathisdu/hf_cache [cite: 2]
export HF_HOME="/fp/projects01/ec12/mathisdu/hf_cache" [cite: 2]
export HF_DATASETS_CACHE="/fp/projects01/ec12/mathisdu/hf_cache" [cite: 2]

# Fix potential OOM fragmentation issues (optional but good practice)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True [cite: 2]

# 4. Run the 4B Training Script
# Make sure this matches the filename of the Python code I provided earlier
python train_gemma_4b.py