#!/bin/bash
#SBATCH --job-name=gemma_adv_4b
#SBATCH --account=ec12
#SBATCH --partition=accel
#SBATCH --gpus=a100:1        # A100 is fast, but an A40/A10 would also work fine for 4B
#SBATCH --mem=32G            # Reduced: 64G -> 32G (4B model fits easily)
#SBATCH --time=01:00:00      # Reduced: 2h -> 1h (Inference is much faster on 4B)
#SBATCH --output=logs/gemma_adv_4b_%j.out

# 1. Load Modules
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# 2. Activate Environment
source /fp/projects01/ec12/mathisdu/llm_env/bin/activate

# Print debug info
echo "Job started on $(hostname)"
echo "GPU Info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# 3. Run the Attack Script with 4B Arguments
# Note: Ensure the adapter path points to where you actually saved the 4B model
python run_gemma_attacks.py \
    --size 4b \
    --adapter /fp/projects01/ec12/mathisdu/gemma/gemma-3-4b-output/gemma-3-4b-finetuned \
    --output results_4b.csv