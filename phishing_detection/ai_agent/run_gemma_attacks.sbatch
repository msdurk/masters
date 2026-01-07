#!/bin/bash
#SBATCH --job-name=gemma_adv
#SBATCH --account=ec12
#SBATCH --partition=accel    
#SBATCH --gpus=a100:1        
#SBATCH --mem=64G            
#SBATCH --time=02:00:00      
#SBATCH --output=logs/gemma_adv_%j.out

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

# 3. Run the Gemma-specific attack script
python run_gemma_attacks.py