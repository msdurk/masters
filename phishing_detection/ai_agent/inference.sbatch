#!/bin/bash
#SBATCH --job-name=phish_eval
#SBATCH --account=ec12
#SBATCH --partition=accel    
#SBATCH --gpus=a100:1        
#SBATCH --mem=32G            
#SBATCH --time=01:00:00      
#SBATCH --output=logs/eval_%j.out

module purge
module load Python/3.12.3-GCCcore-13.3.0
source ~/venvs/myenv/bin/activate

echo "Starting Inference..."
python inference.py