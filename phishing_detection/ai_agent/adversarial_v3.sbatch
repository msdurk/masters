#!/bin/bash
#SBATCH --job-name=adv_v3
#SBATCH --account=ec12
#SBATCH --partition=accel    
#SBATCH --gpus=a100:1        
#SBATCH --mem=32G            
#SBATCH --time=00:30:00      
#SBATCH --output=logs/adv_v3_%j.out

module purge
module load Python/3.12.3-GCCcore-13.3.0
source ~/venvs/myenv/bin/activate

echo "Starting Adversarial V3..."
python adversarial_v3.py