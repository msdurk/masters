#!/bin/bash
#SBATCH --job-name=adv_test
#SBATCH --account=ec12
#SBATCH --partition=accel    
#SBATCH --gpus=a100:1        
#SBATCH --mem=32G            
#SBATCH --time=00:30:00      
#SBATCH --output=logs/adv_%j.out

module purge
module load Python/3.12.3-GCCcore-13.3.0
source ~/venvs/myenv/bin/activate

python adversarial.py