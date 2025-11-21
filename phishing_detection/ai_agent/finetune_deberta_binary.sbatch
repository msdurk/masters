#!/bin/bash
#SBATCH --account=ec12
#SBATCH --partition=accel         # GPU nodes
#SBATCH --gpus=1                  # or: --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

module load Python/3.12.3-GCCcore-13.3.0
source ~/venvs/myenv/bin/activate

# Run your code
python finetune_deberta_binary.py \
  --train_csv data/ephish/train.csv \
  --val_csv data/ephish/val.csv \
  --test_csv data/ephish/test.csv \
  --out_dir models/deberta-phishy \
  --epochs 3 --lr 2e-5 --batch_train 8 --batch_eval 16 --use_fp16
