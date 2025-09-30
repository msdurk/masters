#!/bin/bash
#SBATCH --account=ec12
#SBATCH --partition=accel         # GPU nodes
#SBATCH --gpus=1                  # or: --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

module load Python/3.12.3-GCCcore-13.3.0
source ~/venvs/py312/bin/activate

# (optional but handy for debugging)
which python
python -V
python -c "import numpy; print('NumPy', numpy.__version__)"

# Run your code
python pro_train_embed.py \
  --outdir models/embed_miniLM \
  --embed-backend sentence-transformers \
  --model all-MiniLM-L6-v2

