#!/bin/bash
#SBATCH --job-name=gemma-download
#SBATCH --account=ec12       # <--- REPLACE with your project ID
#SBATCH --time=04:00:00             # 2 hours should be plenty
#SBATCH --mem=16G                    # Download doesn't need much RAM
#SBATCH --partition=normal          # Use CPU partition to save GPU quota
#SBATCH --cpus-per-task=2
#SBATCH --output=download_log_%j.out

set -e  # Exit immediately if a command fails

# --- 1. SETUP STORAGE ---
# Point to your big project folder
export HF_HOME=/fp/projects01/ec12/mathisdu/gemma

# --- 2. AUTHENTICATION ---
# Paste your token here so the Python script can read it
export HF_TOKEN="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"

# --- 3. LOAD PYTHON ---
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
source /fp/projects01/ec12/mathisdu/llm_env/bin/activate
# --- 4. RUN SCRIPT ---
# Assuming you already installed 'huggingface_hub' in your environment.
# If not, uncomment the next line to install it on the fly:
# pip install huggingface_hub

echo "Starting download to $HF_HOME..."

# We use the CLI tool which is more robust than python scripts
huggingface-cli download google/gemma-3-4b-it \
    --token $HF_TOKEN \
    --cache-dir $HF_HOME \
    --resume-download

echo "Download Finished!"
echo "Verifying files..."
ls -R $HF_HOME