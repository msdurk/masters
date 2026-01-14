#!/bin/bash
#SBATCH --job-name=llama-guard-download
#SBATCH --account=ec12       
#SBATCH --time=01:00:00             # Small model, 1 hour is plenty
#SBATCH --mem=8G                    
#SBATCH --partition=normal          
#SBATCH --cpus-per-task=2
#SBATCH --output=download_guard_%j.out

set -e  # Exit immediately if a command fails

# --- 1. SETUP STORAGE ---
# Variable for the specific target location
export TARGET_DIR="/fp/projects01/ec12/mathisdu/llama/Llama-Prompt-Guard-2-86M"

# Make sure the directory exists
mkdir -p $TARGET_DIR

# --- 2. AUTHENTICATION ---
# Paste your token here so the Python script/CLI can read it
export HF_TOKEN="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"

# --- 3. LOAD PYTHON ---
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
source /fp/projects01/ec12/mathisdu/llm_env/bin/activate

# --- 4. RUN DOWNLOAD ---
echo "Starting download to $TARGET_DIR..."

# We use --local-dir to put files exactly in the folder you asked for
huggingface-cli download meta-llama/Llama-Prompt-Guard-2-86M \
    --token $HF_TOKEN \
    --local-dir $TARGET_DIR \
    --local-dir-use-symlinks False \
    --resume-download

echo "Download Finished!"
echo "Verifying files..."
ls -R $TARGET_DIR