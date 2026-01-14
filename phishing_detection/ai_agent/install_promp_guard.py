import os
from huggingface_hub import snapshot_download

# --- CONFIGURATION ---
# Target Model
MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"

# Token from environment (safest) or fallback to strict string
HF_TOKEN = os.getenv("HF_TOKEN")

# Target location explicitly requested
# We use this specific path to ensure files are placed exactly where you want them
project_path = "/fp/projects01/ec12/mathisdu/llama/Llama-Prompt-Guard-2-86M"

# Create the directory if it doesn't exist
os.makedirs(project_path, exist_ok=True)

print(f"--- Starting Download for {MODEL_ID} ---")
print(f"Saving to: {project_path}")

# Using local_dir=project_path ensures the actual model files 
# (config.json, safetensors, etc.) are placed directly in this folder
# rather than in a complex cache structure.
local_dir = snapshot_download(
    repo_id=MODEL_ID,
    token=HF_TOKEN,
    local_dir=project_path,  # Changed from cache_dir to local_dir for visibility
    local_dir_use_symlinks=False, # file copies instead of symlinks
    resume_download=True
)

print(f"--- Success! ---")
print(f"Model downloaded to: {local_dir}")