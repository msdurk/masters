import os
from huggingface_hub import snapshot_download

# --- CONFIGURATION ---
# Replace this with the specific model you want (e.g., gemma-3-27b-it)
MODEL_ID = "google/gemma-3-27b-it"

# Get token from environment variable (safest) or paste it here
HF_TOKEN = os.getenv("HF_TOKEN") 
# If you prefer pasting it directly (delete the line above if so):
# HF_TOKEN = "hf_YourHuggingFaceTokenHere"

# Ensure we are saving to the PROJECT area, not the small Home area
# Educloud Fox path format: /projects/ec-username
project_path = os.environ.get("HF_HOME")

if not project_path:
    raise ValueError("Error: HF_HOME environment variable is not set. The script doesn't know where to save the data.")

print(f"--- Starting Download for {MODEL_ID} ---")
print(f"Saving to: {project_path}")

# This function downloads the model files effectively
local_dir = snapshot_download(
    repo_id=MODEL_ID,
    token=HF_TOKEN,
    cache_dir=project_path,
    resume_download=True  # useful if the connection drops
)

print(f"--- Success! ---")
print(f"Model downloaded to: {local_dir}")