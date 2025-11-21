from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    local_dir="/fp/projects01/ec12/mathisdu/llama/Llama-3.3-70B-Instruct",
    resume_download=True,
    ignore_patterns=["*.md"]  # optional to skip docs
)
