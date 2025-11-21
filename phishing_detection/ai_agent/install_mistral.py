#!/usr/bin/env python3
"""
Download sharded weights for mistralai/Mistral-Small-3.2-24B-Instruct-2506
with high-performance transfers enabled. No CLI arguments required.

Setup first:
  pip install -U huggingface_hub hf-transfer

(Optional) set your token for higher rate limits:
  export HF_TOKEN=hf_xxx
"""

import os
from pathlib import Path
from typing import List, Tuple

REPO_ID = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
LOCAL_DIR = "/fp/projects01/ec12/mathisdu/mistral"  # change if you want a different path

# Sharded layout: index + shards + configs/tokenizer
ALLOW_PATTERNS = [
    "model.safetensors.index.json",
    "model-*.safetensors",
    "config.json",
    "params.json",
    "SYSTEM_PROMPT.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"

def main():
    # Enable high-performance transfer (requires `hf-transfer` installed)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Optional: login via env token
    hf_token = os.environ.get("HF_TOKEN")

    try:
        from huggingface_hub import snapshot_download, login
    except ImportError:
        raise SystemExit(
            "Missing deps. Install with:\n  pip install -U huggingface_hub hf-transfer"
        )

    if hf_token:
        try:
            login(token=hf_token)
        except Exception as e:
            print(f"Warning: login failed ({e}). Proceeding without explicit login.")

    Path(LOCAL_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Repo: {REPO_ID}")
    print("Mode: sharded")
    print(f"High-performance: enabled (HF_HUB_ENABLE_HF_TRANSFER=1)")
    print(f"Destination: {LOCAL_DIR}")

    local_path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,   # place actual files in LOCAL_DIR
        allow_patterns=ALLOW_PATTERNS,  # only sharded files + configs
    )

    # Summarize what we got
    total_bytes = 0
    files: List[Tuple[str, int]] = []
    for p in Path(local_path).rglob("*"):
        if p.is_file():
            sz = p.stat().st_size
            total_bytes += sz
            files.append((p.relative_to(local_path).as_posix(), sz))

    files.sort()
    print("\nDownloaded files:")
    for rel, sz in files:
        print(f"  {rel:<48} {human_bytes(sz):>10}")

    print(f"\nTotal size: {human_bytes(total_bytes)}")
    print("\nDone. You can now load the repo with vLLM or transformers from this folder.")

if __name__ == "__main__":
    main()
