import torch
import os
import argparse
from safetensors.torch import load_file, save_file

def fix_adapter(adapter_path):
    print(f">>> Inspecting adapter at: {adapter_path}")
    
    # 1. Locate the weights file
    safetensors_file = os.path.join(adapter_path, "adapter_model.safetensors")
    bin_file = os.path.join(adapter_path, "adapter_model.bin")
    
    if os.path.exists(safetensors_file):
        print(f"    Found safetensors file.")
        weights = load_file(safetensors_file)
        is_safetensors = True
    elif os.path.exists(bin_file):
        print(f"    Found bin file.")
        weights = torch.load(bin_file, map_location="cpu")
        is_safetensors = False
    else:
        print("!!! Error: No adapter_model.safetensors or adapter_model.bin found.")
        return

    # 2. Analyze and Fix Keys
    new_weights = {}
    renamed_count = 0
    
    print("\n>>> Analyzing keys...")
    
    for k, v in weights.items():
        new_k = k
        # Inject 'language_model' if missing from the path
        if "model.layers" in k and "language_model" not in k:
            new_k = k.replace("model.layers", "model.language_model.layers")
            print(f"    Renaming: {k} -> {new_k}")
            renamed_count += 1
        
        new_weights[new_k] = v

    if renamed_count == 0:
        print(">>> No keys needed renaming. The issue might be elsewhere.")
        return

    # 3. Save Fixed Adapter to a NEW folder
    output_dir = adapter_path.rstrip("/") + "_fixed"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n>>> Saving fixed adapter to: {output_dir}")
    
    if is_safetensors:
        save_file(new_weights, os.path.join(output_dir, "adapter_model.safetensors"))
    else:
        torch.save(new_weights, os.path.join(output_dir, "adapter_model.bin"))
        
    # Copy the config file too
    import shutil
    config_src = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(config_src):
        shutil.copy(config_src, output_dir)
        print("    Copied adapter_config.json")
    
    print(f"\n>>> DONE. Update your script to point to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Pass the folder containing your current (broken) adapter
    parser.add_argument("--adapter", type=str, required=True)
    args = parser.parse_args()
    
    fix_adapter(args.adapter)