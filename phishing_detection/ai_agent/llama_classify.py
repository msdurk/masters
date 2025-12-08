import os
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm  # For progress bar

# --- Config ---
# Using only 1 GPU is faster for 8B unless you use DDP (DistributedDataParallel)
LOCAL_DIR = Path("/fp/projects01/ec12/mathisdu/llama/Llama-3.1-8B-Instruct")
CSV_PATH = "/fp/homes01/u01/ec-mathiassd/phishing_detection/ai_agent/data/ephish/train.csv"
BATCH_SIZE = 16  # Adjust based on GPU memory (try 8, 16, or 32)

# 1. Load Model & Tokenizer
tok = AutoTokenizer.from_pretrained(LOCAL_DIR, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_DIR, 
    device_map="cuda",  # Explicitly put on the GPU
    torch_dtype=torch.bfloat16 # Use bfloat16 for speed/memory efficiency on Ampere+ GPUs
)

# Fix padding for generation
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left" # CRITICAL for batched generation

# 2. Prompting Logic
SYSTEM_PROMPT = """You are an email classifier.
Task: Decide the correct class label for the email.
Labels:
- 0 = not spam / legitimate email
- 1 = spam / unwanted or scam email
Output rules: Respond with ONLY a single character: 0 or 1."""

def create_chat_prompt(email_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(email_text)}
    ]
    # apply_chat_template handles the <|eot_id|> logic automatically
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 3. Batch Processing Function
def process_batches(df):
    prompts = [create_chat_prompt(text) for text in df["text"]]
    preds = []
    
    # Loop through data in chunks
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Classifying"):
        batch_prompts = prompts[i : i + BATCH_SIZE]
        
        # Tokenize batch
        inputs = tok(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048 # Safety limit
        ).to(model.device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=1, # We only need 1 character
                do_sample=False,  # Deterministic
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id
            )
        
        # Decode only the new tokens
        # Slice [:, inputs.shape[1]:] to get only generated part
        generated_ids = out[:, inputs["input_ids"].shape[1]:]
        decoded_batch = tok.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Parse results
        for output in decoded_batch:
            clean_out = output.strip()
            if "1" in clean_out:
                preds.append(1)
            elif "0" in clean_out:
                preds.append(0)
            else:
                # Fallback or log error. defaulting to 0 for now to keep shape
                preds.append(0) 
                
    return preds

# 4. Main Execution
def run_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Optional: Run on a subset first to test!
    # df = df.head(100) 
    
    print(f"Processing {len(df)} rows with Batch Size {BATCH_SIZE}...")
    
    predicted_labels = process_batches(df)
    df["pred"] = predicted_labels

    # Metrics
    y_true = df["label"].astype(int).values
    y_pred = df["pred"].astype(int).values
    
    print("\n--- Results ---")
    print(classification_report(y_true, y_pred, digits=4))

    # Save
    out_path = os.path.splitext(csv_path)[0] + "_with_preds.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    run_csv(CSV_PATH)