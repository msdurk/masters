import os, sys, json, re
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import f1_score 


# --- Config ---
LOCAL_DIR = Path("/fp/projects01/ec12/mathisdu/llama/Llama-3.1-8B-Instruct")
CSV_PATH = "/fp/homes01/u01/ec-mathiassd/phishing_detection/ai_agent/data/ephish/train.csv"

tok = AutoTokenizer.from_pretrained(LOCAL_DIR, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(LOCAL_DIR, device_map="auto")

if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model.config.pad_token_id = tok.pad_token_id
model.generation_config.pad_token_id = tok.pad_token_id
tok.padding_side = "right"  # good default for causal LMs



# --------- Prompting helpers ----------
SYSTEM_PROMPT = """You are an email classifier.

Task:
- Read the email content provided by the user.
- Decide the correct class label for the email.

Labels:
- 0 = not spam / legitimate email
- 1 = spam / unwanted or scam email

Output rules (very important):
- Respond with ONLY a single character: 0 or 1.
- Do NOT include any other text, spaces, punctuation, JSON, or explanation.
"""


def build_prompt(email_text: str) -> str:
    """
    Build a Llama-3.1 chat-style prompt for classification.
    """
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        # The user message is just the email text itself
        f"{email_text}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )



JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def classify_email(email_text: str, max_new_tokens: int = 8) -> int:
    prompt = build_prompt(email_text)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )

    gen = tok.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # Expect "0" or "1"
    if gen.startswith("1"):
        return 1
    else:
        return 0

# --------- Batch over CSV ----------
def run_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    assert {"text", "label"}.issubset(df.columns), "CSV must have 'text' and 'label' columns."

    preds = []
    for i, row in df.iterrows():
        label = classify_email(str(row["text"]))
        preds.append(label)

        # Optional: show progress
        print(f"[{i}] pred={label}")

    df["pred"] = preds

    # Metrics
    y_true = df["label"].astype(int).values
    y_pred = df["pred"].astype(int).values
    acc = (y_true == y_pred).mean()
    f1 = f1_score(y_true, y_pred)

    # Confusion matrix
    import numpy as np
    cm = np.zeros((2,2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    print("\nAccuracy:", round(float(acc), 4))
    print("F1 score:", round(float(f1), 4))
    print("Confusion matrix [[TN FP],[FN TP]]:")
    TN, FP = cm[0,0], cm[0,1]
    FN, TP = cm[1,0], cm[1,1]
    print([[int(TN), int(FP)], [int(FN), int(TP)]])

    # Save results
    out_path = os.path.splitext(csv_path)[0] + "_with_preds.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote predictions to: {out_path}")

if __name__ == "__main__":
    run_csv(CSV_PATH)