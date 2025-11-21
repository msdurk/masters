import os, sys, json, re
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import f1_score 


# --- Config ---
LOCAL_DIR = Path("/fp/projects01/ec12/mathisdu/llama/Llama-3.3-70B-Instruct")
CSV_PATH = "/fp/homes01/u01/ec-mathiassd/phishing_detection/ai_agent/data/ephish/train.csv"


max_memory = {i: "22GiB" for i in range(torch.cuda.device_count())}
max_memory["cpu"] = "220GiB"  # you have 256G, keep some headroom

tok = AutoTokenizer.from_pretrained(LOCAL_DIR, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_DIR,
    max_memory=max_memory,
    device_map="auto",
    low_cpu_mem_usage=True,
)

if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model.config.pad_token_id = tok.pad_token_id
model.generation_config.pad_token_id = tok.pad_token_id
tok.padding_side = "right"  # good default for causal LMs



# --------- Prompting helpers ----------
SYSTEM_PROMPT = (
    "You are a cybersecurity classifier. "
    "Decide if an email is phishing (1) or not phishing (0). "
    "Output ONLY a compact JSON object with this schema:\n"
    '{"label": 0 or 1, "confidence": float between 0 and 1, "reason": "one short sentence"}\n'
    "Do not add anything else before or after the JSON."
)

def build_prompt(email_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            "Classify the following email.\n\n"
            "Return ONLY the JSON object.\n\n"
            f"EMAIL:\n{email_text}"
        },
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def classify_email(email_text: str, max_new_tokens: int = 96):
    prompt = build_prompt(email_text)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=64,   # or even 32; you only need a short JSON answer
            do_sample=False,
            top_p=1.0,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )

    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # Extract first JSON block
    m = JSON_RE.search(gen)
    if not m:
        # Fallback: try to coerce to a minimal dict
        return {"label": 0, "confidence": 0.5, "reason": "Parser fallback (no JSON found)"}, gen
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        # Attempt to fix common trailing commas / quotes
        cleaned = m.group(0).replace("\n", " ").replace("\t", " ")
        obj = json.loads(cleaned)
    return obj, gen

# --------- Batch over CSV ----------
def run_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    assert {"text", "label"}.issubset(df.columns), "CSV must have 'text' and 'label' columns."

    preds, confs, reasons = [], [], []
    for i, row in df.iterrows():
        obj, raw = classify_email(str(row["text"]))
        label = int(obj.get("label", 0))
        conf = float(obj.get("confidence", 0.5))
        reason = str(obj.get("reason", ""))
        preds.append(label)
        confs.append(conf)
        reasons.append(reason)

        # Optional: show progress
        print(f"[{i}] pred={label} conf={conf:.2f} reason={reason}")

    df["pred"] = preds
    df["confidence"] = confs
    df["reason"] = reasons

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
    # Map rows: true class 0->row0, 1->row1
    TN, FP = cm[0,0], cm[0,1]
    FN, TP = cm[1,0], cm[1,1]
    print([[int(TN), int(FP)], [int(FN), int(TP)]])

    # Save results
    out_path = os.path.splitext(csv_path)[0] + "_with_preds.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote predictions to: {out_path}")

if __name__ == "__main__":
    run_csv(CSV_PATH)