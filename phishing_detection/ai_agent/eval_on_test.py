#!/usr/bin/env python3
# usage:
#   python eval_on_test.py --model_dir outputs/deberta-bin/best --test_csv dataset/test.csv

import argparse
import os
import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import evaluate


def get_args():
    ap = argparse.ArgumentParser("Evaluate a fine-tuned HF classifier on test.csv")
    ap.add_argument("--model_dir", required=True, help="Path to fine-tuned model dir (contains model.safetensors, config.json, spm.model, ...)")
    ap.add_argument("--test_csv", required=True, help="Path to test.csv (columns: text,label)")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_csv", default="test_predictions.csv")
    ap.add_argument("--use_fast_tokenizer", action="store_true", help="Use fast tokenizer (optional).")
    return ap.parse_args()


def main():
    args = get_args()

    # 1) Load tokenizer/model
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=args.use_fast_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    # 2) Load test dataset
    ds = load_dataset("csv", data_files={"test": args.test_csv})
    has_labels = "label" in ds["test"].column_names

    # 3) Tokenize (dynamic padding via collator)
    def preprocess(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    ds = ds.map(preprocess, batched=True)
    cols = ["input_ids", "attention_mask"]
    if has_labels:
        ds = ds.rename_column("label", "labels")
        cols += ["labels"]
    ds.set_format(type="torch", columns=cols)

    collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)

    # 4) Minimal Trainer just for predict/evaluate
    ta = TrainingArguments(
        output_dir=os.path.join(args.model_dir, "eval_tmp"),
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
    )
    trainer = Trainer(model=model, args=ta, data_collator=collator)

    # 5) Run prediction
    pred = trainer.predict(ds["test"])
    logits = torch.tensor(pred.predictions)
    if logits.ndim == 1 or logits.shape[-1] == 1:
        # sigmoid (rare if you trained with num_labels=1)
        probs_pos = torch.sigmoid(logits.squeeze(-1))
        probs = torch.stack([1 - probs_pos, probs_pos], dim=1).numpy()
    else:
        probs = torch.softmax(logits, dim=-1).numpy()  # [:,0] = P(class 0), [:,1] = P(class 1)

    preds = probs.argmax(axis=1)

    # 6) Metrics (if labels present)
    metrics = {}
    if has_labels:
        y_true = np.array(ds["test"]["labels"])
        acc = evaluate.load("accuracy").compute(predictions=preds, references=y_true)["accuracy"]
        prec = evaluate.load("precision").compute(predictions=preds, references=y_true, average="macro")["precision"]
        rec = evaluate.load("recall").compute(predictions=preds, references=y_true, average="macro")["recall"]
        f1 = evaluate.load("f1").compute(predictions=preds, references=y_true, average="macro")["f1"]
        metrics = {"accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1}
        print("Test metrics:", metrics)
    else:
        print("No 'label' column found in test CSV — skipping metrics.")

    # 7) Save predictions CSV
    df = pd.read_csv(args.test_csv)
    df_out = pd.DataFrame({
        "text": df["text"],
        **({"label": df["label"]} if "label" in df.columns else {}),
        "prob_0": probs[:, 0],
        "prob_1": probs[:, 1],
        "pred": preds,
    })
    df_out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")

    # Also print a quick head
    print(df_out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
