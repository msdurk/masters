#!/usr/bin/env python3
"""
Pro Predictor (shared-module version)
- Imports TextStatsTransformer from components.py
- Registers it under __main__ so models saved from scripts still unpickle
- Threshold modes: default (0.5), best_f1, p_target (from thresholds.json)
- Input via CLI arg, stdin, or a CSV file


run from phishing_detection:
python data/test_case/pro_predict.py \ 
  --model data/test_case/artifacts_pro/model.joblib \
  --thresholds data/test_case/artifacts_pro/thresholds.json \
  --threshold-mode p_target \
  --csv filtered_raw/Nazario_5_removed_01.csv \
  --text-col body
"""

import argparse, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# --- Ensure the custom class is available before loading the model -----------
from components import TextStatsTransformer  # shared definition
import __main__ as _m
setattr(_m, "TextStatsTransformer", TextStatsTransformer)
# -----------------------------------------------------------------------------

def load_threshold(th_path, mode=None, explicit=None):
    if explicit is not None:
        return explicit
    if th_path is None or not Path(th_path).exists():
        return 0.5
    with open(th_path, "r", encoding="utf-8") as f:
        t = json.load(f)
    if mode == "best_f1":
        return float(t.get("best_f1_threshold", 0.5))
    if mode == "p_target":
        return float(t.get("p_target_threshold", 0.5))
    return 0.5

def predict_texts(pipe, texts, threshold=0.5):
    classes = list(pipe.named_steps["clf"].classes_)
    if "phishing" in classes:
        ph_idx = classes.index("phishing")
        scores = pipe.predict_proba(texts)[:, ph_idx]
    else:
        scores = np.ones(len(texts)) * 0.5
    preds = np.where(scores >= threshold, "phishing", "legit")
    return preds, scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model.joblib")
    ap.add_argument("--thresholds", default=None, help="Path to thresholds.json")
    ap.add_argument("--threshold-mode", choices=["default","best_f1","p_target"], default="default")
    ap.add_argument("--threshold", type=float, default=None, help="Explicit threshold override (0..1)")
    ap.add_argument("--text", default=None, help="Single text via CLI")
    ap.add_argument("--csv", default=None, help="CSV file to score")
    ap.add_argument("--text-col", default=None, help="Column name in CSV containing text")
    args = ap.parse_args()

    pipe = joblib.load(args.model)

    threshold = load_threshold(
        args.thresholds,
        mode=(args.threshold_mode if args.threshold_mode != "default" else None),
        explicit=args.threshold
    )

    # Single text from CLI or stdin
    if args.text is None and args.csv is None:
        data = sys.stdin.read().strip()
        if not data:
            ap.error("Provide --text or --csv, or pipe text via stdin.")
        texts = [data]
        preds, scores = predict_texts(pipe, texts, threshold=threshold)
        print(preds[0])
        return

    if args.text is not None:
        preds, scores = predict_texts(pipe, [args.text], threshold=threshold)
        print(preds[0])
        return

    # CSV batch mode
    df = pd.read_csv(args.csv)
    col = args.text_col or ("text" if "text" in df.columns else df.columns[0])
    texts = df[col].astype(str).tolist()
    preds, scores = predict_texts(pipe, texts, threshold=threshold)
    out = df.copy()
    out["pred"] = preds
    out["score_phishing"] = scores
    out.to_csv("predictions.csv", index=False)
    print("Wrote predictions.csv")

if __name__ == "__main__":
    main()
