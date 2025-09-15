
#!/usr/bin/env python3
# Predict with the saved model (supports both TF-IDF and embedding variants)
# Usage:
#   python pro_predict_embed.py --model-dir models/embed_v1 --in path/to/file.csv --out preds.csv
import os, argparse, json, sys, re, joblib, numpy as np, pandas as pd
from components import TextStatsTransformer
from _loader_utils import read_simple_csv, read_problem_llm_phishing


def load_input(path: str) -> pd.DataFrame:
    name = os.path.basename(path)
    if name == "llm_phishing.csv":
        df = read_problem_llm_phishing(path)
    else:
        df = read_simple_csv(path)
    if "text" not in df.columns:
        subj = df["subject"] if "subject" in df.columns else ""
        body = df["body"] if "body" in df.columns else ""
        df["text"] = (subj.fillna('') + ' ' + body.fillna(''))
    return df[["text"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="preds.csv")
    args = ap.parse_args()

    meta = joblib.load(os.path.join(args.model_dir, "meta.json"))
    df = load_input(args.inp)

    if meta["type"] == "sklearn_tfidf+engineered":
        pipe = joblib.load(os.path.join(args.model_dir, "model.joblib"))
        probs = pipe.predict_proba(df)[:,1]
        preds = (probs >= 0.5).astype(int)
    else:
        # embedding variant
        # For simplicity in this predict script, we don't re-embed (would need same embed backend);
        # recommend using the TF-IDF model for production or extend this with the same embedder.
        raise NotImplementedError("Embedding variant prediction requires embedder; retrain with tfidf or extend embed predict.")

    out = df.copy()
    out["prob_unwanted"] = probs
    out["pred_label"] = preds
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} predictions to {args.out}")

if __name__ == "__main__":
    main()
