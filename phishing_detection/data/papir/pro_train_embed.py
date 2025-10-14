
#!/usr/bin/env python3
"""
Train a classifier on thousands of emails from the semicolon-delimited CSV.

Usage:
  python pro_train_embed.py --input data.csv --outdir models/mailbench_v1
Options:
  --label-from Type           Column for labels (default: Type)
  --positive-value Phishing   Value mapped to 1/unwanted (default: Phishing)
  --limit N                   Optional: train on at most N rows (for quick tests)
"""

import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, average_precision_score
import joblib

from _loader_utils import load_mailbench_semicolon_csv, basic_engineered_features

def build_tfidf():
    return TfidfVectorizer(
        max_features=100000,
        ngram_range=(1,2),
        min_df=2,
        strip_accents="unicode"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label-from", default="Type")
    ap.add_argument("--positive-value", default="Phishing")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_mailbench_semicolon_csv(args.input, label_from=args.label_from, positive_value=args.positive_value)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    if args.limit is not None:
        df = df.iloc[:args.limit].copy()

    if df["label"].isna().any():
        raise ValueError("Some labels are missing. Ensure the label column exists or pass --label-from / --positive-value.")

    X_text = df["text"].astype(str).tolist()
    y = df["label"].astype(int).values

    # Engineered features
    eng = basic_engineered_features(df["text"])

    # Split
    Xtr_text, Xte_text, ytr, yte, eng_tr, eng_te = train_test_split(
        X_text, y, eng, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    vec = build_tfidf()
    from scipy.sparse import hstack, csr_matrix
    Xtr_tfidf = vec.fit_transform(Xtr_text)
    Xte_tfidf = vec.transform(Xte_text)

    Xtr = hstack([Xtr_tfidf, csr_matrix(eng_tr.values)])
    Xte = hstack([Xte_tfidf, csr_matrix(eng_te.values)])

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    clf.fit(Xtr, ytr)

    probs = clf.predict_proba(Xte)[:,1]
    ap = average_precision_score(yte, probs)
    print("PR-AUC (unwanted=1):", round(float(ap), 4))
    print(classification_report(yte, (probs>=0.5).astype(int), target_names=["wanted","unwanted"]))

    joblib.dump({"vectorizer": vec, "clf": clf, "engineered_cols": list(eng.columns)}, os.path.join(args.outdir, "model.joblib"))
    with open(os.path.join(args.outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"type": "tfidf+engineered"}, f, indent=2)

if __name__ == "__main__":
    main()
