#!/usr/bin/env python3
"""
Explain predictions of a trained model on text or on rows sampled from a CSV,
and list the top-N influential words for the whole dataset.

Examples:
# Per-text explanations
python explain_utils_single.py --model artifacts_single/model.joblib --text "Your package is on hold"

# Sample first 10 rows from CSV and explain
python explain_utils_single.py --model artifacts_single/model.joblib --csv Emails.csv --top-k 10

# Global top-N by model weights only
python explain_utils_single.py --model artifacts_single/model.joblib --global-top --top-n 30

# Global top-N weighted by the dataset (coef * mean TF-IDF across CSV)
python explain_utils_single.py --model artifacts_single/model.joblib --csv Emails.csv --global-top --top-n 30
"""
from __future__ import annotations

import argparse
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple

# ========== Robust CSV utilities ==========
def read_email_csv(path: str) -> pd.DataFrame:
    tries = [
        dict(sep=";", engine="python", quotechar='"', doublequote=True, escapechar="\\",
             encoding="utf-8-sig", on_bad_lines="skip"),
        dict(sep=";", engine="python", quotechar='"', doublequote=True, escapechar="\\",
             encoding="utf-8", on_bad_lines="skip"),
        dict(sep=None, engine="python", encoding="utf-8", on_bad_lines="skip"),
    ]
    last_err = None
    for kw in tries:
        try:
            return pd.read_csv(path, **kw)
        except Exception as e:
            last_err = e
    raise last_err if last_err else RuntimeError("Could not read CSV")

def ensure_text_column(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    subj = cols.get("subject", "Subject" if "Subject" in df.columns else None)
    body = cols.get("body", "Body" if "Body" in df.columns else None)
    if subj and body and "text" not in df.columns:
        df["text"] = (df[subj].astype(str) + "\\n\\n" + df[body].astype(str)).str.strip()
    return df

# ========== Model helpers ==========
def _get_union_feature_names(feats) -> List[str]:
    names: List[str] = []
    for name, vec in feats.transformer_list:
        # Some transformers may not expose names; TF-IDF does.
        fn = vec.get_feature_names_out()
        names.extend([f"{name}:{t}" for t in fn])
    return names

def predict_proba_for_text(model, text: str) -> float:
    return float(model.predict_proba([text])[:, 1][0])

def top_tfidf_terms_for_text(model, text: str, top_k=15) -> List[Tuple[str, float]]:
    """
    Works for Pipeline(FeatureUnion -> LogisticRegression). Extracts highest |weight * tfidf| terms.
    """
    pipe = model
    feats = pipe.named_steps["feats"]
    clf = pipe.named_steps["clf"]

    X = feats.transform([text])
    names = _get_union_feature_names(feats)

    w = clf.coef_.ravel()
    vals = X.toarray().ravel()
    contrib = w * vals

    idx = np.argsort(-np.abs(contrib))[:top_k]
    return [(names[i], float(contrib[i])) for i in idx]

def global_top_by_coef(model, top_n=30):
    """
    Top-N by raw coefficient magnitude (model weights only).
    """
    pipe = model
    feats = pipe.named_steps["feats"]
    clf = pipe.named_steps["clf"]
    names = _get_union_feature_names(feats)
    w = clf.coef_.ravel()

    pos_idx = np.argsort(-w)[:top_n]
    neg_idx = np.argsort(w)[:top_n]  # most negative
    top_pos = [(names[i], float(w[i])) for i in pos_idx]
    top_neg = [(names[i], float(w[i])) for i in neg_idx]
    return top_pos, top_neg

def global_top_by_dataset(model, texts, top_n=30):
    """
    Dataset-weighted contributions: coef * mean(tfidf) over the provided texts.
    """
    pipe = model
    feats = pipe.named_steps["feats"]
    clf = pipe.named_steps["clf"]
    names = _get_union_feature_names(feats)

    X = feats.transform(texts)
    mean_tfidf = np.asarray(X.mean(axis=0)).ravel()
    contrib = clf.coef_.ravel() * mean_tfidf

    pos_idx = np.argsort(-contrib)[:top_n]
    neg_idx = np.argsort(contrib)[:top_n]
    top_pos = [(names[i], float(contrib[i])) for i in pos_idx]
    top_neg = [(names[i], float(contrib[i])) for i in neg_idx]
    return top_pos, top_neg

def explain_text(model_path: str, text: str, top_k=15):
    model = joblib.load(model_path)
    p = predict_proba_for_text(model, text)
    top = top_tfidf_terms_for_text(model, text, top_k=top_k)
    print(f"p(phishing) = {p:.4f}")
    print("Top contributions:")
    for term, score in top:
        print(f"  {term:35s}  {score:+.4f}")

def print_table(title: str, rows):
    print(f"\\n{title}")
    print("-" * len(title))
    for term, val in rows:
        print(f"{term:35s}  {val:+.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to joblib model from pro_train_single.py")
    ap.add_argument("--text", default=None, help="Inline text to explain")
    ap.add_argument("--csv", default=None, help="CSV to sample/explain or to compute dataset-weighted globals")
    ap.add_argument("--text-col", default=None, help="Column to read text from (defaults to 'text' if present)")
    ap.add_argument("--top-k", type=int, default=15, help="Top-K features for per-text explanations")
    ap.add_argument("--global-top", action="store_true", help="Show global top-N terms (by weights or dataset-weighted if --csv given)")
    ap.add_argument("--top-n", type=int, default=30, help="Top-N terms for global listing")
    args = ap.parse_args()

    if args.text:
        explain_text(args.model, args.text, top_k=args.top_k)

    if args.csv and not args.global_top:
        df = read_email_csv(args.csv)
        df = ensure_text_column(df)
        col = args.text_col or ("text" if "text" in df.columns else df.columns[0])
        for i, t in enumerate(df[col].astype(str).head(10)):
            print(f"\\n=== Row {i} ===")
            explain_text(args.model, t, top_k=args.top_k)

    if args.global_top:
        model = joblib.load(args.model)
        if args.csv:
            df = read_email_csv(args.csv)
            df = ensure_text_column(df)
            col = args.text_col or ("text" if "text" in df.columns else df.columns[0])
            texts = df[col].astype(str).tolist()
            top_pos, top_neg = global_top_by_dataset(model, texts, top_n=args.top_n)
            print_table(f"Top {args.top_n} POSITIVE (dataset-weighted)", top_pos)
            print_table(f"Top {args.top_n} NEGATIVE (dataset-weighted)", top_neg)
        else:
            top_pos, top_neg = global_top_by_coef(model, top_n=args.top_n)
            print_table(f"Top {args.top_n} POSITIVE (by weight)", top_pos)
            print_table(f"Top {args.top_n} NEGATIVE (by weight)", top_neg)

if __name__ == "__main__":
    main()
