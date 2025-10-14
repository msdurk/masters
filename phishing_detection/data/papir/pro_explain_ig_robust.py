
#!/usr/bin/env python3
"""
Batch scoring, selective explanations, and GLOBAL top-N token contributions.

Examples:
  # Score every row and write predictions.csv
  python pro_explain_ig_robust.py --model-dir models/mailbench_v1 --input data.csv

  # Explain top-20 highest-risk rows
  python pro_explain_ig_robust.py --model-dir models/mailbench_v1 --input data.csv --explain-top 20

  # GLOBAL: top-50 tokens that contribute the most (overall) to phishing classification
  python pro_explain_ig_robust.py --model-dir models/mailbench_v1 --input data.csv --global-top 50
"""

import os, argparse
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix

from _loader_utils import load_mailbench_semicolon_csv

def token_contributions(vec, clf, texts):
    X = vec.transform(texts)
    coef = clf.coef_.ravel()
    vocab = vec.get_feature_names_out()
    return X, coef, vocab

def explain_single_row(X_row, coef, vocab, k=15):
    arr = X_row.toarray().ravel()
    contrib = arr * coef
    idx_pos = np.argsort(-contrib)[:k]
    idx_neg = np.argsort(contrib)[:k]
    top_pos = [(vocab[i], float(contrib[i])) for i in idx_pos]
    top_neg = [(vocab[i], float(contrib[i])) for i in idx_neg]
    return top_pos, top_neg, float(contrib.sum())

def global_top_tokens(X, coef, vocab, topn=50):
    """
    Compute per-feature contributions aggregated over the whole dataset.
    - total_tfidf[i] = sum_j X[j,i]
    - mean_tfidf[i]  = total_tfidf[i] / n_docs
    - df[i]          = number of docs where X[j,i] != 0
    - total_contribution[i] = coef[i] * total_tfidf[i]
    - mean_contribution[i]  = coef[i] * mean_tfidf[i]
    """
    n_docs = X.shape[0]
    total_tfidf = np.asarray(X.sum(axis=0)).ravel()
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    mean_tfidf = total_tfidf / max(1, n_docs)
    total_contribution = coef * total_tfidf
    mean_contribution = coef * mean_tfidf

    out = pd.DataFrame({
        "token": vocab,
        "coef": coef,
        "df": df,
        "mean_tfidf": mean_tfidf,
        "total_tfidf": total_tfidf,
        "total_contribution": total_contribution,
        "mean_contribution": mean_contribution
    })
    out_sorted = out.sort_values("total_contribution", ascending=False)
    return out_sorted.iloc[:topn].reset_index(drop=True), out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--explain-top", type=int, default=0, help="Explain top-N highest-prob unwanted rows")
    ap.add_argument("--k", type=int, default=15, help="Top-K tokens per single-row explanation")
    ap.add_argument("--global-top", type=int, default=0, help="Compute GLOBAL top-N contributing tokens across the entire dataset")
    args = ap.parse_args()

    bundle = joblib.load(os.path.join(args.model_dir, "model.joblib"))
    vec = bundle["vectorizer"]
    clf = bundle["clf"]

    df = load_mailbench_semicolon_csv(args.input)
    texts = df["text"].astype(str).tolist()

    # Always score all rows (also useful for auditing)
    X_all = vec.transform(texts)
    probs = clf.predict_proba(X_all)[:,1]
    preds = (probs >= 0.5).astype(int)
    pred_out = df.copy()
    pred_out.insert(0, "pred_unwanted_prob", probs)
    pred_out.insert(1, "pred_unwanted_label", preds)
    pred_path = os.path.join(args.model_dir, "predictions.csv")
    pred_out.to_csv(pred_path, index=False)
    print(f"Wrote predictions for {len(pred_out)} rows to: {pred_path}")

    if args.global_top and args.global_top > 0:
        top_df, full_df = global_top_tokens(X_all, clf.coef_.ravel(), vec.get_feature_names_out(), topn=args.global_top)
        top_path = os.path.join(args.model_dir, f"global_top_{args.global_top}.csv")
        top_df.to_csv(top_path, index=False)
        full_path = os.path.join(args.model_dir, f"global_all_tokens.csv")
        full_df.to_csv(full_path, index=False)
        print(f"Saved GLOBAL top-{args.global_top} tokens to: {top_path}")
        print(f"(Full token contributions saved to: {full_path})")
        print("\\nTop tokens (total contribution):")
        for i, row in top_df.iterrows():
            print(f"{i+1:>2d}. {row['token']:<25} total_contrib={row['total_contribution']:+.6f}  df={int(row['df'])}  coef={row['coef']:+.6f}")

    if args.explain_top and args.explain_top > 0:
        vocab = vec.get_feature_names_out()
        coef = clf.coef_.ravel()
        top_idx = np.argsort(-probs)[:args.explain_top]
        for rank, row_idx in enumerate(top_idx, start=1):
            X_row = X_all[row_idx]
            pos, neg, score = explain_single_row(X_row, coef, vocab, k=args.k)
            base_name = f"explain_row_{int(row_idx)+1}"
            contrib_df = pd.DataFrame({
                "token": [p[0] for p in pos+neg],
                "contribution": [p[1] for p in pos] + [p[1] for p in neg],
                "sign": ["positive"]*len(pos) + ["negative"]*len(neg)
            })
            out_csv = os.path.join(args.model_dir, base_name + ".csv")
            contrib_df.to_csv(out_csv, index=False)
            print(f"Saved token explanation for row {row_idx+1} -> {out_csv}")

if __name__ == "__main__":
    main()
