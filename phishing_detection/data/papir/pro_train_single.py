#!/usr/bin/env python3
"""
Single-CSV Email Classifier Trainer
- Expects ONE dataset containing both classes (e.g., your Emails.csv)
- Builds a 'text' column from Subject + Body and a binary 'label' from Type
- Trains a TF‑IDF + LogisticRegression model
- Saves model + quick evaluation artifacts

Example:
python pro_train_single.py \
  --csv Emails.csv \
  --subject-col Subject \
  --body-col Body \
  --label-col Type \
  --pos-label Phishing \
  --outdir ./artifacts_single \
  --test-size 0.2
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    precision_recall_curve,
)
import joblib

# ============== Robust CSV loading + schema mapping ==============

def read_email_csv(path: str) -> pd.DataFrame:
    """
    Read a semicolon-delimited email CSV with quoted multiline Body.
    Tolerant of BOM and minor irregularities.
    """
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

def build_text_and_label(df: pd.DataFrame,
                         subject_col="Subject",
                         body_col="Body",
                         label_col="Type",
                         pos_label="Phishing"):
    """
    Create a unified text column and a binary label.
    text = Subject + 2 newlines + Body
    label = 1 if value equals pos_label (case-insensitive), else 0
    """
    # case-insensitive column resolution
    cols = {c.lower(): c for c in df.columns}
    subject_col = cols.get(str(subject_col).lower(), subject_col)
    body_col = cols.get(str(body_col).lower(), body_col)
    label_col = cols.get(str(label_col).lower(), label_col)

    for need in [subject_col, body_col, label_col]:
        if need not in df.columns:
            raise KeyError(f"Required column '{need}' not found in CSV. Available: {list(df.columns)}")

    df["text"] = (df[subject_col].astype(str) + "\n\n" + df[body_col].astype(str)).str.strip()

    def to_bin(x):
        return 1 if str(x).strip().lower() == str(pos_label).strip().lower() else 0

    df["label"] = df[label_col].map(to_bin)
    df = df.dropna(subset=["text", "label"])
    return df[["text", "label"]].reset_index(drop=True)

# ============== Model training/evaluation ==============

def make_pipeline():
    # Word and character channels
    word_tfidf = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=2,
        max_df=0.9,
        strip_accents="unicode",
        lowercase=True
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3,5),
        min_df=2,
        max_df=0.95,
        lowercase=False
    )
    features = FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
    ])
    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=200
    )
    pipe = Pipeline([("feats", features), ("clf", clf)])
    return pipe

def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, digits=3, zero_division=0)
    return dict(acc=acc, precision=p, recall=r, f1=f1, auc=auc, confusion_matrix=cm, report=report)

def best_threshold_for_f1(y_true, y_prob):
    # search thresholds on PR curve points
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Align lengths (thresholds has one fewer element)
    thr = np.r_[thresholds, 1.0]
    f1s = 2*precisions*recalls/(precisions+recalls+1e-12)
    i = int(f1s.argmax())
    return float(thr[i]), float(f1s[i])

def best_threshold_for_precision(y_true, y_prob, target_precision=0.9):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    thr = np.r_[thresholds, 1.0]
    diffs = np.abs(precisions - target_precision)
    i = int(diffs.argmin())
    return float(thr[i]), float(precisions[i]), float(recalls[i])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Single dataset with both classes")
    ap.add_argument("--subject-col", default="Subject")
    ap.add_argument("--body-col", default="Body")
    ap.add_argument("--label-col", default="Type")
    ap.add_argument("--pos-label", default="Phishing")
    ap.add_argument("--outdir", default="./artifacts_single")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--target-precision", type=float, default=0.90)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load and map schema
    raw = read_email_csv(args.csv)
    df = build_text_and_label(raw, args.subject_col, args.body_col, args.label_col, args.pos_label)
    X = df["text"].astype(str).values
    y = df["label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    pipe = make_pipeline()
    pipe.fit(X_train, y_train)

    # Probabilities on test
    y_prob = pipe.predict_proba(X_test)[:, 1]

    # Default threshold 0.5
    eval_default = evaluate(y_test, y_prob, threshold=0.5)

    # Best F1 threshold
    thr_f1, best_f1 = best_threshold_for_f1(y_test, y_prob)
    eval_f1 = evaluate(y_test, y_prob, threshold=thr_f1)

    # Target precision threshold
    thr_p, got_p, got_r = best_threshold_for_precision(y_test, y_prob, target_precision=args.target_precision)
    eval_p = evaluate(y_test, y_prob, threshold=thr_p)

    # Save model and thresholds
    model_path = outdir / "model.joblib"
    joblib.dump(pipe, model_path)

    thresholds = {
        "default": 0.5,
        "best_f1": thr_f1,
        "target_precision": {"threshold": thr_p, "target": args.target_precision}
    }
    with open(outdir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    spec = {
        "dataset": str(Path(args.csv).resolve()),
        "subject_col": args.subject_col,
        "body_col": args.body_col,
        "label_col": args.label_col,
        "pos_label": args.pos_label,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_balance": {"pos_train": int(y_train.sum()), "neg_train": int((1-y_train).sum())},
        "vectorizers": {"word": {"ngram": (1,2)}, "char": {"ngram": (3,5)}},
        "clf": "LogisticRegression(liblinear, class_weight=balanced, max_iter=200)"
    }
    with open(outdir / "spec.json", "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    # Write reports
    def save_eval(name, ev):
        with open(outdir / f"report_{name}.txt", "w", encoding="utf-8") as f:
            f.write(f"Accuracy: {ev['acc']:.4f}\n")
            f.write(f"Precision: {ev['precision']:.4f}\n")
            f.write(f"Recall: {ev['recall']:.4f}\n")
            f.write(f"F1: {ev['f1']:.4f}\n")
            f.write(f"AUC: {ev['auc']:.4f}\n\n")
            f.write("Confusion matrix [ [tn, fp], [fn, tp] ]:\n")
            f.write(json.dumps(ev["confusion_matrix"]) + "\n\n")
            f.write(ev["report"] + "\n")

    save_eval("default_0p5", eval_default)
    save_eval("best_f1", eval_f1)
    save_eval("target_precision", eval_p)

    print(f"Saved model: {model_path}")
    print(f"Saved thresholds: {outdir / 'thresholds.json'}")
    print(f"Saved spec: {outdir / 'spec.json'}")
    print(f"Test results written to: {outdir}")
    print(f"Best F1 threshold: {thr_f1:.4f} (F1={best_f1:.4f})")
    print(f"Target precision threshold ~{args.target_precision:.2f}: {thr_p:.4f} (P={got_p:.3f}, R={got_r:.3f})")

if __name__ == "__main__":
    main()
