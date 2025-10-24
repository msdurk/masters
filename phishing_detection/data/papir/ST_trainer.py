#!/usr/bin/env python3
"""
Email Type Classifier using Sentence-Transformers embeddings + scikit-learn.

- Trains a classifier on Emails.csv (must contain a target column named 'Type').
- Text features are built by concatenating common email text fields (Subject, Body, Snippet, etc.).
- Saves a ready-to-use pipeline to `email_type_classifier.joblib`.
- Provides a CLI for training/evaluation and single-text prediction.

Usage:
python email_type_classifier.py \
  --data Emails.csv \
  --train \
  --binary \
  --positive-label "Phishing" \
  --negative-label-name "Other" \
  --ignore-labels "Unknown,NA" \
  --undersample-neg 5000

Notes:
- If the SentenceTransformer model cannot be loaded (e.g., offline), the script will
  automatically fall back to a TF-IDF baseline so you can still proceed.
- The default model is 'sentence-transformers/all-MiniLM-L6-v2'.
"""
from __future__ import annotations
import argparse
import os
import sys
import json
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

# Optional: if a helper exists in your repo
try:
    # If your _loader_utils.py defines helpful loaders, import it.
    # This file is optional. If not present or fails, we continue without it.
    import _loader_utils  # type: ignore
except Exception:  # pragma: no cover - best-effort import
    _loader_utils = None

SAVED_MODEL_PATH = "models/email_type_classifier.joblib"
DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEncoder(BaseEstimator, TransformerMixin):
    """Sklearn-compatible wrapper around sentence-transformers.

    Falls back to TF-IDF if sentence-transformers can't be loaded (e.g., offline).
    """
    def __init__(self, model_name: str = DEFAULT_ST_MODEL):
        self.model_name = model_name
        self._st_model = None
        self._using_fallback = False
        self._tfidf = None

    def fit(self, X: List[str], y=None):
        X = self._ensure_list(X)
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name)
            # Warm up encode a tiny sample to ensure model is ready
            _ = self._st_model.encode(X[:2] if len(X) >= 2 else X, normalize_embeddings=True)
            self._using_fallback = False
        except Exception as e:
            warnings.warn(
                f"Falling back to TF-IDF because SentenceTransformer failed to load: {e}\n"
                "Install or cache the model for best performance."
            )
            self._using_fallback = True
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
            self._tfidf.fit(X)
        return self

    def transform(self, X: List[str]):
        X = self._ensure_list(X)
        if not self._using_fallback and self._st_model is not None:
            # Normalize embeddings improves linear classifier performance
            emb = self._st_model.encode(X, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(emb, dtype=np.float32)
        # Fallback
        return self._tfidf.transform(X)

    @staticmethod
    def _ensure_list(X):
        if isinstance(X, (pd.Series, np.ndarray)):
            return X.tolist()
        return list(X)


def read_dataset(csv_path: str) -> pd.DataFrame:
    """Robust CSV reader that handles messy files.

    - Tries multiple encodings.
    - Tries multiple delimiters and engines.
    - Falls back to skipping bad lines if needed.
    """
    if _loader_utils and hasattr(_loader_utils, "load_emails_csv"):
        try:
            return _loader_utils.load_emails_csv(csv_path)  # type: ignore
        except Exception:
            pass

    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    delimiters = [None, ",", ";", "	", "|"]  # None => let pandas infer (python engine)
    engines = ["c", "python"]

    last_err = None
    for enc in encodings:
        for eng in engines:
            for sep in delimiters:
                try:
                    kwargs = {"encoding": enc, "engine": eng}
                    if eng == "python":
                        # In python engine, allow delimiter inference when sep is None
                        if sep is None:
                            kwargs.update({"sep": None})  # pandas will try to sniff
                        else:
                            kwargs.update({"sep": sep})
                        # If file is very messy, skip bad lines rather than crashing
                        kwargs.update({"on_bad_lines": "skip"})
                    else:
                        if sep is not None:
                            kwargs.update({"sep": sep})
                    df = pd.read_csv(csv_path, **kwargs)
                    if df.shape[1] == 1 and sep in (",", ";"):
                        # Heuristic: if only 1 column produced with a common sep, try tab
                        continue
                    return df
                except Exception as e:  # keep trying combos
                    last_err = e
                    continue
    # If we got here, re-raise the last error for visibility
    raise last_err if last_err else ValueError("Failed to read CSV: unknown error")


def combine_text_columns(df: pd.DataFrame) -> pd.Series:
    # Prefer common email fields if present
    candidate_cols = [
        "Subject", "Body", "Snippet", "From", "To", "Cc", "Bcc", "Preview", "Text", "Content"
    ]
    present = [c for c in candidate_cols if c in df.columns]
    if not present:
        # Fallback: use every non-target, non-id column as text
        present = [c for c in df.columns if c.lower() not in {"type", "label", "target", "id"}]
    # Clean and concat
    text = (
        df[present]
        .astype(str)
        .replace({"nan": "", "None": ""}, regex=False)
        .agg(". ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return text


def build_pipeline(model_name: str = DEFAULT_ST_MODEL) -> Pipeline:
    if model_name.lower() == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        encoder = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        clf = LogisticRegression(max_iter=2000, n_jobs=None, class_weight="balanced")
        pipe = Pipeline([
            ("tfidf", encoder),
            ("clf", clf),
        ])
        return pipe

    encoder = SentenceTransformerEncoder(model_name=model_name)
    clf = LogisticRegression(max_iter=2000, n_jobs=None, class_weight="balanced")
    pipe = Pipeline([
        ("embed", encoder),
        ("clf", clf),
    ])
    return pipe



def load_model(path: str = SAVED_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file '{path}' not found. Train it first with --train."
        )
    return joblib.load(path)


def predict_labels(texts: List[str], model_path: str = SAVED_MODEL_PATH) -> List[str]:
    bundle = load_model(model_path)
    pipe: Pipeline = bundle["pipeline"]
    le: LabelEncoder = bundle["label_encoder"]

    # Ensure list and basic cleaning
    if isinstance(texts, str):
        texts = [texts]
    texts = [" ".join(str(t).split()) for t in texts]

    preds_enc = pipe.predict(texts)
    preds = le.inverse_transform(preds_enc)
    return preds.tolist()



# -----------------------------
# Helper/diagnostic utilities
# -----------------------------

def _text_stats(series: pd.Series) -> dict:
    lens = series.fillna("").map(lambda s: len(str(s)))
    words = series.fillna("").map(lambda s: len(str(s).split()))
    return {
        "count": int(series.shape[0]),
        "empty_pct": float((series.fillna("") == "").mean()),
        "min_chars": int(lens.min() if len(lens) else 0),
        "mean_chars": float(lens.mean() if len(lens) else 0.0),
        "median_chars": float(lens.median() if len(lens) else 0.0),
        "max_chars": int(lens.max() if len(lens) else 0),
        "min_words": int(words.min() if len(words) else 0),
        "mean_words": float(words.mean() if len(words) else 0.0),
        "median_words": float(words.median() if len(words) else 0.0),
        "max_words": int(words.max() if len(words) else 0),
    }


def debug_read_dataset(csv_path: str) -> pd.DataFrame:
    """Run the robust reader and print a quick audit of the result.

    Returns the loaded DataFrame so you can keep using it in a REPL.
    """
    print(f"[read_dataset] Trying to load: {csv_path}")
    df = read_dataset(csv_path)
    print("[read_dataset] Loaded!")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print("Dtypes:", df.dtypes)

    # Show a tiny preview
    with pd.option_context('display.max_colwidth', 120, 'display.width', 120):
        print("Head(5):", df.head(5))

    # Null diagnostics
    nulls = df.isna().mean().sort_values(ascending=False)
    print("Null fraction by column (top 10):", nulls.head(10))

    # Basic heuristic check for separators left in the text (commas/semicolons/pipes)
    sample_cols = [c for c in df.columns if df[c].dtype == object][:5]
    if sample_cols:
        suspicious = {}
        for c in sample_cols:
            s = df[c].astype(str).head(200).str.contains(r"[,;|	]", regex=True, na=False).mean()
            suspicious[c] = float(s)
        print("Heuristic: fraction of sample cells containing delimiters (might indicate bad parsing):", suspicious)

    return df


def debug_combine_text_columns(df: pd.DataFrame, target_col: str = "Type", show_samples: int = 5, random_state: int = 42) -> pd.Series:
    """Validate `combine_text_columns` output and print quality stats.

    Returns the combined text Series.
    """
    if target_col not in df.columns:
        print(f"[combine_text_columns] WARNING: target column '{target_col}' not found.")

    combined = combine_text_columns(df)
    print("[combine_text_columns] Combined text created.")
    stats = _text_stats(combined)
    print("Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Correlate with labels if present
    if target_col in df.columns:
        y = df[target_col].astype(str).fillna("")
        # Show label distribution
        dist = y.value_counts(normalize=True).head(20)
        print("Label distribution (top 20):", dist)

        # Show a few random (text, label) examples
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(df), size=min(show_samples, len(df)), replace=False)
        print("Sample combined text + label pairs:")
        for i in idx:
            t = combined.iloc[int(i)]
            lab = y.iloc[int(i)]
            print("- ", repr(t[:200] + ("..." if len(t) > 200 else "")), "->", lab)
    else:
        # Show a few random text samples only
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(df), size=min(show_samples, len(df)), replace=False)
        print("Sample combined texts:")
        for i in idx:
            t = combined.iloc[int(i)]
            print("- ", repr(t[:200] + ("..." if len(t) > 200 else "")))

    # Check for fully empty rows (after combination)
    empty_rows = combined.fillna("") == ""
    if empty_rows.any():
        print(f"[combine_text_columns] WARNING: {int(empty_rows.sum())} empty combined texts.")
        empties = df.loc[empty_rows]
        print("Example empty rows (up to 3):", empties.head(3))

    return combined


# CLI hooks for diagnostics
#   --check-read to sanity check CSV loading
#   --check-combine to sanity check text combination

def _add_debug_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--check-read", action="store_true", help="Run read_dataset diagnostics and exit")
    parser.add_argument("--check-combine", action="store_true", help="Run combine_text_columns diagnostics and exit")



# -----------------------------
# Binary classification helpers
# -----------------------------
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

def _binarize_targets(
    df: pd.DataFrame,
    positive_label: str,
    target_col: str = "Type",
    negative_label_name: str = "Other",
    ignore_labels: Optional[List[str]] = None,
):
    if target_col not in df.columns:
        raise ValueError(f"Dataset must contain target column '{target_col}'.")
    ignore = set([str(x) for x in (ignore_labels or [])])
    # Drop ignored labels first
    if ignore:
        df = df[~df[target_col].astype(str).isin(ignore)].copy()
    y_raw = df[target_col].astype(str)
    pos = str(positive_label)
    y_bin = y_raw.apply(lambda v: pos if v == pos else negative_label_name)
    return df, y_bin


def _maybe_undersample_neg(X: pd.Series, y: pd.Series, negative_label_name: str, n_neg: int):
    if n_neg <= 0:
        return X, y
    neg_idx = y[y == negative_label_name].index
    if len(neg_idx) <= n_neg:
        return X, y
    pos_idx = y[y != negative_label_name].index
    keep_neg = np.random.default_rng(42).choice(neg_idx, size=n_neg, replace=False)
    keep_idx = np.concatenate([pos_idx.values, keep_neg])
    return X.loc[keep_idx], y.loc[keep_idx]


def predict_positive_proba(texts: List[str], model_path: str = SAVED_MODEL_PATH) -> List[float]:
    bundle = load_model(model_path)
    meta_task = bundle.get("task", "multiclass")
    pipe: Pipeline = bundle["pipeline"]
    le: LabelEncoder = bundle["label_encoder"]
    pos_label = bundle.get("positive_label")
    if meta_task != "binary" or pos_label is None:
        raise ValueError("This model is not a binary model with a defined positive class.")
    # Ensure list
    if isinstance(texts, str):
        texts = [texts]
    texts = [" ".join(str(t).split()) for t in texts]
    # Get positive class column
    proba = pipe.predict_proba(texts)
    pos_idx = int(np.where(le.classes_ == pos_label)[0][0])
    return proba[:, pos_idx].tolist()


# Re-define train_and_evaluate with binary support (overrides earlier def)
def train_and_evaluate(
    data_path: str,
    model_name: str = DEFAULT_ST_MODEL,
    test_size: float = 0.2,
    random_state: int = 42,
    save_path: str = SAVED_MODEL_PATH,
    binary: bool = False,
    positive_label: Optional[str] = None,
    negative_label_name: str = "Other",
    ignore_labels: Optional[List[str]] = None,
    undersample_neg: int = 0,
):
    df = read_dataset(data_path)

    if binary:
        if not positive_label:
            raise ValueError("--positive-label is required when --binary is set.")
        df2, y = _binarize_targets(
            df,
            positive_label=positive_label,
            negative_label_name=negative_label_name,
            ignore_labels=ignore_labels
        )
        X = combine_text_columns(df2)
        X, y = _maybe_undersample_neg(X, y, negative_label_name, undersample_neg)
    else:
        if "Type" not in df.columns:
            raise ValueError("Dataset must contain a 'Type' column as the target.")
        y = df["Type"].astype(str).fillna("")
        X = combine_text_columns(df)
    print("Label distribution before split:", pd.Series(y).value_counts())


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if pd.Series(y).nunique() > 1 else None
    )

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    pipe = build_pipeline(model_name)
    pipe.fit(X_train, y_train_enc)

    y_pred  = pipe.predict(X_test)

    # ====== BINARY-ONLY DIAGNOSTICS ======
    if binary:
        if len(le.classes_) != 2:
            raise RuntimeError(
                f"Binary diagnostics require exactly 2 classes after mapping; got {le.classes_}"
            )

        classes = list(le.classes_)
        pos_name = positive_label
        neg_name = negative_label_name

        if pos_name not in classes or neg_name not in classes:
            # fall back to whichever two labels exist (keeps code safe if names differ)
            pos_idx = int(np.where(np.array(classes) == pos_name)[0][0]) if pos_name in classes else 1
            neg_idx = 1 - pos_idx
            pos_name, neg_name = classes[pos_idx], classes[neg_idx]
        else:
            pos_idx = classes.index(pos_name)
            neg_idx = classes.index(neg_name)

        # try probabilities for AUC metrics
        try:
            y_proba_pos = pipe.predict_proba(X_test)[:, pos_idx]
        except Exception:
            y_proba_pos = None

        # core binary metrics (pos vs other)
        acc = accuracy_score(y_test_enc, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test_enc, y_pred, average="binary", pos_label=pos_idx, zero_division=0
        )

        print("\n=== Binary Diagnostics ({} vs {}) ===".format(pos_name, neg_name))
        print(f"Accuracy              : {acc:.4f}")
        print(f"Precision (pos={pos_name}): {prec:.4f}")
        print(f"Recall    (pos={pos_name}): {rec:.4f}")
        print(f"F1        (pos={pos_name}): {f1:.4f}")

        if y_proba_pos is not None:
            try:
                roc  = roc_auc_score((y_test_enc == pos_idx).astype(int), y_proba_pos)
                prau = average_precision_score((y_test_enc == pos_idx).astype(int), y_proba_pos)
                print(f"ROC-AUC               : {roc:.4f}")
                print(f"PR-AUC                : {prau:.4f}")
            except Exception:
                pass

        # 2×2 confusion matrix with explicit label order [pos, neg]
        cm = confusion_matrix(y_test_enc, y_pred, labels=[pos_idx, neg_idx])
        # rows = actual, cols = predicted
        tp, fn = cm[0, 0], cm[0, 1]
        fp, tn = cm[1, 0], cm[1, 1]

        print("\nConfusion Matrix (rows=actual, cols=predicted)")
        print(f"             Pred {pos_name:>8} | Pred {neg_name:>8}")
        print(f"Actual {pos_name:>8}: {tp:10d} | {fn:12d}")
        print(f"Actual {neg_name:>8}: {fp:10d} | {tn:12d}")

        # Optional: binary-only classification report (just the two classes, in [pos, neg] order)
        print("\nClassification Report (binary only):")
        print(classification_report(
            y_test_enc, y_pred,
            labels=[pos_idx, neg_idx],
            target_names=[pos_name, neg_name],
            zero_division=0
        ))

    else:
        # (unchanged) multiclass summary if you ever call without --binary
        print("\n=== Multiclass Evaluation ===")
        print(classification_report(y_test_enc, y_pred, target_names=[str(c) for c in le.classes_], zero_division=0))
        print("Confusion Matrix:\n", confusion_matrix(y_test_enc, y_pred))

    # ====== persist bundle as before ======
    bundle = {
        "pipeline": pipe,
        "label_encoder": le,
        "model_name": model_name,
        "columns": df.columns.tolist(),
        "task": "binary" if binary else "multiclass",
        "positive_label": positive_label if binary else None,
        "negative_label_name": negative_label_name if binary else None,
        "ignore_labels": ignore_labels or [],
    }
    joblib.dump(bundle, save_path)
    print(f"\nSaved trained model to: {save_path}")



def main(argv: Optional[List[str]] = None):  # type: ignore[override]
    parser = argparse.ArgumentParser(description="Email Type Classifier")
    parser.add_argument("--data", type=str, default="Emails.csv", help="Path to CSV with a 'Type' column")
    parser.add_argument("--model-name", type=str, default=DEFAULT_ST_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--train", action="store_true", help="Train and evaluate the model")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test size fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--save-path", type=str, default=SAVED_MODEL_PATH, help="Where to save the trained model")
    parser.add_argument("--predict", type=str, default=None, help="Predict the Type for a single email text")
    parser.add_argument("--predict-file", type=str, default=None, help="Path to text file with one email per line to predict")
    _add_debug_args(parser)
    # Binary-classification flags
    parser.add_argument("--binary", action="store_true", help="Train a binary classifier for one positive label")
    parser.add_argument("--positive-label", type=str, default=None, help="Name of the positive class in the Type column (required with --binary)")
    parser.add_argument("--negative-label-name", type=str, default="Other", help="Name to assign to all non-positive examples")
    parser.add_argument("--ignore-labels", type=str, default=None, help="Comma-separated list of labels to drop before binarizing")
    parser.add_argument("--undersample-neg", type=int, default=0, help="Optional cap on number of negative examples to keep for training speed")
    parser.add_argument("--proba", action="store_true", help="When predicting with a binary model, also print positive-class probability")

    args = parser.parse_args(argv)

    # Diagnostics shortcuts
    if args.check_read:
        df = debug_read_dataset(args.data)
        # If they also want to check combine in one go
        if args.check_combine:
            debug_combine_text_columns(df)
        return

    if args.check_combine:
        df = read_dataset(args.data)
        debug_combine_text_columns(df)
        return
    if args.binary:
        if args.model_name != DEFAULT_ST_MODEL:
            args.save_path = f"models/binary_{args.model_name}.joblib"
        else:
            args.save_path = "models/binary_email_type_classifier.joblib"

    # Default behavior unchanged
    if args.train:
        ignore_list = [s.strip() for s in (args.ignore_labels.split(",") if args.ignore_labels else []) if s.strip()]
        train_and_evaluate(
            data_path=args.data,
            model_name=args.model_name,
            test_size=args.test_size,
            random_state=args.random_state,
            save_path=args.save_path,
            binary=args.binary,
            positive_label=args.positive_label,
            negative_label_name=args.negative_label_name,
            ignore_labels=args.ignore_labels,
            undersample_neg=args.undersample_neg
        )
        if args.predict:
            preds = predict_labels([args.predict], model_path=args.save_path)
            print(f"Prediction: {preds[0]}")
            if args.proba:
                try:
                    p = predict_positive_proba([args.predict], model_path=args.save_path)[0]
                    print(f"Positive-class probability: {p:.4f}")
                except Exception as e:
                    print(f"[proba] {e}")
        if args.predict_file and os.path.exists(args.predict_file):
            with open(args.predict_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            preds = predict_labels(lines, model_path=args.save_path)
            if args.proba:
                try:
                    probs = predict_positive_proba(lines, model_path=args.save_path)
                except Exception as e:
                    probs = None
                    print(f"[proba] {e}")
            else:
                probs = None
            for i, (t, p) in enumerate(zip(lines, preds)):
                if probs is not None:
                    print(f"PRED	{p}	{probs[i]:.4f}	{t[:80]}{'...' if len(t) > 80 else ''}")
                else:
                    print(f"PRED	{p}	{t[:80]}{'...' if len(t) > 80 else ''}")
        return

    if args.predict is not None or args.predict_file is not None:
        texts: List[str] = []
        if args.predict is not None:
            texts.append(args.predict)
        if args.predict_file is not None:
            if not os.path.exists(args.predict_file):
                sys.exit(f"Predict file not found: {args.predict_file}")
            with open(args.predict_file, "r", encoding="utf-8") as f:
                texts.extend([line.strip() for line in f if line.strip()])
        preds = predict_labels(texts, model_path=args.save_path)
        if args.proba:
            try:
                probs = predict_positive_proba(texts, model_path=args.save_path)
            except Exception as e:
                probs = None
                print(f"[proba] {e}")
        else:
            probs = None
        for i, (t, p) in enumerate(zip(texts, preds)):
            if probs is not None:
                print(f"PRED	{p}	{probs[i]:.4f}	{t[:80]}{'...' if len(t) > 80 else ''}")
            else:
                print(f"PRED	{p}	{t[:80]}{'...' if len(t) > 80 else ''}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()

