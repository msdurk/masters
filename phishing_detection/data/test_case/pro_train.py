#!/usr/bin/env python3
"""
Pro Email Classifier Trainer
- Word + character TF-IDF channels
- Hand-crafted URL/HTML features
- Small hyperparameter sweep (GridSearchCV) with correct positive label
- Threshold tuning for best F1 and target phishing precision
- Final model refit on train+val with best params


use:
python pro_train.py \
  --legit human_legit.csv llm_legit.csv \
  --phish humen_phishing.csv llm_phishing.csv \
  --outdir ./artifacts_pro \
  --test-size 0.2 \
  --val-size 0.25 \
  --min-df 2 \
  --max-features 80000 \
  --word-ngrams 1 2 \
  --char-ngrams 3 5 \
  --grid \
  --target-precision 0.95
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    f1_score,
    make_scorer,
)

import joblib

PREFERRED_TEXT_COLS = ["text", "body", "content", "email", "message", "subject", "body_text"]

def load_csv_lenient(path: str) -> pd.DataFrame:
    tries = [
        {"encoding": "utf-8", "sep": ",", "engine": "python", "on_bad_lines": "skip", "quotechar": '"', "escapechar": "\\"},
        {"encoding": "utf-8", "sep": None, "engine": "python", "on_bad_lines": "skip", "quotechar": '"', "escapechar": "\\"},
        {"encoding": "latin-1", "sep": ",", "engine": "python", "on_bad_lines": "skip", "quotechar": '"', "escapechar": "\\"},
        {"encoding": "latin-1", "sep": None, "engine": "python", "on_bad_lines": "skip", "quotechar": '"', "escapechar": "\\"},
    ]
    last_err = None
    for kw in tries:
        try:
            return pd.read_csv(path, **kw)
        except Exception as e:
            last_err = e
    raise last_err

def pick_text_column(df: pd.DataFrame) -> str:
    for c in PREFERRED_TEXT_COLS:
        if c in df.columns:
            return c
    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not object_cols:
        return df.columns[0]
    avg = [(c, df[c].astype(str).str.len().mean()) for c in object_cols]
    avg.sort(key=lambda x: x[1], reverse=True)
    return avg[0][0]

def normalize_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    col = pick_text_column(df)
    out = df[[col]].rename(columns={col: "text"}).copy()
    out["text"] = out["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.dropna(subset=["text"])
    out = out[~out["text"].eq("")]
    out["label"] = label
    return out

# ---------- Hand-crafted feature transformer ----------
class TextStatsTransformer(BaseEstimator, TransformerMixin):
    URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
    HTML_TAG_RE = re.compile(r"<[^>]+>")
    SUSPICIOUS_TLDS = (".ru",".cn",".top",".xyz",".tk",".icu",".click",".link",".work",".country")
    URGENT_WORDS = (
        "urgent","verify","suspend","suspended","login","immediately","click",
        "password","account","confirm","update","security","expire","expired","invoice","payment"
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for text in X:
            t = str(text)
            length = max(len(t), 1)

            urls = self.URL_RE.findall(t)
            url_count = len(urls)

            html_tags = len(self.HTML_TAG_RE.findall(t))

            suspicious = sum(1 for u in urls if any(u.lower().endswith(tld) or tld in u.lower() for tld in self.SUSPICIOUS_TLDS))

            exclam = t.count("!")

            letters = [ch for ch in t if ch.isalpha()]
            upper = sum(1 for ch in letters if ch.isupper())
            upper_ratio = upper / max(len(letters), 1)

            digits = sum(ch.isdigit() for ch in t)
            digit_ratio = digits / length

            money = t.count("$") + t.count("€") + t.count("£")

            at_count = t.count("@")

            urgent_hits = sum(t.lower().count(w) for w in self.URGENT_WORDS)

            feats.append([
                url_count, html_tags, suspicious, exclam,
                upper_ratio, digit_ratio, money, at_count, urgent_hits, length
            ])
        mat = np.asarray(feats, dtype=float)
        mat[:, 0] = np.log1p(mat[:, 0])   # url_count
        mat[:, 1] = np.log1p(mat[:, 1])   # html_tags
        mat[:, 2] = np.log1p(mat[:, 2])   # suspicious
        mat[:, 3] = np.log1p(mat[:, 3])   # exclam
        mat[:, 6] = np.log1p(mat[:, 6])   # money
        mat[:, 7] = np.log1p(mat[:, 7])   # at_count
        mat[:, 8] = np.log1p(mat[:, 8])   # urgent_hits
        mat[:, 9] = np.log1p(mat[:, 9])   # length
        return sparse.csr_matrix(mat)

def build_pipeline(word_ngrams=(1,2), char_ngrams=(3,5), min_df=2, max_features=100_000, use_char=True):
    word_vec = TfidfVectorizer(
        ngram_range=word_ngrams,
        min_df=min_df,
        max_features=max_features,
        strip_accents="unicode",
        analyzer="word",
    )
    blocks = [("word", word_vec)]
    if use_char:
        char_vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=char_ngrams,
            min_df=2,
            max_features=max_features // 2,
            strip_accents=None,
        )
        blocks.append(("char", char_vec))

    blocks.append(("stats", Pipeline([
        ("stats", TextStatsTransformer()),
        ("scale", StandardScaler(with_mean=False))
    ])))

    feat_union = FeatureUnion(blocks)
    clf = LogisticRegression(
        solver="liblinear",
        penalty="l2",
        C=1.0,
        max_iter=400,
        class_weight="balanced",
    )
    pipe = Pipeline([("features", feat_union), ("clf", clf)])
    return pipe

def find_thresholds(y_true, y_scores, target_precision=0.95, positive_label="phishing"):
    precision, recall, thresholds = precision_recall_curve(
        [1 if y==positive_label else 0 for y in y_true], y_scores
    )
    f1s = []
    for i, th in enumerate(thresholds):
        p = precision[i+1]
        r = recall[i+1]
        f1s.append(0.0 if (p+r)==0 else 2*p*r/(p+r))
    best_idx = int(np.argmax(f1s)) if len(f1s) else 0
    best_f1_threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5
    best_f1 = float(np.max(f1s)) if len(f1s) else 0.0

    p_target_threshold = 0.5
    for i in range(len(thresholds)-1, -1, -1):
        p = precision[i+1]
        if p >= target_precision:
            p_target_threshold = float(thresholds[i])
            break

    return {
        "best_f1_threshold": best_f1_threshold,
        "best_f1": best_f1,
        "p_target_threshold": p_target_threshold,
        "target_precision": float(target_precision),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legit", nargs="+", default=[], help="CSV files with legit emails")
    ap.add_argument("--phish", nargs="+", default=[], help="CSV files with phishing emails")
    ap.add_argument("--outdir", default="artifacts_pro")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.25, help="Fraction of train used as validation (after test split)")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--max-features", type=int, default=100000)
    ap.add_argument("--word-ngrams", type=int, nargs=2, default=[1,2])
    ap.add_argument("--char-ngrams", type=int, nargs=2, default=[3,5])
    ap.add_argument("--no-char", action="store_true", help="Disable character n-gram channel")
    ap.add_argument("--grid", action="store_true", help="Run a small hyperparameter sweep")
    ap.add_argument("--target-precision", type=float, default=0.95, help="Precision target for phishing threshold")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load and combine
    frames = []
    for p in args.legit:
        df = load_csv_lenient(p); frames.append(normalize_df(df, "legit"))
    for p in args.phish:
        df = load_csv_lenient(p); frames.append(normalize_df(df, "phishing"))
    if not frames:
        raise SystemExit("No data provided. Use --legit/--phish with CSV paths.")

    data = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["text","label"]).reset_index(drop=True)

    # Split: test holdout
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        data["text"], data["label"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=data["label"]
    )
    # Validation split from training portion
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=y_train_full
    )

    # Build pipeline
    pipe = build_pipeline(
        word_ngrams=tuple(args.word_ngrams),
        char_ngrams=tuple(args.char_ngrams),
        min_df=args.min_df,
        max_features=args.max_features,
        use_char=not args.no_char
    )

    # Small grid search (with correct positive label)
    best_params = {}
    if args.grid:
        param_grid = {
            "clf__C": [0.25, 0.5, 1.0, 2.0],
            "features__word__ngram_range": [(1,1), (1,2)],
        }
        if not args.no_char:
            param_grid["features__char__ngram_range"] = [(3,5), (3,6)]
        scorer = make_scorer(f1_score, pos_label='phishing')
        try:
            gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1, refit=True, scoring=scorer)
            gs.fit(X_train, y_train)
            pipe = gs.best_estimator_
            best_params = gs.best_params_
        except Exception as e:
            print(f"[WARN] Grid search failed: {e}\nFalling back to baseline training.")
            pipe.fit(X_train, y_train)
    else:
        pipe.fit(X_train, y_train)

    # Validation scores & threshold search
    classes = list(pipe.named_steps["clf"].classes_)
    if "phishing" in classes:
        ph_idx = classes.index("phishing")
        y_val_scores = pipe.predict_proba(X_val)[:, ph_idx]
    else:
        y_val_scores = np.ones(len(X_val)) * 0.5

    thres = find_thresholds(y_val, y_val_scores, target_precision=args.target_precision, positive_label="phishing")

    # Refit on train+val with best hyperparams
    # Rebuild a fresh pipeline mirroring the structure actually present in `pipe`
    used_word_ngrams = pipe.get_params().get("features__word__ngram_range", (1,2))
    has_char = "features__char__ngram_range" in pipe.get_params()
    used_char_ngrams = pipe.get_params().get("features__char__ngram_range", (3,5))
    final_pipe = build_pipeline(
        word_ngrams=used_word_ngrams,
        char_ngrams=used_char_ngrams,
        min_df=args.min_df,
        max_features=args.max_features,
        use_char=has_char
    )
    # Apply best params only if valid for the new pipeline
    valid_keys = set(final_pipe.get_params().keys())
    filtered = {k: v for k, v in best_params.items() if k in valid_keys}
    if filtered:
        final_pipe.set_params(**filtered)

    final_pipe.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    # Evaluate on test set with both thresholds
    if "phishing" in final_pipe.named_steps["clf"].classes_:
        ph_idx = list(final_pipe.named_steps["clf"].classes_).index("phishing")
        y_test_scores = final_pipe.predict_proba(X_test)[:, ph_idx]
    else:
        y_test_scores = np.ones(len(X_test)) * 0.5

    def eval_with_threshold(scores, y_true, t):
        y_pred = np.where(scores >= t, "phishing", "legit")
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label="phishing", zero_division=0)
        try:
            auc = roc_auc_score([1 if y=="phishing" else 0 for y in y_true], scores)
        except Exception:
            auc = None
        cm = confusion_matrix(y_true, y_pred, labels=["legit","phishing"]).tolist()
        return {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1), "roc_auc": float(auc) if auc is not None else None, "confusion_matrix": cm}

    eval_f1 = eval_with_threshold(y_test_scores, y_test, thres["best_f1_threshold"])
    eval_p = eval_with_threshold(y_test_scores, y_test, thres["p_target_threshold"])
    eval_default = eval_with_threshold(y_test_scores, y_test, 0.5)

    # Save artifacts
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_model = outdir / "model.joblib"
    joblib.dump(final_pipe, out_model)

    thresholds = {
        "best_f1_threshold": thres["best_f1_threshold"],
        "p_target_threshold": thres["p_target_threshold"],
        "target_precision": thres["target_precision"],
    }
    with open(outdir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    spec = {
        "best_params": best_params,
        "classes": list(final_pipe.named_steps["clf"].classes_),
        "thresholds": thresholds,
        "test_eval": {
            "default_0.5": eval_default,
            "best_f1": eval_f1,
            "p_target": eval_p,
        }
    }
    with open(outdir / "spec.json", "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    # CSV confusion matrices
    pd.DataFrame(eval_default["confusion_matrix"], index=["true_legit","true_phishing"], columns=["pred_legit","pred_phishing"]).to_csv(outdir / "confusion_matrix_default.csv")
    pd.DataFrame(eval_f1["confusion_matrix"], index=["true_legit","true_phishing"], columns=["pred_legit","pred_phishing"]).to_csv(outdir / "confusion_matrix_best_f1.csv")
    pd.DataFrame(eval_p["confusion_matrix"], index=["true_legit","true_phishing"], columns=["pred_legit","pred_phishing"]).to_csv(outdir / "confusion_matrix_p_target.csv")

    print(f"Saved model: {out_model}")
    print(f"Saved thresholds: {outdir / 'thresholds.json'}")
    print(f"Saved spec: {outdir / 'spec.json'}")
    print(f"Saved test confusion matrices (CSV) in: {outdir}")

if __name__ == "__main__":
    main()
