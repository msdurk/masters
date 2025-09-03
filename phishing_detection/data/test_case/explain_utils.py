"""
run from data:
python test_case/explain_utils.py --model test_case/artifacts_pro/model.joblib --global-top --top-n 50
"""

import math
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy import sparse  # noqa: F401

# If using the pro pipeline, this import will succeed; otherwise it's fine too.
try:
    from components import TextStatsTransformer  # noqa: F401
    HAS_STATS = True
except Exception:
    HAS_STATS = False

# Stats feature names must match TextStatsTransformer.transform order
STATS_FEATURE_NAMES = [
    "url_count_log", "html_tags_log", "suspicious_url_tld_log", "exclamations_log",
    "upper_ratio", "digit_ratio", "money_symbols_log", "at_count_log", "urgent_hits_log", "length_log"
]

def _safe_names_for_transformer(name, transformer):
    """Return feature names for a single transformer within a FeatureUnion block."""
    # 1) Try the normal API if present
    if hasattr(transformer, "get_feature_names_out"):
        try:
            out = transformer.get_feature_names_out()
            return np.asarray(out, dtype=np.str_)  # force uniform string dtype
        except Exception:
            # e.g., Pipeline(..., StandardScaler()) raises because last step has no names
            pass

    # 2) Special-case the 'stats' block (Pipeline of TextStatsTransformer -> StandardScaler)
    if name == "stats" and HAS_STATS:
        return np.asarray(STATS_FEATURE_NAMES, dtype=np.str_)

    # 3) Last resort: try to infer the dimension by transforming a dummy string
    try:
        X = transformer.transform(["dummy"])
        dim = X.shape[1]
        return np.asarray([f"{name}:{i}" for i in range(dim)], dtype=np.str_)
    except Exception:
        # Give up: empty (shouldn’t happen in practice)
        return np.asarray([], dtype=np.str_)

def _get_block_and_names(pipe):
    """Return a list of (block_name, feature_names) matching FeatureUnion order."""
    feat = pipe.named_steps["features"]  # FeatureUnion
    out = []
    for name, transformer in feat.transformer_list:
        names = _safe_names_for_transformer(name, transformer)
        # Prefix with the block name; use list comprehension to avoid np.char.add dtype issues
        prefixed = np.asarray([f"{name}:{s}" for s in names], dtype=np.str_)
        out.append((name, prefixed))
    return out

def global_top_features(model_path, top_n=30):
    """Show top + and - features globally (toward 'phishing' vs 'legit')."""
    pipe = joblib.load(model_path)
    clf = pipe.named_steps["clf"]

    classes = list(clf.classes_)
    if len(classes) != 2:
        raise ValueError("This script assumes binary classification.")
    # Orient weights so positive always means 'phishing'
    w = clf.coef_[0] if classes[1] == "phishing" else -clf.coef_[0]

    blocks = _get_block_and_names(pipe)
    feat_names = np.concatenate([n for _, n in blocks], axis=0)

    # Safety: align lengths if something odd happens
    D = min(len(feat_names), w.shape[0])
    feat_names = feat_names[:D]
    w = w[:D]

    coefs = pd.DataFrame({
        "feature": feat_names,
        "weight_towards_phishing": w
    })
    coefs["abs_weight"] = coefs["weight_towards_phishing"].abs()

    top_pos = coefs.sort_values("weight_towards_phishing", ascending=False).head(top_n)
    top_neg = coefs.sort_values("weight_towards_phishing", ascending=True).head(top_n)

    print("\n=== Top features pushing TOWARD 'phishing' ===")
    print(top_pos[["feature","weight_towards_phishing"]].to_string(index=False))

    print("\n=== Top features pushing TOWARD 'legit' ===")
    print(top_neg[["feature","weight_towards_phishing"]].to_string(index=False))

    return top_pos, top_neg

def explain_text(model_path, text, top_k=15):
    """Per-email explanation: list features with biggest contributions."""
    pipe = joblib.load(model_path)
    clf = pipe.named_steps["clf"]
    feat = pipe.named_steps["features"]  # FeatureUnion

    classes = list(clf.classes_)
    if classes[1] == "phishing":
        w = clf.coef_[0]; b = clf.intercept_[0]
    else:
        w = -clf.coef_[0]; b = -clf.intercept_[0]

    blocks = _get_block_and_names(pipe)
    feat_names = np.concatenate([n for _, n in blocks], axis=0)

    # Vectorize one example
    X = feat.transform([text])  # 1 x D sparse
    row = X.tocoo()

    # Safety: align lengths
    D = min(len(feat_names), w.shape[0])
    feat_names = feat_names[:D]
    # If X has more columns than w (shouldn't), drop extras
    if X.shape[1] > D:
        X = X[:, :D]
        row = X.tocoo()

    # Contributions to the log-odds = value * weight
    contrib = {}
    for j, v in zip(row.col, row.data):
        contrib[int(j)] = (v * w[j], v, w[j])

    margin = (row.data @ w[row.col]) + b
    prob_phish = 1.0 / (1.0 + math.exp(-margin))
    pred = "phishing" if prob_phish >= 0.5 else "legit"

    items = [(feat_names[j], c, v, wj) for j, (c, v, wj) in contrib.items()]
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    top = items[:top_k]

    print(f"\nText: {text[:120].replace('\\n',' ')}{'...' if len(text)>120 else ''}")
    print(f"Decision margin (toward 'phishing'): {margin:.4f}")
    print(f"Probability('phishing'): {prob_phish:.4f} | Pred: {pred}")
    print("\nTop feature contributions (name, contribution=value*weight, value, weight_towards_phishing):")
    for name, c, v, wj in top:
        print(f"  {name:40s}  contrib={c:+.4f}   value={v:.4f}   weight={wj:+.4f}")

    return {
        "prob_phishing": float(prob_phish),
        "margin": float(margin),
        "pred": pred,
        "top_contributions": top
    }

# Optional CLI
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts_pro/model.joblib")
    ap.add_argument("--text", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--global-top", action="store_true", help="Print global top features")
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--top-k", type=int, default=15)
    args = ap.parse_args()

    if args.global_top:
        global_top_features(args.model, top_n=args.top_n)

    if args.text:
        explain_text(args.model, args.text, top_k=args.top_k)

    if args.csv:
        df = pd.read_csv(args.csv)
        col = args.text_col or ("text" if "text" in df.columns else df.columns[0])
        for i, t in enumerate(df[col].astype(str).head(10)):
            print(f"\n=== Row {i} ===")
            explain_text(args.model, t, top_k=args.top_k)
