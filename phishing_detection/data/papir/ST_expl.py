#!/usr/bin/env python3
"""
pro_explain_ig_robust.py — Explanations compatible with email_type_classifier.joblib

This script provides model-faithful explanations for the classifier artifact saved by
`email_type_classifier.py`. It supports both feature pipelines:

  1) TF‑IDF → Logistic Regression
     - Top-K n-grams per class (positive and negative).
     - Document-level contribution via linear surrogate (optional).

  2) SBERT (Sentence-Transformers) embeddings → Logistic Regression
     - Top-K *prototypical training emails* per class (requires the CSV to re-embed).
     - Token-level attributions for a single text using **Integrated Gradients** (IG)
       computed against the Transformer backbone that matches `model_name` in the bundle.

Usage examples:
  # Show top 50 n-grams for class "Phishing" (when model used TF-IDF fallback)
  python pro_explain_ig_robust.py --model email_type_classifier.joblib \
      --top-ngrams --class "Phishing" --k 50

  # Show top 50 prototypes (highest margins) for class using the CSV
  python pro_explain_ig_robust.py --model email_type_classifier.joblib \
      --prototypes --class "Phishing" --data Emails.csv --k 50

  # Token-level IG attributions for a single text (SBERT models)
  python pro_explain_ig_robust.py --model email_type_classifier.joblib \
      --ig --class "Phishing" --text "Invoice attached for September"

  # Save HTML with highlighted tokens
  python pro_explain_ig_robust.py --model email_type_classifier.joblib \
      --ig --class "Phishing" --text "..." --save-html ig_vis.html

Dependencies:
  pip install joblib numpy pandas scikit-learn transformers torch
  (If you use prototypes, also ensure pandas is available to load your CSV.)

Notes:
- For SBERT IG we approximate the Sentence-Transformers pooling with mean pooling
  of the last hidden states, then L2 normalize, which matches the common ST config
  (e.g., all-MiniLM-L6-v2). This yields logits comparable to the linear layer that
  was trained on normalized embeddings.
- Prototypes require your CSV (`--data`) because the original training texts are
  not stored inside the model artifact.
"""
from __future__ import annotations
import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import joblib

# -------------------------------------------------
# Unpickle compatibility shim for custom transformer
# -------------------------------------------------
# If the model was trained by running email_type_classifier.py as a script,
# the custom class `SentenceTransformerEncoder` may have been pickled under
# the module name `__main__`. When we load from a different script, pickle
# tries to resolve `__main__.SentenceTransformerEncoder` and fails.
#
# We fix this by providing a class with the same name in this module. The
# saved instance state (_st_model/_tfidf/_using_fallback/...) will be
# restored by joblib; we only need compatible methods.
try:
    from ST_trainer.py import SentenceTransformerEncoder  # use the original if importable
except Exception:
    from sklearn.base import BaseEstimator, TransformerMixin
    import warnings
    import numpy as np
    import pandas as pd

    class SentenceTransformerEncoder(BaseEstimator, TransformerMixin):  # type: ignore
        def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
            self.model_name = model_name
            self._st_model = None
            self._using_fallback = False
            self._tfidf = None
        def fit(self, X, y=None):  # not used during explain, but keep for API completeness
            warnings.warn("SentenceTransformerEncoder.fit called in explainer context; no-op.")
            return self
        def transform(self, X):
            # During unpickle, the fitted state is restored; just dispatch
            X = self._ensure_list(X)
            if not getattr(self, "_using_fallback", False) and getattr(self, "_st_model", None) is not None:
                return self._st_model.encode(X, normalize_embeddings=True, show_progress_bar=False)
            # Fallback: use restored TF-IDF vectorizer
            return self._tfidf.transform(X)
        @staticmethod
        def _ensure_list(X):
            if isinstance(X, (pd.Series, np.ndarray)):
                return X.tolist()
            return list(X)

# Optional deps for data/prototypes
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

# For IG on SBERT path
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception:  # pragma: no cover
    torch = None
    AutoTokenizer = None
    AutoModel = None

# ---------------------------------------------
# Utilities to load the saved bundle
# ---------------------------------------------

def load_bundle(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    bundle = joblib.load(model_path)
    # expected keys: pipeline, label_encoder, model_name, task, positive_label, ...
    return bundle


# ---------------------------------------------
# Path 1: TF-IDF explanations
# ---------------------------------------------

def is_tfidf_pipeline(bundle) -> bool:
    pipe = bundle["pipeline"]
    embed = getattr(pipe, "named_steps", {}).get("embed", None)
    return hasattr(embed, "get_feature_names_out")


def top_ngrams_for_class(bundle, class_name: str, k: int = 50) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    pipe = bundle["pipeline"]
    le = bundle["label_encoder"]
    clf = pipe.named_steps["clf"]
    vec = pipe.named_steps["embed"]

    if not hasattr(vec, "get_feature_names_out"):
        raise ValueError("Top n-grams only apply when the model used TF-IDF fallback.")

    classes = list(le.classes_)
    if class_name not in classes:
        raise ValueError(f"Unknown class '{class_name}'. Available: {classes}")
    kidx = classes.index(class_name)

    vocab = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[kidx]  # shape [V]

    top_pos_idx = np.argsort(coefs)[-k:][::-1]
    top_neg_idx = np.argsort(coefs)[:k]
    top_pos = [(vocab[i], float(coefs[i])) for i in top_pos_idx]
    top_neg = [(vocab[i], float(coefs[i])) for i in top_neg_idx]
    return top_pos, top_neg


# ---------------------------------------------
# Path 2: Prototypes (both pipelines)
# ---------------------------------------------
def read_dataset_robust(csv_path: str, sep: str = None, encoding: str = None):
    """
    Robust CSV reader:
      - Tries provided sep/encoding if given.
      - Otherwise tries multiple encodings, engines, and delimiters.
      - Skips bad lines when using the python engine.
    """
    import pandas as pd
    # If user forces settings, honor them first
    if sep is not None or encoding is not None:
        return pd.read_csv(csv_path, sep=sep, encoding=encoding, engine="python", on_bad_lines="skip")

    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    delimiters = [None, ",", ";", "\t", "|"]  # None => sniff with python engine
    engines = ["c", "python"]

    last_err = None
    for enc in encodings:
        for eng in engines:
            for d in delimiters:
                try:
                    kwargs = {"encoding": enc, "engine": eng}
                    if eng == "python":
                        kwargs["on_bad_lines"] = "skip"
                        kwargs["sep"] = d  # allow sniff (None) or explicit
                    else:
                        if d is not None:
                            kwargs["sep"] = d
                    df = pd.read_csv(csv_path, **kwargs)
                    # heuristic: if only 1 column with common sep, keep searching
                    if df.shape[1] == 1 and d in (",", ";"):
                        continue
                    return df
                except Exception as e:
                    last_err = e
                    continue
    raise last_err if last_err else ValueError("Failed to read CSV")

def require_pandas():
    if pd is None:
        raise RuntimeError("pandas is required for prototype explanations.")


def combine_text_columns(df: 'pd.DataFrame') -> 'pd.Series':
    """Light reimplementation (aligned with email_type_classifier) for stand-alone use."""
    # Prefer common email fields
    preferred = ["Subject", "Body", "Sender", "Preview", "Text", "Content"]
    present = [c for c in preferred if c in df.columns]
    if not present:
        blacklist = {"type", "label", "target", "id", "no.", "year", "source", "created by", "url(s)", "file"}
        present = [c for c in df.columns if c.lower() not in blacklist]
    parts = [df[c].astype(str).fillna("") for c in present]
    if not parts:
        return pd.Series(["" for _ in range(len(df))], index=df.index)
    s = pd.Series([" ".join(t).strip() for t in zip(*parts)], index=df.index)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def top_prototypes(bundle, data_path: str, class_name: str, k: int = 50, sep: str = None, encoding: str = None):
    """Return top-k training examples by class margin w_k^T x + b.
    Requires the original CSV to rebuild texts. Works for both TF-IDF and SBERT.
    """
    df = read_dataset_robust(data_path, sep=sep, encoding=encoding)
    if "Type" not in df.columns:
        raise ValueError("CSV must contain a 'Type' column.")
    if "Type" not in df.columns:
        raise ValueError("CSV must contain a 'Type' column.")

    X = combine_text_columns(df)
    y = df["Type"].astype(str)

    pipe = bundle["pipeline"]
    le = bundle["label_encoder"]

    if class_name not in le.classes_:
        raise ValueError(f"Unknown class '{class_name}'. Available: {list(le.classes_)}")
    kidx = int(np.where(le.classes_ == class_name)[0][0])

    # decision_function gives margins for LR; for some solvers, .predict_proba is also fine
    try:
        Z = pipe.decision_function(X)
        margins = Z[:, kidx]
    except Exception:
        P = pipe.predict_proba(X)
        margins = P[:, kidx]

    idx = np.argsort(margins)[-k:][::-1]
    rows = []
    for i in idx:
        rows.append({
            "index": int(i),
            "Type": y.iloc[i],
            "margin": float(margins[i]),
            "text": X.iloc[i],
        })
    return rows


# ---------------------------------------------
# Integrated Gradients for SBERT + LR
# ---------------------------------------------

def is_sbert_pipeline(bundle) -> bool:
    return not is_tfidf_pipeline(bundle)


def _mean_pool(last_hidden: 'torch.Tensor', attention_mask: 'torch.Tensor') -> 'torch.Tensor':
    # last_hidden: [B, T, H], mask: [B, T]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # [B,T,1]
    summed = (last_hidden * mask).sum(dim=1)                   # [B,H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                  # [B,1]
    return summed / counts


def _l2_normalize(x: 'torch.Tensor', eps: float = 1e-12) -> 'torch.Tensor':
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def integrated_gradients_tokens(bundle, text: str, class_name: str, steps: int = 50):
    """Compute token-level attributions via IG for SBERT pipelines.

    We reconstruct the embedding using HF transformers (AutoModel) with
    mean pooling over last_hidden_state and L2 normalization, then apply
    the scikit-learn LR weights to get the class logit.
    """
    if torch is None:
        raise RuntimeError("torch/transformers are required for IG.")

    model_name = bundle.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name)
    enc.eval()

    pipe = bundle["pipeline"]
    le = bundle["label_encoder"]
    clf = pipe.named_steps["clf"]

    if class_name not in le.classes_:
        raise ValueError(f"Unknown class '{class_name}'. Available: {list(le.classes_)}")
    kidx = int(np.where(le.classes_ == class_name)[0][0])

    with torch.no_grad():
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
        out = enc(**inputs)
        pooled = _mean_pool(out.last_hidden_state, inputs["attention_mask"])  # [1,H]
        z = _l2_normalize(pooled)                                             # [1,H]
        # Quick check: this path should produce a vector compatible with LR

    # Prepare linear layer from scikit model
    W = torch.tensor(clf.coef_, dtype=torch.float32)  # [K,H]
    b = torch.tensor(clf.intercept_, dtype=torch.float32)  # [K]

    # IG over input embeddings (path from baseline 0 to actual embeddings)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]  # [1,T]
    attention_mask = inputs["attention_mask"]

    # Get token embeddings
    emb_layer = enc.get_input_embeddings()  # nn.Embedding
    # Build embeddings and a zero baseline
    with torch.no_grad():
        emb = emb_layer(input_ids)  # [1,T,H]
    baseline = torch.zeros_like(emb)

    # We integrate on the embedding space and backprop to get attributions per token
    total_grads = torch.zeros_like(emb)

    for s in range(1, steps + 1):
        alpha = s / steps
        emb_alpha = baseline + alpha * (emb - baseline)  # [1,T,H]
        emb_alpha.requires_grad_(True)

        # Forward: replace input embeddings via hooks
        def inputs_embeds_forward(**kwargs):
            return enc(inputs_embeds=emb_alpha, attention_mask=attention_mask)

        out_alpha = inputs_embeds_forward()
        pooled_alpha = _mean_pool(out_alpha.last_hidden_state, attention_mask)
        z_alpha = _l2_normalize(pooled_alpha)  # [1,H]
        logit_k = (W[kidx] @ z_alpha.squeeze(0)) + b[kidx]
        logit_k.backward()

        total_grads += emb_alpha.grad.detach()
        enc.zero_grad()

    # Average gradient along path, multiply by input diff per IG definition
    avg_grads = total_grads / steps  # [1,T,H]
    ig = (emb - baseline) * avg_grads  # [1,T,H]
    # Aggregate per token by L2 norm across hidden dim
    token_importance = ig.norm(dim=-1).squeeze(0)  # [T]

    # Map to tokens (avoid special tokens in ranking but keep for alignment)
    tokens = tok.convert_ids_to_tokens(input_ids.squeeze(0))
    # Rank by importance (skip CLS/SEP/PAD)
    skip = set([tok.cls_token, tok.sep_token, tok.pad_token])
    ranked = [
        (i, t, float(token_importance[i]))
        for i, t in enumerate(tokens)
        if t not in skip and attention_mask[0, i].item() == 1
    ]
    ranked.sort(key=lambda x: x[2], reverse=True)

    return {
        "tokens": tokens,
        "importances": token_importance.tolist(),
        "ranked": ranked,
    }
# ---------- IG corpus aggregation (SBERT models) ----------
def ig_corpus_top_tokens(
    bundle,
    data_path: str,
    class_name: str,
    k: int = 50,
    steps: int = 24,
    limit: int = None,
    sample_p: float = 1.0,
    predicted_only: bool = True,
    prob_thresh: float = 0.5,
    min_len: int = 2,
    strip_stop: bool = True,
    save_csv: str = None,
):
    """
    Aggregate token-level IG across many emails to get a global Top-N list.

    Parameters
    ----------
    bundle : joblib-loaded dict from email_type_classifier.joblib
    data_path : CSV file with emails
    class_name : class to explain (e.g., "Phishing")
    k : how many tokens to return
    steps : IG integration steps (reduce for speed)
    limit : cap the number of emails processed (int). If None, process all (or sampled)
    sample_p : randomly sample this fraction of eligible emails (0<sample_p<=1)
    predicted_only : if True, compute IG only on emails predicted as class_name
    prob_thresh : probability threshold for predicted_only filter
    min_len : min token length to keep after normalization
    strip_stop : drop common stopwords
    save_csv : optional path to write a CSV with [token,total_ig,count,mean_ig]

    Returns
    -------
    List of tuples: (token, total_ig, count, mean_ig) sorted by total_ig desc.
    """
    import numpy as np
    import pandas as pd
    import math

    # Need torch/transformers for IG and pandas for the dataset
    try:
        import torch  # noqa
        from transformers import AutoTokenizer  # noqa
    except Exception:
        raise RuntimeError("torch/transformers are required for IG corpus aggregation.")
    if pd is None:
        raise RuntimeError("pandas is required for IG corpus aggregation.")

    # Robust load (reuse your classifier's robust reader if available)
    try:
        df = read_dataset_robust(data_path)  # if you added it earlier
    except Exception:
        df = pd.read_csv(data_path, engine="python", on_bad_lines="skip")

    # Rebuild the same text field your classifier used
    X = combine_text_columns(df)

    pipe = bundle["pipeline"]
    le = bundle["label_encoder"]
    classes = list(le.classes_)
    if class_name not in classes:
        raise ValueError(f"Unknown class '{class_name}'. Available: {classes}")
    kidx = int(np.where(le.classes_ == class_name)[0][0])

    # Filter to predicted positives for quality and speed
    idx_all = np.arange(len(X))
    if predicted_only:
        try:
            probs = pipe.predict_proba(X)[:, kidx]
            mask = probs >= prob_thresh
            idx_all = np.where(mask)[0]
        except Exception:
            pass  # fallback: use all

    # Sample/limit
    rng = np.random.default_rng(42)
    if sample_p < 1.0:
        n = int(max(1, math.floor(len(idx_all) * sample_p)))
        idx_all = rng.choice(idx_all, size=n, replace=False)
    if limit is not None:
        idx_all = idx_all[:limit]

    # Stoplist
    stop = set()
    if strip_stop:
        stop.update({
            "the","a","an","and","or","to","of","in","for","on","at","with",
            "from","by","as","is","are","be","this","that","it","you","your",
            "we","our","us","re","fw","fwd","de"
        })

    agg = {}
    cnt = {}

    for i in idx_all:
        txt = str(X.iloc[i])
        res = integrated_gradients_tokens(bundle, txt, class_name, steps=steps)
        toks = res.get("norm_tokens") or res["tokens"]  # prefer normalized if available
        imps = res["importances"]
        for t, v in zip(toks, imps):
            t_norm = (t or "").replace("▁","").replace("##","").lstrip("Ġ#").lower()
            if not t_norm or len(t_norm) < min_len:
                continue
            if strip_stop and t_norm in stop:
                continue
            val = abs(float(v))
            agg[t_norm] = agg.get(t_norm, 0.0) + val
            cnt[t_norm] = cnt.get(t_norm, 0) + 1

    if not agg:
        return []

    items = [(t, agg[t], cnt[t], agg[t] / max(1, cnt[t])) for t in agg.keys()]
    # Sort by total IG, then mean IG
    items.sort(key=lambda x: (x[1], x[3]), reverse=True)
    top = items[:k]

    if save_csv:
        pd.DataFrame(top, columns=["token","total_ig","count","mean_ig"]).to_csv(save_csv, index=False)

    return top


def render_html_highlight(text: str, ranked: List[Tuple[int, str, float]], top_k: int = 50, cmap=(255, 0, 0)) -> str:
    """Render a simple HTML highlight using the tokenizer's whitespace split as a proxy.
    For SBERT WordPiece/BPE tokens, we join subwords when possible, but for a quick
    visualization we simply color the top_k ranked tokens by intensity.
    """
    # Normalize scores 0..1 for top_k tokens
    top = ranked[:top_k]
    if not top:
        return f"<p>{text}</p>"
    maxv = max(v for _, _, v in top) or 1.0
    # Mark token indices to color
    idx2alpha = {i: v / maxv for i, _, v in top}

    # Very simple whitespace tokenization to display; this will not perfectly
    # align with subword tokens, but provides a readable heatmap.
    import html
    words = text.split()
    # Color the whole text proportionally by distributing top token scores nearby
    # (fallback simplistic mapping)
    colored = []
    for w in words:
        score = 0.0
        # naive: if a ranked token substring is in w, accumulate (best-effort)
        for _, tok, v in top:
            if tok.strip("#▁") and tok.strip("#▁").lower() in w.lower():
                score = max(score, v / maxv)
        r, g, b = cmap
        alpha = score
        style = f"background-color: rgba({r},{g},{b},{alpha:.2f}); padding:2px; border-radius:4px;"
        colored.append(f"<span style=\"{style}\">{html.escape(w)}</span>")
    return "<p>" + " ".join(colored) + "</p>"


# ---------------------------------------------
# CLI
# ---------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Explanations for email_type_classifier models")
    ap.add_argument("--model", required=True, help="Path to email_type_classifier.joblib")
    ap.add_argument("--class", dest="class_name", required=False, help="Class name to explain")
    ap.add_argument("--k", type=int, default=50, help="Top-K to display")
    ap.add_argument("--sep", type=str, default=None, help="Force delimiter: ',', ';', '\\t', '|'")
    ap.add_argument("--encoding", type=str, default=None, help="Force encoding, e.g. 'utf-8', 'latin-1'")


    # Modes
    ap.add_argument("--top-ngrams", action="store_true", help="Show top-K n-grams for a class (TF-IDF models only)")
    ap.add_argument("--prototypes", action="store_true", help="Show top-K prototypical training emails for a class")
    ap.add_argument("--data", help="CSV with emails (needed for prototypes)")

    ap.add_argument("--ig", action="store_true", help="Run Integrated Gradients token attribution for a single text (SBERT models)")
    ap.add_argument("--text", type=str, help="Text to explain with IG")
    ap.add_argument("--save-html", type=str, default=None, help="Optional path to save a simple HTML heatmap for IG")
    ap.add_argument("--ig-corpus", action="store_true",
                help="Aggregate IG across many emails to get a top-N token list (SBERT only)")
    
    ap.add_argument("--steps", type=int, default=24, help="IG steps (lower is faster)")
    ap.add_argument("--limit", type=int, default=None, help="Cap number of emails processed")
    ap.add_argument("--sample-p", type=float, default=1.0, help="Randomly sample this fraction of eligible emails")
    ap.add_argument("--predicted-only", action="store_true", help="Only run IG on emails predicted as the target class")
    ap.add_argument("--prob-thresh", type=float, default=0.5, help="Probability threshold for predicted-only filtering")
    ap.add_argument("--min-len", type=int, default=2, help="Minimum token length to keep")
    ap.add_argument("--no-stop", action="store_true", help="Do not strip stopwords")
    ap.add_argument("--save-csv", type=str, default=None, help="Where to save the aggregated top-N as CSV")

    args = ap.parse_args()

    bundle = load_bundle(args.model)

    if args.ig_corpus:
        if not is_sbert_pipeline(bundle):
            raise SystemExit("IG corpus aggregation is only available for SBERT-based models.")
        if not args.class_name:
            raise SystemExit("--class is required with --ig-corpus")
        if not args.data:
            raise SystemExit("--data CSV is required with --ig-corpus")

    top = ig_corpus_top_tokens(
        bundle=bundle,
        data_path=args.data,
        class_name=args.class_name,
        k=args.k,
        steps=args.steps,
        limit=args.limit,
        sample_p=args.sample_p,
        predicted_only=args.predicted_only,
        prob_thresh=args.prob_thresh,
        min_len=args.min_len,
        strip_stop=(not args.no_stop),
        save_csv=args.save_csv,
    )
    print(f"\nTop {args.k} tokens by aggregated IG for class = {args.class_name}")
    print("(Columns: token  total_ig  count  mean_ig)")
    for t, tot, c, mean in top:
        print(f"{t}\t{tot:.6f}\t{c}\t{mean:.6f}")
    return


    if args.top_ngrams:
        if not args.class_name:
            raise SystemExit("--class is required with --top-ngrams")
        pos, neg = top_ngrams_for_class(bundle, args.class_name, k=args.k)
        print(f"\n[Top +{args.k}] features for class = {args.class_name}")
        for f, w in pos:
            print(f"  + {f}: {w:.4f}")
        print(f"\n[Top -{args.k}] features (push away) for class = {args.class_name}")
        for f, w in neg:
            print(f"  - {f}: {w:.4f}")
        return

    if args.prototypes:
        if not args.class_name:
            raise SystemExit("--class is required with --prototypes")
        if not args.data:
            raise SystemExit("--data CSV is required with --prototypes")
        rows = top_prototypes(bundle, args.data, args.class_name, k=args.k,
                      sep=args.sep, encoding=args.encoding)

        print(f"\nTop {args.k} prototypes for class = {args.class_name}")
        for r in rows:
            text = r["text"]
            preview = (text[:160] + ("..." if len(text) > 160 else ""))
            print(f"  [margin {r['margin']:.4f}] {preview}")
        return

    if args.ig:
        if not is_sbert_pipeline(bundle):
            raise SystemExit("IG is only available for SBERT-based models (not TF-IDF).")
        if not args.class_name:
            raise SystemExit("--class is required with --ig")
        if not args.text:
            raise SystemExit("--text is required with --ig")
        res = integrated_gradients_tokens(bundle, args.text, args.class_name, steps=50)
        print(f"\nTop {args.k} token attributions for class = {args.class_name}")
        for i, tok, score in res["ranked"][: args.k]:
            print(f"  {tok}\t{score:.6f}")
        if args.save_html:
            html = render_html_highlight(args.text, res["ranked"], top_k=args.k)
            with open(args.save_html, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"Saved HTML visualization to: {args.save_html}")
        return

    # If no mode selected
    ap.print_help()


if __name__ == "__main__":
    main()
