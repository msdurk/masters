#!/usr/bin/env python3
"""
pro_explain_ig_robust.py — Explanations compatible with email_type_classifier.joblib

This script provides model-faithful explanations for the classifier artifact saved by
`email_type_classifier.py`. It supports both feature pipelines:

  1) TF-IDF → Logistic Regression
     - Top-K n-grams per class (positive and negative).
     - Document-level contribution via linear surrogate (optional).

  2) SBERT (Sentence-Transformers) embeddings → Logistic Regression
     - Top-K *prototypical training emails* per class (requires the CSV to re-embed).
     - Token/word-level attributions for a single text using **Integrated Gradients** (IG).
     - **Corpus IG aggregation** to get a global Top-N token list.

Usage examples:
  # Top 50 n-grams for a class (TF-IDF fallback models only)
  python pro_explain_ig_robust.py --model email_type_classifier.joblib \
      --top-ngrams --class "Phishing" --k 50

  # Top 50 prototypes (highest margins) for a class using the CSV
  python pro_explain_ig_robust.py --model email_type_classifier.joblib \
      --prototypes --class "Phishing" --data Emails.csv --k 50

  # Token/word-level IG attributions for a single text (SBERT models)
  python pro_explain_ig_robust.py --model email_type_classifier.joblib \
      --ig --class "Phishing" --text "Invoice attached for September" --k 20 --save-html ig_vis.html

  # Aggregate IG across the dataset to get a Top-N token list (SBERT models)
  python pro_explain_ig_robust.py --model email_type_classifier.joblib \
      --ig-corpus --class "Phishing" --data Emails.csv --k 50 --steps 16 --predicted-only --prob-thresh 0.7 --save-csv ig_top_tokens.csv

Dependencies:
  pip install joblib numpy pandas scikit-learn transformers torch

Notes:
- For SBERT IG we approximate Sentence-Transformers pooling with mean pooling of the
  last hidden states and then L2 normalize, which matches common ST configs
  (e.g., all-MiniLM-L6-v2). This yields logits compatible with the linear head.
- Prototypes require your CSV (`--data`) because training texts aren’t stored in the artifact.
"""
from __future__ import annotations
import argparse
import os
from typing import List, Optional, Tuple
from typing import Tuple
import numpy as np
import joblib

# -------------------------------------------------
# Unpickle compatibility shim for custom transformer
# -------------------------------------------------
# If the model was trained by running email_type_classifier.py as a script,
# the custom class `SentenceTransformerEncoder` may have been pickled under
# the module name `__main__`. When we load from a different script, pickle
# tries to resolve `__main__.SentenceTransformerEncoder` and fails.
try:
    from email_type_classifier import SentenceTransformerEncoder  # use the original if importable
except Exception:
    from sklearn.base import BaseEstimator, TransformerMixin
    import warnings
    import numpy as _np
    import pandas as _pd

    class SentenceTransformerEncoder(BaseEstimator, TransformerMixin):  # type: ignore
        def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
            self.model_name = model_name
            self._st_model = None
            self._using_fallback = False
            self._tfidf = None
        def fit(self, X, y=None):
            warnings.warn("SentenceTransformerEncoder.fit called in explainer context; no-op.")
            return self
        def transform(self, X):
            X = self._ensure_list(X)
            if not getattr(self, "_using_fallback", False) and getattr(self, "_st_model", None) is not None:
                return self._st_model.encode(X, normalize_embeddings=True, show_progress_bar=False)
            return self._tfidf.transform(X)
        @staticmethod
        def _ensure_list(X):
            if isinstance(X, (_pd.Series, _np.ndarray)):
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
# Utilities to load the saved bundle & robust CSV
# ---------------------------------------------

def load_bundle(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    bundle = joblib.load(model_path)
    return bundle

def read_dataset_robust(csv_path: str, sep: str | None = None, encoding: str | None = None):
    """Robust CSV reader with delimiter/encoding fallbacks and bad-line skipping."""
    if pd is None:
        raise RuntimeError("pandas is required to read CSV files.")
    # honor explicit args first
    if sep is not None or encoding is not None:
        return pd.read_csv(csv_path, sep=sep, encoding=encoding, engine="python", on_bad_lines="skip")
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    delimiters = [None, ",", ";", "\t", "|"]  # None => sniff in python engine
    engines = ["c", "python"]
    last_err = None
    for enc in encodings:
        for eng in engines:
            for d in delimiters:
                try:
                    kw = {"encoding": enc, "engine": eng}
                    if eng == "python":
                        kw["on_bad_lines"] = "skip"
                        kw["sep"] = d
                    elif d is not None:
                        kw["sep"] = d
                    df = pd.read_csv(csv_path, **kw)
                    if df.shape[1] == 1 and d in (",", ";"):
                        continue
                    return df
                except Exception as e:
                    last_err = e
                    continue
    raise last_err if last_err else ValueError("Failed to read CSV")

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

def require_pandas():
    if pd is None:
        raise RuntimeError("pandas is required for prototype explanations.")

def combine_text_columns(df: 'pd.DataFrame') -> 'pd.Series':
    """Light reimplementation (aligned with email_type_classifier) for stand-alone use."""
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

def top_prototypes(bundle, data_path: str, class_name: str, k: int = 50, sep: str | None = None, encoding: str | None = None):
    """Return top-k training examples by class margin w_k^T x + b.
    Requires the original CSV to rebuild texts. Works for both TF-IDF and SBERT.
    """
    require_pandas()
    df = read_dataset_robust(data_path, sep=sep, encoding=encoding)
    if "Type" not in df.columns:
        raise ValueError("CSV must contain a 'Type' column.")

    X = combine_text_columns(df)
    y = df["Type"].astype(str)

    pipe = bundle["pipeline"]
    le = bundle["label_encoder"]

    if class_name not in le.classes_:
        raise ValueError(f"Unknown class '{class_name}'. Available: {list(le.classes_)}")
    kidx = int(np.where(le.classes_ == class_name)[0][0])

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
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # [B,T,1]
    summed = (last_hidden * mask).sum(dim=1)                   # [B,H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                  # [B,1]
    return summed / counts

def _l2_normalize(x: 'torch.Tensor', eps: float = 1e-12) -> 'torch.Tensor':
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def integrated_gradients_tokens(bundle, text: str, class_name: str, steps: int = 50):
    """Compute token-level IG for SBERT pipelines, returning signed attributions and encoding."""
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

    # include offsets for later word aggregation
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)

    with torch.no_grad():
        out = enc(**{k: v for k, v in inputs.items() if k != "offset_mapping"})
        pooled = _mean_pool(out.last_hidden_state, inputs["attention_mask"])
        _ = _l2_normalize(pooled)

    W = torch.tensor(clf.coef_, dtype=torch.float32)
    b = torch.tensor(clf.intercept_, dtype=torch.float32)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    emb_layer = enc.get_input_embeddings()
    with torch.no_grad():
        emb = emb_layer(input_ids)
    baseline = torch.zeros_like(emb)

    total_grads = torch.zeros_like(emb)

    for s in range(1, steps + 1):
        alpha = s / steps
        emb_alpha = baseline + alpha * (emb - baseline)
        emb_alpha.requires_grad_(True)

        out_alpha = enc(inputs_embeds=emb_alpha, attention_mask=attention_mask)
        pooled_alpha = _mean_pool(out_alpha.last_hidden_state, attention_mask)
        z_alpha = _l2_normalize(pooled_alpha)
        logit_k = (W[kidx] @ z_alpha.squeeze(0)) + b[kidx]
        logit_k.backward()

        total_grads += emb_alpha.grad.detach()
        enc.zero_grad()

    avg_grads = total_grads / steps
    ig = (emb - baseline) * avg_grads

    # Unsigned (L2) and signed (sum) per token
    token_importance_unsigned = ig.norm(dim=-1).squeeze(0)     # [T]
    token_importance_signed   = ig.sum(dim=-1).squeeze(0)      # [T]

    tokens = tok.convert_ids_to_tokens(input_ids.squeeze(0))
    skip = set([tok.cls_token, tok.sep_token, tok.pad_token])

    ranked = [
        (i, t, float(abs(token_importance_signed[i])))
        for i, t in enumerate(tokens)
        if t not in skip and attention_mask[0, i].item() == 1
    ]
    ranked.sort(key=lambda x: x[2], reverse=True)

    return {
        "tokens": tokens,
        "importances": token_importance_unsigned.tolist(),
        "importances_signed": token_importance_signed.tolist(),
        "ranked": ranked,
        "encoding": inputs,  # contains offset_mapping and word ids
    }


from typing import Tuple  # if not already imported

def aggregate_word_level(text: str, ig_result: dict, tokenizer) -> List[Tuple[str, float, float, int]]:
    """
    Merge subword IG to word-level using tokenizer word_ids/offsets.
    Returns (word_text, total_abs_ig, total_signed_ig, token_count), sorted by total_abs_ig desc.
    """
    tokens = ig_result["tokens"]
    signed = ig_result["importances_signed"]
    enc = ig_result.get("encoding")

    try:
        word_ids = enc.word_ids()
    except Exception:
        try:
            word_ids = enc.word_ids(batch_index=0)
        except Exception:
            word_ids = list(range(len(tokens)))

    offsets = enc.get("offset_mapping", None)
    if offsets is not None:
        offsets = offsets[0].tolist()

    by_word = {}
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        s = float(signed[i])
        a = abs(s)
        w = by_word.setdefault(wid, {"abs": 0.0, "signed": 0.0, "count": 0, "start": 10**9, "end": -1})
        w["abs"] += a
        w["signed"] += s
        w["count"] += 1
        if offsets:
            start, end = offsets[i]
            w["start"] = min(w["start"], start)
            w["end"]   = max(w["end"], end)

    rows = []
    for wid, stats in by_word.items():
        if offsets:
            start, end = stats["start"], stats["end"]
            word_text = text[start:end]
        else:
            subtoks = [tokens[i] for i, w in enumerate(word_ids) if w == wid]
            word_text = "".join(t.replace("##","").lstrip("▁").lstrip("Ġ") for t in subtoks)
        rows.append((word_text, stats["abs"], stats["signed"], stats["count"]))

    rows.sort(key=lambda r: (r[1], abs(r[2])), reverse=True)
    return rows

def split_signed_lists(items: List[Tuple[str, float, float, int]], k: int = 50):
    """Return top-K TOWARD (positive signed) and top-K AWAY (negative signed)."""
    toward = [it for it in items if it[2] > 0]
    away   = [it for it in items if it[2] < 0]
    toward.sort(key=lambda r: r[2], reverse=True)
    away.sort(key=lambda r: r[2])  # most negative first
    return toward[:k], away[:k]

def print_signed(name: str, items: List[Tuple[str, float, float, int]]):
    print(f"\n{name} (name\ttotal_abs_ig\ttotal_signed_ig\tcount)")
    for w, tot_abs, tot_signed, c in items:
        print(f"{w}\t{tot_abs:.6f}\t{tot_signed:.6f}\t{c}")
def ig_corpus_top_words_signed(
    bundle,
    data_path: str,
    class_name: str,
    k: int = 50,
    steps: int = 24,
    limit: Optional[int] = None,
    sample_p: float = 1.0,
    predicted_only: bool = True,
    prob_thresh: float = 0.5,
    min_len: int = 2,
    strip_stop: bool = True,
    save_csv_toward: Optional[str] = None,
    save_csv_away: Optional[str] = None,
):
    """Aggregate word-level signed IG across many emails. Returns (top_toward, top_away)."""
    if torch is None:
        raise RuntimeError("torch/transformers are required for IG corpus aggregation.")
    if pd is None:
        raise RuntimeError("pandas is required for IG corpus aggregation.")

    # Robust CSV read (reuse pd.read_csv if you prefer)
    try:
        df = pd.read_csv(data_path, engine="python", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(data_path)

    # Rebuild text column (reuse your existing helper)
    if "Subject" in df.columns or "Body" in df.columns:
        X = combine_text_columns(df)
    else:
        # fallback: assume a single text column
        X = df.iloc[:, 0].astype(str)

    pipe = bundle["pipeline"]
    le = bundle["label_encoder"]
    classes = list(le.classes_)
    if class_name not in classes:
        raise ValueError(f"Unknown class '{class_name}'. Available: {classes}")
    kidx = int(np.where(le.classes_ == class_name)[0][0])

    # filter to predicted positives if requested
    idx_all = np.arange(len(X))
    if predicted_only:
        try:
            probs = pipe.predict_proba(X)[:, kidx]
            mask = probs >= prob_thresh
            idx_all = np.where(mask)[0]
        except Exception:
            pass

    rng = np.random.default_rng(42)
    if sample_p < 1.0:
        n = int(max(1, np.floor(len(idx_all) * sample_p)))
        idx_all = rng.choice(idx_all, size=n, replace=False)
    if limit is not None:
        idx_all = idx_all[:limit]

    model_name = bundle.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    tok = AutoTokenizer.from_pretrained(model_name)

    # Stoplist
    stop = set()
    if strip_stop:
        stop.update({'the','a','an','and','or','to','of','in','for','on','at','with',
                     'from','by','as','is','are','be','this','that','it','you','your',
                     'we','our','us','re','fw','fwd','de'})

    # Global word-level sums
    agg_abs = {}     # word -> total |IG|
    agg_signed = {}  # word -> total signed IG
    counts = {}      # word -> occurrences (subtokens merged)

    for i in idx_all:
        txt = str(X.iloc[i])
        ig = integrated_gradients_tokens(bundle, txt, class_name, steps=steps)
        # merge to words
        words = aggregate_word_level(txt, ig, tok)
        for w, tot_abs, tot_signed, cnt in words:
            w_norm = w.strip().lower()
            if len(w_norm) < min_len:
                continue
            if strip_stop and w_norm in stop:
                continue
            agg_abs[w_norm] = agg_abs.get(w_norm, 0.0) + float(tot_abs)
            agg_signed[w_norm] = agg_signed.get(w_norm, 0.0) + float(tot_signed)
            counts[w_norm] = counts.get(w_norm, 0) + 1

    # Build rows: (word, total_abs, total_signed, count, mean_abs)
    rows = []
    for w in agg_abs.keys():
        tot_abs = agg_abs[w]
        tot_signed = agg_signed.get(w, 0.0)
        c = counts.get(w, 1)
        rows.append((w, tot_abs, tot_signed, c, tot_abs / max(1, c)))

    # Toward / away
    toward = [r for r in rows if r[2] > 0]
    away   = [r for r in rows if r[2] < 0]
    toward.sort(key=lambda r: (r[2], r[1]), reverse=True)  # sort by signed (desc), tie on abs
    away.sort(key=lambda r: (r[2], -r[1]))                 # most negative first

    top_toward = [(w, ta, ts, c) for (w, ta, ts, c, _) in toward[:k]]
    top_away   = [(w, ta, ts, c) for (w, ta, ts, c, _) in away[:k]]

    # Save if requested
    if save_csv_toward:
        pd.DataFrame(top_toward, columns=["word","total_abs_ig","total_signed_ig","count"]).to_csv(save_csv_toward, index=False)
    if save_csv_away:
        pd.DataFrame(top_away, columns=["word","total_abs_ig","total_signed_ig","count"]).to_csv(save_csv_away, index=False)

    return top_toward, top_away

# ---------- IG corpus aggregation (SBERT models) ----------

def ig_corpus_top_tokens(
    bundle,
    data_path: str,
    class_name: str,
    k: int = 50,
    steps: int = 24,
    limit: Optional[int] = None,
    sample_p: float = 1.0,
    predicted_only: bool = True,
    prob_thresh: float = 0.5,
    min_len: int = 2,
    strip_stop: bool = True,
    save_csv: Optional[str] = None,
):
    """Aggregate token-level IG across many emails to get a global Top-N list.

    Returns a list of (token, total_ig, count, mean_ig) sorted by total_ig desc.
    """
    if torch is None:
        raise RuntimeError("torch/transformers are required for IG corpus aggregation.")
    if pd is None:
        raise RuntimeError("pandas is required for IG corpus aggregation.")

    model_name = bundle.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    tok = AutoTokenizer.from_pretrained(model_name)

    df = read_dataset_robust(data_path)
    X = combine_text_columns(df)

    pipe = bundle["pipeline"]
    le = bundle["label_encoder"]
    classes = list(le.classes_)
    if class_name not in classes:
        raise ValueError(f"Unknown class '{class_name}'. Available: {classes}")
    kidx = int(np.where(le.classes_ == class_name)[0][0])

    idx_all = np.arange(len(X))
    if predicted_only:
        try:
            probs = pipe.predict_proba(X)[:, kidx]
            mask = probs >= prob_thresh
            idx_all = np.where(mask)[0]
        except Exception:
            pass

    rng = np.random.default_rng(42)
    if sample_p < 1.0:
        n = int(max(1, np.floor(len(idx_all) * sample_p)))
        idx_all = rng.choice(idx_all, size=n, replace=False)
    if limit is not None:
        idx_all = idx_all[:limit]

    stop = set()
    if strip_stop:
        stop.update({
            'the','a','an','and','or','to','of','in','for','(cid)','on','at','with','from','by','as','is','are','be','this','that','it','you','your','we','our','us','re','fw','fwd','de'
        })

    agg = {}
    cnt = {}

    for i in idx_all:
        txt = str(X.iloc[i])
        res = integrated_gradients_tokens(bundle, txt, class_name, steps=steps)
        toks = res.get("tokens")
        imps = res.get("importances")
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
    items.sort(key=lambda x: (x[1], x[3]), reverse=True)
    top = items[:k]

    if save_csv:
        pd.DataFrame(top, columns=["token","total_ig","count","mean_ig"]).to_csv(save_csv, index=False)

    return top

def render_html_highlight(text: str, ranked: List[Tuple[int, str, float]], top_k: int = 50, cmap=(255, 0, 0)) -> str:
    """Render a simple HTML heatmap using whitespace tokens as a proxy.
    This won’t perfectly align with subwords, but it’s readable.
    """
    top = ranked[:top_k]
    if not top:
        return f"<p>{text}</p>"
    maxv = max(v for _, _, v in top) or 1.0
    import html
    words = text.split()
    colored = []
    for w in words:
        score = 0.0
        for _, tok, v in top:
            if tok.strip("#▁") and tok.strip("#▁").lower() in w.lower():
                score = max(score, v / maxv)
        r, g, b = cmap
        style = f"background-color: rgba({r},{g},{b},{score:.2f}); padding:2px; border-radius:4px;"
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
    ap.add_argument("--sep", type=str, default=None, help="Force delimiter: ',', ';', '\\t', '|' (for CSV read)")
    ap.add_argument("--encoding", type=str, default=None, help="Force encoding, e.g. 'utf-8', 'latin-1' (for CSV read)")

    # Modes
    ap.add_argument("--top-ngrams", action="store_true", help="Show top-K n-grams for a class (TF-IDF models only)")
    ap.add_argument("--prototypes", action="store_true", help="Show top-K prototypical training emails for a class")
    ap.add_argument("--data", help="CSV with emails (needed for prototypes or ig-corpus)")

    ap.add_argument("--ig", action="store_true", help="Run IG token/word attribution for a single text (SBERT models)")
    ap.add_argument("--text", type=str, help="Text to explain with IG")
    ap.add_argument("--save-html", type=str, default=None, help="Optional path to save a simple HTML heatmap for IG")

    ap.add_argument("--ig-corpus-words", action="store_true",
    help="Aggregate IG across dataset and output Top-N WORDS pushing TOWARD and AWAY (SBERT only)")
    ap.add_argument("--ig-corpus", action="store_true", help="Aggregate IG across many emails to get a Top-N token list (SBERT only)")
    ap.add_argument("--steps", type=int, default=24, help="IG steps (lower is faster)")
    ap.add_argument("--limit", type=int, default=None, help="Cap number of emails processed for ig-corpus")
    ap.add_argument("--sample-p", type=float, default=1.0, help="Randomly sample this fraction of eligible emails (ig-corpus)")
    ap.add_argument("--predicted-only", action="store_true", help="Only run IG on emails predicted as the target class (ig-corpus)")
    ap.add_argument("--prob-thresh", type=float, default=0.5, help="Probability threshold for predicted-only filtering (ig-corpus)")
    ap.add_argument("--min-len", type=int, default=2, help="Minimum token length to keep (ig-corpus)")
    ap.add_argument("--no-stop", action="store_true", help="Do not strip stopwords in ig-corpus aggregation")
    ap.add_argument("--save-csv-toward", type=str, default=None, help="CSV path for toward words")
    ap.add_argument("--save-csv-away", type=str, default=None, help="Where to save the aggregated Top-N as CSV (ig-corpus)")

    args = ap.parse_args()

    bundle = load_bundle(args.model)

    # ---------- IG-corpus aggregation ----------
    if args.ig_corpus_words:
        if not is_sbert_pipeline(bundle):
            raise SystemExit("IG-corpus words is only available for SBERT-based models.")
        if not args.class_name:
            raise SystemExit("--class is required with --ig-corpus-words")
        if not args.data:
            raise SystemExit("--data CSV is required with --ig-corpus-words")

    top_toward, top_away = ig_corpus_top_words_signed(
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
        save_csv_toward=args.save_csv_toward,
        save_csv_away=args.save_csv_away,
    )
    print_signed(f"Top {args.k} WORDS pushing TOWARD '{args.class_name}' (dataset)", top_toward)
    print_signed(f"Top {args.k} WORDS pushing AWAY from '{args.class_name}' (dataset)", top_away)
    return


    # ---------- Top n-grams (TF-IDF only) ----------
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

    # ---------- Prototypes (both pipelines) ----------
    if args.prototypes:
        if not args.class_name:
            raise SystemExit("--class is required with --prototypes")
        if not args.data:
            raise SystemExit("--data CSV is required with --prototypes")
        rows = top_prototypes(bundle, args.data, args.class_name, k=args.k, sep=args.sep, encoding=args.encoding)
        print(f"\nTop {args.k} prototypes for class = {args.class_name}")
        for r in rows:
            text = r["text"]
            preview = (text[:160] + ("..." if len(text) > 160 else ""))
            print(f"  [margin {r['margin']:.4f}] {preview}")
        return

    # ---------- Single-text IG (SBERT only) ----------
    if args.ig:
        if not is_sbert_pipeline(bundle):
            raise SystemExit("IG is only available for SBERT-based models (not TF-IDF).")
        if not args.class_name:
            raise SystemExit("--class is required with --ig")
        if not args.text:
            raise SystemExit("--text is required with --ig")
        res = integrated_gradients_tokens(bundle, args.text, args.class_name, steps=args.steps)

        # Word-level rollup (merge subwords)
        model_name = bundle.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        tok = AutoTokenizer.from_pretrained(model_name)
        words = aggregate_word_level(args.text, res, tok)
        top_toward, top_away = split_signed_lists(words, k=args.k)
        print_signed(f"Top {args.k} WORDS pushing TOWARD '{args.class_name}'", top_toward)
        print_signed(f"Top {args.k} WORDS pushing AWAY from '{args.class_name}'", top_away)

        # Token-level table (|signed| magnitude)
        print(f"\nTop {args.k} token attributions for class = {args.class_name}")
        for i, tok_str, score in res["ranked"][: args.k]:
            print(f"  {tok_str}\t{score:.6f}")
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
