#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explain Sentence-Transformers + LogisticRegression classifiers with Integrated Gradients.
- Loads artifacts from pro_train_embed.py (embedding branch).
- Rebuilds the ST encoder and LR head, includes StandardScaler and engineered stats.
- Attributes the unwanted-class logit to token embeddings using Captum LayerIntegratedGradients.
- Aggregates subword attributions into words and saves a CSV.

This version includes a robust CSV reader that:
  * Tries normal pandas.read_csv first.
  * Falls back to tolerant parsers.
  * Finally handles "text,label" files where the last comma separates text from label, with many commas in text.

Usage examples:
  python pro_explain_ig_robust.py \
  --model-dir models/embed_miniLM \
  --input-csv llm_phishing.csv --text-col text \
  --aggregate p90 --topk 20 --steps 64 --batch-size 16 \
  --out attributions.csv


"""
import os, sys, argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import LayerIntegratedGradients
from transformers import AutoModel, AutoTokenizer
import joblib
import math

# import engineered features
try:
    from components import TextStatsTransformer
except Exception:
    TextStatsTransformer = None

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts

class STLRModel(nn.Module):
    """
    Hugging Face encoder + (optional) StandardScaler + Linear head copied from sklearn LR.
    Forward returns the unwanted-class logit.
    """
    def __init__(self, encoder, lr_coef, lr_bias, scale_vec=None, feat_order="embed_then_stats"):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("scale_vec", None if scale_vec is None else torch.from_numpy(scale_vec.astype(np.float32)))
        self.head = nn.Linear(lr_coef.shape[1], 1, bias=True)
        with torch.no_grad():
            self.head.weight.copy_(torch.from_numpy(lr_coef.astype(np.float32)))
            self.head.bias.copy_(torch.from_numpy(lr_bias.astype(np.float32)))
        assert feat_order in ("embed_then_stats",)
        self.feat_order = feat_order
        for p in self.head.parameters():
            p.requires_grad_(False)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(self, input_ids, attention_mask, stats_tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        sent = mean_pool(out.last_hidden_state, attention_mask)
        feats = torch.cat([sent, stats_tensor], dim=1)
        if self.scale_vec is not None:
            feats = feats / torch.clamp(self.scale_vec, min=1e-6)
        logit = self.head(feats).squeeze(-1)
        return logit

def tokens_to_words(tokens, scores):
    """Merge WordPiece tokens into words by averaging their scores."""
    words, vals = [], []
    buf, agg, cnt = "", 0.0, 0
    special = {"[CLS]","[SEP]","[PAD]"}
    for tok, s in zip(tokens, scores):
        if tok in special: 
            continue
        if tok.startswith("##"):
            piece = tok[2:]
            buf += piece; agg += s; cnt += 1
        else:
            if buf:
                words.append(buf); vals.append(agg / max(cnt, 1))
            buf = tok; agg = s; cnt = 1
    if buf:
        words.append(buf); vals.append(agg / max(cnt, 1))
    words = [w.replace("▁","").replace("_"," ") for w in words]
    return words, np.array(vals, dtype=float)

def build_from_dir(model_dir):
    bundle = joblib.load(os.path.join(model_dir, "model.joblib"))
    meta = joblib.load(os.path.join(model_dir, "meta.json"))
    if meta.get("type") != "embed+engineered":
        raise RuntimeError(f"This explainer expects an embed+engineered model. Got: {meta.get('type')}")
    embed_meta = meta.get("embed_meta", {})
    backend = embed_meta.get("backend", "sentence-transformers")
    model_name = embed_meta.get("model", "sentence-transformers/all-MiniLM-L6-v2")

    if backend == "sentence-transformers":
        # normalize common shorthand like "all-MiniLM-L6-v2"
        if "/" not in model_name:
            model_name = f"sentence-transformers/{model_name}"
    else:
        raise RuntimeError(f"Only sentence-transformers backend supported here. Got: {backend}")

    if backend != "sentence-transformers":
        raise RuntimeError(f"Only sentence-transformers backend supported here. Got: {backend}")
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name)
    enc.eval()
    scaler = bundle.get("scaler", None)
    clf = bundle.get("clf", None)
    if clf is None:
        raise RuntimeError("clf missing in model.joblib")
    lr_coef = clf.coef_.astype(np.float32)
    lr_bias = clf.intercept_.astype(np.float32)
    scale_vec = None
    if scaler is not None and hasattr(scaler, "scale_"):
        scale_vec = scaler.scale_.astype(np.float32)
    return tok, enc, lr_coef, lr_bias, scale_vec

def compute_stats(texts):
    if TextStatsTransformer is None:
        return np.zeros((len(texts), 10), dtype=np.float32)
    t = TextStatsTransformer()
    X = t.fit_transform(texts)  # csr
    return X.toarray().astype(np.float32)

def robust_read_csv(path, text_col):
    """Try multiple ways to read messy CSVs. Return a DataFrame with at least `text_col`."""
    import pandas as pd, csv

    # 1) Straightforward
    try:
        df = pd.read_csv(path)
        if text_col in df.columns:
            return df
    except Exception:
        pass

    # 2) Tolerant engines / alt separators
    for kw in [
        dict(engine='python'),
        dict(engine='python', sep='|'),
        dict(engine='python', sep=',', quotechar='"', escapechar='\\'),
        dict(engine='python', on_bad_lines='skip'),
    ]:
        try:
            df = pd.read_csv(path, **kw)
            if text_col in df.columns:
                return df
        except Exception:
            continue

    # 3) "text,label" with commas inside text -> split on the last comma
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        first = f.readline()
        # detect header
        has_header = first.lower().strip().startswith(text_col)
        if not has_header:
            # include first line in processing
            if first:
                f.seek(0)
        for line in f:
            line = line.rstrip('\n')
            if not line: 
                continue
            if ',' in line:
                text, *_tail = line.rsplit(',', 1)
            else:
                text = line
            rows.append({text_col: text})
    return pd.DataFrame(rows)


def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

def score_texts(model_dir, texts, batch_size=32):
    """
    Fast batched scoring (no IG). Returns per-text probabilities (unwanted) as a numpy array.
    """
    tok, enc, lr_coef, lr_bias, scale_vec = build_from_dir(model_dir)

    # build the same torch model head we use for IG
    model = STLRModel(enc, lr_coef, lr_bias, scale_vec=scale_vec)
    model.eval()

    probs = []
    # simple batching to avoid OOM
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        batch = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # engineered stats for the whole chunk
        stats_np = compute_stats(chunk)                # [B,S]
        stats_tensor = torch.from_numpy(stats_np)      # float32

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, stats_tensor=stats_tensor)  # [B]
            p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p)

    return np.concatenate(probs, axis=0) if probs else np.array([], dtype=float)

def explain_texts(model_dir, texts, topk=15, n_steps=64, out_csv=None, batch_size=16):
    """
    Original IG explainer, now with lightweight batching inside (per text for IG attribution).
    Also returns per-row probability vector so the caller can aggregate.
    """
    tok, enc, lr_coef, lr_bias, scale_vec = build_from_dir(model_dir)
    model = STLRModel(enc, lr_coef, lr_bias, scale_vec=scale_vec)
    model.eval()
    lig = LayerIntegratedGradients(model.forward, model.encoder.get_input_embeddings())

    rows = []
    per_row_probs = []

    for text in texts:
        batch = tok(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        stats_np = compute_stats([text])
        stats_tensor = torch.from_numpy(stats_np)

        # baseline = PAD of same length
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        baseline_ids = torch.full_like(input_ids, fill_value=pad_id)

        # probability for this row
        with torch.no_grad():
            logit = model(input_ids=input_ids, attention_mask=attention_mask, stats_tensor=stats_tensor)
            prob = torch.sigmoid(logit).item()
        per_row_probs.append(prob)

        # IG attribution
        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask, stats_tensor),
            n_steps=n_steps,
            internal_batch_size=batch_size,
            return_convergence_delta=True,
        )
        token_attr = attributions.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()
        token_attr = token_attr / max(token_attr.max(), 1e-6)
        tokens = tok.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        words, scores = tokens_to_words(tokens, token_attr)
        order = np.argsort(scores)[::-1]
        top_words = [(words[i], float(scores[i])) for i in order[:topk]]
        for w, s in top_words:
            rows.append({"text": text, "unit": w, "score": s, "type": "word"})

    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df, np.array(per_row_probs, dtype=float)

def _aggregate(scores, how="mean"):
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return float("nan")
    how = how.lower()
    if how == "mean":
        return float(scores.mean())
    if how == "median":
        return float(np.median(scores))
    if how == "p90":
        return float(np.quantile(scores, 0.90))
    raise ValueError(f"Unknown aggregate: {how}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Dir with model.joblib and meta.json from pro_train_embed.py (embedding branch).")
    ap.add_argument("--text", help="Explain/score a single text.")
    ap.add_argument("--input-csv", help="CSV of texts to process.")
    ap.add_argument("--text-col", default="text", help="Column name containing the text (default: text).")
    ap.add_argument("--out", default=None, help="Write per-row outputs (IG words or per-row probs) to this CSV.")
    ap.add_argument("--topk", type=int, default=15, help="Top words per text for IG.")
    ap.add_argument("--steps", type=int, default=64, help="IG steps.")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for scoring/IG internals.")
    ap.add_argument("--score-only", action="store_true", help="Skip IG; compute probs for all rows and aggregate.")
    ap.add_argument("--aggregate", default="mean", choices=["mean","median","p90"], help="How to combine row scores into a dataset score.")
    args = ap.parse_args()

    # Collect texts
    texts = []
    if args.text:
        texts.append(args.text)
    if args.input_csv:
        df_in = robust_read_csv(args.input_csv, args.text_col)
        if args.text_col not in df_in.columns:
            raise SystemExit(f"Column '{args.text_col}' not found or could not be derived from {args.input_csv}")
        texts.extend(df_in[args.text_col].astype(str).tolist())
    if not texts:
        raise SystemExit("Provide --text or --input-csv")

    if args.score_only:
        # Fast path: no IG, just probabilities (batched)
        probs = score_texts(args.model_dir, texts, batch_size=args.batch_size)
        dataset_score = _aggregate(probs, how=args.aggregate)
        print(f"\nDataset score ({args.aggregate} of prob_unwanted) = {dataset_score:.6f}  over N={len(probs)} rows")
        if args.out:
            pd.DataFrame({"text": texts, "prob_unwanted": probs}).to_csv(args.out, index=False)
        return

    # IG + also return per-row probs (slower)
    df_out, probs = explain_texts(
        args.model_dir, texts, topk=args.topk, n_steps=args.steps, out_csv=args.out, batch_size=args.batch_size
    )
    dataset_score = _aggregate(probs, how=args.aggregate)
    print(f"\nDataset score ({args.aggregate} of prob_unwanted) = {dataset_score:.6f}  over N={len(probs)} rows")

    # Pretty print the first few explanations
    with pd.option_context('display.max_colwidth', 80):
        for t in texts[:5]:
            sub = df_out[df_out["text"] == t].sort_values("score", ascending=False).head(args.topk)
            print("\nTEXT:", t[:120].replace("\n"," ") + ("..." if len(t)>120 else ""))
            if not sub.empty:
                print(sub[["unit","score"]].to_string(index=False))
            else:
                print("(no IG rows)")

if __name__ == "__main__":
    main()
