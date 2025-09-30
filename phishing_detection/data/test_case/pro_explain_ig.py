#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explain Sentence-Transformers + LogisticRegression classifiers with Integrated Gradients.
- Loads artifacts from pro_train_embed.py (embedding branch).
- Rebuilds the ST encoder and LR head, includes StandardScaler and engineered stats.
- Attributes the unwanted-class logit to token embeddings using Captum LayerIntegratedGradients.
- Aggregates subword attributions into words and saves a CSV.
Usage examples:
  python pro_explain_ig.py --model-dir models/embed_v1 --text "URGENT: Verify your account now..."
  python pro_explain_ig.py --model-dir models/embed_v1 --input-csv emails.csv --text-col text --out attributions.csv

  python pro_explain_ig.py --model-dir models/embed_v1 \
  --input-csv emails.csv --text-col text --out attributions.csv --topk 20 --steps 64
Requirements:
  pip install torch transformers sentence-transformers captum scikit-learn joblib pandas numpy
"""
import os, sys, argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import LayerIntegratedGradients
from transformers import AutoModel, AutoTokenizer
import joblib

# import engineered features
try:
    from components import TextStatsTransformer
except Exception:
    TextStatsTransformer = None

def mean_pool(last_hidden_state, attention_mask):
    # last_hidden_state: [B,T,H], attention_mask: [B,T]
    mask = attention_mask.unsqueeze(-1)  # [B,T,1]
    summed = (last_hidden_state * mask).sum(dim=1)  # [B,H]
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

        # freeze head for attribution (we attribute to embeddings only)
        for p in self.head.parameters():
            p.requires_grad_(False)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(self, input_ids, attention_mask, stats_tensor):
        # Encode -> mean-pool
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        sent = mean_pool(out.last_hidden_state, attention_mask)  # [B,H]
        # Concatenate engineered features
        if self.feat_order == "embed_then_stats":
            feats = torch.cat([sent, stats_tensor], dim=1)  # [B, H+S]
        else:
            feats = torch.cat([stats_tensor, sent], dim=1)

        # Apply StandardScaler (with_mean=False) if provided
        if self.scale_vec is not None:
            feats = feats / torch.clamp(self.scale_vec, min=1e-6)

        logit = self.head(feats).squeeze(-1)
        return logit

def tokens_to_words(tokens, scores):
    """Merge WordPiece tokens into words by averaging their scores."""
    words = []
    vals = []
    buf = ""
    agg = 0.0
    cnt = 0
    special = set(["[CLS]","[SEP]","[PAD]"])
    for tok, s in zip(tokens, scores):
        if tok in special:
            continue
        if tok.startswith("##"):
            piece = tok[2:]
            buf += piece
            agg += s
            cnt += 1
        else:
            if buf:
                words.append(buf)
                vals.append(agg / max(cnt, 1))
            buf = tok
            agg = s
            cnt = 1
    if buf:
        words.append(buf)
        vals.append(agg / max(cnt, 1))

    words = [w.replace("▁","").replace("_"," ") for w in words]
    return words, np.array(vals, dtype=float)

def build_from_dir(model_dir):
    # Load model.joblib and meta (pickled dicts)
    bundle = joblib.load(os.path.join(model_dir, "model.joblib"))
    meta = joblib.load(os.path.join(model_dir, "meta.json"))  # saved via joblib in training code
    if meta.get("type") != "embed+engineered":
        raise RuntimeError("This explainer expects an embed+engineered model. Got: %s" % meta)

    embed_meta = meta.get("embed_meta", {})
    backend = embed_meta.get("backend", "sentence-transformers")
    model_name = embed_meta.get("model", "sentence-transformers/all-MiniLM-L6-v2")

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
        # Fallback: zeros if component is unavailable
        # You can replace with your own engineered_features if needed.
        return np.zeros((len(texts), 10), dtype=np.float32)
    t = TextStatsTransformer()
    X = t.fit_transform(texts)  # csr
    return X.toarray().astype(np.float32)

def explain_texts(model_dir, texts, topk=15, n_steps=64, out_csv=None):
    tok, enc, lr_coef, lr_bias, scale_vec = build_from_dir(model_dir)
    model = STLRModel(enc, lr_coef, lr_bias, scale_vec=scale_vec)
    model.eval()
    lig = LayerIntegratedGradients(model.forward, model.encoder.get_input_embeddings())

    rows = []
    for text in texts:
        batch = tok(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        # Stats
        stats_np = compute_stats([text])  # [1,S]
        stats_tensor = torch.from_numpy(stats_np)

        # Baseline = PAD tokens of same length
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        baseline_ids = torch.full_like(input_ids, fill_value=pad_id)

        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask, stats_tensor),
            n_steps=n_steps,
            internal_batch_size=16,
            return_convergence_delta=True,
        )
        # Aggregate embedding-dim attributions to per-token
        token_attr = attributions.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()
        token_attr = token_attr / max(token_attr.max(), 1e-6)

        tokens = tok.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        words, scores = tokens_to_words(tokens, token_attr)
        order = np.argsort(scores)[::-1]
        top_words = [(words[i], float(scores[i])) for i in order[:topk]]

        # Save per-token and per-word rows
        for w, s in top_words:
            rows.append({"text": text, "unit": w, "score": s, "type": "word"})
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Directory with model.joblib and meta.json (pickled) from pro_train_embed.py")
    ap.add_argument("--text", help="Explain a single text inline")
    ap.add_argument("--input-csv", help="CSV of texts to explain")
    ap.add_argument("--text-col", default="text", help="Column in CSV containing text")
    ap.add_argument("--out", default=None, help="Where to save the per-text top words CSV")
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--steps", type=int, default=64)
    args = ap.parse_args()

    texts = []
    if args.text:
        texts.append(args.text)
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        if args.text_col not in df.columns:
            raise SystemExit(f"Column '{args.text_col}' not found in {args.input_csv}")
        texts.extend(df[args.text_col].astype(str).tolist())
    if not texts:
        raise SystemExit("Provide --text or --input-csv")

    df = explain_texts(args.model_dir, texts, topk=args.topk, n_steps=args.steps, out_csv=args.out)
    # Print a compact view
    with pd.option_context('display.max_colwidth', 80):
        for t in texts[:5]:
            sub = df[df["text"] == t].sort_values("score", ascending=False).head(args.topk)
            print("\nTEXT:", t[:120].replace("\n"," ") + ("..." if len(t)>120 else ""))
            print(sub[["unit","score"]].to_string(index=False))

if __name__ == "__main__":
    main()
