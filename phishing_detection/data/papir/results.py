
#!/usr/bin/env python3
"""results_phishing_only.py (multi-threshold + dual-encoder cosine runs, optional second model)

- PRIMARY model (--model) is always used for *classification*.
- Cosine encoder for the PRIMARY run comes from the PRIMARY model (forced by --force-sim-encoder).
- Cosine encoder for the OTHER run:
    * If --cosine-model is provided, encoder steps are extracted from that SECOND model
      (optionally forced by --force-other-sim-encoder).
    * Otherwise, we use the "opposite" encoder from the PRIMARY model (tfidf <-> sbert).
- Thresholds are fixed to [0.90, 0.80, 0.70, 0.60]. For each threshold we run PRIMARY then OTHER.
- No scikit-learn Pipeline construction during similarity (avoids 'Pipeline not fitted' warning).
- Each run prints ONLY four lines: rephrased_accuracy, cosine_threshold, cosine_encoder_used, num_emails_evaluated.
- CSV/Summary outputs get suffixes to avoid overwrite. If --cosine-model is used for OTHER, filenames include 'cos2'.

"""
from ST_trainer import RegexReplacer, TokenDropper, SentenceTransformerEncoder
import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import List, Tuple, Optional

import joblib
import numpy as np

# Optional sklearn utilities for vector-space cosine
try:
    from sklearn.preprocessing import normalize as sk_normalize
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    _SK_OK = True
except Exception:  # pragma: no cover
    _SK_OK = False
    TfidfVectorizer = object  # placeholder


# --- Fallback (legacy) token overlap cosine ---

def tokenize_for_cosine(s: str) -> Counter:
    s = (s or "").lower()
    tokens = []
    cur = []
    for ch in s:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                tokens.append(''.join(cur))
                cur = []
    if cur:
        tokens.append(''.join(cur))
    from collections import Counter as _Counter
    return _Counter(tokens)


def cosine_from_counters(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = 0
    for k, v in a.items():
        dot += v * b.get(k, 0)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def canonicalize_label(label: str) -> str:
    if label is None:
        return 'Other'
    lab = str(label).lower()
    if lab.startswith('phish') or 'phish' in lab or 'malicious' in lab or 'spam' == lab:
        return 'phishing'
    return 'Other'


def get_probs_from_model(model, texts: List[str]) -> np.ndarray:
    """Return an (N, C) array of probabilities for each text.
    If the model supports predict_proba, use it. Otherwise try decision_function -> softmax.
    As a last resort, use one-hot from predict.
    """
    if hasattr(model, 'predict_proba'):
        return np.asarray(model.predict_proba(texts))

    if hasattr(model, 'decision_function'):
        dec = np.asarray(model.decision_function(texts))
        if dec.ndim == 1:
            dec = np.vstack([-dec, dec]).T
        exp = np.exp(dec - dec.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    preds = model.predict(texts)
    if hasattr(model, 'classes_'):
        classes = list(model.classes_)
    else:
        classes = sorted(set(preds))
    class_index = {c: i for i, c in enumerate(classes)}
    probs = np.zeros((len(preds), len(classes)), dtype=float)
    for i, p in enumerate(preds):
        probs[i, class_index[p]] = 1.0
    return probs


def load_pipeline_and_encoder(path: str):
    obj = joblib.load(path)
    if isinstance(obj, dict):
        pipeline = obj.get('pipeline') or obj.get('model') or obj
        le = obj.get('label_encoder')
    else:
        pipeline = obj
        le = None
    return pipeline, le


# === Encoder detection helpers ===

def _is_classifier_step(name: str, step) -> bool:
    lname = name.lower()
    if lname in ('clf', 'classifier', 'final', 'estimator'):
        return True
    return (hasattr(step, 'fit') and hasattr(step, 'predict') and not hasattr(step, 'transform'))


def _is_tfidf_step(name: str, step) -> bool:
    if isinstance(step, TfidfVectorizer):
        return True
    n = name.lower()
    if 'tfidf' in n or 'tf-idf' in n:
        return True
    return hasattr(step, 'get_feature_names_out') and hasattr(step, 'transform')


def _is_sbert_step(name: str, step) -> bool:
    n = name.lower()
    if any(tag in n for tag in ('sbert', 'sentence', 'st_embed', 'embed', 'embedding')):
        if not _is_tfidf_step(name, step) and hasattr(step, 'transform'):
            return True
    cls = type(step).__name__.lower()
    if any(t in cls for t in ('sentence', 'sbert', 'transformer')) and hasattr(step, 'transform'):
        return True
    return hasattr(step, 'encode') or hasattr(step, 'model')


def _iter_feature_steps_auto(full_pipeline) -> Tuple[List[Tuple[str, object]], Optional[str]]:
    if not hasattr(full_pipeline, 'steps'):
        return [], None
    collected = []
    encoder_kind = None
    for name, step in full_pipeline.steps:
        if _is_classifier_step(name, step):
            break
        collected.append((name, step))
        if _is_tfidf_step(name, step):
            encoder_kind = 'tfidf'
        elif _is_sbert_step(name, step):
            encoder_kind = 'sbert'
    return collected, encoder_kind


def _iter_feature_steps_forced(full_pipeline, mode: str) -> Tuple[List[Tuple[str, object]], Optional[str]]:
    if not hasattr(full_pipeline, 'steps'):
        return [], None
    collected = []
    found = False
    for name, step in full_pipeline.steps:
        collected.append((name, step))
        if mode == 'tfidf' and _is_tfidf_step(name, step):
            found = True
            return collected, 'tfidf'
        if mode == 'sbert' and _is_sbert_step(name, step):
            found = True
            return collected, 'sbert'
        if _is_classifier_step(name, step):
            break
    return (collected if found else []), (mode if found else None)


def _transform_through_steps(steps: List[Tuple[str, object]], texts: List[str]):
    X = texts
    for name, step in steps:
        if step is None or step == 'drop' or step == 'passthrough':
            continue
        if hasattr(step, 'transform'):
            X = step.transform(X)
        elif hasattr(step, 'encode'):
            X = step.encode(list(X))
        else:
            try:
                X = step(X)
            except Exception:
                pass
    return X


# === Vector-space cosine ===

def pairwise_row_cosines(A, B) -> np.ndarray:
    if not _SK_OK:
        raise RuntimeError("scikit-learn not available for vector cosine")
    A_n = sk_normalize(A, norm='l2', copy=False)
    B_n = sk_normalize(B, norm='l2', copy=False)
    try:
        import scipy.sparse as sp  # type: ignore
        if sp.issparse(A_n) and sp.issparse(B_n):
            return np.asarray((A_n.multiply(B_n)).sum(axis=1)).ravel()
    except Exception:
        pass
    sims = []
    for i in range(A_n.shape[0]):
        s = sk_cosine_similarity(A_n[i], B_n[i])[0, 0]
        sims.append(float(s))
    return np.array(sims, dtype=float)


def embedding_cosine_filter(
    feature_steps: List[Tuple[str, object]],
    original_texts: List[str],
    rephr_texts: List[str],
    threshold: float
):
    A = _transform_through_steps(feature_steps, original_texts)
    B = _transform_through_steps(feature_steps, rephr_texts)
    sims = pairwise_row_cosines(A, B)
    keep_idx = [i for i, s in enumerate(sims) if s < threshold]
    discarded = int((sims >= threshold).sum())
    return keep_idx, discarded, sims


# === Evaluation helper (single run) ===

def run_once(pipeline,
             rows: List[dict],
             feature_steps: List[Tuple[str, object]],
             used_encoder_kind: str,
             threshold: float,
             model_class_names: List[str],
             canonical_index: dict,
             out_csv: Optional[str] = None,
             out_summary: Optional[str] = None):
    # Prepare texts
    originals = [r['original_text'] for r in rows]
    rephrased = [r['rephrased_text'] for r in rows]

    # Similarity-based filtering
    discarded_count = 0
    sims = None
    if feature_steps and _SK_OK:
        try:
            keep_idx, discarded_count, sims = embedding_cosine_filter(feature_steps, originals, rephrased, threshold)
            kept_rows = [rows[i] for i in keep_idx]
        except Exception:
            feature_steps = []
            kept_rows = []
    else:
        kept_rows = []

    if not feature_steps:
        # token fallback
        kept = []
        local_sims = []
        for row in rows:
            a = tokenize_for_cosine(row['original_text'])
            b = tokenize_for_cosine(row['rephrased_text'])
            sim = cosine_from_counters(a, b)
            local_sims.append(sim)
            if sim >= threshold:
                discarded_count += 1
                continue
            kept.append(row)
        kept_rows = kept
        sims = np.array(local_sims, dtype=float)

    # Predictions
    originals_kept = [r['original_text'] for r in kept_rows]
    rephrased_kept = [r['rephrased_text'] for r in kept_rows]

    if len(kept_rows) > 0:
        orig_probs = get_probs_from_model(pipeline, originals_kept)
        rephr_probs = get_probs_from_model(pipeline, rephrased_kept)
    else:
        orig_probs = np.zeros((0, 2))
        rephr_probs = np.zeros((0, 2))

    # Rephrased accuracy
    y_true = []
    y_pred_reph = []
    for i, r in enumerate(kept_rows):
        true_c = r['true_canonical']
        idx_true = canonical_index.get(true_c, canonical_index.get('Other', 0))
        pred_idx_reph = int(np.argmax(rephr_probs[i])) if len(kept_rows) > 0 else 0
        pred_reph_name = model_class_names[pred_idx_reph] if model_class_names else str(pred_idx_reph)
        def _is_phish_name(name):
            return isinstance(name, str) and ('phish' in name.lower() or 'spam' == name.lower())
        y_true.append(1 if true_c == 'phishing' else 0)
        y_pred_reph.append(1 if (_is_phish_name(pred_reph_name) and true_c != 'Other') or (pred_idx_reph == idx_true and true_c == 'phishing') else (1 if _is_phish_name(pred_reph_name) else 0))

    acc_reph = (sum(1 for yt, yp in zip(y_true, y_pred_reph) if yt == yp) / len(y_true)) if y_true else 0.0

    # Optional outputs
    if out_csv:
        import csv
        with open(out_csv, 'w', newline='', encoding='utf8') as outf:
            fieldnames = ['true_label', 'source_raw', 'original_pred', 'rephrased_pred', 'original_true_prob', 'rephrased_true_prob', 'delta_true_prob']
            writer = csv.DictWriter(outf, fieldnames=fieldnames)
            writer.writeheader()
            for i, r in enumerate(kept_rows):
                true_c = r['true_canonical']
                # The above accidental unicode glitch fixed below

                idx_true = canonical_index.get(true_c, canonical_index.get('Other', 0))
                pred_idx_orig = int(np.argmax(orig_probs[i])) if len(kept_rows) > 0 else 0
                pred_idx_reph = int(np.argmax(rephr_probs[i])) if len(kept_rows) > 0 else 0
                pred_orig_name = model_class_names[pred_idx_orig] if model_class_names else str(pred_idx_orig)
                pred_reph_name = model_class_names[pred_idx_reph] if model_class_names else str(pred_idx_reph)
                prob_true_orig = float(orig_probs[i, idx_true]) if len(kept_rows) > 0 and idx_true < orig_probs.shape[1] else 0.0
                prob_true_reph = float(rephr_probs[i, idx_true]) if len(kept_rows) > 0 and idx_true < rephr_probs.shape[1] else 0.0
                writer.writerow({
                    'true_label': true_c,
                    'source_raw': r['source_raw'],
                    'original_pred': pred_orig_name,
                    'rephrased_pred': pred_reph_name,
                    'original_true_prob': prob_true_orig,
                    'rephrased_true_prob': prob_true_reph,
                    'delta_true_prob': prob_true_reph - prob_true_orig,
                })

    if out_summary:
        summary = {
            'num_evaluated': len(kept_rows),
            'cosine_threshold': threshold,
            'used_encoder_kind': used_encoder_kind,
            'rephrased_accuracy': acc_reph,
            'discarded_for_similarity': int(discarded_count),
        }
        with open(out_summary, 'w', encoding='utf8') as jf:
            json.dump(summary, jf, indent=2)

    # Print only the four requested lines
    print(f"rephrased_accuracy: {acc_reph:.4f}")
    print(f"cosine_threshold: {threshold:.4f}")
    print(f"cosine_encoder_used: {used_encoder_kind}")
    print(f"num_emails_evaluated: {len(kept_rows)}")

    return acc_reph, len(kept_rows), used_encoder_kind


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='path to PRIMARY joblib model (used for classification)')
    parser.add_argument('--json', required=True, help='path to dataset json')
    parser.add_argument('--out-csv', default='json_eval_probabilities.csv')
    parser.add_argument('--out-summary', default=None)
    parser.add_argument('--only-phishing', action='store_true', default=False, help='only evaluate rows whose true label canonicalizes to "phishing"')
    parser.add_argument('--force-sim-encoder', choices=['auto', 'tfidf', 'sbert'], default='auto',
                        help='PRIMARY run cosine encoder selection (from PRIMARY model).')
    parser.add_argument('--cosine-model', default=None, help='OPTIONAL: SECOND joblib model to supply encoder for the OTHER cosine run.')
    parser.add_argument('--force-other-sim-encoder', choices=['auto', 'tfidf', 'sbert'], default='auto',
                        help='When --cosine-model is provided, choose which encoder to use from that SECOND model.')
    args = parser.parse_args(argv)

    # Load primary (classification) model
    pipeline, label_encoder = load_pipeline_and_encoder(args.model)

    # Determine class names from primary model
    if label_encoder is not None:
        model_class_names = list(label_encoder.classes_)
    elif hasattr(pipeline, 'classes_'):
        model_class_names = list(pipeline.classes_)
    elif hasattr(pipeline, 'steps'):
        final = pipeline.steps[-1][1]
        model_class_names = list(getattr(final, 'classes_', []))
    else:
        model_class_names = []

    canonical_names = ['Other', 'phishing']

    # Load data
    with open(args.json, 'r', encoding='utf8') as f:
        raw = json.load(f)

    rows = []
    for item in raw:
        o = item.get('original_text') or item.get('original') or item.get('text')
        r = item.get('rephrased_text') or item.get('rephrased') or item.get('rephrased_text', '')
        src = item.get('source') or item.get('label') or item.get('y')
        if o is None or r is None or src is None:
            continue
        true_cname = canonicalize_label(src)
        rows.append({'original_text': o, 'rephrased_text': r, 'source_raw': src, 'true_canonical': true_cname})

    if args.only_phishing:
        rows = [r for r in rows if r['true_canonical'] == 'phishing']

    # Canonical index for primary model
    class_index = {c: i for i, c in enumerate(model_class_names)} if model_class_names else {}
    canonical_index = {}
    for cname in canonical_names:
        if cname in class_index:
            canonical_index[cname] = class_index[cname]
        else:
            if len(model_class_names) == 2:
                canonical_index['Other'] = 0
                canonical_index['phishing'] = 1
                break
            else:
                found = None
                for k in model_class_names:
                    if 'phish' in str(k).lower() or 'spam' in str(k).lower():
                        found = k
                        break
                if found is not None and model_class_names:
                    canonical_index['phishing'] = class_index[found]
                    for k2 in model_class_names:
                        if k2 != found:
                            canonical_index['Other'] = class_index[k2]
                            break
                else:
                    canonical_index['phishing'] = max(len(model_class_names) - 1, 0) if model_class_names else 1
                    canonical_index['Other'] = 0

    # Helper to get feature steps + kind from a given pipeline and mode
    def steps_for(p, mode: str):
        if p is None:
            return [], 'token-fallback'
        if mode == 'auto':
            steps, inferred = _iter_feature_steps_auto(p)
            kind = inferred or ('auto-encoder' if steps else 'token-fallback')
            return steps, kind
        elif mode in ('tfidf', 'sbert'):
            steps, kind = _iter_feature_steps_forced(p, mode)
            kind = kind or 'token-fallback'
            return steps, kind
        else:
            return [], 'token-fallback'

    # PRIMARY cosine steps from PRIMARY model
    primary_steps, primary_kind = steps_for(pipeline, args.force_sim_encoder)

    # Determine OTHER cosine steps/kind
    other_steps = []
    other_kind = 'token-fallback'
    other_source_tag = ''

    if args.cosine_model:
        cos_pipeline, _ = load_pipeline_and_encoder(args.cosine_model)
        other_steps, other_kind = steps_for(cos_pipeline, args.force_other_sim_encoder)
        other_source_tag = 'cos2'
        # If auto couldn't infer, try opposite of primary as heuristic
        if other_kind in ('auto-encoder', 'token-fallback'):
            if primary_kind == 'tfidf':
                tmp_steps, tmp_kind = steps_for(cos_pipeline, 'sbert')
            elif primary_kind == 'sbert':
                tmp_steps, tmp_kind = steps_for(cos_pipeline, 'tfidf')
            else:
                tmp_steps, tmp_kind = steps_for(cos_pipeline, 'sbert')
            if tmp_steps:
                other_steps, other_kind = tmp_steps, tmp_kind
    else:
        if primary_kind == 'tfidf':
            other_steps, other_kind = steps_for(pipeline, 'sbert')
        elif primary_kind == 'sbert':
            other_steps, other_kind = steps_for(pipeline, 'tfidf')
        else:
            other_steps, other_kind = steps_for(pipeline, 'auto')

    thresholds = [0.90, 0.80, 0.70]

    # Per-threshold runs
    base_csv_root, base_csv_ext = os.path.splitext(args.out_csv) if args.out_csv else (None, None)
    base_sum_root, base_sum_ext = os.path.splitext(args.out_summary) if args.out_summary else (None, None)

    for thr in thresholds:
        pct = int(round(thr * 100))

        # PRIMARY run filenames
        out_csv_primary = f"{base_csv_root}_enc-{primary_kind}_t-{pct}{base_csv_ext}" if base_csv_root else None
        out_sum_primary = f"{base_sum_root}_enc-{primary_kind}_t-{pct}{base_sum_ext}" if base_sum_root else None

        # OTHER run filenames
        if base_csv_root:
            tag = f"enc-{other_kind}_t-{pct}"
            if other_source_tag:
                tag = f"{tag}_{other_source_tag}"
            out_csv_other = f"{base_csv_root}_{tag}{base_csv_ext}"
        else:
            out_csv_other = None

        if base_sum_root:
            tag = f"enc-{other_kind}_t-{pct}"
            if other_source_tag:
                tag = f"{tag}_{other_source_tag}"
            out_sum_other = f"{base_sum_root}_{tag}{base_sum_ext}"
        else:
            out_sum_other = None

        # PRIMARY run
        run_once(pipeline, rows, primary_steps, primary_kind, thr, model_class_names, canonical_index,
                 out_csv=out_csv_primary, out_summary=out_sum_primary)

        # OTHER run
        run_once(pipeline, rows, other_steps, other_kind, thr, model_class_names, canonical_index,
                 out_csv=out_csv_other, out_summary=out_sum_other)


if __name__ == '__main__':
    raise SystemExit(main())
