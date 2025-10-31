#!/usr/bin/env python3
"""
Evaluate a pretrained text classification model on a JSON dataset that
contains pairs of `original_text` and `rephrased_text`, along with a
`source` field holding the true label.

This version automatically remaps labels with suffixes like `_human`
(e.g. 'phishing_human' → 'phishing', 'legit_human' → 'legit') if they
are not recognized by the model's LabelEncoder.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from ST_trainer import TokenDropper, SentenceTransformerEncoder
from sklearn.feature_extraction.text import TfidfVectorizer



try:
    from sklearn.metrics import classification_report, confusion_matrix
except Exception:  # pragma: no cover
    classification_report = None
    confusion_matrix = None

from collections import Counter
import math
import re

def _tokenize_for_cosine(s: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z0-9]+", str(s).lower()) if t]

def _tf(tokens: list[str]) -> Counter:
    return Counter(tokens)

def _cosine_from_counters(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b.get(t, 0) for t in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def reduce_pairs_by_cosine(data: list[dict], threshold: float = 0.92) -> list[dict]:
    """Drop items where original_text and rephrased_text are too similar."""
    reduced = []
    for d in data:
        o = _tf(_tokenize_for_cosine(d.get("original_text", "")))
        r = _tf(_tokenize_for_cosine(d.get("rephrased_text", "")))
        cos = _cosine_from_counters(o, r)
        if cos < threshold:
            reduced.append(d)
    return reduced

def load_bundle(model_path: str) -> Dict[str, Any]:
    bundle = joblib.load(model_path)
    if isinstance(bundle, dict) and "pipeline" in bundle:
        return bundle
    return {"pipeline": bundle, "label_encoder": getattr(bundle, "label_encoder", None)}


def predict_proba_safe(pipe, X: List[str]) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)
    if hasattr(pipe, "decision_function"):
        from scipy.special import softmax
        scores = pipe.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        return softmax(scores, axis=1)
    hard = pipe.predict(X)
    classes_ = getattr(pipe, "classes_", None)
    if classes_ is None:
        raise ValueError("Model has neither predict_proba nor classes_.")
    proba = np.zeros((len(hard), len(classes_)), dtype=float)
    idx = {c: i for i, c in enumerate(classes_)}
    for r, h in enumerate(hard):
        proba[r, idx[h]] = 1.0
    return proba


def evaluate(model_path: str, json_path: str, out_csv: str | None, out_summary: str | None) -> Dict[str, Any]:
    print(f"[evaluate] Loading model: {model_path}")
    bundle = load_bundle(model_path)
    pipe = bundle["pipeline"]
    le = bundle.get("label_encoder")

    if le is not None and hasattr(le, "classes_"):
        class_names = list(le.classes_)
    elif hasattr(pipe, "classes_"):
        class_names = list(pipe.classes_)
        le = None
    else:
        raise ValueError("Could not determine class names.")
    class_index = {c: i for i, c in enumerate(class_names)}

    print(f"[evaluate] Reading JSON: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Added: filter too-similar pairs using cosine
    try:
        threshold = float(os.environ.get("COSINE_SIM_THRESHOLD", "0.92"))
    except Exception:
        threshold = 0.92
    orig_len = len(data)
    data = reduce_pairs_by_cosine(data, threshold)
    print(f"[evaluate] Cosine filter threshold={threshold:.2f}: kept {len(data)}/{orig_len}")


    original_texts = [d["original_text"] for d in data]
    rephrased_texts = [d["rephrased_text"] for d in data]
    true_labels_names = [d["source"] for d in data]

    # Canonicalize labels to your 2 classes: 'Other' and 'phishing' (case-insensitive)
    canonical_map = {
        "phishing": "phishing",
        "phishing_human": "phishing",
        "phishing_llm": "phishing",
        "legit_llm": "Other",
        "phish": "phishing",
        "other": "Other",
        "legit": "Other",
        "legit_human": "Other",
        "ham": "Other",
        "benign": "Other",
        "non_phishing": "Other",
        "safe": "Other",
    }
    def to_canonical(lbl: str) -> str:
        return canonical_map.get(str(lbl).strip().lower(), str(lbl))

    # First pass: canonicalize all incoming labels
    true_labels_names = [to_canonical(lbl) for lbl in true_labels_names]

    # If the model has a LabelEncoder, ensure the canonical labels exist; if not, try to nudge
    if le is not None:
        want = {"Other", "phishing"}
        have = set(map(str, getattr(le, "classes_", [])))
        # If we still see anything outside the model's classes, try a gentle fix for capitalization
        remapped = []
        for lbl in true_labels_names:
            if lbl not in have and lbl.lower() == "other" and "Other" in have:
                remapped.append("Other")
            else:
                remapped.append(lbl)
        true_labels_names = remapped

        # Final guardrail: surface any remaining unknowns clearly
        unknown = sorted({lbl for lbl in true_labels_names if lbl not in have})
        if unknown:
            print(f"[WARN] Labels not in model classes {sorted(have)}: {unknown}")
            print("      Tip: retrain or extend mapping above to cover these variants.")

        y_true = le.transform(true_labels_names)
    else:
        y_true = np.array([class_index[name] for name in true_labels_names], dtype=int)

    y_pred_orig = pipe.predict(original_texts)
    if np.issubdtype(np.array(y_pred_orig).dtype, np.integer):
        y_pred_orig_idx = np.asarray(y_pred_orig, dtype=int)
        y_pred_orig_names = [class_names[i] for i in y_pred_orig_idx]
    else:
        y_pred_orig_names = list(map(str, y_pred_orig))
        y_pred_orig_idx = np.array([class_index[n] for n in y_pred_orig_names], dtype=int)

    proba_orig = predict_proba_safe(pipe, original_texts)
    acc_orig = float(np.mean(y_pred_orig_idx == y_true))

    y_pred_reph = pipe.predict(rephrased_texts)
    if np.issubdtype(np.array(y_pred_reph).dtype, np.integer):
        y_pred_reph_idx = np.asarray(y_pred_reph, dtype=int)
        y_pred_reph_names = [class_names[i] for i in y_pred_reph_idx]
    else:
        y_pred_reph_names = list(map(str, y_pred_reph))
        y_pred_reph_idx = np.array([class_index[n] for n in y_pred_reph_names], dtype=int)

    proba_reph = predict_proba_safe(pipe, rephrased_texts)
    acc_reph = float(np.mean(y_pred_reph_idx == y_true))

    mean_proba_per_class_original = {c: float(proba_orig[:, i].mean()) for i, c in enumerate(class_names)}
    mean_proba_per_class_rephrased = {c: float(proba_reph[:, i].mean()) for i, c in enumerate(class_names)}

    idx = np.arange(len(y_true))
    mean_trueclass_proba_original = float(proba_orig[idx, y_true].mean())
    mean_trueclass_proba_rephrased = float(proba_reph[idx, y_true].mean())

    average_accuracy = float(np.mean([acc_orig, acc_reph]))
    average_trueclass_proba = float(np.mean([mean_trueclass_proba_original, mean_trueclass_proba_rephrased]))

    by_label_true_p = {}
    for c, i in class_index.items():
        mask = (y_true == i)
        if mask.any():
            by_label_true_p[c] = {
                "n": int(mask.sum()),
                "mean_true_p_original": float(proba_orig[mask, i].mean()),
                "mean_true_p_rephrased": float(proba_reph[mask, i].mean()),
            }

    rows = []
    for i in range(len(data)):
        row = {
            "id": i,
            "true_label": true_labels_names[i],
            "pred_original": y_pred_orig_names[i],
            "pred_rephrased": y_pred_reph_names[i],
            "correct_original": bool(y_pred_orig_idx[i] == y_true[i]),
            "correct_rephrased": bool(y_pred_reph_idx[i] == y_true[i]),
        }
        for j, cname in enumerate(class_names):
            row[f"proba_original::{cname}"] = float(proba_orig[i, j])
            row[f"proba_rephrased::{cname}"] = float(proba_reph[i, j])
        true_cname = true_labels_names[i]
        row["proba_original::true_class"] = float(proba_orig[i, class_index[true_cname]])
        row["proba_rephrased::true_class"] = float(proba_reph[i, class_index[true_cname]])
        row["delta_trueclass_proba"] = row["proba_rephrased::true_class"] - row["proba_original::true_class"]
        rows.append(row)

    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[evaluate] Wrote per-item probabilities to: {out_csv}")

    if classification_report is not None:
        print("\n=== Metrics ===")
        print(f"Accuracy (Original): {acc_orig:.4f}")
        print(f"Accuracy (Rephrased): {acc_reph:.4f}")
        print(f"Average Accuracy: {average_accuracy*100:.2f}%")
        print(f"Mean TRUE-class p (Original): {mean_trueclass_proba_original:.4f}")
        print(f"Mean TRUE-class p (Rephrased): {mean_trueclass_proba_rephrased:.4f}")
        print(f"Avg TRUE-class p (General %): {average_trueclass_proba*100:.2f}%")

    result = {
        "classes": class_names,
        "accuracy_original": acc_orig,
        "accuracy_rephrased": acc_reph,
        "average_accuracy": average_accuracy,
        "mean_trueclass_proba_original": mean_trueclass_proba_original,
        "mean_trueclass_proba_rephrased": mean_trueclass_proba_rephrased,
        "average_trueclass_proba": average_trueclass_proba,
        "mean_proba_per_class_original": mean_proba_per_class_original,
        "mean_proba_per_class_rephrased": mean_proba_per_class_rephrased,
        "by_label_trueclass_proba": by_label_true_p,
        "per_item_csv": out_csv,
    }

    if out_summary:
        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[evaluate] Wrote summary JSON to: {out_summary}")

    return result


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pretrained model on JSON with auto label remap.")
    p.add_argument("--model", required=True)
    p.add_argument("--json", required=True)
    p.add_argument("--out-csv", default="json_eval_probabilities.csv")
    p.add_argument("--out-summary", default=None)
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    evaluate(args.model, args.json, args.out_csv, args.out_summary)


if __name__ == "__main__":
    main()
