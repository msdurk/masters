#!/usr/bin/env python3
# save as make_csv_from_json.py
# usage:
#   pip install pandas scikit-learn
#   python make_csv_from_json.py --in /mnt/data/ephishLLM.json --out ./dataset

import json
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_json(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of objects (records).")
    df = pd.DataFrame(data)
    expected = {"Subject", "Body", "type", "Language"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Input JSON missing required fields: {missing}")
    return df


def english_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Language"].astype(str).str.lower().eq("en")].copy()


def build_text(df: pd.DataFrame) -> pd.DataFrame:
    def make_text(row):
        subj = str(row.get("Subject", "")).strip()
        body = str(row.get("Body", "")).strip()
        if subj and body:
            return f"Subject: {subj}\n\n{body}"
        return (subj or "") + ("\n\n" + body if body else "")
    df["text"] = df.apply(make_text, axis=1).str.strip()
    return df


def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    # ensure integer labels
    df["label"] = df["type"].astype(int)
    # drop unusable rows
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df[["text", "label"]]


def split_80_10_10(df: pd.DataFrame, seed: int = 42):
    # stratified 10% test
    if df["label"].value_counts().min() >= 2 and len(df) >= 20:
        trainval, test = train_test_split(
            df, test_size=0.10, stratify=df["label"], random_state=seed
        )
        # from remaining 90%, take 1/9 ≈ 11.111% as val → overall 10%
        val_rel = 1 / 9
        train, val = train_test_split(
            trainval, test_size=val_rel, stratify=trainval["label"], random_state=seed
        )
    else:
        # fallback (no stratify) for tiny or skewed datasets
        trainval, test = train_test_split(df, test_size=0.10, random_state=seed)
        train, val = train_test_split(trainval, test_size=0.1111111, random_state=seed)
    return train, val, test


def main():
    ap = argparse.ArgumentParser(description="Convert phishing JSON to English-only CSV splits.")
    ap.add_argument("--in", dest="inp", required=True, help="Path to input JSON (list of records).")
    ap.add_argument("--out", dest="out_dir", default="./dataset", help="Output directory for CSVs.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splits.")
    args = ap.parse_args()

    inp = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_json(inp)
    df = english_only(df)
    df = build_text(df)
    df = prepare_labels(df)

    train, val, test = split_80_10_10(df, seed=args.seed)

    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    # Quick summary
    def counts(name, dff):
        c0 = int((dff["label"] == 0).sum())
        c1 = int((dff["label"] == 1).sum())
        return f"{name}: n={len(dff)} | label0={c0}, label1={c1}"

    print(counts("TRAIN", train))
    print(counts("VAL  ", val))
    print(counts("TEST ", test))
    print(f"\nWrote:\n  {out_dir/'train.csv'}\n  {out_dir/'val.csv'}\n  {out_dir/'test.csv'}")


if __name__ == "__main__":
    main()
