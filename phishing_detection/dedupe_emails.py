#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import radius_neighbors_graph
from collections import defaultdict
"""
run from phishing_detection

python dedupe_emails.py \
  --input data/raw/Nigerian_5.csv \
  --text-cols subject body \
  --threshold 0.1 \
  --output-prefix Nigerian_Fraud \
  --out-dir filtered_raw  
"""
class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

def combine_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    combined = df[cols].fillna("").astype(str).agg("\n".join, axis=1)
    combined = combined.str.replace(r"\s+", " ", regex=True).str.lower()
    return combined

def pick_representative(indices, strategy, texts):
    if strategy == "first":
        return min(indices)
    if strategy == "shortest":
        return min(indices, key=lambda i: len(texts.iloc[i]))
    if strategy == "longest":
        return max(indices, key=lambda i: len(texts.iloc[i]))
    return min(indices)

def dedupe(df, text_cols, threshold, min_df, max_df, ngram_max, rep_strategy):
    texts = combine_columns(df, text_cols)

    vectorizer = TfidfVectorizer(ngram_range=(1, ngram_max), min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(texts)

    radius = 1.0 - float(threshold)
    G = radius_neighbors_graph(
        X, radius=radius, mode="distance", metric="cosine", include_self=False
    ).tocsr()

    n = X.shape[0]
    dsu = DSU(n)

    G.eliminate_zeros()
    indptr, indices = G.indptr, G.indices
    for i in range(n):
        start, end = indptr[i], indptr[i + 1]
        for idx in range(start, end):
            j = indices[idx]
            if i < j:
                dsu.union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[dsu.find(i)].append(i)

    reps = []
    for comp_indices in clusters.values():
        rep = pick_representative(comp_indices, rep_strategy, texts)
        reps.append(rep)
    reps = sorted(set(reps))

    mapping_records = []
    for comp_indices in clusters.values():
        rep = pick_representative(comp_indices, rep_strategy, texts)
        for i in comp_indices:
            mapping_records.append(
                {"row_index": i, "cluster_rep": rep, "is_representative": int(i == rep)}
            )
    map_df = pd.DataFrame(mapping_records).sort_values(["cluster_rep", "row_index"]).reset_index(drop=True)
    clean_df = df.iloc[reps].reset_index(drop=True)

    stats = {
        "original_rows": int(n),
        "remaining_rows": int(len(reps)),
        "duplicates_removed": int(n - len(reps)),
    }
    return clean_df, map_df, stats

def parse_args():
    p = argparse.ArgumentParser(
        description="Deduplicate emails using TF-IDF + cosine similarity threshold."
    )
    p.add_argument("--input", required=True, help="Path to input CSV.")
    p.add_argument("--text-cols", nargs="+", default=["subject", "body"],
                   help="Columns to combine as the text (default: subject body).")
    p.add_argument("--threshold", type=float, default=0.85,
                   help="Cosine similarity threshold to consider duplicates (default: 0.85).")
    p.add_argument("--min-df", type=float, default=1,
                   help="TF-IDF min_df (int count or float proportion). Default: 1.")
    p.add_argument("--max-df", type=float, default=0.95,
                   help="TF-IDF max_df (fraction). Default: 0.95.")
    p.add_argument("--ngram-max", type=int, default=2,
                   help="Use up to this n-gram size (default: 2 = unigrams+bigrams).")
    p.add_argument("--rep-strategy", choices=["first", "shortest", "longest"], default="first",
                   help="Which item to keep per cluster (default: first).")
    p.add_argument("--output-prefix", default=None,
                   help="Prefix for output files (default: input filename without extension).")
    p.add_argument("--out-dir", default="filtered_01",
                   help="Directory to save results (default: filtered_01).")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    base = args.output_prefix or os.path.splitext(os.path.basename(args.input))[0]
    df = pd.read_csv(args.input)

    clean_df, map_df, stats = dedupe(
        df,
        text_cols=args.text_cols,
        threshold=args.threshold,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_max=args.ngram_max,
        rep_strategy=args.rep_strategy,
    )

    # --- create output folder ---
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    suffix = str(args.threshold).replace(".", "")

    clean_path   = os.path.join(out_dir, f"{base}_dedup_{suffix}.csv")
    removed_path = os.path.join(out_dir, f"{base}_removed_{suffix}.csv")
    map_path     = os.path.join(out_dir, f"{base}_clusters_{suffix}.csv")

    clean_df.to_csv(clean_path, index=False)
    removed_df = df.drop(clean_df.index).reset_index(drop=True)
    removed_df.to_csv(removed_path, index=False)
    map_df.to_csv(map_path, index=False)

    print(f"Original rows: {stats['original_rows']}")
    print(f"Remaining rows (kept): {stats['remaining_rows']}")
    print(f"Duplicates removed: {stats['duplicates_removed']}")
    print(f"Wrote cleaned dataset to: {clean_path}")
    print(f"Wrote removed dataset to: {removed_path}")
    print(f"Wrote cluster mapping to: {map_path}")


if __name__ == "__main__":
    main()
