#!/usr/bin/env python3
"""
merge_custom_email_sets.py

Tailored merger for the datasets you described:
 - LLM datasets: (col1=text, col2=label). The label column in these files may be a constant;
   we ignore that and set Type based on filename (contains 'legit' or 'phish').
 - Human datasets: columns = sender,receiver,date,subject,body,urls,label. We only keep 'body'
   and map/clean the label to 'legit' or 'phishing'.

Output CSV has at least columns: Body, Type
Optionally keeps other columns if present.

Usage examples:
  python merge_custom_email_sets.py --out /mnt/data/merged_emails.csv
  # Or explicit inputs:
  python merge_custom_email_sets.py --inputs "/mnt/data/llm_legit.csv:llm,/mnt/data/llm_phishing.csv:llm,/mnt/data/human_legit.csv:human,/mnt/data/humen_phishing.csv:human" --out merged.csv
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def read_dataset_robust(path: str) -> pd.DataFrame:
    """Robust CSV reader that tries a few encodings/delimiters and falls back gracefully."""
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    seps = [",", ";", "\t", None]
    engines = ["python", "c"]
    last_err = None
    for enc in encodings:
        for sep in seps:
            for eng in engines:
                try:
                    kwargs = dict(encoding=enc, engine=eng)
                    if eng == "python":
                        kwargs["on_bad_lines"] = "skip"
                        if sep is None:
                            kwargs["sep"] = None
                        else:
                            kwargs["sep"] = sep
                    else:
                        if sep is not None:
                            kwargs["sep"] = sep
                    df = pd.read_csv(path, **kwargs)
                    if df.shape[1] == 1 and sep in (",", ";"):
                        # probably wrong delimiter, try next
                        continue
                    return df
                except Exception as e:
                    last_err = e
    raise RuntimeError(f"Could not read {path}. Last error: {last_err}")


def infer_type_from_filename(path: str) -> str:
    name = Path(path).name.lower()
    name = re.sub(r"humen", "human", name)  # fix common typo
    if "phish" in name or "phishing" in name:
        return "phishing"
    if "legit" in name or "valid" in name or "ham" in name:
        return "legit"
    # default
    return "legit"


def normalize_label_value(val: str) -> Optional[str]:
    """Try to map a label string to 'legit' or 'phishing'. Return None if unknown."""
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    if s in ("phish", "phishing", "spam", "malicious", "bad"):
        return "phishing"
    if s in ("ham", "legit", "valid", "good", "notspam", "genuine"):
        return "legit"
    # numeric labels like "0"/"1" can't be reliably mapped generically
    if s in ("0", "1"):
        # caller should maybe use filename inference; returning None to indicate unknown
        return None
    return None


def process_llm_format(path: str, target_type: Optional[str] = None) -> pd.DataFrame:
    """
    Expect first column = text, second column = label (may be constant).
    We'll create columns: Body (text) and Type (target_type or inferred from filename).
    """
    df = read_dataset_robust(path)
    if df.shape[1] < 1:
        raise ValueError(f"LLM file {path} has no columns")
    # Take first column as text content
    first_col = df.columns[0]
    body = df[first_col].astype(str)
    inferred = infer_type_from_filename(path)
    final_type = target_type if target_type is not None else inferred
    out = pd.DataFrame({"Body": body, "Type": [final_type] * len(body)})
    return out


def process_human_format(path: str, target_type: Optional[str] = None) -> pd.DataFrame:
    """
    Expect columns like sender,receiver,date,subject,body,urls,label.
    We'll attempt to locate a 'body' column (case-insensitive) and a 'label' column.
    Keep only Body and Type (mapped/cleaned).
    """
    df = read_dataset_robust(path)
    cols = {c.lower(): c for c in df.columns}
    # find body column
    body_col = None
    for candidate in ("body", "message", "content", "text", "email"):
        if candidate in cols:
            body_col = cols[candidate]
            break
    if body_col is None:
        # fallback to second-to-last column if it looks like text, else first column
        if len(df.columns) >= 3:
            body_col = df.columns[3] if len(df.columns) > 3 else df.columns[1]
        else:
            body_col = df.columns[0]
    # find label column
    label_col = None
    for candidate in ("label", "type", "class"):
        if candidate in cols:
            label_col = cols[candidate]
            break
    # build Body series
    body = df[body_col].astype(str)
    # derive Type per-row
    types = []
    for idx, row in df.iterrows():
        mapped = None
        if label_col is not None:
            mapped = normalize_label_value(row[label_col])
        if mapped is None:
            # if label couldn't be mapped, infer from filename (file-level) or provided target_type
            mapped = target_type if target_type is not None else infer_type_from_filename(path)
        types.append(mapped)
    out = pd.DataFrame({"Body": body, "Type": types})
    return out


def merge_custom(inputs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    inputs: list of (path, format_hint) where format_hint is 'llm' or 'human' or 'auto'.
    'auto' will choose processing by inspecting filename for 'llm' or 'human' or default rules.
    """
    pieces = []
    for path, fmt in inputs:
        p = Path(path)
        if not p.exists():
            print(f"[warn] file not found: {path} (skipping)")
            continue
        fmt_use = fmt.lower() if fmt else "auto"
        if fmt_use == "auto":
            name = p.name.lower()
            if "llm" in name:
                fmt_use = "llm"
            elif "human" in name or "humen" in name:
                fmt_use = "human"
            else:
                # fallback: choose human if many columns (>2), else llm
                try:
                    tmp = read_dataset_robust(path)
                    fmt_use = "human" if tmp.shape[1] > 2 else "llm"
                except Exception:
                    fmt_use = "llm"
        print(f"[merge] Processing {path} as {fmt_use} format")
        if fmt_use == "llm":
            df_piece = process_llm_format(path)
        else:
            df_piece = process_human_format(path)
        pieces.append(df_piece)

    if not pieces:
        raise ValueError("No valid input files processed.")
    merged = pd.concat(pieces, ignore_index=True, sort=False)
    # Normalize Type column (fix typos)
    merged["Type"] = merged["Type"].astype(str).str.lower().str.replace(r"phising", "phishing", regex=True)
    merged["Type"] = merged["Type"].astype(str).str.replace(r"humen", "human", regex=True)
    # keep only Body and Type
    merged = merged[["Body", "Type"]]
    # drop rows with empty body
    merged["Body"] = merged["Body"].astype(str).str.strip()
    merged = merged[merged["Body"].str.len() > 0].reset_index(drop=True)
    return merged


def main():
    ap = argparse.ArgumentParser(description="Merge custom LLM/human email datasets into Body+Type CSV.")
    ap.add_argument(
        "--inputs",
        type=str,
        default=",".join([
            "llm_legit.csv:llm",
            "llm_phishing_fixed.csv:llm",
            "human_legit.csv:human",
            "humen_phishing.csv:human",
        ]),
        help="Comma-separated list of PATH:FORMAT entries where FORMAT is 'llm' or 'human' or 'auto'.",
    )
    ap.add_argument("--out", type=str, default="/mnt/data/merged_emails.csv", help="Output CSV path")
    args = ap.parse_args()

    items = []
    for item in (s.strip() for s in args.inputs.split(",") if s.strip()):
        if ":" in item:
            path, fmt = item.split(":", 1)
            items.append((path.strip(), fmt.strip()))
        else:
            items.append((item, "auto"))

    merged = merge_custom(items)
    merged.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[merge] Saved {len(merged)} rows to {args.out}")


if __name__ == "__main__":
    main()
