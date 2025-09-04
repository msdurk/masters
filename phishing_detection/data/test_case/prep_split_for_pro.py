
"""
Split a single labeled CSV into two files for pro_train.py:
  - <prefix>_legit.csv
  - <prefix>_phishing.csv

Auto-detects text/label columns, with CLI overrides.
Usage:
  python prep_split_for_pro.py Phishing_Email_hugging.csv --out-prefix hug
  # (optional) override columns if needed:
  python prep_split_for_pro.py Phishing_Email_hugging.csv --text-col "Email Text" --label-col Label --out-prefix hug
"""
import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np

PREFERRED_TEXT_COLS = ["text","body","content","email","message","subject","body_text","Email Text","Email","Message","Body"]
POSSIBLE_LABEL_COLS = ["label","labels","class","target","y","category","is_phishing","spam","phish","Label","Class","Category"]

PHISH_VALUES = {"phishing","spam","malicious","fraud","scam","phish","bad","1","true","yes"}
LEGIT_VALUES  = {"legit","ham","benign","legitimate","safe","clean","good","0","false","no","not phishing","not_phishing"}

def load_csv_lenient(path: Path) -> pd.DataFrame:
    tries = [
        dict(encoding="utf-8", sep=",", engine="python", on_bad_lines="skip"),
        dict(encoding="utf-8", sep=None, engine="python", on_bad_lines="skip"),
        dict(encoding="latin-1", sep=",", engine="python", on_bad_lines="skip"),
        dict(encoding="latin-1", sep=None, engine="python", on_bad_lines="skip"),
    ]
    last_err = None
    for kw in tries:
        try:
            return pd.read_csv(path, **kw)
        except Exception as e:
            last_err = e
    raise last_err

def pick_text_column(df: pd.DataFrame, override: str|None) -> str:
    if override and override in df.columns:
        return override
    for c in PREFERRED_TEXT_COLS:
        if c in df.columns:
            return c
    obj = [c for c in df.columns if df[c].dtype == "object"]
    if not obj:
        return df.columns[0]
    # longest average string length
    avg = [(c, df[c].astype(str).str.len().mean()) for c in obj]
    avg.sort(key=lambda x: x[1], reverse=True)
    return avg[0][0]

def pick_label_column(df: pd.DataFrame, override: str|None) -> str|None:
    if override and override in df.columns:
        return override
    for c in POSSIBLE_LABEL_COLS:
        if c in df.columns:
            return c
    # heuristic: any column with only few unique values
    candidates = [c for c in df.columns if df[c].nunique(dropna=True) <= 6 and c != "index"]
    return candidates[0] if candidates else None

def normalize_labels(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    mapping = {}
    # map common numeric/bool-like to strings first
    s = s.replace({"1":"1","0":"0","true":"true","false":"false"})
    for v in s.unique():
        if v in PHISH_VALUES:
            mapping[v] = "phishing"
        elif v in LEGIT_VALUES:
            mapping[v] = "legit"
    # fallbacks for pure integers
    if not mapping and set(s.unique()) <= {"0","1"}:
        mapping = {"1":"phishing","0":"legit"}
    # if still empty, try majority rules by keyword
    if not mapping:
        mapping = {v: ("phishing" if any(k in v for k in ["phish","spam","fraud","scam","bad"]) else "legit")
                   for v in s.unique()}
    return s.map(mapping)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to single labeled CSV (e.g., Phishing_Email_hugging.csv)")
    ap.add_argument("--text-col", default=None, help="Name of the text column if auto-detect fails")
    ap.add_argument("--label-col", default=None, help="Name of the label column if auto-detect fails")
    ap.add_argument("--out-prefix", default="hug", help="Prefix for output files")
    args = ap.parse_args()

    path = Path(args.csv)
    df = load_csv_lenient(path)
    if df.empty:
        sys.exit("Input CSV is empty.")

    text_col = pick_text_column(df, args.text_col)
    label_col = pick_label_column(df, args.label_col)
    if label_col is None:
        sys.exit(f"Could not find a label column. Pass --label-col. Columns: {list(df.columns)}")

    y = normalize_labels(df[label_col])
    if y.isna().mean() > 0.05:
        # too many unknown values
        uniq = df[label_col].dropna().unique().tolist()
        sys.exit(f"Could not map many label values. Add --label-col and ensure values are one of "
                 f"phishing/legit, spam/ham, or 1/0. Found: {uniq[:20]}")

    texts = df[text_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out = pd.DataFrame({"text": texts, "label": y})

    legit = out[out["label"] == "legit"][["text"]]
    phish = out[out["label"] == "phishing"][["text"]]
    o_legit = Path(f"{args.out_prefix}_legit.csv")
    o_phish = Path(f"{args.out_prefix}_phishing.csv")
    legit.to_csv(o_legit, index=False)
    phish.to_csv(o_phish, index=False)

    print(f"Detected text column:  {text_col}")
    print(f"Detected label column: {label_col}")
    print(f"Wrote {len(legit)} legit rows -> {o_legit}")
    print(f"Wrote {len(phish)} phishing rows -> {o_phish}")

if __name__ == "__main__":
    main()
