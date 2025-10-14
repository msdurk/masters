#!/usr/bin/env python3
# loader.py — robust loader for your semicolon-delimited email CSV

from __future__ import annotations
import re
from typing import Tuple, Iterable, Optional
import pandas as pd

# --------- header normalization ---------

MAILBENCH_CANON_COLS = {
    "no.": "no",
    "no": "no",
    "subject": "subject",
    "body": "body",
    "sender": "sender",
    "url(s)": "urls",
    "url_s": "urls",
    "urls": "urls",
    "file": "file",
    "motivation": "motivation",
    "human evaluated emotion": "human_evaluated_emotion",
    "human_evaluated_emotion": "human_evaluated_emotion",
    "llm detected emotion": "llm_detected_emotion",
    "llm_detected_emotion": "llm_detected_emotion",
    "type": "type",
    "created by": "created_by",
    "created_by": "created_by",
    "source": "source",
    "year": "year",
}

_CTRL_CHARS = re.compile(r"[\u0000-\u001f\u007f-\u009f]")
_BOM = "\ufeff"

def _normalize_col(name: str) -> str:
    """Remove BOM/control chars, lowercase, collapse spaces, non-word→underscore, trim underscores."""
    if name is None:
        return ""
    s = str(name)
    s = s.replace(_BOM, "")
    s = _CTRL_CHARS.sub("", s)
    s = s.strip().lower()
    if s == "url(s)":
        s = "urls"
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w]+", "_", s)
    s = s.strip("_")
    return s

def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize headers and map common variants to canonical names."""
    df = df.copy()
    std_cols = [_normalize_col(c) for c in df.columns]
    if all(c == "" for c in std_cols):
        # last-resort fallback
        std_cols = [str(c).strip().lower() for c in df.columns]
    df.columns = std_cols
    df.rename(columns=MAILBENCH_CANON_COLS, inplace=True)
    return df

# --------- text building ---------

def _build_text(df: pd.DataFrame, text_cols: Tuple[str, ...]) -> pd.Series:
    """Concatenate selected columns into a single 'text' Series, robust if some are missing."""
    text = pd.Series([""] * len(df), index=df.index)
    for c in text_cols:
        c_std = c.lower()
        if c_std in df.columns:
            col = df[c_std].fillna("").astype(str)
            text = (text + " " + col).str.strip()
    return text.fillna("")

# --------- main loader (full-file) ---------

def load_mailbench_semicolon_csv(
    path: str,
    label_from: Optional[str] = "type",
    positive_value: str = "phishing",
    text_cols: Tuple[str, ...] = ("subject", "body"),
    encoding: str = "utf-8-sig",
    passthrough_cols: Tuple[str, ...] = ("subject","body","sender","urls","motivation",
                                         "year","type","created_by","source","no","file"),
) -> pd.DataFrame:
    """
    Load a semicolon-delimited CSV with quoted multi-line fields (UTF-8 with BOM).
    Returns a DataFrame with at least a 'text' column; if label_from is provided,
    also returns a binary 'label' column (1 if matches positive_value else 0).
    """
    df = pd.read_csv(
        path,
        sep=";",
        engine="python",
        quotechar='"',
        dtype=str,
        on_bad_lines="warn",
        encoding=encoding,
        header=0,
        skip_blank_lines=False,
    )

    # Recover headers if pandas produced blank column names
    bad_headers = (len(df.columns) > 0) and all((str(c).strip() == "" for c in df.columns))
    if bad_headers:
        with open(path, "r", encoding=encoding, newline="") as f:
            first = ""
            for line in f:
                if line.strip():
                    first = line
                    break
        header_tokens = [h.strip().strip('"') for h in first.rstrip("\r\n").split(";")]
        if len(header_tokens) == len(df.columns):
            df.columns = header_tokens
        else:
            expected = [
                "No.", "Subject", "Body", "Sender", "URL(s)", "File",
                "Motivation", "Human evaluated Emotion", "LLM detected emotion",
                "Type", "Created by", "Source", "Year",
            ]
            if len(df.columns) == len(expected):
                df.columns = expected
            else:
                raise ValueError(
                    f"Could not recover headers. Found {len(df.columns)} columns; "
                    f"first-line tokens={len(header_tokens)}"
                )

    df = _standardize_cols(df)

    # Build 'text'
    text = _build_text(df, text_cols)

    out = pd.DataFrame({"text": text})

    # Optional label
    if label_from is not None:
        labcol = label_from.lower()
        if labcol not in df.columns:
            raise ValueError(
                f"Label column '{label_from}' not found after standardization. "
                f"Available columns: {sorted(df.columns.tolist())}"
            )
        out["label"] = (
            df[labcol].fillna("").str.strip().str.lower() == str(positive_value).lower()
        ).astype(int)

    # Passthrough useful columns if present
    for keep in passthrough_cols:
        if keep in df.columns:
            out[keep] = df[keep]

    return out

# --------- chunked iterator (for very large files) ---------

def iter_mailbench_csv(
    path: str,
    chunksize: int = 5000,
    label_from: Optional[str] = "type",
    positive_value: str = "phishing",
    text_cols: Tuple[str, ...] = ("subject", "body"),
    encoding: str = "utf-8-sig",
    passthrough_cols: Tuple[str, ...] = ("subject","body","sender","urls","motivation",
                                         "year","type","created_by","source","no","file"),
) -> Iterable[pd.DataFrame]:
    """
    Yield normalized chunks with at least 'text' (and optionally 'label').
    """
    for chunk in pd.read_csv(
        path,
        sep=";",
        engine="python",
        quotechar='"',
        dtype=str,
        on_bad_lines="warn",
        encoding=encoding,
        header=0,
        skip_blank_lines=False,
        chunksize=chunksize,
    ):
        chunk = _standardize_cols(chunk)
        text = _build_text(chunk, text_cols)

        out = pd.DataFrame({"text": text})
        if label_from is not None:
            labcol = label_from.lower()
            if labcol in chunk.columns:
                out["label"] = (
                    chunk[labcol].fillna("").str.strip().str.lower() == str(positive_value).lower()
                ).astype(int)
            else:
                # keep going but without label
                out["label"] = pd.NA

        for keep in passthrough_cols:
            if keep in chunk.columns:
                out[keep] = chunk[keep]

        yield out

# --------- small helper (optional) ---------

def basic_engineered_features(texts: pd.Series) -> pd.DataFrame:
    """Tiny feature set you may want later; safe to remove if not needed."""
    s = texts.fillna("").astype(str)
    n_chars = s.str.len()
    n_urls = s.str.count(r"http[s]?://|www\.")
    cap_ratio = s.apply(lambda t: (sum(ch.isupper() for ch in t) / max(1, len(t))) if t else 0.0)
    return pd.DataFrame({"n_chars": n_chars, "n_urls": n_urls, "cap_ratio": cap_ratio})
