#!/usr/bin/env python3
"""
download_zenodo_csvs.py
Download all .csv / .csv.gz files referenced in a Zenodo record JSON into a target folder.

Usage:
  python download_zenodo_csvs.py --meta data/raw/8339691.json --out data/raw
"""
import argparse, json, os, re, sys, hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    import requests
except ImportError:
    raise SystemExit("Missing dependency: requests. Install with `pip install requests`")

def iter_files_nodes(obj: Any):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() == "files" and isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        yield it
            elif isinstance(v, (dict, list)):
                yield from iter_files_nodes(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from iter_files_nodes(x)

def pick_download_url(entry: Dict[str, Any]) -> Optional[str]:
    links = entry.get("links") or {}
    return links.get("download") or links.get("self") or entry.get("download_url") or entry.get("href")

def parse_checksum(s: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not s:
        return None, None
    if ":" in s:
        algo, hexd = s.split(":", 1)
        return algo.strip().lower(), hexd.strip().lower()
    return "md5", s.strip().lower()

def compute_checksum(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download_file(url: str, dest: Path, expected_algo: Optional[str], expected_hex: Optional[str]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "zenodo-csv-downloader/1.0"}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0"))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) if tqdm else None
        tmp_path = dest.with_suffix(dest.suffix + ".part")
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    if bar: bar.update(len(chunk))
        if bar: bar.close()
    if expected_algo and expected_hex:
        got = compute_checksum(tmp_path, expected_algo)
        if got.lower() != expected_hex.lower():
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {dest.name}: expected {expected_algo}:{expected_hex}, got {got}")
    tmp_path.rename(dest)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="Path to Zenodo JSON (e.g., data/raw/8339691.json)")
    ap.add_argument("--out", required=True, help="Folder to save CSV files (e.g., data/raw)")
    args = ap.parse_args()

    meta_path = Path(args.meta)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    entries = list(iter_files_nodes(doc))
    if not entries:
        raise SystemExit("No 'files' entries found in JSON.")

    to_get = []  # (name, url, algo, hex)
    for e in entries:
        name = e.get("key") or e.get("filename") or e.get("name") or ""
        url = pick_download_url(e)
        if not name or not url:
            continue
        lower = name.lower()
        if lower.endswith(".csv") or lower.endswith(".csv.gz"):
            algo, hexv = parse_checksum(e.get("checksum") or e.get("md5") or e.get("sha256"))
            to_get.append((name, url, algo, hexv))

    if not to_get:
        raise SystemExit("No .csv or .csv.gz files found in the JSON 'files' list.")

    print(f"Found {len(to_get)} CSV-like files. Downloading to {out_dir} ...")
    ok = fail = 0
    for name, url, algo, hexv in to_get:
        dest = out_dir / name
        try:
            if dest.exists():
                if algo and hexv:
                    got = compute_checksum(dest, algo)
                    if got.lower() == hexv.lower():
                        print(f"[skip] {name} exists and checksum matches.")
                        ok += 1
                        continue
                    else:
                        print(f"[redo] {name} checksum mismatch; re-downloading.")
                else:
                    print(f"[skip] {name} exists (no checksum to verify).");
                    ok += 1
                    continue
            download_file(url, dest, algo, hexv)
            print(f"[ok] {name}")
            ok += 1
        except Exception as e:
            print(f"[fail] {name}: {e}")
            fail += 1
    print(f"Done. {ok} ok, {fail} failed.")
    if fail:
        sys.exit(1)

if __name__ == "__main__":
    main()
