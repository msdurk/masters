#!/usr/bin/env python3
"""
download_from_local_json.py
Download all .csv / .csv.gz from a Zenodo record JSON that has:
  "files": { "entries": { "<name>.csv": { ... "links": {"content": "<url>"} } } }

Usage (from phishing_detection/data/raw/):
  python download_from_local_json.py --meta 8339691.json --out .
"""
import argparse, json, hashlib, sys
from pathlib import Path
from typing import Optional, Tuple

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: requests. Install with `pip install requests`")

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # progress bar optional

CSV_EXTS = (".csv", ".csv.gz", ".CSV", ".CSV.GZ")

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
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def download(url: str, dest: Path, algo: Optional[str], hexd: Optional[str]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "zenodo-downloader/1.0"}
    with requests.get(url, stream=True, headers=headers, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0"))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) if tqdm else None
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1048576):
                if chunk:
                    f.write(chunk)
                    if bar: bar.update(len(chunk))
        if bar: bar.close()
    if algo and hexd:
        got = compute_checksum(tmp, algo)
        if got.lower() != hexd.lower():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {dest.name}: expected {algo}:{hexd}, got {got}")
    tmp.rename(dest)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="Path to local JSON (e.g., 8339691.json)")
    ap.add_argument("--out", required=True, help="Output folder (e.g., .)")
    args = ap.parse_args()

    meta_path = Path(args.meta)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = json.loads(meta_path.read_text(encoding="utf-8"))

    # Your JSON layout: files -> entries -> {name: { ... links: {content, self} ...}}
    files = []
    files_node = (doc.get("files") or {}).get("entries") or {}
    for name, info in files_node.items():
        if not any(name.endswith(ext) for ext in CSV_EXTS):
            continue
        links = info.get("links") or {}
        url = links.get("content") or links.get("self")
        if not url:
            continue
        algo, hexd = parse_checksum(info.get("checksum"))
        files.append((name, url, algo, hexd))

    if not files:
        sys.exit("No .csv or .csv.gz entries found under files.entries.")

    print(f"Found {len(files)} CSV-like files. Downloading to {out_dir.resolve()} ...")
    ok = fail = 0
    for name, url, algo, hexd in files:
        dest = out_dir / name
        try:
            if dest.exists():
                if algo and hexd:
                    got = compute_checksum(dest, algo)
                    if got.lower() == hexd.lower():
                        print(f"[skip] {name} already exists (checksum OK).")
                        ok += 1
                        continue
                    else:
                        print(f"[redo] {name} exists but checksum mismatch; re-downloading.")
                else:
                    print(f"[skip] {name} already exists (no checksum to verify).")
                    ok += 1
                    continue
            download(url, dest, algo, hexd)
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
