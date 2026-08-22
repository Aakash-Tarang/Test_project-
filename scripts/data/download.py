#!/usr/bin/env python3
"""Download raw market data.

TWO MODES (pick with --source):

  --source github   (DEFAULT, works in this sandbox)
      Downloads the classic NYSE (Kaggle) S&P-500-style dataset from GitHub as a
      repo tarball into data/raw/nyse/. Real data: 501 US securities, split-adjusted
      daily OHLCV, GICS sectors, quarterly fundamentals, span 2010-01-04..2016-12-30.
      The GitHub API / codeload tarball route is the only data path reachable in this
      sandbox (Yahoo/stooq/Nasdaq are blocked at runtime here).

  --source yahoo   (RUN ON YOUR OWN MACHINE for a >=10 year primary panel)
      Fetches daily OHLCV + Adj Close from Yahoo Finance (via `yfinance`) for each
      ticker in --ticker-file (default scripts/data/sp500_tickers.txt), for
      --start..--end (default 2010-01-01..today), saving one CSV per ticker to
      data/raw/yfinance/. Yahoo's Adj Close is fully adjusted for splits AND
      dividends, so it satisfies the corporate-action-adjustment requirement.
      Requires internet + `pip install yfinance`. Use:  python3 scripts/data/download.py --source yahoo

Always writes a provenance manifest (source, url/ref, retrieval date, checksums) to
data/manifest/. Idempotent; safe to re-run.
"""
import argparse
import hashlib
import json
import os
import sys
import tarfile
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "raw")
MANIFEST_DIR = os.path.join(ROOT, "data", "manifest")

# Source of truth for the in-sandbox GitHub dataset.
GITHUB_REPO = "ashishpatel26/NYSE-STOCK_MARKET-ANALYSIS-USING-LSTM"
GITHUB_REF = "HEAD"


def sha256(path, blocksize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(blocksize), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_checksums(dirpath):
    out = {}
    for root, _, files in os.walk(dirpath):
        for name in sorted(files):
            p = os.path.join(root, name)
            rel = os.path.relpath(p, dirpath)
            out[rel] = sha256(p)
    return out


def write_manifest(source, meta, files_dir):
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    manifest = {
        "source": source,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        "files": manifest_checksums(files_dir),
    }
    path = os.path.join(MANIFEST_DIR, f"{source}.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [manifest] wrote {path}")


def download_github():
    import subprocess, urllib.request
    dest_dir = os.path.join(RAW, "nyse")
    os.makedirs(RAW, exist_ok=True)
    tarball = os.path.join(RAW, "_nyse_src.tar.gz")
    url = f"https://api.github.com/repos/{GITHUB_REPO}/tarball/{GITHUB_REF}"
    print(f"==> downloading {GITHUB_REPO}@{GITHUB_REF} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "statarbsim-data"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = r.read()
    with open(tarball, "wb") as f:
        f.write(data)
    print(f"  downloaded {len(data)} bytes")

    # Extract, stripping the single top-level repo folder.
    if os.path.isdir(dest_dir):
        import shutil
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)
    with tarfile.open(tarball, "r:gz") as tf:
        members = tf.getmembers()
        # remove top-level dir component (Python <3.12 has no filter=, so sanitize manually)
        safe = []
        for m in members:
            parts = m.name.split("/", 1)
            m.name = parts[1] if len(parts) == 2 else ""
            if not m.name or m.name.startswith("..") or "/.." in m.name:
                continue
            safe.append(m)
        tf.extractall(dest_dir, members=safe)
    os.remove(tarball)
    print("  extracted to data/raw/nyse/")

    write_manifest("github_nyse", {"repo": GITHUB_REPO, "ref": GITHUB_REF, "url": url}, dest_dir)
    return dest_dir


def download_yahoo(ticker_file, start, end, interval):
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance not installed. Run: python3 -m pip install yfinance   (then re-run)")
    with open(ticker_file) as f:
        tickers = [t.strip() for t in f if t.strip()]
    dest_dir = os.path.join(RAW, "yfinance")
    os.makedirs(dest_dir, exist_ok=True)
    failed = []
    print(f"==> fetching {len(tickers)} tickers from Yahoo Finance {start}..{end}")
    for i, tk in enumerate(tickers, 1):
        out = os.path.join(dest_dir, f"{tk}.csv")
        try:
            df = yf.download(tk, start=start, end=end, interval=interval,
                             auto_adjust=False, progress=False, threads=False)
            if df is None or df.empty:
                failed.append(tk); continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.reset_index().rename(columns={"Date": "date", "Open": "open",
                                                  "High": "high", "Low": "low",
                                                  "Close": "close", "Adj Close": "adj_close",
                                                  "Volume": "volume"})
            df.insert(0, "ticker", tk)
            df.to_csv(out, index=False)
        except Exception as e:
            failed.append(f"{tk}:{type(e).__name__}")
        if i % 25 == 0:
            print(f"  ... {i}/{len(tickers)} done")
    print(f"  done. failed/empty: {len(failed)}")
    if failed:
        print("  failed:", failed[:50])
    write_manifest("yahoo_yfinance", {"ticker_file": ticker_file, "start": start,
                                      "end": end, "interval": interval}, dest_dir)
    return dest_dir


def main():
    import pandas as pd  # noqa: F401  (used by yahoo path)
    ap = argparse.ArgumentParser(description="Download raw market data.")
    ap.add_argument("--source", choices=["github", "yahoo"], default="github")
    ap.add_argument("--ticker-file", default=os.path.join(ROOT, "scripts", "data", "sp500_tickers.txt"))
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="")
    ap.add_argument("--interval", default="1d")
    args = ap.parse_args()
    if args.source == "github":
        download_github()
    else:
        end = args.end or datetime.now().strftime("%Y-%m-%d")
        download_yahoo(args.ticker_file, args.start, end, args.interval)
    print("done.")


if __name__ == "__main__":
    main()
