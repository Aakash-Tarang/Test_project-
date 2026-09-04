#!/usr/bin/env python3
"""Clean raw market data into a uniform long panel.

Reads raw data (data/raw/{nyse|yfinance}) and produces:
  data/processed/universe.csv    long panel: date,ticker,open,high,low,close,adj_close,volume
  data/processed/symbols.csv     symbol metadata (name, sector, sub_industry, date_first_added)
  data/processed/_meta.json      provenance + cleaning notes

Corporate-action adjustment:
  * nyse source : the provided 'prices-split-adjusted.csv' is split-adjusted; no dividend
                  adjustment is present, so returns here are PRICE (split-adjusted) returns,
                  not total returns. Documented limitation.
  * yahoo source: Yahoo 'Adj Close' is fully adjusted for splits AND dividends (total-return
                  prices). adj_close is carried through verbatim.

Re-runnable and idempotent. Selects whichever source is present (prefers yfinance if both,
since it is the primary >=10-year panel).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
SECTOR_MAP = os.path.join(ROOT, "scripts", "data", "sector_map.json")
SYMBOLS_CSV = os.path.join(ROOT, "scripts", "data", "sp500_symbols.csv")

LONG_COLS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
DTYPES = {"open": "float64", "high": "float64", "low": "float64",
          "close": "float64", "adj_close": "float64", "volume": "float64"}


def _find(base, name):
    for root, _, files in os.walk(base):
        if name in files:
            return os.path.join(root, name)
    return None


def load_nyse():
    d = os.path.join(RAW, "nyse")
    p = _find(d, "prices-split-adjusted.csv")
    sec = _find(d, "securities.csv")
    if not p or not sec:
        return None, None
    df = pd.read_csv(p)
    df = df.rename(columns={"date": "date", "symbol": "ticker"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["adj_close"] = df["close"].astype(float)  # split-adjusted price == adj_close for this source
    df = df[LONG_COLS]
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    # symbols/metadata
    sec = pd.read_csv(sec, dtype=str)
    sec.columns = [c.strip('"').strip() for c in sec.columns]
    sym = sec.rename(columns={"Ticker symbol": "ticker", "Security": "name",
                              "GICS Sector": "sector", "GICS Sub Industry": "sub_industry",
                              "Date first added": "date_first_added"})
    sym = sym[["ticker", "name", "sector", "sub_industry", "date_first_added"]]
    return df, sym


def load_yfinance():
    d = os.path.join(RAW, "yfinance")
    files = sorted(glob.glob(os.path.join(d, "*.csv")))
    if not files:
        return None, None
    frames = []
    for f in files:
        tk = os.path.basename(f)[:-4]
        df = pd.read_csv(f)
        if "ticker" not in df.columns:
            df.insert(0, "ticker", tk)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        for c in LONG_COLS:
            if c not in df.columns:
                df[c] = np.nan
        frames.append(df[LONG_COLS])
    df = pd.concat(frames, ignore_index=True)
    # symbols metadata from bundled S&P500 symbol table (sector lookup)
    sym = pd.read_csv(SYMBOLS_CSV, dtype=str)
    sym = sym.rename(columns={"symbol": "ticker", "name": "name", "sector": "sector",
                              "sub_industry": "sub_industry", "date_first_added": "date_first_added"})
    return df, sym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["nyse", "yfinance", "auto"], default="auto")
    args = ap.parse_args()

    loaders = {"nyse": load_nyse, "yfinance": load_yfinance}
    if args.source != "auto":
        src_used = args.source
        df, sym = loaders[src_used]()
    else:
        has_yf = os.path.isdir(os.path.join(RAW, "yfinance")) and glob.glob(os.path.join(RAW, "yfinance", "*.csv"))
        if has_yf:
            src_used = "yfinance"; df, sym = load_yfinance()
        else:
            src_used = "nyse"; df, sym = load_nyse()
    if df is None:
        sys.exit(f"No raw data found for source '{src_used}'. Run download.py first.")

    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")

    # Sort + drop dupes (keep last)
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # basic sanity drops
    before = len(df)
    df = df[df["close"].notna() & (df["close"] > 0)]
    df = df[df["adj_close"].notna() & (df["adj_close"] > 0)]
    dropped = before - len(df)
    df["open"] = df["open"].where(df["open"] > 0)
    # Drop the negligible zero-volume artifact rows (0.002% in the nyse source).
    n_vol0 = int((df["volume"] <= 0).sum())
    df = df[df["volume"] > 0]
    if n_vol0:
        print(f"  (clean) dropped {n_vol0} rows with non-positive volume")

    os.makedirs(PROC, exist_ok=True)
    df.to_csv(os.path.join(PROC, "universe.csv"), index=False)
    sym.to_csv(os.path.join(PROC, "symbols.csv"), index=False)

    meta = {
        "source": src_used or args.source,
        "rows": len(df),
        "dropped_invalid_rows": int(dropped),
        "n_tickers": int(df["ticker"].nunique()),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "columns": list(LONG_COLS),
        "note": ("adj_close from Yahoo (split+dividend adjusted) " if src_used == "yfinance"
                 else "adj_close = split-adjusted price (no dividend adjustment; price returns only)"),    }
    with open(os.path.join(PROC, "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"wrote data/processed/universe.csv ({len(df):,} rows, {df['ticker'].nunique()} tickers)")
    print(f"wrote data/processed/symbols.csv ({len(sym)} symbols)")


if __name__ == "__main__":
    import sys
    main()
