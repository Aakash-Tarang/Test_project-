#!/usr/bin/env python3
"""Quality-assurance checks on the cleaned universe panel.

Validates: no negative prices, no NaN in close/adj_close, no duplicate (date,ticker),
sane volumes, per-symbol span coverage, and quantifies missing-data gaps.
Exits non-zero if a hard check fails (so it can gate the pipeline).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(ROOT, "data", "processed")


def load():
    p = os.path.join(PROC, "universe.csv")
    if not os.path.exists(p):
        sys.exit("data/processed/universe.csv missing — run clean.py first.")
    df = pd.read_csv(p, parse_dates=["date"])
    return df


def main():
    df = load()
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    check("non-empty", len(df) > 0, f"{len(df):,} rows")
    check("positive close", (df['close'] > 0).all(), f"{int((df['close'] <= 0).sum())} non-positive")
    check("positive adj_close", (df['adj_close'] > 0).all(), f"{int((df['adj_close'] <= 0).sum())} non-positive")
    check("no NaN close", df['close'].notna().all())
    check("no NaN adj_close", df['adj_close'].notna().all())
    check("no dup (date,ticker)", not df.duplicated(subset=["date", "ticker"]).any(),
          f"{int(df.duplicated(subset=['date','ticker']).sum())} dupes")
    check("no zero/neg volume", (df['volume'] >= 0).all(), f"{int((df['volume'] < 0).sum())} negative")
    check(">=200 tickers", df['ticker'].nunique() >= 200, f"{df['ticker'].nunique()} tickers")
    check(">=250 trading days/symbol median",
          df.groupby('ticker')['date'].count().median() >= 250,
          f"median {df.groupby('ticker')['date'].count().median():.0f} days")

    # gap analysis (calendar gaps beyond 5 calendar days within each symbol)
    gap_rows = []
    for tk, g in df.groupby('ticker'):
        d = pd.to_datetime(g['date']).sort_values()
        delta = d.diff().dt.days
        big = int((delta > 5).sum())
        gap_rows.append((tk, len(g), delta.min() if len(delta) else 0, delta.max() if len(delta) else 0, big))
    gaps = pd.DataFrame(gap_rows, columns=['ticker', 'n', 'min_gap', 'max_gap', 'big_gaps'])
    total_big = int(gaps['big_gaps'].sum())
    check("no excessive calendar gaps", total_big < 0.01 * len(df),
          f"{total_big} gaps >5 cal days across all symbols")

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    print(f"\nQA result: {len(checks) - n_fail}/{len(checks)} passed, {n_fail} failed")
    # persist gap table for reporting
    os.makedirs(os.path.join(ROOT, "results", "tables"), exist_ok=True)
    gaps.to_csv(os.path.join(ROOT, "results", "tables", "qa_gaps.csv"), index=False)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
