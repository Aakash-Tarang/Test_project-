#!/usr/bin/env python3
"""Export a wide matrix of adjusted-close prices for a target + basket to feed the
C++ backtest engine, from the cleaned universe panel.

Writes (to data/processed/engine/, git-ignored):
  <target>.csv   columns: date,target,basket1..basketN  (adjusted close, common dates)
  <target>.json  {target, basket, n_assets, n_obs, date_first, date_last, window...}
"""
import argparse
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(ROOT, "data", "processed")
ENGINE = os.path.join(PROC, "engine")
UNI = os.path.join(PROC, "universe.csv")
SYM = os.path.join(PROC, "symbols.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="ADBE")
    ap.add_argument("--basket", nargs="+", default=["CRM", "ADSK", "INTU"])
    args = ap.parse_args()
    tickers = [args.target] + list(args.basket)

    u = pd.read_csv(UNI, parse_dates=["date"])
    piv = u.pivot_table(index="date", columns="ticker", values="adj_close")
    # drop rows with any missing across our tickers (keep a strict balanced panel)
    sub = piv[tickers].dropna()
    sub = sub[(sub > 0).all(axis=1)]
    sym = pd.read_csv(SYM)

    os.makedirs(ENGINE, exist_ok=True)
    csv_path = os.path.join(ENGINE, f"{args.target}.csv")
    sub.reset_index().to_csv(csv_path, index=False)

    sector = {}
    for _, r in sym.iterrows():
        sector[r["ticker"]] = r["sector"]
    cfg = {
        "target": args.target,
        "basket": list(args.basket),
        "sectors": {t: sector.get(t, "?") for t in tickers},
        "n_obs": int(len(sub)),
        "date_first": str(sub.index.min().date()),
        "date_last": str(sub.index.max().date()),
        "csv": csv_path,
    }
    with open(os.path.join(ENGINE, f"{args.target}.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(json.dumps(cfg, indent=2))
    print(f"wrote {csv_path} ({len(sub)} rows x {len(tickers)} cols)")
    print(f"date range {cfg['date_first']} .. {cfg['date_last']}")


if __name__ == "__main__":
    main()
