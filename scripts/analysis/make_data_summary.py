#!/usr/bin/env python3
"""Produce the dataset summary tables and figures for report Section 3 (Data).

Outputs:
  results/tables/data_summary.csv      summary statistics by metric
  results/tables/universe_by_year.csv  number of assets traded each calendar year
  results/tables/sector_counts.csv     sector composition
  results/figures/universe_by_year.png  assets per year bar chart
  results/figures/sector_comp.png       sector composition
All numbers derive from data/processed/universe.csv (+symbols.csv). Nothing is
hand-typed here.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(ROOT, "data", "processed")
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def main():
    df = pd.read_csv(os.path.join(PROC, "universe.csv"), parse_dates=["date"])
    sym = pd.read_csv(os.path.join(PROC, "symbols.csv"))
    with open(os.path.join(PROC, "_meta.json")) as f:
        meta = json.load(f)

    df = df.sort_values(["ticker", "date"])
    df["year"] = df["date"].dt.year
    df["ret"] = df.groupby("ticker")["adj_close"].transform(lambda s: s.pct_change())

    # ---- universe by year ----
    by_year = df.groupby("year")["ticker"].nunique().reset_index()
    by_year.columns = ["year", "n_tickers"]
    by_year.to_csv(os.path.join(TAB, "universe_by_year.csv"), index=False)

    # ---- sector composition ----
    sec = sym[["ticker", "sector"]].drop_duplicates().dropna(subset=["sector"])
    sector_counts = sec["sector"].value_counts().rename_axis("sector").reset_index(name="n_tickers")
    sector_counts.to_csv(os.path.join(TAB, "sector_counts.csv"), index=False)

    # ---- per-symbol span + return stats ----
    g = df.groupby("ticker")
    span = g["date"].agg(["min", "max", "count"]).rename(columns={"min": "first", "max": "last", "count": "n_days"})
    span = span.join(g["ret"].mean().rename("mean_daily_ret"))
    span = span.join(g["ret"].std().rename("std_daily_ret"))
    span["n_years"] = (span["last"] - span["first"]).dt.days / 365.25

    summary_rows = [
        ("rows", int(len(df))),
        ("n_tickers", int(df["ticker"].nunique())),
        ("date_min", str(df["date"].min().date())),
        ("date_max", str(df["date"].max().date())),
        ("span_years", round((df["date"].max() - df["date"].min()).days / 365.25, 2)),
        ("median_n_days_per_ticker", int(span["n_days"].median())),
        ("median_n_years_per_ticker", round(float(span["n_years"].median()), 2)),
        ("pct_tickers_ge5yrs", round(float((span["n_years"] >= 5).mean() * 100), 2)),
        ("pct_tickers_ge7yrs", round(float((span["n_years"] >= 7).mean() * 100), 2)),
        ("cross_sectional_median_ann_vol_pct", round(float((span["std_daily_ret"].median() * np.sqrt(252) * 100)), 2)),
        ("n_sectors", int(sector_counts.shape[0])),
        ("source", meta.get("source", "?")),
        ("adj_close_note", meta.get("note", "?")),
    ]
    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])
    summary.to_csv(os.path.join(TAB, "data_summary.csv"), index=False)

    # ---- figures ----
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_year["year"].astype(str), by_year["n_tickers"], color="#4C72B0")
    ax.set_title("Number of assets with valid daily data, by year")
    ax.set_xlabel("Year"); ax.set_ylabel("Tickers")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "universe_by_year.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sc = sector_counts.sort_values("n_tickers")
    ax.barh(sc["sector"], sc["n_tickers"], color="#55A868")
    ax.set_title("Sector composition of the universe"); ax.set_xlabel("Tickers")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "sector_comp.png"), dpi=150); plt.close(fig)

    print("wrote results/tables/{data_summary,universe_by_year,sector_counts}.csv")
    print("wrote results/figures/{universe_by_year,sector_comp}.png")


if __name__ == "__main__":
    main()
