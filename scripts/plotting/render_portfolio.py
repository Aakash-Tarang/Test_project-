#!/usr/bin/env python3
"""Render Part 7 portfolio figures + LaTeX tables from the live portfolio run.

Reads results/tables/portfolio_{summary,daily,regime,heat_ez,heat_ridge}.csv and
writes:
  results/figures/portfolio_equity.png     portfolio NAV (inv-vol vs equal, net/gross)
  results/figures/portfolio_allocation.png inverse-vol weight allocations over time
  results/figures/portfolio_risk.png       daily P&L with VaR + drawdown
  results/figures/portfolio_heat_ez.png    lookback x entry-z net Sharpe (ADBE)
  results/figures/portfolio_heat_ridge.png lookback x ridge-lambda net Sharpe (ADBE)
  report/tables/portfolio.tex              per-strategy + portfolio + regime tables
"""
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
REPORT_FIG = os.path.join(ROOT, "report", "figures")
REPORT_TAB = os.path.join(ROOT, "report", "tables")
KEYS = ["ADBE", "AVGO", "GS", "CAT"]
COLS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
STRESS = [("COVID-2020", "2020-02-19", "2020-12-31"),
          ("2022 selloff", "2022-01-03", "2022-12-30")]


def main():
    summ = pd.read_csv(os.path.join(TAB, "portfolio_summary.csv"))
    daily = pd.read_csv(os.path.join(TAB, "portfolio_daily.csv"), parse_dates=["date"])
    reg = pd.read_csv(os.path.join(TAB, "portfolio_regime.csv"))
    heat_ez = pd.read_csv(os.path.join(TAB, "portfolio_heat_ez.csv"))
    heat_rl = pd.read_csv(os.path.join(TAB, "portfolio_heat_ridge.csv"))

    os.makedirs(FIG, exist_ok=True)
    os.makedirs(REPORT_FIG, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    nav_iv = np.cumsum(daily["pnl_iv"].values)
    nav_eq = np.cumsum(daily["pnl_eq"].values)
    # gross (net-of-cost per-strategy sum under equal weight is net; gross = equal of
    # per-strategy gross). We recompute gross only if available; else plot net only.
    date = daily["date"]

    # ---------- fig 1: portfolio NAV ----------
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(date, nav_iv, label="Portfolio net (inverse-vol, capped)", color="#C44E52", lw=1.6)
    ax.plot(date, nav_eq, label="Portfolio net (equal weight)", color="#4C72B0", lw=1.2, alpha=0.85)
    for _, lb, a, b in [("s", *s) for s in STRESS]:
        m = (date >= a) & (date <= b)
        if m.any():
            ax.axvspan(date[m].min(), date[m].max(), color="0.85", alpha=0.5)
    ax.set_title("Book of four market-neutral spread strategies (net of 10 bps) — "
                 "ADBE, AVGO semis, GS-BAC, CAT-HON")
    ax.set_ylabel("Cumulative log P&L")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "portfolio_equity.png"), dpi=150)
    plt.close(fig)

    # ---------- fig 2: allocation weights ----------
    fig, ax = plt.subplots(figsize=(10, 4))
    W = daily[[f"w_{k}" for k in KEYS]].values
    ax.stackplot(date, W.T, labels=KEYS, colors=COLS, alpha=0.85)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of book capital")
    ax.set_title("Inverse-vol allocation (monthly rebalance, per-name cap 45%)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "portfolio_allocation.png"), dpi=150)
    plt.close(fig)

    # ---------- fig 3: daily P&L + VaR + drawdown ----------
    pnl = daily["pnl_iv"].values
    var95 = np.quantile(pnl, 0.05)
    dd = nav_iv - np.maximum.accumulate(nav_iv)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1]})
    axes[0].bar(date, pnl, color=np.where(pnl >= 0, "#55A868", "#C44E52"), width=1.0)
    axes[0].axhline(var95, color="#4C72B0", ls="--", lw=1.5, label=f"VaR95 = {var95:.4f}")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("daily P&L"); axes[0].legend(fontsize=9)
    axes[0].set_title("Portfolio daily P&L and drawdown (inverse-vol book)")
    axes[1].fill_between(date, dd, 0, color="#8172B3", alpha=0.6)
    axes[1].set_ylabel("drawdown (log)"); axes[1].set_xlabel("")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "portfolio_risk.png"), dpi=150)
    plt.close(fig)

    # ---------- fig 4/5: heatmaps ----------
    def heat(ax, df, xlab, ylab, title):
        X = df["window"].values
        cols = [c for c in df.columns if c != "window"]
        vals = df[cols].values.astype(float)
        im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=-0.6, vmax=0.3)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=8)
        ax.set_yticks(range(len(X))); ax.set_yticklabels(X, fontsize=9)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax)
        return ax

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    heat(ax, heat_ez, "entry z-threshold", "lookback window",
         "Net Sharpe vs window x entry-z (ADBE book)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "portfolio_heat_ez.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    heat(ax, heat_rl, "ridge lambda", "lookback window",
         "Net Sharpe vs window x ridge-lambda (ADBE book)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "portfolio_heat_ridge.png"), dpi=150)
    plt.close(fig)

    for f in ("portfolio_equity.png", "portfolio_allocation.png", "portfolio_risk.png",
              "portfolio_heat_ez.png", "portfolio_heat_ridge.png"):
        shutil.copy2(os.path.join(FIG, f), os.path.join(REPORT_FIG, f))

    # ---------- LaTeX tables ----------
    os.makedirs(REPORT_TAB, exist_ok=True)
    s = summ.set_index(["strategy", "alloc"])
    L = ["% Auto-generated by scripts/plotting/render_portfolio.py",
         "\\begin{table}[H]\\centering\\small",
         "\\caption{Book of four concurrent market-neutral spread strategies (each "
         "OLS 120-bar causal hedge, 10 bps one-way cost, net P&L). Individual books "
         "and the combined book under equal-weight and inverse-vol (monthly-rebalanced, "
         "per-name cap 45\\%) allocation. VaR/CVaR are historical on daily P&L at 95\\%.}",
         "\\label{tab:portfolio_summary}",
         "\\begin{tabular}{llccccc}\\toprule",
         "\\textbf{Book} & \\textbf{Alloc} & \\textbf{Sharpe} & \\textbf{ann. ret} "
         "& \\textbf{maxDD} & \\textbf{VaR95} & \\textbf{CVaR95} \\\\ \\midrule"]
    for k in KEYS + ["portfolio"]:
        for alloc in (["solo"] if k != "portfolio" else ["equal_wt", "inv_vol"]):
            try:
                row = s.loc[(k, alloc)]
            except KeyError:
                continue
            L.append("%s & %s & %.2f & %.4f & %.3f & %.5f & %.5f \\\\" % (
                k.replace("_", "\\_"), alloc.replace("_", "\\_"),
                row["sharpe"], row["ann_ret"], row["maxdd"], row["var95"], row["cvar95"]))
    L.append("\\bottomrule\\end{tabular}\\end{table}")

    # regime table
    L.append("")
    L.append("\\begin{table}[H]\\centering\\small")
    L.append("\\caption{Portfolio performance by realized-volatility regime and in "
             "the stress windows present in the sample. 2008 is outside the "
             "2010--2026 panel and is disclosed as not testable here. Regimes are "
             "terciles of rolling 63-day portfolio volatility.}")
    L.append("\\label{tab:portfolio_regime}")
    L.append("\\begin{tabular}{lcccc}\\toprule")
    L.append("\\textbf{Regime} & \\textbf{days} & \\textbf{Sharpe} & "
             "\\textbf{ann. ret} & \\textbf{VaR95} \\\\ \\midrule")
    for _, r in reg.iterrows():
        L.append("%s & %d & %.2f & %.4f & %.5f \\\\" % (
            r["regime"], r["days"], r["sharpe"], r["ann_ret"], r["var95"]))
    L.append("\\bottomrule\\end{tabular}\\end{table}")
    with open(os.path.join(REPORT_TAB, "portfolio.tex"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote 5 portfolio figures + report/tables/portfolio.tex")


if __name__ == "__main__":
    main()
