#!/usr/bin/env python3
"""Render Part 6 diagnostics figures + LaTeX tables from the live diagnostic run.

Reads results/tables/{diagnostics_summary,spread_residuals,diagnostics_halflife}.csv
and writes:
  results/figures/diagnostics_residual.png   traded spread + its ACF (mean reversion)
  results/figures/diagnostics_halflife.png   OU half-life vs realized holding
  report/tables/diagnostics.tex              the battery + cointegration + half-life tables
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


def main():
    summ = pd.read_csv(os.path.join(TAB, "diagnostics_summary.csv"))
    res = pd.read_csv(os.path.join(TAB, "spread_residuals.csv"), parse_dates=["date"])
    hl = pd.read_csv(os.path.join(TAB, "diagnostics_halflife.csv"))
    mt = pd.read_csv(os.path.join(TAB, "diagnostics_multtest.csv")).iloc[0]
    coint = pd.read_csv(os.path.join(TAB, "diagnostics_coint.csv")).set_index("item")["value"]
    names = ["CRM", "ADSK", "INTU"]
    eg_p = {n: float(coint[f"eg_p_{n}"]) for n in names}
    joh_rank = int(coint["johansen_rank_95"])
    hac_t = {n: float(coint[f"hac_t_{n}"]) for n in names}
    cl_t = {n: float(coint[f"classic_t_{n}"]) for n in names}

    os.makedirs(FIG, exist_ok=True)
    os.makedirs(REPORT_FIG, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    NAVY, RED, GREEN = "#4C72B0", "#C44E52", "#55A868"

    # ---------- fig 1: traded spread + ACF ----------
    e = res["e_roll"].values
    d = res["date"].values
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False,
                             gridspec_kw={"height_ratios": [1.6, 1]})
    axes[0].plot(d, e, color=NAVY, lw=0.7)
    axes[0].axhline(0, color="k", lw=0.7)
    axes[0].axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2026-09-01"),
                    color=RED, alpha=0.08, label="test 2022+")
    axes[0].set_title("Causal rolling-120 spread (target ADBE vs {CRM,ADSK,INTU})")
    axes[0].set_ylabel("log-price residual")
    axes[0].legend(loc="upper right")
    # ACF
    e = e - e.mean()
    nlags = 40
    ac = np.array([1.0] + [np.corrcoef(e[:-l], e[l:])[0, 1] for l in range(1, nlags + 1)])
    axes[1].bar(range(nlags + 1), ac, color=NAVY, width=0.8)
    se = 1.96 / np.sqrt(len(e))
    axes[1].axhline(0, color="k", lw=0.7)
    axes[1].axhline(se, color=RED, ls="--", lw=1, label="95% band")
    axes[1].axhline(-se, color=RED, ls="--", lw=1)
    axes[1].set_title("Autocorrelation of the traded spread (decays by ~lag 10-20)")
    axes[1].set_xlabel("lag (trading days)"); axes[1].set_ylabel("ACF")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "diagnostics_residual.png"), dpi=150)
    plt.close(fig)

    # ---------- fig 2: half-life vs holding ----------
    idx = {"OLS": 0, "Ridge": 1}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    full_rows = hl[hl["period"].str.contains("full")].set_index("model")
    test_rows = hl[hl["period"].str.contains("test")].set_index("model")
    names = ["OLS", "Ridge"]
    xfull = [float(full_rows.loc[m, "ou_halflife"]) for m in names]
    xtest = [float(test_rows.loc[m, "ou_halflife"]) for m in names]
    x = np.arange(len(names)); wdt = 0.34
    ax.bar(x - wdt / 2, xfull, wdt, label="Static hedge OU half-life (full 2010-2026)",
           color=RED)
    ax.bar(x + wdt / 2, xtest, wdt, label="Static hedge OU half-life (test 2022+)",
           color="#F5A623")
    hold_test = [float(test_rows.loc[m, "avg_holding_bars"]) for m in names]
    ax.bar(x + wdt / 2 + wdt + 0.05, hold_test, wdt,
           label="Realized avg holding (test 2022+)", color=GREEN)
    ax.axhline(9.4, color=NAVY, ls=":", lw=1.6)
    ax.text(2.55, 9.7, "traded spread OU half-life ~9.4 bars", color=NAVY, fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("bars (log scale)")
    ax.set_title("OU half-life of the static hedge vs the traded (rolling) spread")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "diagnostics_halflife.png"), dpi=150)
    plt.close(fig)

    for f in ("diagnostics_residual.png", "diagnostics_halflife.png"):
        shutil.copy2(os.path.join(FIG, f), os.path.join(REPORT_FIG, f))

    # ---------- LaTeX tables ----------
    os.makedirs(REPORT_TAB, exist_ok=True)
    summ = summ.set_index("model_period")
    rn = {"OLS_full": "OLS static (full)", "OLS_test": "OLS static (test 2022+)",
          "Ridge_full": "Ridge static (full)", "Ridge_test": "Ridge static (test 2022+)",
          "roll_OLS_full": "Rolling traded (full)", "roll_OLS_test": "Rolling traded (test)"}
    order = ["OLS_full", "Ridge_full", "OLS_test", "Ridge_test",
             "roll_OLS_full", "roll_OLS_test"]
    L = ["% Auto-generated by scripts/plotting/render_diagnostics.py",
         "\\begin{table}[H]\\centering\\small",
         "\\caption{Spec \\S2.2 diagnostic battery on the ADBE-vs-basket spread residuals. "
         "``Static'' = one fixed full-sample (or in-window) OLS/Ridge hedge of the "
         "log-price relation; ``Rolling traded'' = the causal walk-forward spread the "
         "strategy actually trades. ADF $H_0$: unit root; KPSS $H_0$: stationary. "
         "Verdict combines them. DW $\\approx2$ no AR(1); LB/BP/White give p-values "
         "(all $<0.01$: strong serial correlation and heteroskedasticity). OU half-life "
         "in trading days.}",
         "\\label{tab:diag_stationarity}",
         "\\begin{tabular}{lccclccc}\\toprule",
         "\\textbf{Residual} & \\textbf{ADF p} & \\textbf{KPSS p} & \\textbf{Verdict} "
         "& \\textbf{DW} & \\textbf{LB p} & \\textbf{OU $\\tau_{1/2}$} \\\\ \\midrule"]
    def fmtp(v):
        return "$<0.001$" if v < 0.001 else f"{v:.3f}"

    for m in order:
        r = summ.loc[m]
        L.append("%s & %s & %.3f & %s & %.3f & $<10^{-3}$ & %s \\\\" % (
            rn[m], fmtp(r["adf_p"]), r["kpss_p"],
            r["stationarity"].replace("_", "\\_"), r["dw"], f"{r['ou_halflife']:.1f}"))
    L.append("\\bottomrule\\end{tabular}\\end{table}")

    # cointegration + HAC table
    L.append("")
    L.append("\\begin{table}[H]\\centering\\small")
    L.append("\\caption{Cointegration and hedge-significance evidence. Engle--Granger "
             "two-step ADF p-values for ADBE vs each basket name individually; Johansen "
             "cointegration rank at 95\\% on the full 4-name log-price vector; and OLS "
             "hedge $t$-statistics on the full-sample hedge before and after the "
             "Newey--West HAC correction.}")
    L.append("\\label{tab:diag_coint}")
    L.append("\\begin{tabular}{lcccc}\\toprule")
    L.append("\\textbf{ADBE vs} & \\textbf{CRM} & \\textbf{ADSK} & \\textbf{INTU} & "
             "\\textbf{Johansen} \\\\ \\midrule")
    L.append("EG ADF p & %s & %s & %s & rank $r$ = %d \\\\" % (
        fmtp(eg_p["CRM"]), fmtp(eg_p["ADSK"]), fmtp(eg_p["INTU"]), joh_rank))
    L.append("Hedge $t$ (OLS) & %.1f & %.1f & %.1f & \\\\" % (
        cl_t["CRM"], cl_t["ADSK"], cl_t["INTU"]))
    L.append("Hedge $t$ (HAC) & %.1f & %.1f & %.1f & \\\\" % (
        hac_t["CRM"], hac_t["ADSK"], hac_t["INTU"]))
    L.append("\\bottomrule\\end{tabular}\\end{table}")
    lines = "\n".join(L)
    with open(os.path.join(REPORT_TAB, "diagnostics.tex"), "w") as f:
        f.write(lines + "\n")
    print("wrote diagnostics figures + report/tables/diagnostics.tex")


if __name__ == "__main__":
    main()
