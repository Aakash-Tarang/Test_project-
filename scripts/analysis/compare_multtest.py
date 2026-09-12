#!/usr/bin/env python3
"""compare_multtest.py — Part 10 multiple-testing correction + Diebold-Mariano audit.

Adversarial self-critique of the headline net Sharpe: after Parts 4-9 scanned many
(basket x hedge) strategies, how much of the best full-sample net Sharpe survives a
data-snooping correction, and are the pairwise differences statistically separable?

Design (all net of 10 bps, strictly causal, common evaluation window):
  1. Build, per basket, the daily *net* P&L series for every hedge model actually
     reported/traded across Parts 4-9:
         ADBE: Part-5 grid (OLS/Ridge/Lasso/ElasticNet/PCA @ w120), rolling OLS
               w60/w120/w180, and Kalman q in {1e-5,1e-4,1e-3,1e-2};
         AVGO, GS, CAT: rolling OLS w60/w120/w180, Ridge w120 a10, Kalman q1e-5/1e-4.
  2. Align all columns on a common daily calendar and run:
         - per-strategy HAC t-test that mean daily net P&L > 0  (Bonferroni bar);
         - White (2000) Reality Check and Hansen (2005) SPA on the (T,M) matrix;
         - explicit hypothesis count M and corrected significance threshold.
  3. Pairwise Diebold-Mariano (return-difference) forest among the ADBE hedge
     variants over the same window, for scripts/plotting/render_multtest.py.

Writes (git-ignored): results/tables/multtest_audit.csv,
results/tables/multtest_corrections.csv, results/tables/multtest_dm.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("scripts", "scripts/model", "scripts/statkit"):
    sys.path.insert(0, os.path.join(ROOT, p))
import baseline_models as bm  # noqa: E402
import kalman as km  # noqa: E402
import statkit  # noqa: E402

ENGINE = os.path.join(ROOT, "data", "processed", "engine")
TAB = os.path.join(ROOT, "results", "tables")
os.makedirs(TAB, exist_ok=True)

ENTRY, EXIT, COST = 2.0, 0.5, 0.001       # 10 bps one-way
TEST_START = "2022-01-01"
EVAL_A = 210                               # common start index: > max warm-up (180)
N_BOOT = 1200                              # stationary-bootstrap draws for RC / SPA
SEED = 11

BASKETS = {
    "ADBE": ["CRM", "ADSK", "INTU"],
    "AVGO": ["ADI", "INTC", "AAPL"],
    "GS": ["BAC"],
    "CAT": ["HON"],
}
KAL_Q = [1e-5, 1e-4, 1e-3, 1e-2]


def _hedge_models(_basket):
    """(label, callable->wf) candidate hedge models — the SAME symmetric grid for every
    basket, so the data-snooping universe is the clean product basket x model.
    wf = dict with causal {z, beta}."""
    return {
        "OLS-w60": lambda lp: bm.walk_forward(lp, 60, "OLS", {}),
        "OLS-w120": lambda lp: bm.walk_forward(lp, 120, "OLS", {}),
        "OLS-w180": lambda lp: bm.walk_forward(lp, 180, "OLS", {}),
        "Ridge-w120": lambda lp: bm.walk_forward(lp, 120, "Ridge", {"alpha": 10.0}),
        "Lasso-w120": lambda lp: bm.walk_forward(lp, 120, "Lasso", {"alpha": 1e-3}),
        "EN-w120": lambda lp: bm.walk_forward(lp, 120, "ElasticNet",
                                              {"alpha": 1e-3, "l1_ratio": 0.5}),
        "PCA-w120": lambda lp: bm.walk_forward(lp, 120, "PCA", {"k": 2}),
        "Kalman-q1e-05": lambda lp: km.kalman_hedge(lp, q=1e-5, warm=120),
        "Kalman-q1e-04": lambda lp: km.kalman_hedge(lp, q=1e-4, warm=120),
        "Kalman-q1e-03": lambda lp: km.kalman_hedge(lp, q=1e-3, warm=120),
        "Kalman-q1e-02": lambda lp: km.kalman_hedge(lp, q=1e-2, warm=120),
    }


def _net_series(lp, dates, wf, window, a=EVAL_A):
    """Daily net P&L over [a+1, T) aligned to its date; pandas Series."""
    T = lp.shape[1]
    sim = bm.simulate(wf["z"], wf["beta"], lp, a, T, window, ENTRY, EXIT, COST)
    nav = sim["net"]                       # len T-a, cumulative from a
    ret = np.diff(nav)                     # per-bar returns for bars a+1..T-1
    dts = pd.to_datetime(dates)[a + 1:a + 1 + len(ret)]
    return pd.Series(ret, index=dts)


def _load(basket):
    raw = pd.read_csv(os.path.join(ENGINE, f"{basket}.csv"))
    cols = [c for c in raw.columns if c != "date"]
    dates = pd.to_datetime(raw["date"]).values
    lp = np.log(raw[cols].values.astype(float)).T
    return lp, dates


def _hac_t_pvalue_one_sided(daily):
    """One-sided (mean>0) HAC t-test p-value on a daily-return series."""
    import statsmodels.api as sm
    n = len(daily)
    maxlags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    X = np.ones((n, 1))
    fit = sm.OLS(np.asarray(daily, float), X).fit(cov_type="HAC",
                                                  cov_kwds={"maxlags": maxlags})
    t = float(fit.params[0]) / float(fit.bse[0])
    from scipy import stats as _st
    return t, float(1.0 - _st.norm.cdf(t))


def main():
    frames = {}      # column label -> Series of daily net returns (full window)
    test_start_dt = np.datetime64(TEST_START)
    audit_rows = []

    for basket, names in BASKETS.items():
        lp, dates = _load(basket)
        models = _hedge_models(basket)
        for label, fn in models.items():
            wf = fn(lp)
            s = _net_series(lp, dates, wf, 120)
            col = f"{basket}::{label}"
            frames[col] = s
            tfull, pfull = _hac_t_pvalue_one_sided(s.values)
            # test (2022+) sub-series for context
            stest = s[s.index >= test_start_dt]
            sh_full = bm.metrics_from_nav(s.cumsum().values)["sharpe"]
            ann_full = float(s.mean()) * 252
            audit_rows.append({
                "basket": basket, "model": label,
                "net_sharpe_full": round(sh_full, 3),
                "ann_ret_full": round(ann_full, 4),   # annualized net return (frac)
                "mean_daily_bps": round(float(s.mean()) * 1e4, 2),
                "hac_t_full": round(tfull, 3),
                "p_mean_gt0": round(pfull, 4),
                "test_n": int(len(stest)),
                "test_sharpe_net": round(
                    bm.metrics_from_nav(stest.cumsum().values)["sharpe"], 3),
            })

    audit = pd.DataFrame(audit_rows)
    audit = audit.sort_values("net_sharpe_full", ascending=False).reset_index(drop=True)
    audit.to_csv(os.path.join(TAB, "multtest_audit.csv"), index=False)

    # ---- (T,M) matrix aligned on the common calendar ----
    mat = pd.DataFrame(frames).dropna()          # drop rows where any column missing
    T, M = mat.shape
    labels = list(mat.columns)
    perf = mat.values                            # (T, M) daily net returns
    # Persist the aligned (T, M) daily net-P&L matrix so downstream analyses
    # (power / deflated-Sharpe, report tables) never have to re-simulate.
    mat.to_csv(os.path.join(TAB, "multtest_pnl.csv"))

    # ---- Bonferroni over per-strategy HAC p-values (mean>0) ----
    pvals = audit.set_index(["basket", "model"])["p_mean_gt0"].values
    # align audit order to matrix column order
    p_order = np.array([audit[(audit.basket == lab.split("::")[0]) &
                              (audit.model == lab.split("::", 1)[1])]["p_mean_gt0"].iloc[0]
                        for lab in labels])
    bonf = statkit.multtest.bonferroni(p_order, alpha=0.05)
    best_idx = int(np.argmax(audit["net_sharpe_full"]))
    best = audit.iloc[best_idx]

    # ---- White Reality Check & Hansen SPA on the (T,M) matrix ----
    rc = statkit.multtest.white_reality_check(perf, n_boot=N_BOOT, mean_block=10,
                                              seed=SEED)
    spa = statkit.multtest.hansen_spa(perf, n_boot=N_BOOT, mean_block=10, seed=SEED)

    corr_rows = [{
        "quantity": "n_hypotheses_M", "value": M,
        "detail": "distinct (basket x hedge) net-return strategies scanned",
    }, {
        "quantity": "n_days_T", "value": T,
        "detail": f"common daily calendar {mat.index[0].date()}..{mat.index[-1].date()}",
    }, {
        "quantity": "bonferroni_bar_alpha_05", "value": round(0.05 / M, 6),
        "detail": "marginal significance threshold after Bonferroni",
    }, {
        "quantity": "best_strategy", "value": f"{best.basket}::{best.model}",
        "detail": f"best full-sample net Sharpe {best.net_sharpe_full}",
    }, {
        "quantity": "best_p_raw_hac", "value": round(best.p_mean_gt0, 4),
        "detail": "uncorrected one-sided HAC p (mean net P&L > 0)",
    }, {
        "quantity": "best_p_bonferroni", "value": round(min(float(best.p_mean_gt0) * M, 1.0), 4),
        "detail": "Bonferroni-adjusted p for the best strategy",
    }, {
        "quantity": "n_survive_bonferroni", "value": int(len(bonf["significant_indices"])),
        "detail": "how many strategies reject H0 mean<=0 after Bonferroni",
    }, {
        "quantity": "white_rc_p", "value": round(rc["pvalue"], 4),
        "detail": "White (2000) Reality Check p over M scanned strategies",
    }, {
        "quantity": "hansen_spa_p", "value": round(spa["pvalue"], 4),
        "detail": "Hansen (2005) SPA p over M scanned strategies",
    }, {
        "quantity": "n_boot", "value": int(N_BOOT),
        "detail": "stationary-bootstrap draws (mean_block=10)",
    }]
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(os.path.join(TAB, "multtest_corrections.csv"), index=False)

    # ---- Pairwise Diebold-Mariano on ADBE (full window) ----
    adbe_cols = [c for c in labels if c.startswith("ADBE::")]
    dm_rows = []
    ref = [c for c in adbe_cols if "Kalman-q1e-05" in c][0]
    for c in adbe_cols:
        r = statkit.forecast.dm_return_pair(mat[ref].values, mat[c].values)
        dm_rows.append({
            "ref": ref.split("::", 1)[1], "other": c.split("::", 1)[1],
            "mean_diff_daily": round(r["mean_diff"] * 1e4, 3),   # bps/day ref-other
            "hac_se_daily": round(r["hac_se"] * 1e4, 3),         # bps/day
            "dm": round(r["dm_statistic"], 3),
            "p": round(r["pvalue"], 4),
            "ref_better": bool(r["mean_diff"] > 0),
        })
    dm_df = pd.DataFrame(dm_rows)
    dm_df.to_csv(os.path.join(TAB, "multtest_dm.csv"), index=False)

    print("== Part 10 multiple-testing audit ==")
    print(f"matrix: T={T} days, M={M} strategies (common calendar)")
    print(audit.head(6).to_string(index=False))
    print("\n== corrections ==")
    print(corr.to_string(index=False))
    print("\n== ADBE DM vs best (Kalman q1e-5), full window ==")
    print(dm_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
