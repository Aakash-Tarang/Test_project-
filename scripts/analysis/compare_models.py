#!/usr/bin/env python3
"""compare_models.py — Part 5 baseline model comparison on the real basket.

Walk-forward, no-lookahead comparison of the five linear hedge estimators
(OLS / Ridge / Lasso / ElasticNet / PCR=PCA) on the ADBE-vs-{CRM,ADSK,INTU}
basket. For every (method, hyper-parameter) candidate we run the causal
windowed fit exactly as in the Part 4 engine and feed the resulting residual
z through the same one-bar-delay, cost-aware trade logic. Hyper-parameters are
selected on a validation split (2019-2021, maximizing net Sharpe); headline
results are then reported out-of-sample on the test split (2022+).

Writes (git-ignored):
  results/tables/model_compare_summary.csv  per-method test metrics
  results/tables/model_compare_sweep.csv    val net Sharpe vs hyper-parameter
  results/model/beta_<method>.csv           chosen hedge weights over time
  results/model/net_test_<method>.csv       test-period log NAV per method
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "model"))
from baseline_models import (walk_forward, simulate, metrics_from_nav,
                             information_coefficient)  # noqa: E402

ENGINE_CSV = os.path.join(ROOT, "data", "processed", "engine", "ADBE.csv")
TAB = os.path.join(ROOT, "results", "tables")
MODEL = os.path.join(ROOT, "results", "model")

WINDOW = 120
ENTRY, EXIT = 2.0, 0.5
COST = 0.001            # 10 bps one-way (Part 4 baseline)
VAL_START = "2019-01-01"
TEST_START = "2022-01-01"

CANDIDATES = {
    "OLS": [{}],
    "Ridge": [{"alpha": a} for a in (0.1, 1.0, 10.0)],
    "Lasso": [{"alpha": a} for a in (1e-4, 1e-3, 1e-2)],
    "ElasticNet": [{"alpha": a, "l1_ratio": 0.5} for a in (1e-4, 1e-3, 1e-2)],
    "PCA": [{"k": k} for k in (1, 2, 3)],
}


def param_label(method, p):
    if not p:
        return "none"
    if method == "PCA":
        return f"k={p['k']}"
    if method == "ElasticNet":
        return f"a={p['alpha']:.0e},r={p['l1_ratio']}"
    return f"a={p['alpha']:.0e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="limit bars for a fast smoke run")
    args = ap.parse_args()

    raw = pd.read_csv(ENGINE_CSV)
    dates = pd.to_datetime(raw["date"]).values
    cols = [c for c in raw.columns if c != "date"]
    target = cols[0]
    basket = cols[1:]
    prices = raw[cols].values.astype(float)          # (T, n_assets) target first
    prices = prices.transpose()                       # (n_assets, T)
    lp = np.log(prices)
    T = lp.shape[1]
    n_assets, k = lp.shape[0], lp.shape[1] - 1
    if args.quick:
        T = min(T, 900)
        lp = lp[:, :T]
        dates = dates[:T]

    warm = WINDOW
    if T <= warm + 60:
        raise SystemExit(f"too few bars ({T}) for window {warm}")

    idx_val0 = int(np.searchsorted(dates.astype("datetime64[D]"),
                                   np.datetime64(VAL_START))) if dates.size else 0
    idx_test0 = int(np.searchsorted(dates.astype("datetime64[D]"),
                                    np.datetime64(TEST_START))) if dates.size else T
    idx_val0 = max(idx_val0, warm)
    idx_test0 = max(idx_test0, warm + 20)
    val_range = (idx_val0, idx_test0)
    test_range = (idx_test0, T)

    # ---- run every candidate once over the full sample (causal), score on val ----
    results = {}          # method -> list of (params, meta)
    for method, cands in CANDIDATES.items():
        results[method] = []
        for p in cands:
            wf = walk_forward(lp, WINDOW, method, p)
            simv = simulate(wf["z"], wf["beta"], lp, *val_range, WINDOW,
                            ENTRY, EXIT, COST)
            mv = metrics_from_nav(simv["net"])
            results[method].append({
                "params": p, "label": param_label(method, p),
                "ms_per_fit": wf["ms_per_fit"], "val_net_sharpe": mv["sharpe"],
                "wf": wf,
            })

    # ---- choose best hyper-parameter per method on validation net Sharpe ----
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(MODEL, exist_ok=True)
    chosen = {}
    summary_rows = []
    sweep_rows = []
    for method, cands in CANDIDATES.items():
        lst = results[method]
        best = max(lst, key=lambda r: r["val_net_sharpe"])
        chosen[method] = best
        for r in lst:
            sweep_rows.append({
                "method": method, "params": r["label"],
                "ms_per_fit": round(r["ms_per_fit"], 4),
                "val_net_sharpe": round(r["val_net_sharpe"], 4),
            })
        # Headline metrics on the (out-of-sample) test split with the chosen params.
        wf = best["wf"]
        st = simulate(wf["z"], wf["beta"], lp, *test_range, WINDOW, ENTRY, EXIT, COST)
        mt = metrics_from_nav(st["net"])
        mg = metrics_from_nav(st["gross"])
        # Full-sample context (2010-2026, same span as the Part 4 OLS baseline).
        sfull = simulate(wf["z"], wf["beta"], lp, warm, T, WINDOW, ENTRY, EXIT, COST)
        mfulln = metrics_from_nav(sfull["net"])
        mfullg = metrics_from_nav(sfull["gross"])
        ic = information_coefficient(wf["z"], wf["beta"], lp,
                                     test_range[0], test_range[1], WINDOW, h=21)
        # coefficient stability over the test split
        beta_ts = wf["beta"][test_range[0]:test_range[1]]
        beta_ts = beta_ts[~np.isnan(beta_ts).any(axis=1)]
        nz = np.count_nonzero(beta_ts, axis=1)
        sparsity = float((nz > 0).mean()) if nz.size else 0.0
        mean_abs = np.abs(beta_ts).mean(axis=0)
        std_abs = beta_ts.std(axis=0)
        denom = np.where(mean_abs > 1e-12, mean_abs, 1.0)
        coef_instab = float(np.mean(std_abs / denom)) if beta_ts.size else 0.0
        avg_hold = st["total_hold"] / st["n_opens"] if st["n_opens"] else 0.0

        summary_rows.append({
            "method": method, "params": best["label"],
            "val_net_sharpe": round(best["val_net_sharpe"], 4),
            "ms_per_fit": round(best["ms_per_fit"], 4),
            "test_ann_ret_gross": round(mg["ann_ret"], 4),
            "test_sharpe_gross": round(mg["sharpe"], 3),
            "test_ann_ret_net": round(mt["ann_ret"], 4),
            "test_sharpe_net": round(mt["sharpe"], 3),
            "test_maxdd_net": round(mt["maxdd"], 3),
            "test_trades": int(st["n_trades"]),
            "test_avg_hold": round(avg_hold, 1),
            "test_cost_log": round(st["cost_log"], 4),
            "test_ic_h21": round(ic, 4),
            "coef_instab": round(coef_instab, 3),
            "nonzero_frac": round(sparsity, 3),
            "full_sharpe_gross": round(mfullg["sharpe"], 3),
            "full_sharpe_net": round(mfulln["sharpe"], 3),
            "full_ann_ret_net": round(mfulln["ann_ret"], 4),
        })
        # persist chosen weights + test NAV
        bdf = pd.DataFrame(wf["beta"], columns=basket)
        bdf.insert(0, "date", pd.to_datetime(dates))
        bdf.to_csv(os.path.join(MODEL, f"beta_{method.lower()}.csv"), index=False)
        ndf = pd.DataFrame({
            "date": pd.to_datetime(dates)[test_range[0]:test_range[1]],
            "net": st["net"], "gross": st["gross"],
        })
        ndf.to_csv(os.path.join(MODEL, f"net_test_{method.lower()}.csv"), index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(TAB, "model_compare_summary.csv"), index=False)
    pd.DataFrame(sweep_rows).to_csv(os.path.join(TAB, "model_compare_sweep.csv"),
                                    index=False)

    # quick validation against the Part 4 OLS baseline (full sample, default cfg)
    if not args.quick and "OLS" in chosen:
        wf = chosen["OLS"]["wf"]
        simfull = simulate(wf["z"], wf["beta"], lp, warm, T, WINDOW, ENTRY, EXIT, COST)
        mf = metrics_from_nav(simfull["net"])
        print(f"[check] OLS full-sample net Sharpe = {mf['sharpe']:.3f} "
              f"(Part 4 engine reported ~0.050); net ann ret = {mf['ann_ret']:.4f}")

    print("wrote results/tables/model_compare_summary.csv + model_compare_sweep.csv")
    print(summary[["method", "params", "test_sharpe_gross", "test_sharpe_net",
                   "test_ic_h21"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
