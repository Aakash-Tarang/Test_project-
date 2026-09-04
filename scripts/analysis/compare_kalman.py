#!/usr/bin/env python3
"""compare_kalman.py — Part 8 Kalman-filtered time-varying beta vs rolling OLS.

On the ADBE-vs-{CRM,ADSK,INTU} log-price panel we run the Kalman-filtered time-varying
hedge (scripts/model/kalman.py) through the *same* no-lookahead 10-bps cost-aware
pipeline as the rolling OLS/Ridge hedges of Part 5, and compare honestly on both the
full sample (2010-2026) and the out-of-sample 2022+ test period.

Hyper-parameters:
  * rolling refs are run at their Part-5 settings (windows 60/120/180 OLS, and Ridge
    window120 alpha=10 = best net performer).
  * Kalman's single state-noise hyper-parameter q is swept over a grid and reported at
    every value on both periods (no single cherry-picked q); the table also marks the
    value chosen on the 2019-2021 validation split by net Sharpe.

Also writes the filtered-beta path and runs Part-6-style diagnostics (ADF/KPSS + OU
half-life) on the Kalman residual so the adaptive spread is judged on the same footing.

Writes (git-ignored):
  results/tables/kalman_summary.csv   per-config metrics (full + test)
  results/model/beta_kalman.csv       filtered beta + z over time
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "model"))
from baseline_models import (walk_forward, simulate, metrics_from_nav,  # noqa: E402
                             information_coefficient)
from kalman import kalman_hedge  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "scripts", "statkit"))
import statkit  # noqa: E402

ENGINE_CSV = os.path.join(ROOT, "data", "processed", "engine", "ADBE.csv")
TAB = os.path.join(ROOT, "results", "tables")
MODEL = os.path.join(ROOT, "results", "model")
os.makedirs(TAB, exist_ok=True)
os.makedirs(MODEL, exist_ok=True)

ENTRY, EXIT, COST = 2.0, 0.5, 0.001
VAL_START = "2019-01-01"
TEST_START = "2022-01-01"
Q_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
BASKET = ["CRM", "ADSK", "INTU"]


def _metrics(wf, lp, a, b, window):
    sim = simulate(wf["z"], wf["beta"], lp, a, b, window, ENTRY, EXIT, COST)
    mg = metrics_from_nav(sim["gross"])
    mn = metrics_from_nav(sim["net"])
    return sim, mg, mn


def main():
    raw = pd.read_csv(ENGINE_CSV)
    dates = pd.to_datetime(raw["date"]).values
    cols = [c for c in raw.columns if c != "date"]
    lp = np.log(raw[cols].values.astype(float)).T
    T = lp.shape[1]
    idx_val0 = int(np.searchsorted(dates.astype("datetime64[D]"),
                                   np.datetime64(VAL_START)))
    idx_test0 = int(np.searchsorted(dates.astype("datetime64[D]"),
                                    np.datetime64(TEST_START)))
    full_range = (max(idx_val0 if False else 0, 60), T)   # score only from warm
    val_range = (idx_val0, max(idx_test0, 200))
    test_range = (max(idx_test0, 200), T)
    full_a = 60

    rows = []

    # ---- reference rolling hedges (Part 5 settings) ----
    refs = [
        ("rolling OLS w60", walk_forward(lp, 60, "OLS", {}), 60),
        ("rolling OLS w120", walk_forward(lp, 120, "OLS", {}), 120),
        ("rolling OLS w180", walk_forward(lp, 180, "OLS", {}), 180),
        ("rolling Ridge w120 a=10", walk_forward(lp, 120, "Ridge", {"alpha": 10.0}), 120),
    ]
    for name, wf, w in refs:
        sf, mfg, mfn = _metrics(wf, lp, full_a, T, w)
        st, mtg, mtn = _metrics(wf, lp, *test_range, w)
        rows.append({"model": name, "kind": "rolling",
                     "full_sharpe_gross": round(mfg["sharpe"], 3),
                     "full_sharpe_net": round(mfn["sharpe"], 3),
                     "test_sharpe_net": round(mtn["sharpe"], 3),
                     "test_ann_net": round(mtn["ann_ret"], 4),
                     "full_trades": int(sf["n_trades"])})

    # ---- Kalman sweep on full + test; also validate-selection ----
    val_best = (-1e9, None)
    for q in Q_GRID:
        wf = kalman_hedge(lp, q=q, warm=120)
        sf, mfg, mfn = _metrics(wf, lp, 120, T, 120)
        st, mtg, mtn = _metrics(wf, lp, *test_range, 120)
        sv, _, mvn = _metrics(wf, lp, *val_range, 120)
        if mvn["sharpe"] > val_best[0]:
            val_best = (mvn["sharpe"], q)
        rows.append({"model": f"Kalman q={q:.0e}", "kind": "kalman",
                     "full_sharpe_gross": round(mfg["sharpe"], 3),
                     "full_sharpe_net": round(mfn["sharpe"], 3),
                     "test_sharpe_net": round(mtn["sharpe"], 3),
                     "test_ann_net": round(mtn["ann_ret"], 4),
                     "full_trades": int(sf["n_trades"])})
    # chosen q = validation-selected
    chosen = val_best[1]
    rows.append({"model": "=> Kalman q* (val-selected)", "kind": "meta",
                 "note": f"q*={chosen:.0e}, chosen on 2019-2021 val"})

    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(TAB, "kalman_summary.csv"), index=False)

    # ---- filtered beta path + diagnostics for the val-selected Kalman ----
    wf_k = kalman_hedge(lp, q=chosen, warm=120)
    bdf = pd.DataFrame(wf_k["beta"], columns=BASKET)
    bdf.insert(0, "z", wf_k["z"])
    bdf.insert(0, "date", pd.to_datetime(dates))
    bdf.to_csv(os.path.join(MODEL, "beta_kalman.csv"), index=False)

    res = wf_k["innovation"][~np.isnan(wf_k["innovation"])]
    st = statkit.stationarity.interpret_stationarity(res)
    ou = statkit.ou.fit_ou(res)
    ic_test = information_coefficient(wf_k["z"], wf_k["beta"], lp,
                                      test_range[0], test_range[1], 120, h=21)
    ic_full = information_coefficient(wf_k["z"], wf_k["beta"], lp,
                                      full_a, T, 120, h=21)
    print(f"[Kalman q*={chosen:.0e}] residual: ADF p={st['adf_pvalue']:.4f} "
          f"verdict={st['label']} OU half-life={ou['half_life']:.1f} bars | "
          f"IC(h21) full={ic_full:.3f} test={ic_test:.3f}")
    print("\n== Kalman vs rolling OLS (full sample / test 2022+) ==")
    print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
