#!/usr/bin/env python3
"""Record reproducible synthetic-validation results of the statkit toolkit.

Demonstrates on simulated data (spec permits synthetic data for isolated math
checks) that the statistical routines behave correctly, and writes the numbers
to results/tables/statkit_validation.csv so the report's Methodology section
cites reproducible evidence rather than hand-typed claims.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

TAB = os.path.join(ROOT, "results", "tables")
os.makedirs(TAB, exist_ok=True)


def main():
    from statkit.ou import fit_ou
    from statkit.stationarity import interpret_stationarity
    from statkit.cointegration import engle_granger
    from statkit.forecast import diebold_mariano
    from statkit.multtest import white_reality_check

    rng = np.random.default_rng(0)
    rows = []

    # 1) OU parameter recovery
    theta_true, mu_true, sigma_true = 0.4, 0.0, 0.15
    n = 20000
    x = np.empty(n); x[0] = 0.0
    for t in range(1, n):
        x[t] = x[t - 1] + theta_true * (mu_true - x[t - 1]) + sigma_true * rng.standard_normal()
    ou = fit_ou(x)
    rows.append(("ou_true_theta", round(theta_true, 4)))
    rows.append(("ou_est_theta", round(ou["theta"], 4)))
    rows.append(("ou_est_halflife", round(ou["half_life"], 4)))
    rows.append(("ou_halflife_true_ln2/theta", round(np.log(2) / theta_true, 4)))

    # 2) joint stationarity on a stationary AR residual
    ar = np.empty(3000); ar[0] = 0.0
    for t in range(1, 3000):
        ar[t] = 0.6 * ar[t - 1] + rng.standard_normal()
    rows.append(("stationarity_AR1_label", interpret_stationarity(ar)["label"]))

    # 3) cointegration detection
    trend = np.cumsum(rng.standard_normal(2000))
    spread = np.empty(2000); s = 0.0
    for i in range(2000):
        s = 0.9 * s + rng.standard_normal() * 0.2; spread[i] = s
    eg = engle_granger(2.0 * trend + spread, trend)
    rows.append(("coint_hedge_beta_est", round(float(eg["beta"][1]), 3)))
    rows.append(("coint_eg_adf_pvalue", round(eg["adf"]["pvalue"], 5)))

    # 4) DM detects superior forecast
    e1 = rng.standard_normal(1000) * 0.5
    e2 = rng.standard_normal(1000) * 1.5
    dm = diebold_mariano(e1, e2)
    rows.append(("dm_statistic", round(dm["dm_statistic"], 3)))
    rows.append(("dm_pvalue", round(dm["pvalue"], 5)))

    # 5) Reality Check: null panel (all true-mean zero) stays non-significant
    perf_null = rng.standard_normal((800, 20))
    rc_null = white_reality_check(perf_null, n_boot=600, seed=1)
    rows.append(("rc_pvalue_all_null", round(rc_null["pvalue"], 3)))
    # and a genuine edge is detected
    perf_edge = rng.standard_normal((800, 20)); perf_edge[:, 0] += 0.35
    rc_edge = white_reality_check(perf_edge, n_boot=600, seed=1)
    rows.append(("rc_pvalue_with_real_edge", round(rc_edge["pvalue"], 3)))

    import pandas as pd
    df = pd.DataFrame(rows, columns=["check", "value"])
    out = os.path.join(TAB, "statkit_validation.csv")
    df.to_csv(out, index=False)
    print("wrote", out)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
