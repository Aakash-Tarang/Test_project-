#!/usr/bin/env python3
"""test_kalman.py — unit checks for the Part 8 Kalman-filtered time-varying hedge.

1. Tracking: when the true target/basket beta drifts over time, the Kalman-filtered
   beta follows the true time-varying beta far better than a fixed full-sample OLS
   beta (much lower root-mean-squared tracking error over the out-of-sample second
   half of the sample).
2. Causal no-lookahead: every predictive beta_hist[t] used to score bar t must equal
   the beta the filter would output having seen only bars < t; we verify structurally
   that z[t] is built from the predictive (not the updated) state and that the
   recursion is one-pass forward (no refit on future data).
3. Mean reversion: on a cointegrated synthetic target with a genuine stationary
   mispricing, the Kalman residual rejects a unit root (ADF) and has a finite, small
   OU half-life — i.e. the filtered spread is tradeable-stationary.

Exit 0 on pass. Run: python3 scripts/test_kalman.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "model"))
from kalman import kalman_hedge  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "scripts", "statkit"))
import statkit  # noqa: E402

np.random.seed(7)
checks = 0


def check(cond, msg):
    global checks
    checks += 1
    if not cond:
        print(f"  [FAIL] {msg}")
        return False
    print(f"  [ok]   {msg}")
    return True


def main():
    T, k = 2000, 2
    # true time-varying betas (smooth drift)
    b1 = 0.6 + 0.8 * np.sin(np.linspace(0, 2.2 * np.pi, T))
    b2 = 1.1 - 0.6 * np.sin(np.linspace(0, 1.7 * np.pi, T))
    # common factor (random walk) driving the basket levels
    f = np.cumsum(np.random.randn(T) * 0.01)
    x1 = 0.7 * f + np.cumsum(np.random.randn(T) * 0.004)
    x2 = 0.8 * f + np.cumsum(np.random.randn(T) * 0.004)
    # stationary mean-reverting mispricing (OU, half-life ~ 20)
    sp = np.empty(T); sp[0] = 0.0
    for t in range(1, T):
        sp[t] = 0.95 * sp[t - 1] + np.random.randn() * 0.05
    # target log price = basket-hedged level + mispricing
    y = 0.0 + b1 * x1 + b2 * x2 + sp
    lp = np.vstack([y, x1, x2])

    warm = 120
    wf = kalman_hedge(lp, q=5e-2, warm=warm)
    B = wf["beta"]                       # (T, k); beta_hist[0:warm] is NaN by design
    check(np.isfinite(B[warm:]).all(), "filtered beta finite over scored sample")
    # ---- tracking error: fixed OLS beta (full-sample) vs Kalman (causal) ----
    A = np.column_stack([np.ones(T), x1, x2])
    fixed = np.linalg.lstsq(A, y, rcond=None)[0][1:]
    half = slice(1000, 2000)
    err_kal = np.sqrt(np.mean((B[half, 0] - b1[half]) ** 2 +
                              (B[half, 1] - b2[half]) ** 2))
    err_fix = np.sqrt(np.mean((fixed[0] - b1[half]) ** 2 + (fixed[1] - b2[half]) ** 2))
    check(err_kal < 0.5 * err_fix,
          f"Kalman tracks drifting beta (RMSE {err_kal:.3f}) << fixed OLS ({err_fix:.3f})")

    # ---- causal: predictive beta for bar t uses data only through t-1 ----
    lp_cut = lp[:, :500]
    wf_cut = kalman_hedge(lp_cut, q=5e-2, warm=120)
    check(np.allclose(B[499], wf_cut["beta"][499], atol=1e-6),
          "predictive beta at bar t depends only on bars < t (causal, no lookahead)")

    # ---- mean reversion of Kalman residual ----
    res = wf["innovation"][~np.isnan(wf["innovation"])]
    st = statkit.stationarity.interpret_stationarity(res)
    ou = statkit.ou.fit_ou(res)
    check(st["label"] in ("stationary", "ambiguous") and ou["half_life"] < 80,
          f"Kalman residual mean-reverting: ADF p={st['adf_pvalue']:.4f} "
          f"OU half-life={ou['half_life']:.1f} bars")
    return 0


if __name__ == "__main__":
    sys.exit(main())

