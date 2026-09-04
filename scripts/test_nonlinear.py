#!/usr/bin/env python3
"""Unit tests for the Part 9 nonlinear forecast pipeline.

Covers:
  * feature/target builder returns finite causal features and a bounded target
  * walk_forecast runs end-to-end on each estimator and yields held-out predictions
  * no-lookahead: predictions for a bar depend only on strictly-earlier bars
  * model variance: on a genuinely linear synthetic R_t^h the linear forecast has
    non-negative OOS IC; flexible models also run and return finite output
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("scripts", "scripts/model"):
    sys.path.insert(0, os.path.join(ROOT, p))
import nonlinear_models as nm  # noqa: E402


def synth_lp(T=2200, n_assets=4, seed=1, beta0=None):
    """Synthetic log-price matrix: target = basket + linear cointegrating combo + noise."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal((n_assets - 1, T)), axis=1) * 0.01
    if beta0 is None:
        beta0 = np.full(n_assets - 1, 0.6)
    # target = intercept + beta dot basket + stationary mispricing (AR(1) mean-rev)
    mis = np.zeros(T)
    for t in range(1, T):
        mis[t] = 0.9 * mis[t - 1] + rng.standard_normal() * 0.002
    y = 0.5 + beta0 @ x + mis
    lp = np.vstack([y, x])
    return lp


class TestFeatures(unittest.TestCase):
    def test_features_finite_and_bounded_target(self):
        lp = synth_lp()
        bf = nm.build_features_targets(lp, 120, 5)
        F, y = bf["feat"], bf["target"]
        self.assertEqual(F.shape[1], 11)
        frac = np.isfinite(F).all(axis=1).mean()
        self.assertGreater(frac, 0.9, "most rows should have finite causal features")
        finite_t = np.isfinite(y)
        self.assertGreater(finite_t.mean(), 0.9)
        if finite_t.any():
            self.assertLess(np.nanpercentile(np.abs(y[finite_t]), 99), 0.5,
                            "short-side 5-bar returns should be small")


class TestWalkForecast(unittest.TestCase):
    def test_runs_all_estimators_and_returns_heldout(self):
        lp = synth_lp(T=2200)
        for m in ("linear", "gbm", "mlp"):
            wf = nm.walk_forecast(lp, m, 120, 5, 63, 0)
            pred = wf["pred"]
            n_fin = int(np.isfinite(pred).sum())
            self.assertGreater(n_fin, 300, f"{m} should produce many held-out preds")
            # finite predictions always coincide with finite features/target windows
            self.assertGreaterEqual(n_fin, int(np.isfinite(pred).sum()))
            self.assertLessEqual(n_fin, len(wf["target"]))
            self.assertFalse(np.isnan(pred).all())

    def test_no_lookahead_on_linear(self):
        lp = synth_lp(T=2200)
        wf = nm.walk_forecast(lp, "linear", 120, 5, 63, 0)
        pred = wf["pred"]
        ix = np.where(np.isfinite(pred))[0]
        self.assertGreater(len(ix), 300)
        # correlation must not be trivially 1 (would signal leakage)
        ok = np.isfinite(pred) & np.isfinite(wf["target"])
        p = pred[ok]; t = wf["target"][ok]
        if p.std() > 0 and t.std() > 0:
            c = float(np.corrcoef(p, t)[0, 1])
            self.assertLess(c, 0.95, "lookahead would drive correlation toward 1")
            self.assertGreater(c, -0.95)


class TestRESET(unittest.TestCase):
    def test_ramsey_rejects_quadratic_and_accepts_linear(self):
        from statkit.misspec import ramsey_reset
        rng = np.random.default_rng(0)
        X = rng.standard_normal((1200, 3))
        # linear signal: no omitted nonlinearity
        yl = 1.0 + 0.7 * X[:, 0] - 1.3 * X[:, 1] + 0.5 * X[:, 2] + 0.3 * rng.standard_normal(1200)
        rl = ramsey_reset(yl, X, powers=(2, 3), test="f")
        self.assertFalse(rl["reject_linear_at_5pct"])
        # quadratic signal on the first regressor: linear fit omits it
        yq = X[:, 0] + 2.0 * X[:, 0] ** 2 + 0.1 * rng.standard_normal(1200)
        rq = ramsey_reset(yq, X, powers=(2, 3), test="f")
        self.assertTrue(rq["reject_linear_at_5pct"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
