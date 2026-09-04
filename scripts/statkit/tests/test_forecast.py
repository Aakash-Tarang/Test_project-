"""Tests for Diebold-Mariano forecast / return-pair comparisons."""
import numpy as np

from ..forecast import diebold_mariano, dm_return_pair


def test_diebold_mariano_forecast_detects_better_forecast():
    # target with two forecasts; e1 smaller squared error than e2
    rng = np.random.default_rng(0)
    n = 600
    truth = rng.standard_normal(n)
    e1 = 0.2 * rng.standard_normal(n)      # good forecast
    e2 = 1.0 * rng.standard_normal(n)      # poor forecast
    r = diebold_mariano(e1, e2, h=1, loss="square")
    # negative dm => model1 (e1) better
    assert r["dm_statistic"] < -3.0, f"expected strong negative DM: {r['dm_statistic']}"
    assert r["pvalue"] < 0.01


def test_dm_return_pair_detects_dominant_strategy():
    rng = np.random.default_rng(1)
    n = 2000
    common = rng.standard_normal(n)          # market factor, cancels in the difference
    r1 = 0.10 + common + 0.5 * rng.standard_normal(n)   # drift + market + noise
    r2 = common + 0.5 * rng.standard_normal(n)          # market + noise, no drift
    r = dm_return_pair(r1, r2)
    assert r["dm_statistic"] > 3.0, f"expected strategy1 better: {r['dm_statistic']}"
    assert r["pvalue"] < 0.01
    assert r["mean_diff"] > 0


def test_dm_return_pair_non_significant_for_equal_strategies():
    rng = np.random.default_rng(2)
    n = 600
    base = rng.standard_normal(n)
    r1 = 0.0 + base
    r2 = 0.0 + base + 0.05 * rng.standard_normal(n)   # ~same mean
    r = dm_return_pair(r1, r2)
    assert r["pvalue"] > 0.05, f"equal strategies should not reject: p={r['pvalue']}"
