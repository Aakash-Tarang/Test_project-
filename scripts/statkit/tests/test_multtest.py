"""Tests for multiple-testing correction (Bonferroni, White RC, Hansen SPA)
and the stationary bootstrap on synthetic performance panels."""
import numpy as np

from ..multtest import bonferroni, white_reality_check, hansen_spa
from ..bootstrap import stationary_bootstrap, bootstrap_ci


def test_bonferroni_controls_inflation():
    # 100 tests all truly null with p~U(0,1): none should survive strict Bonferroni at 0.05
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 100)
    res = bonferroni(p, alpha=0.05)
    assert res["n_tests"] == 100
    # P(min adjusted p < 0.05) under all-null = 1-(1-0.05/100)^100 ~ 0.0487; allow 0 or occasionally 1
    assert len(res["significant_indices"]) <= 1


def test_bonferroni_finds_real_effect():
    p = np.array([1e-9, 0.2, 0.3, 0.5])
    res = bonferroni(p, alpha=0.05)
    assert res["significant_indices"] == [0]


def _make_null_panel(n_strat, T, seed, true_mean=0.0):
    """n_strat independent strategies with mean true_mean (all equal -> best is null)."""
    rng = np.random.default_rng(seed)
    return true_mean + rng.standard_normal((T, n_strat))


def test_white_reality_check_no_false_signal_when_all_null():
    # 20 strategies all with true mean 0. Best observed mean should NOT look
    # significant once we correct for the max over 20.
    perf = _make_null_panel(20, 600, seed=1, true_mean=0.0)
    res = white_reality_check(perf, n_boot=800, seed=3)
    assert res["pvalue"] > 0.05, f"RC pvalue should be non-significant under null: {res['pvalue']:.3f}"


def test_white_reality_check_detects_genuine_best():
    # 20 strategies, one genuinely good (mean 0.3), rest null. RC should be significant.
    rng = np.random.default_rng(5)
    T, M = 800, 20
    perf = rng.standard_normal((T, M))
    perf[:, 0] += 0.35
    res = white_reality_check(perf, n_boot=800, seed=6)
    assert res["pvalue"] < 0.05, f"RC should reject null when a real edge exists: {res['pvalue']:.3f}"


def test_hansen_spa_runs():
    perf = _make_null_panel(10, 400, seed=7)
    res = hansen_spa(perf, n_boot=500, seed=8)
    assert 0.0 <= res["pvalue"] <= 1.0


def test_stationary_bootstrap_shapes():
    idx = stationary_bootstrap(100, 50, mean_block=10, rng=np.random.default_rng(0))
    assert idx.shape == (50, 100)
    assert (idx >= 0).all() and (idx < 100).all()


def test_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(9)
    x = rng.standard_normal(2000)
    res = bootstrap_ci(x, lambda s: np.mean(s), n_boot=600, mean_block=20, seed=1)
    assert res["ci_low"] <= 0.0 <= res["ci_high"]
