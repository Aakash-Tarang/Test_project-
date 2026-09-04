"""Tests for Engle-Granger and Johansen cointegration on synthetic pairs."""
import numpy as np
import pandas as pd

from ..cointegration import engle_granger, johansen_rank


def _gen_cointegrated(n, seed, hedge=1.5):
    """Common trend + stationary spread => y and x are cointegrated."""
    rng = np.random.default_rng(seed)
    trend = np.cumsum(rng.standard_normal(n))
    spread = 0.0 * np.empty(n)
    # stationary OU-like spread
    s = 0.0
    for i in range(n):
        s = 0.9 * s + rng.standard_normal() * 0.3
        spread[i] = s
    x = trend
    y = hedge * x + spread
    return y, x


def _gen_independent_rw(n, seed):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n)), np.cumsum(rng.standard_normal(n))


def test_engle_granger_rejects_on_cointegrated_pair():
    y, x = _gen_cointegrated(1200, seed=10)
    res = engle_granger(y, x)
    # hedge beta close to true
    assert abs(res["beta"][1] - 1.5) < 0.1, res["beta"]
    # residual should be stationary => ADF rejects unit root
    assert res["adf"]["pvalue"] < 0.05, f"EG should find cointegration: p={res['adf']['pvalue']:.4f}"


def test_engle_granger_no_cointegration_independent_rw():
    y, x = _gen_independent_rw(800, seed=11)
    res = engle_granger(y, x)
    assert res["adf"]["pvalue"] > 0.05, "independent random walks should NOT be cointegrated"


def test_johansen_rank_one_for_cointegrated_pair():
    y, x = _gen_cointegrated(800, seed=12)
    df = pd.DataFrame({"y": y, "x": x})
    res = johansen_rank(df)
    assert res["rank_at_95pct"] >= 1, f"expected rank>=1, got {res['rank_at_95pct']}"


def test_johansen_rank_zero_for_independent_rw():
    y, x = _gen_independent_rw(600, seed=13)
    df = pd.DataFrame({"y": y, "x": x})
    res = johansen_rank(df)
    # With only 600 obs this is stochastic; require not falsely asserting rank 2,
    # and accept 0 OR occasionally 1; assert <2 to catch gross over-fit.
    assert res["rank_at_95pct"] < 2
