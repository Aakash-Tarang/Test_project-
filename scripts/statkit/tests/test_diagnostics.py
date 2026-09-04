"""Tests for autocorrelation, heteroskedasticity, HAC and DM routines."""
import numpy as np

from ..autocorr import durbin_watson, ljung_box
from ..heterosk import breusch_pagan, white_test
from ..hac import hac_ols
from ..forecast import diebold_mariano


def test_dw_around_2_for_white_noise():
    rng = np.random.default_rng(0)
    e = rng.standard_normal(1000)
    dw = durbin_watson(e)
    assert abs(dw - 2.0) < 0.3, f"DW should be ~2 for white noise, got {dw:.3f}"


def test_dw_low_for_positive_ar1():
    rng = np.random.default_rng(1)
    e = np.empty(1000); e[0] = rng.standard_normal()
    for t in range(1, 1000):
        e[t] = 0.9 * e[t - 1] + rng.standard_normal()
    dw = durbin_watson(e)
    assert dw < 1.5, f"positive AR(1) => DW<2, got {dw:.3f}"


def test_ljung_box_rejects_autocorrelated():
    rng = np.random.default_rng(2)
    e = np.empty(1000); e[0] = rng.standard_normal()
    for t in range(1, 1000):
        e[t] = 0.5 * e[t - 1] + rng.standard_normal()
    r = ljung_box(e, lags=10)
    assert r["pvalue"] < 0.05, f"LB should reject white noise on AR(1): p={r['pvalue']:.4f}"


def test_ljung_box_accepts_white_noise():
    rng = np.random.default_rng(3)
    e = rng.standard_normal(1500)
    r = ljung_box(e, lags=10)
    assert r["pvalue"] > 0.05


def test_bp_rejects_heteroskedastic():
    # variance scales ~linearly with x (positive range) so BP's linear-in-x
    # auxiliary regression of squared residuals has power.
    rng = np.random.default_rng(4)
    x = rng.uniform(0.5, 3.0, 3000)
    y = 1.0 + 0.5 * x + rng.standard_normal(len(x)) * x
    r = breusch_pagan(y, x)
    assert r["lm_pvalue"] < 0.05, f"BP should reject homosk under heterosk: p={r['lm_pvalue']:.4f}"


def test_white_rejects_heteroskedastic():
    rng = np.random.default_rng(5)
    x = np.linspace(-2, 2, 1500)
    y = 1.0 + 0.5 * x + rng.standard_normal(len(x)) * np.abs(x)
    r = white_test(y, x)
    assert r["lm_pvalue"] < 0.05, f"White should reject homosk: p={r['lm_pvalue']:.4f}"


def test_hac_runs_and_is_finite():
    rng = np.random.default_rng(6)
    x = rng.standard_normal(500)
    y = 2.0 + 1.0 * x + rng.standard_normal(500)
    r = hac_ols(y, x)
    assert np.all(np.isfinite(r["hac_t"]))
    assert r["hac_se"].shape == r["classic_se"].shape


def test_dm_detects_better_model():
    rng = np.random.default_rng(7)
    target = rng.standard_normal(500)
    # model 1 slightly better than model 2
    e1 = target - (0.3 * target)          # correlated, small error
    # make e2 clearly worse
    e2 = target - (target * 0 + rng.standard_normal()*0 + np.roll(target, 1)) - rng.standard_normal(500)
    e2 = target - np.zeros_like(target)   # worst: no signal
    # simpler: e1 small noise, e2 big noise
    e1 = rng.standard_normal(500) * 0.5
    e2 = rng.standard_normal(500) * 1.5
    dm = diebold_mariano(e1, e2, h=1)
    # e1 has much smaller squared loss => strongly negative DM
    assert dm["dm_statistic"] < -2.0, f"DM should strongly favor model1: {dm['dm_statistic']:.2f}"
    assert dm["pvalue"] < 0.05
