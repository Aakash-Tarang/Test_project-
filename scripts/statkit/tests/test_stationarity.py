"""Tests for ADF/KPSS joint interpretation on synthetic stationary vs unit-root."""
import numpy as np

from ..stationarity import adf_test, kpss_test, interpret_stationarity


def test_adf_rejects_unitroot_on_stationary_series():
    rng = np.random.default_rng(0)
    x = np.cumsum(0.5 * rng.standard_normal(1000)) + 10 * rng.standard_normal(1000)
    x = x - np.arange(len(x)) * 0.0  # stationary around driftless level w/ shocks
    # Better: a truly stationary AR process
    ar = np.empty(1000)
    ar[0] = 0
    for t in range(1, 1000):
        ar[t] = 0.7 * ar[t - 1] + rng.standard_normal()
    r = adf_test(ar)
    assert r["pvalue"] < 0.05, f"ADF should reject unit root on stationary AR: p={r['pvalue']:.4f}"


def test_adf_does_not_reject_on_random_walk():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(500))
    r = adf_test(x)
    assert r["pvalue"] > 0.05, f"ADF should NOT reject unit root on RW: p={r['pvalue']:.4f}"


def test_kpss_accepts_stationary():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(800)
    r = kpss_test(x)
    assert r["pvalue"] > 0.05, f"KPSS should accept stationary null: p={r['pvalue']:.4f}"


def test_kpss_rejects_on_random_walk():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.standard_normal(500))
    r = kpss_test(x)
    # statsmodels caps KPSS p-value at 0.1 by default in old versions; check stat vs 5% crit
    assert r["statistic"] > r["critical"][0.05], "KPSS should reject stationary on RW"


def test_joint_label_stationary_and_unitroot():
    rng = np.random.default_rng(4)
    ar = np.empty(1500); ar[0] = 0.0
    for t in range(1, 1500):
        ar[t] = 0.6 * ar[t - 1] + rng.standard_normal()
    assert interpret_stationarity(ar)["label"] == "stationary"

    rw = np.cumsum(rng.standard_normal(800))
    assert interpret_stationarity(rw)["label"] == "unit_root"
