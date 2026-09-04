"""Synthetic-data tests for OU fitting (spec permits synthetic data for math checks)."""
import numpy as np

from ..ou import fit_ou, half_life


def simulate_ou(theta, mu, sigma, n, dt=1.0, seed=0, x0=None):
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = mu if x0 is None else x0
    sdt = np.sqrt(dt)
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + sigma * sdt * rng.standard_normal()
    return x


def test_ou_recovers_theta_and_halflife():
    # strong reversion
    theta_true, mu_true, sigma_true = 0.5, 0.0, 0.1
    x = simulate_ou(theta_true, mu_true, sigma_true, 20000, seed=1)
    res = fit_ou(x, dt=1.0)
    # discretization + noise: allow tolerance
    assert abs(res["theta"] - theta_true) < 0.02, f"theta={res['theta']:.4f}"
    assert abs(res["mu"] - mu_true) < 0.02, f"mu={res['mu']:.4f}"
    assert abs(res["sigma"] - sigma_true) < 0.01, f"sigma={res['sigma']:.4f}"
    assert abs(res["half_life"] - np.log(2) / theta_true) < 0.1, res["half_life"]


def test_halflife_math():
    assert abs(half_life(1.0) - np.log(2.0)) < 1e-9
    # faster reversion => shorter half-life
    assert half_life(2.0) < half_life(0.5)


def test_ou_zero_speed_gives_infinite_halflife():
    assert np.isinf(half_life(0.0))


def test_ou_fit_short_series_rejects():
    try:
        fit_ou(np.array([1.0, 2.0]))
        assert False, "should have raised"
    except ValueError:
        pass


def test_ou_nonstationary_series_slow_theta():
    # random walk (theta~0) => OU theta near 0, half-life huge
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.standard_normal(4000))
    res = fit_ou(x)
    assert res["theta"] < 0.05, f"theta={res['theta']:.4f} (random walk should have theta~0)"
    assert res["half_life"] > 100
