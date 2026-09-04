"""Unit tests for the Ramsey RESET misspecification test."""
import numpy as np

from ..misspec import ramsey_reset


def test_reset_accepts_truly_linear():
    # y is exactly linear in X => RESET should NOT reject the linear model (p>0.05).
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.standard_normal((n, 2))
    y = 1.0 + 0.7 * X[:, 0] - 1.3 * X[:, 1] + 0.3 * rng.standard_normal(n)
    r = ramsey_reset(y, X, powers=(2, 3))
    assert not r["reject_linear_at_5pct"], f"linear data rejected RESET: p={r['pvalue']:.4f}"


def test_reset_rejects_quadratic():
    # y contains a strong quadratic term => RESET should reject the linear model.
    rng = np.random.default_rng(1)
    n = 1500
    x = rng.standard_normal(n)
    X = x.reshape(-1, 1)
    # quadratic signal: y = x^2 + noise; a linear-in-x fit leaves x^2 structure
    y = 0.0 + 0.0 * x + x ** 2 + 0.1 * rng.standard_normal(n)
    r = ramsey_reset(y, X, powers=(2, 3))
    assert r["reject_linear_at_5pct"], f"quadratic data accepted RESET: p={r['pvalue']:.4f}"


def test_reset_f_form_matches_lm_decision():
    rng = np.random.default_rng(2)
    n = 1200
    X = rng.standard_normal((n, 2))
    y = 1.0 + 0.5 * X[:, 0] + X[:, 0] ** 2 + 0.5 * X[:, 1] + 0.1 * rng.standard_normal(n)
    rf = ramsey_reset(y, X, test="f")
    rl = ramsey_reset(y, X, test="lm")
    assert rf["reject_linear_at_5pct"] == rl["reject_linear_at_5pct"]
    assert rf["n_added"] == 2 and rl["n_added"] == 2
