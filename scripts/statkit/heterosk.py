"""Heteroskedasticity tests: Breusch-Pagan and White.

  * Breusch-Pagan: regress squared OLS residuals on the original regressors
    (+const). LM = n*R^2 ~ chi2(k). H0: homoskedasticity.
  * White: regress squared residuals on regressors, their squares and cross
    products (test of general heteroskedasticity / misspecification).
    H0: homoskedasticity.
Both are available through statsmodels; we wrap them with clean return dicts.
"""
import numpy as np
import statsmodels.api as _sm

from .cointegration import _ols_residual

__all__ = ["breusch_pagan", "white_test"]


def _prep(y, X):
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    # het_* tests require exog with a constant column; add it if missing.
    if not np.any(np.all(X == 1.0, axis=0)):
        X = _sm.add_constant(X)
    return y, X


def breusch_pagan(y, X):
    """Breusch-Pagan test. H0: homoskedastic residuals from y ~ X."""
    from statsmodels.stats.diagnostic import het_breuschpagan
    y, X = _prep(y, X)
    resid, _ = _ols_residual(y, X)
    lm, lm_p, fstat, f_p = het_breuschpagan(resid, X)
    return {"lm_statistic": float(lm), "lm_pvalue": float(lm_p),
            "f_statistic": float(fstat), "f_pvalue": float(f_p)}


def white_test(y, X):
    """White test (general heteroskedasticity, includes squares/interactions).
    H0: homoskedasticity."""
    from statsmodels.stats.diagnostic import het_white
    y, X = _prep(y, X)
    resid, _ = _ols_residual(y, X)
    lm, lm_p, fstat, f_p = het_white(resid, X)
    return {"lm_statistic": float(lm), "lm_pvalue": float(lm_p),
            "f_statistic": float(fstat), "f_pvalue": float(f_p)}
