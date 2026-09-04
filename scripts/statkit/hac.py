"""Newey-West (HAC) standard errors.

When residuals are autocorrelated and/or heteroskedastic, the OLS covariance
matrix is inconsistent; Newey & West (1987) provide a positive semi-definite
HAC covariance estimator. We expose a helper that fits y ~ X and reports
coefficients with both classic (i.i.d.) and HAC standard errors so the change in
t-statistics can be shown before/after correction.
"""
import numpy as np

__all__ = ["hac_ols", "fit_and_compare_se"]


def hac_ols(y, X, maxlags=None):
    """OLS with Newey-West HAC standard errors using statsmodels.

    Returns dict with beta, classic_se, hac_se, classic_t, hac_t,
    plus n, k, maxlags.
    """
    import statsmodels.api as sm
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = sm.add_constant(X)
    if maxlags is None:
        n = len(y)
        maxlags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))  # common rule of thumb
    model = sm.OLS(y, X).fit()
    hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {
        "beta": np.asarray(model.params),
        "classic_se": np.asarray(model.bse),
        "hac_se": np.asarray(hac.bse),
        "classic_t": np.asarray(model.tvalues),
        "hac_t": np.asarray(hac.tvalues),
        "n_obs": int(len(y)),
        "maxlags": int(maxlags),
    }


def fit_and_compare_se(y, X, maxlags=None):
    """Convenience: returns the compact comparison of classic vs HAC inference."""
    r = hac_ols(y, X, maxlags=maxlags)
    return {
        "coefficients": r["beta"].tolist(),
        "classic_t": r["classic_t"].tolist(),
        "hac_t": r["hac_t"].tolist(),
        "classic_se": r["classic_se"].tolist(),
        "hac_se": r["hac_se"].tolist(),
        "note": "HAC (Newey-West) standard errors correct for autocorrelation and heteroskedasticity.",
    }
