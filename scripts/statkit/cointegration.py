"""Cointegration tests.

  * engle_granger : two-step (Engle & Granger 1987). Step 1 regress y on x (OLS)
    and obtain the residual; step 2 run an ADF test on the residual. Rejecting a
    unit root in the residual indicates the series are cointegrated (the residual
    is a stationary spread that can be traded).
  * johansen      : Johansen (1991) trace / max-eigenvalue test on a vector of
    series to infer the cointegration RANK of a basket.
"""
import numpy as np
import pandas as pd

__all__ = ["engle_granger", "johansen_rank"]


def _ols_residual(y, X, add_const=True):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if add_const:
        X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return resid, beta


def engle_granger(y, x, adf_kwargs=None):
    """Engle-Granger two-step cointegration test between y and x.

    Returns dict with the estimated hedge beta, the residual, and the ADF result
    (with the DF/EG critical values caveat that standard ADF tables are approximate
    for the two-step estimator).
    """
    from .stationarity import adf_test
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    resid, beta = _ols_residual(y, x)
    adf_kwargs = adf_kwargs or {}
    adf = adf_test(resid, **adf_kwargs)
    return {"beta": beta, "residual": resid, "adf": adf}


def johansen_rank(df, det_order=0, k_ar_diff=1, significance_level=0.05):
    """Johansen cointegration rank test on a DataFrame of I(1) series (columns).

    Returns the trace statistic result: number of cointegrating relationships
    implied at the given level, plus the trace/statistics/critical table.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    df = df.dropna()
    data = df.values.astype(float)
    n, p = data.shape
    if p < 2:
        raise ValueError("Need >= 2 series")
    res = coint_johansen(data, det_order, k_ar_diff)
    # Eigenvalues, trace stats and 95% critical values (rows = r=0..p-1)
    evals = res.eig
    trace_stat = res.lr1
    trace_crit = res.cvt  # columns correspond to 90,95,99? cvt shape (p,3)
    # statsmodels cvt: critical values for trace at 90/95/99 as columns 0/1/2
    cv95 = trace_crit[:, 1]
    # choose rank r* = smallest r such that trace_stat[r] > cv95[r] (reject r<=r-1)
    rank = int(np.sum(trace_stat > cv95))
    return {
        "n_series": p,
        "n_obs": n,
        "rank_at_95pct": rank,
        "trace_statistic": trace_stat.tolist(),
        "trace_crit_95": cv95.tolist(),
        "eigenvalues": evals.tolist(),
        "significance_level": significance_level,
    }
