"""Residual autocorrelation tests: Durbin-Watson and Ljung-Box.

  * Durbin-Watson tests first-order serial correlation in OLS residuals.
    DW in [0,4]; ~2 => no AR(1) autocorrelation; <2 => positive autocorrelation.
    H0: no first-order autocorrelation.
  * Ljung-Box tests whether any of the first `lags` autocorrelations are jointly
    non-zero (portmanteau). H0: the series is white noise / no serial correlation.
"""
import numpy as np

__all__ = ["durbin_watson", "ljung_box"]


def durbin_watson(resid):
    """Durbin-Watson statistic for first-order serial correlation."""
    from statsmodels.stats.stattools import durbin_watson as _dw
    resid = np.asarray(resid, dtype=float).ravel()
    resid = resid[~np.isnan(resid)]
    return float(_dw(resid))


def ljung_box(resid, lags=None):
    """Ljung-Box portmanteau test for residual autocorrelation up to `lags`.

    Returns dict: lb_statistic, pvalue, lags_used.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    resid = np.asarray(resid, dtype=float).ravel()
    resid = resid[~np.isnan(resid)]
    if lags is None:
        lags = min(10, max(1, len(resid) // 5))
    if len(resid) <= lags:
        raise ValueError("Not enough observations for requested lags")
    lb = acorr_ljungbox(resid, lags=[lags], return_df=True)
    stat = float(lb["lb_stat"].iloc[0])
    p = float(lb["lb_pvalue"].iloc[0])
    return {"statistic": stat, "pvalue": p, "lags": int(lags)}
