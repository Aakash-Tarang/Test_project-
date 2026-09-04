"""Stationarity tests: Augmented Dickey-Fuller (ADF) and KPSS.

The two tests have OPPOSITE null hypotheses, so they are read jointly:
  * ADF   H0: the series has a unit root (is integrated / non-stationary).
          Small p => reject H0 => evidence of stationarity.
  * KPSS  H0: the series is (trend/level) stationary.
          Small p => reject H0 => evidence of a unit root / non-stationarity.
A residual series is judged consistent with stationarity when ADF rejects AND
KPSS does NOT reject. Inconsistent outcomes indicate an ambiguous sample
(too few observations, borderline process, breaks, etc.).
"""
import warnings

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

__all__ = ["adf_test", "kpss_test", "interpret_stationarity"]


def _clean(x):
    x = np.asarray(x, dtype=float).ravel()
    return x[~np.isnan(x)]


def _normalize_crit_vals(crit):
    """Normalize critical-value lookup to float-keyed dict {0.01,0.05,0.10}.

    statsmodels returns these either as {'1%':..,'5%':..,'10%':..} (dict), a bare
    tuple/array ordered (1%,5%,10%) for KPSS, or a dict with the same info. This
    helper normalizes to {0.01,0.05,0.10} so downstream code is version-stable.
    """
    keys = ("1%", "5%", "10%")
    order = (0.01, 0.05, 0.10)
    if hasattr(crit, "items"):
        items = {k: float(v) for k, v in crit.items()}
        out = {}
        for label, level in zip(keys, order):
            out[level] = items.get(label, items.get(str(level), float("nan")))
        return out
    arr = np.asarray(crit, dtype=float).ravel()  # KPSS: ordered 1%,5%,10%
    return {level: float(arr[i]) for i, level in enumerate(order)} if len(arr) == 3 else {
        level: float("nan") for level in order}



def adf_test(x, regression="c", autolag="AIC", maxlag=None):
    """Augmented Dickey-Fuller test.

    Parameters
    ----------
    x : array-like
    regression : {'c','ct','ctt','n'}  constant / +trend / +trend² / none.
    autolag : lag selection ('AIC','BIC','t-stat', None).

    Returns dict with statistic, pvalue, usedlag, critical values.
    """
    x = _clean(x)
    if len(x) < 3:
        raise ValueError("ADF requires at least 3 observations")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pvalue, usedlag, nobs, crit, _icbest = adfuller(
            x, maxlag=maxlag, regression=regression, autolag=autolag)
    return {"statistic": float(stat), "pvalue": float(pvalue),
            "usedlag": int(usedlag), "n_obs": int(nobs),
            "critical": _normalize_crit_vals(crit)}


def kpss_test(x, regression="c", nlags="auto"):
    """Kwiatkowski-Phillips-Schmidt-Shin test. H0: stationary.

    regression 'c' => level stationarity; 'ct' => trend stationarity.
    """
    x = _clean(x)
    if len(x) < 3:
        raise ValueError("KPSS requires at least 3 observations")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pvalue, lags, crit = kpss(x, regression=regression, nlags=nlags)
    return {"statistic": float(stat), "pvalue": float(pvalue),
            "lags": int(lags), "critical": _normalize_crit_vals(crit)}


def interpret_stationarity(x, alpha=0.05, regression="c"):
    """Joint ADF+KPSS verdict. Returns a dict describing both tests and a label:
      'stationary'   ADF rejects H0 and KPSS does not reject H0.
      'unit_root'    ADF does not reject H0 and KPSS rejects H0.
      'ambiguous'    otherwise (insufficient/conflicting evidence).
    """
    a = adf_test(x, regression=regression)
    k = kpss_test(x, regression=regression)
    adf_reject = a["pvalue"] < alpha
    kpss_reject = k["pvalue"] < alpha
    if adf_reject and not kpss_reject:
        label = "stationary"
    elif (not adf_reject) and kpss_reject:
        label = "unit_root"
    else:
        label = "ambiguous"
    return {
        "label": label,
        "adf_pvalue": a["pvalue"], "adf_statistic": a["statistic"], "adf_reject_h0_unitroot": bool(adf_reject),
        "kpss_pvalue": k["pvalue"], "kpss_statistic": k["statistic"], "kpss_reject_h0_stationary": bool(kpss_reject),
    }
