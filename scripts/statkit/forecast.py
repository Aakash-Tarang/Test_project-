"""Diebold-Mariano test for comparing two forecasts.

Given forecast errors e1_t and e2_t for the same target, define the loss
differential d_t = loss(e1_t) - loss(e2_t) (loss is squared or absolute error
typically). Under the null of equal predictive accuracy, E[d]=0. The DM
statistic is the mean of d_t divided by a HAC estimate of its long-run variance:

    DM = mean(d) / sqrt( (1/n) * long-run var(d) )

which is asymptotically standard normal (Diebold & Mariano 1995). Negative DM
means forecast 1 is better (smaller mean loss).
"""
import numpy as np

__all__ = ["diebold_mariano", "dm_return_pair"]


def _loss(e, loss):
    e = np.asarray(e, dtype=float)
    if loss == "square":
        return e ** 2
    if loss == "abs":
        return np.abs(e)
    raise ValueError("loss must be 'square' or 'abs'")


def _long_run_var(d, h):
    """Newey-West estimate of long-run variance of mean, with truncation lag h-1."""
    n = len(d)
    d = d - d.mean()
    gamma0 = np.dot(d, d) / n
    gamma = 0.0
    maxlag = max(0, h - 1)
    for lag in range(1, maxlag + 1):
        w = 1.0 - lag / (maxlag + 1.0)  # Bartlett
        cov = np.dot(d[lag:], d[:n - lag]) / n
        gamma += 2.0 * w * cov
    return (gamma0 + gamma) / n


def diebold_mariano(e1, e2, h=1, loss="square"):
    """DM test statistic and two-sided p-value (asymptotic normal).

    e1, e2: equal-length arrays of out-of-sample forecast errors.
    h: forecast horizon (>=1); used for the HAC long-run variance.
    Returns dict: dm_statistic, pvalue, mean_loss_diff, direction.
    """
    e1 = np.asarray(e1, dtype=float).ravel()
    e2 = np.asarray(e2, dtype=float).ravel()
    n = len(e1)
    if len(e2) != n or n < 2:
        raise ValueError("e1 and e2 must be equal length, length>=2")
    d = _loss(e1, loss) - _loss(e2, loss)
    mean_d = d.mean()
    lrv = _long_run_var(d, h)
    if lrv <= 0:
        # degenerate (perfectly correlated losses) -> treat as no difference
        return {"dm_statistic": float("nan"), "pvalue": float("nan"),
                "mean_loss_diff": float(mean_d), "n_obs": int(n)}
    from scipy import stats as _st
    dm = mean_d / np.sqrt(lrv)
    p = 2.0 * (1.0 - _st.norm.cdf(abs(dm)))
    return {"dm_statistic": float(dm), "pvalue": float(p),
            "mean_loss_diff": float(mean_d), "n_obs": int(n),
            "note": "Negative DM => model 1 (e1) has lower loss (better forecast)."}


def dm_return_pair(r1, r2, maxlags=None, two_sided=True):
    """Diebold--Mariano-style test on a pair of *return* (P&L) series.

    The forecast-error form of DM (\`diebold_mariano\`) squares the loss, which is the
    right object for forecast-accuracy comparison but not for comparing two trading
    strategies, where higher mean P&L is better. Here we take the return differential

        d_t = r1_t - r2_t

    and test H0: E[d] = 0 (equal mean P&L) against the alternative that one strategy
    dominates, dividing the sample mean of d by a Newey--West HAC standard error of the
    mean (correlated daily P&L from multi-day holds invalidates an i.i.d. standard
    error). Positive dm => strategy 1 has higher mean daily P&L.

    r1, r2 : equal-length arrays of daily net returns.
    maxlags: HAC truncation lag (default: Newey-West rule of thumb).
    Returns dict: dm_statistic, pvalue (two-sided unless two_sided=False), mean_diff,
    hac_se, n_obs.
    """
    import statsmodels.api as sm
    r1 = np.asarray(r1, dtype=float).ravel()
    r2 = np.asarray(r2, dtype=float).ravel()
    n = len(r1)
    if len(r2) != n or n < 2:
        raise ValueError("r1 and r2 must be equal length, length>=2")
    d = r1 - r2
    if maxlags is None:
        maxlags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    # HAC regression of the differential on a lone intercept (no extra constant):
    # beta is the HAC sample mean, bse is its Newey-West standard error.
    X = np.ones((n, 1))
    fit = sm.OLS(d, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    mean_diff = float(fit.params[0])
    se = float(fit.bse[0])
    dm = mean_diff / se if se > 0 else float("nan")
    from scipy import stats as _st
    if np.isnan(dm):
        pvalue = float("nan")
    else:
        pvalue = 2.0 * (1.0 - _st.norm.cdf(abs(dm))) if two_sided \
            else (1.0 - _st.norm.cdf(dm))
    return {"dm_statistic": float(dm), "pvalue": float(pvalue),
            "mean_diff": mean_diff, "hac_se": float(se),
            "n_obs": int(n), "maxlags": int(maxlags),
            "note": "Positive DM => strategy 1 (r1) has higher mean daily P&L."}
