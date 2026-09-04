"""Time-series bootstrap for dependent data.

For returns/forecast-loss series we must NOT use an i.i.d. bootstrap (it destroys
the temporal dependence and gives overconfident intervals). Instead we use the
stationary bootstrap of Politis & Romano (1994): resample blocks of random
(geometric) length so that the resampled series is (weakly) stationary.

Provides `stationary_bootstrap` (return resampled indices) and
`bootstrap_ci` (percentile confidence interval of a statistic), used for
confidence intervals on Sharpe ratio, mean return, etc.
"""
import numpy as np

__all__ = ["stationary_bootstrap", "bootstrap_ci"]


def stationary_bootstrap(n, n_boot=1, mean_block=10, rng=None):
    """Return a (n_boot, n) int array of stationary-bootstrap indices.

    Each series: start uniformly, then take a block of geometric length with
    mean `mean_block`; when the index runs past n it wraps around.
    """
    if rng is None:
        rng = np.random.default_rng()
    p = 1.0 / mean_block
    out = np.empty((n_boot, n), dtype=int)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(0, n)
        for t in range(n):
            idx[t] = i
            if rng.random() < p:
                i = rng.integers(0, n)  # start a new block (geometric length)
            else:
                i = (i + 1) % n        # continue block, wrapping
        out[b] = idx
    return out


def bootstrap_ci(x, statistic, n_boot=1000, mean_block=10, alpha=0.05, seed=None):
    """Percentile confidence interval for `statistic(x)` via stationary bootstrap.

    statistic: callable x->scalar (e.g. lambda s: s.mean(), or a Sharpe fn).
    Returns dict: estimate (on original data), ci_low, ci_high, bootstrap_sd.
    """
    x = np.asarray(x, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 2:
        raise ValueError("need >=2 observations")
    idx = stationary_bootstrap(n, n_boot, mean_block=mean_block, rng=rng)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = statistic(x[idx[b]])
    est = statistic(x)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": float(est), "ci_low": float(lo), "ci_high": float(hi),
            "bootstrap_sd": float(stats.std(ddof=1)), "n_boot": int(n_boot)}
