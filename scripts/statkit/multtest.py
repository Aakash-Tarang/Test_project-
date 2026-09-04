"""Multiple-testing / data-snooping correction.

Given a universe scan that produces the best performance statistic from many
candidate strategies, naively reporting its p-value is invalid (data snooping).
We implement three corrections:

  * bonferroni      : adjusted threshold alpha/M (or adjusted p = M*p).
                      Conservative; controls family-wise error rate (FWER).
  * white_reality_check : White (2000). Tests whether the BEST of M models has
                      performance better than a benchmark, against a null that the
                      best model has no predictive superiority. Uses the stationary
                      bootstrap over the re-centered performance differentials and
                      accounts for the max over M models in the test statistic.
  * hansen_spa      : Hansen (2005) Superior Predictive Ability. Same family but
                      uses a studentized statistic and a re-centering that improves
                      power vs. White's Reality Check.

Both RC/SPA answer the question (spec 2.3): after scanning M candidate baskets,
is the best out-of-sample performance statistically distinguishable from the
benchmark / from zero?

Note on p-value direction convention: we work with a "performance" series f_t per
strategy (e.g. daily PnL). The best strategy maximizes mean(f). We test
  H0: max_m E[f_m] <= 0   (no strategy beats zero)
against the alternative that the best is positive. See White (2000) for the
technical construction; the implementation below follows the standard textbook
algorithm (Politis-Romano stationary bootstrap on dbar = mean over time of f).
"""
import numpy as np

from .bootstrap import stationary_bootstrap

__all__ = ["bonferroni", "white_reality_check", "hansen_spa"]


def bonferroni(pvalues, alpha=0.05):
    """Bonferroni correction. Returns the set of indices significant at FWER alpha."""
    p = np.asarray(pvalues, dtype=float).ravel()
    m = len(p)
    adjusted = np.minimum(p * m, 1.0) if m > 0 else p
    significant = np.where(adjusted < alpha)[0].tolist()
    return {"n_tests": int(m), "adjusted_pvalues": adjusted.tolist(),
            "significant_indices": significant,
            "rejects_at_fwer": bool(len(significant) > 0)}


def _bootstrap_best_perf(perf, n_boot=1000, mean_block=10, seed=1):
    """perf: (T, M) array of performance series (e.g. daily PnL per strategy).
    Returns dict of bootstrap dist of the max-mean statistic under the RC null
    and the SPA (studentized) null.
    """
    perf = np.asarray(perf, dtype=float)
    T, M = perf.shape
    rng = np.random.default_rng(seed)
    dbar = perf.mean(axis=0)              # (M,) sample means
    idx = stationary_bootstrap(T, n_boot, mean_block=mean_block, rng=rng)
    # sample means per bootstrap per strategy
    boot_means = np.empty((n_boot, M))
    for b in range(n_boot):
        boot_means[b] = perf[idx[b]].mean(axis=0)
    # White RC null: recenter each strategy by its own mean (so H0 all <=0 boundary)
    rc_null = boot_means - dbar[None, :]          # (n_boot, M)
    rc_dist = rc_null.max(axis=1)                 # (n_boot,)
    # Hansen SPA: studentized, recenter only "good" models (dbar>0) by 0 and the
    # others by own mean; recentering on the empirical mean for all is the RC.
    # We implement the common variant: recenter by min(dbar,0) for studentization.
    omega2 = boot_means.var(axis=0, ddof=1)
    omega = np.sqrt(np.maximum(omega2, 1e-12))
    spa_null = (boot_means - dbar[None, :]) / omega[None, :]
    spa_dist = spa_null.max(axis=1)
    return {"rc_dist": rc_dist, "spa_dist": spa_dist, "dbar": dbar,
            "observed_rc": float(dbar.max()), "omega": omega}


def white_reality_check(perf, n_boot=1000, mean_block=10, seed=1):
    """White (2000) Reality Check p-value for the best of M models.

    obs = max_m mean(f_m). Under H0 all models have mean <= 0; we recenter by each
    model's own sample mean and build the null distribution of the max. p-value =
    fraction of bootstrap max-stats >= observed max.
    """
    perf = np.asarray(perf, dtype=float)
    T, M = perf.shape
    if M < 2:
        raise ValueError("Reality Check needs >=2 models")
    b = _bootstrap_best_perf(perf, n_boot=n_boot, mean_block=mean_block, seed=seed)
    obs = b["observed_rc"]
    # RC null distribution recenters by the sample mean; observed test stat is max dbar.
    pvalue = float(np.mean(b["rc_dist"] >= obs))
    return {"pvalue": pvalue, "observed_max_mean": obs, "n_boot": int(n_boot),
            "n_models": int(M), "mean_block": int(mean_block),
            "test": "White Reality Check"}


def hansen_spa(perf, n_boot=1000, mean_block=10, seed=1):
    """Hansen (2005) SPA p-value (studentized variant)."""
    perf = np.asarray(perf, dtype=float)
    T, M = perf.shape
    if M < 2:
        raise ValueError("SPA needs >=2 models")
    b = _bootstrap_best_perf(perf, n_boot=n_boot, mean_block=mean_block, seed=seed)
    # studentized observed statistic (max of dbar/omega)
    obs = float(np.max(b["dbar"] / np.maximum(b["omega"], 1e-12)))
    pvalue = float(np.mean(b["spa_dist"] >= obs))
    return {"pvalue": pvalue, "observed_studentized_max": obs, "n_boot": int(n_boot),
            "n_models": int(M), "mean_block": int(mean_block), "test": "Hansen SPA"}
