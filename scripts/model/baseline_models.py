#!/usr/bin/env python3
"""baseline_models.py — walk-forward estimators for the Part 5 model comparison.

Each estimator maps a trailing window of the target's LOG price (y) and the basket's
LOG prices (X) to a fitted linear hedge y ~ intercept + X^T beta. All methods fit
with an intercept and share the causal convention of the Part 4 engine: the window
used to score bar t ends at bar t-1 (no lookahead).

Estimator implementations (sklearn, thin wrappers so the C++ basket_selector.h
library and this harness are independently checkable):
    OLS        LinearRegression (raw features) — must reproduce the C++ engine baseline.
    Ridge      Ridge on column-standardized features (alpha in standardized units).
    Lasso      Lasso on standardized features (sparse hedge).
    ElasticNet ElasticNet on standardized features (l1_ratio in (0,1)).
    PCA/PCR    k principal components of the standardized basket, then OLS on scores.

Standardizing the penalized / PCA variants makes penalties comparable across
windows and is the conventional formulation; beta / intercept are mapped back to the
raw log-price scale so the resulting spread is directly comparable across methods.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

# Daily bars per year used to annualize Sharpe (matches Part 4 engine).
BARS_PER_YEAR = 252.0


def _raw_from_std(coef_std, intercept_std, mu, scale):
    """Map a model fit on column-standardized features to raw-feature units.

    Xs = (X - mu)/scale ; pred = intercept_std + sum coef_std_k * Xs_k
       => beta_raw_k = coef_std_k / scale_k
       => intercept_raw = intercept_std - sum_k coef_std_k*mu_k/scale_k
    """
    scale = np.where(scale == 0, 1.0, scale)
    beta_raw = coef_std / scale
    intercept_raw = intercept_std - np.sum(coef_std * mu / scale)
    return intercept_raw, beta_raw


def _standardize(Xw):
    sc = StandardScaler().fit(Xw)
    Xs = sc.transform(Xw)
    return sc, Xs


def make_fitter(method, params):
    """Return fit(Xw, yw) -> (intercept_raw, beta_raw, resid_std) for a method.

    resid_std is sqrt(sum(resid^2)/(W-p-1)) computed over the training window
    (the C++ RollingRegression's degrees-of-freedom convention), used to form the
    standardized signal z.
    """
    params = params or {}

    if method == "OLS":
        def fit_ols(Xw, yw):
            m = LinearRegression().fit(Xw, yw)
            return m.intercept_, m.coef_.copy(), None
        return fit_ols

    if method == "Ridge":
        alpha = float(params.get("alpha", 1.0))
        def fit_ridge(Xw, yw):
            sc, Xs = _standardize(Xw)
            m = Ridge(alpha=alpha).fit(Xs, yw)
            inter, beta = _raw_from_std(m.coef_, m.intercept_, sc.mean_, sc.scale_)
            return inter, beta, None
        return fit_ridge

    if method == "Lasso":
        alpha = float(params.get("alpha", 1e-3))
        def fit_lasso(Xw, yw):
            sc, Xs = _standardize(Xw)
            m = Lasso(alpha=alpha, max_iter=20000, tol=1e-6).fit(Xs, yw)
            inter, beta = _raw_from_std(m.coef_, m.intercept_, sc.mean_, sc.scale_)
            return inter, beta, None
        return fit_lasso

    if method == "ElasticNet":
        alpha = float(params.get("alpha", 1e-3))
        rho = float(params.get("l1_ratio", 0.5))
        def fit_en(Xw, yw):
            sc, Xs = _standardize(Xw)
            m = ElasticNet(alpha=alpha, l1_ratio=rho, max_iter=20000, tol=1e-6).fit(Xs, yw)
            inter, beta = _raw_from_std(m.coef_, m.intercept_, sc.mean_, sc.scale_)
            return inter, beta, None
        return fit_en

    if method == "PCA":
        k = int(params.get("k", 2))
        def fit_pca(Xw, yw):
            sc, Xs = _standardize(Xw)
            nf = Xw.shape[1]
            ncomp = min(k, nf)
            if ncomp < 1:
                ncomp = nf
            pca = PCA(n_components=ncomp).fit(Xs)
            Z = pca.transform(Xs)
            m = LinearRegression().fit(Z, yw)
            # effective slope in the standardized-feature space: coef_std = V^T @ coef_pc
            coef_std = pca.components_.T @ m.coef_
            inter, beta = _raw_from_std(coef_std, m.intercept_, sc.mean_, sc.scale_)
            return inter, beta, None
        return fit_pca

    raise ValueError(f"unknown method {method}")


def walk_forward(lprices, window, method, params):
    """Causal walk-forward: fit window ending t-1, predict bar t.

    lprices: (n_assets, T) log prices, row 0 = target.
    Returns dict with arrays over T (NaN before the window is warm) and timings.
    """
    n_assets, T = lprices.shape
    k = n_assets - 1
    Xf = lprices[1:, :]          # basket log prices, (k, T)
    yf = lprices[0, :]           # target log price
    fit = make_fitter(method, params)

    z = np.full(T, np.nan)
    beta_hist = np.full((T, k), np.nan)
    inter_hist = np.full(T, np.nan)
    r2_hist = np.full(T, np.nan)
    n_fits = 0
    import time
    t0 = time.perf_counter()

    # residual std denominator convention: W - p - 1 (free intercept), C++ dof.
    dof = max(int(window) - k - 1, 1)

    for t in range(window, T):
        Xw = Xf[:, t - window:t].T          # (W, k)
        yw = yf[t - window:t]
        inter, beta, _ = fit(Xw, yw)
        pred = inter + Xf[:, t] @ beta
        resid_t = yf[t] - pred
        # in-sample residual std over the fit window (C++ sigma_residual)
        pred_tr = inter + Xw @ beta
        ss_res = float(np.sum((yw - pred_tr) ** 2))
        sigma = np.sqrt(ss_res / dof) if ss_res >= 0 else 0.0
        z[t] = resid_t / sigma if sigma > 0 else 0.0
        beta_hist[t] = beta
        inter_hist[t] = inter
        # training R^2
        ss_tot = float(np.sum((yw - yw.mean()) ** 2))
        r2_hist[t] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        n_fits += 1

    el = time.perf_counter() - t0
    return {
        "z": z, "beta": beta_hist, "intercept": inter_hist, "r2": r2_hist,
        "n_fits": n_fits, "secs": el, "ms_per_fit": 1000.0 * el / max(n_fits, 1),
        "window": window, "k": k, "T": T,
    }


def simulate(z, beta_hist, lprices, a, b, window, entry_z=2.0, exit_z=0.5,
             cost_one_way=0.001):
    """Replicate the Part 4 single-book, one-bar-delay trade logic over bars [a,b).

    Returns dict of gross/net NAV increments within the segment (book empty at a),
    plus trade/statistics.
    """
    n_assets, T = lprices.shape
    k = n_assets - 1
    lr = np.diff(lprices, axis=1)         # (n_assets, T-1); lr[:, s] is return into bar s+1
    # lr for bar s: price at s vs s-1
    # Build returns indexed by bar t (t>=1): ret_bar[t] = lr[:, t-1]
    # We'll just index with t-1 below.
    warm = int(window)
    a = max(a, 0)

    pos = 0
    beta = np.zeros(k)
    g = 1.0
    open_bar = -1
    lg = 0.0
    cost = 0.0
    gross = np.zeros(b - a)
    net = np.zeros(b - a)
    pos_arr = np.zeros(b - a, dtype=int)
    total_hold = 0
    n_opens = 0
    n_trades = 0
    cost_log = 0.0

    for idx, t in enumerate(range(a, b)):
        # P&L over bar t from the book active during bar t.
        ret = 0.0
        if t >= 1 and pos != 0:
            s = -lr[0, t - 1]
            s = s + float(beta @ lr[1:, t - 1])
            ret = (float(pos) / g) * s
            total_hold += 1
        lg += ret
        pos_arr[idx] = pos

        # Decision for bar t+1 from causal z[t].
        if t >= warm and not np.isnan(z[t]):
            zt = z[t]
            if pos == 0:
                newpos = 0
                if zt >= entry_z:
                    newpos = 1
                elif zt <= -entry_z:
                    newpos = -1
                if newpos != 0:
                    pos = newpos
                    beta = beta_hist[t].copy()
                    g = 1.0 + float(np.abs(beta).sum())
                    open_bar = t
                    n_opens += 1
                    n_trades += 1
                    cost_log += cost_one_way * g
            else:
                do_exit = False
                if pos > 0 and zt <= exit_z:
                    do_exit = True
                elif pos < 0 and zt >= -exit_z:
                    do_exit = True
                if do_exit:
                    cost_log += cost_one_way * g
                    n_trades += 1
                    pos = 0
                    beta = np.zeros(k)
                    g = 1.0
                    open_bar = -1
        gross[idx] = lg
        net[idx] = lg - cost_log

    return {
        "gross": gross, "net": net, "pos": pos_arr,
        "n_trades": n_trades, "n_opens": n_opens, "total_hold": total_hold,
        "cost_log": cost_log,
    }


def metrics_from_nav(nav):
    """Annualized mean/sharpe/maxdd from a log-NAV increment series."""
    d = np.diff(nav)
    if d.size == 0:
        return {"ann_ret": 0.0, "sharpe": 0.0, "maxdd": 0.0}
    sd = d.std(ddof=1)
    mean = d.mean()
    years = d.size / BARS_PER_YEAR
    ann_ret = nav[-1] / years if years > 0 else 0.0
    sharpe = (mean / sd) * np.sqrt(BARS_PER_YEAR) if sd > 0 else 0.0
    nav_c = np.concatenate([[0.0], nav])
    peak = np.maximum.accumulate(nav_c)
    maxdd = float(np.max(peak - nav_c))
    return {"ann_ret": ann_ret, "sharpe": sharpe, "maxdd": maxdd}


def information_coefficient(z, beta_hist, lprices, a, b, window, h=21):
    """Gross OOS IC: corr(z_t, P&L of entering at t+1 and holding h bars).

    P&L uses the sign of the mispricing (short target when z>0), fixed beta_hist[t].
    """
    n_assets, T = lprices.shape
    lr = np.diff(lprices, axis=1)
    k = n_assets - 1
    zs = []
    rs = []
    for t in range(max(a, int(window)), b):
        if t + h >= T or np.isnan(z[t]):
            continue
        beta = beta_hist[t]
        g = 1.0 + float(np.abs(beta).sum())
        sign = 1.0 if z[t] > 0 else -1.0
        tot = 0.0
        for s in range(t + 1, t + 1 + h):
            if s >= T:
                break
            leg = -lr[0, s - 1] + float(beta @ lr[1:, s - 1])
            tot += (sign / g) * leg
        zs.append(z[t])
        rs.append(tot)
    zs = np.array(zs); rs = np.array(rs)
    if zs.size < 5:
        return 0.0
    if np.std(zs) == 0 or np.std(rs) == 0:
        return 0.0
    return float(np.corrcoef(zs, rs)[0, 1])
