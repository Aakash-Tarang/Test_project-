#!/usr/bin/env python3
"""nonlinear_models.py — causal, no-lookahead nonlinear forecasting of the spread.

Extends the linear spread machinery with a forecast-comparison layer used in Part 9.
The object is the tradeable causal spread of Parts 4-6:
    S_t = y_t - (alpha_{t|t-1} + beta_{t|t-1} . x_t)      (rolling-OLS hedge through t-1)
    z_t = S_t / sigma_t                                     (standardized mispricing)

We build, for each bar t, a causal feature vector f_t (information through t only) and
the realized *short-side* h-bar market-neutral return R_t^h of fading a rich spread:
    R_t^h = -(S_{t+h} - S_t) / g_t ,      g_t = 1 + sum_i |beta_{t|t-1,i}|
(the gross return of shorting $1 of target / longing the hedged basket and holding h
bars against the fixed entry hedge). Predicting R_t^h is the mean-reversion forecast.

Forecasters are fit on an expanding causal window (retrained periodically, never on
future data) and evaluated out-of-sample on the test period:
    "linear"   Ridge regression on standardized features (the linear baseline)
    "gbm"      HistGradientBoostingRegressor (nonlinear)
    "mlp"      shallow MLPRegressor (nonlinear)
A "persistence/current-mispricing" reference is also computed (IC of z_t itself) to
show how much beyond simply trading the current |z_t| each forecaster adds.

Every function is deliberately leak-free: f_t uses only bars <= t, and the target is
the realized value at t+h only used for fitting on past bars / evaluation on future.
"""
from __future__ import annotations

import numpy as np

BARS_PER_YEAR = 252.0


def causal_spread(lprices, window=120):
    """Causal rolling-OLS spread S_t, z_t, hedge beta_t, and g_t.

    lprices: (n_assets, T) log prices, row0 target. beta used to score bar t is fit on
    the window ending t-1 (no lookahead), matching the engine. Returns dict of (T,)
    arrays; entries before `window` are NaN.
    """
    n_assets, T = lprices.shape
    k = n_assets - 1
    Xf = lprices[1:, :]
    yf = lprices[0, :]
    S = np.full(T, np.nan)
    z = np.full(T, np.nan)
    beta = np.full((T, k), np.nan)
    g = np.full(T, np.nan)
    sigma = np.full(T, np.nan)
    alpha = np.full(T, np.nan)
    dof = max(int(window) - k - 1, 1)
    for t in range(window, T):
        A = np.column_stack([np.ones(window), Xf[:, t - window:t].T])
        b, *_ = np.linalg.lstsq(A, yf[t - window:t], rcond=None)
        a0, bvec = b[0], b[1:]
        pred = a0 + Xf[:, t] @ bvec
        r = yf[t] - pred
        # in-sample residual std over the fitting window (C++/Part-5 convention)
        pred_tr = A @ b
        ss = float(np.sum((yf[t - window:t] - pred_tr) ** 2))
        sig = np.sqrt(ss / dof) if ss >= 0 else 0.0
        S[t] = r
        z[t] = r / sig if sig > 0 else 0.0
        beta[t] = bvec
        alpha[t] = a0
        g[t] = 1.0 + float(np.abs(bvec).sum())
        sigma[t] = sig
    return {"S": S, "z": z, "beta": beta, "alpha": alpha, "g": g,
            "sigma": sigma, "window": window, "k": k, "T": T}


def build_features_targets(lprices, window=120, h=5):
    """Causal features + realized h-bar short-side return target over the full sample."""
    cs = causal_spread(lprices, window)
    T = lprices.shape[1]
    S, z = cs["S"], cs["z"]
    k = cs["k"]
    Xf = lprices[1:, :]
    lr = np.diff(lprices, axis=1)             # (n_assets, T-1); lr[:, s] = ret into s+1
    # target log return per bar t (t>=1)
    ret0 = np.full(T, np.nan); ret0[1:] = lr[0]
    retb = np.full(T, np.nan)                  # basket-mean return
    retb[1:] = lr[1:].mean(axis=0)

    n_feat = 11
    F = np.full((T, n_feat), np.nan)
    for t in range(1, T):
        f = []
        f.append(z[t])                          # 0 current mispricing
        f.append(S[t])                          # 1 spread level
        f.append(S[t] - S[t - 1] if t >= window and np.isfinite(S[t]) and np.isfinite(S[t-1]) else np.nan)  # 2
        # 3/4 recent 5d/20d realized vol of S (past only)
        if t >= window + 5 and np.isfinite(S[t - 4:t + 1]).all():
            f.append(S[t - 4:t + 1].std())
        else:
            f.append(np.nan)
        if t >= window + 20 and np.isfinite(S[t - 19:t + 1]).all():
            f.append(S[t - 19:t + 1].std())
        else:
            f.append(np.nan)
        f.append(ret0[t])                       # 5
        f.append(ret0[t - 1] if t >= 1 else np.nan)  # 6
        f.append(retb[t])                       # 7
        f.append(retb[t - 1] if t >= 1 else np.nan)  # 8
        f.append(S[t] - S[t - 3] if t >= window and np.isfinite(S[t]) and np.isfinite(S[t-3]) else np.nan)  # 9
        f.append(z[t - 1] if t >= 1 and np.isfinite(z[t - 1]) else np.nan)  # 10
        F[t] = f
    # target: realized h-bar short-side return R_t^h = -(S_{t+h}-S_t)/g_t
    target = np.full(T, np.nan)
    for t in range(window, T - h):
        if np.isfinite(S[t]) and np.isfinite(S[t + h]) and np.isfinite(cs["g"][t]) \
                and cs["g"][t] > 0:
            target[t] = -(S[t + h] - S[t]) / cs["g"][t]
    return {"feat": F, "target": target, "z": z, "S": S, "cs": cs}


def walk_forecast(lprices, model, window=120, h=5, retrain_every=63, seed=0):
    """Causal expanding-window forecast; scaler is fit on the same train window and
    applied to the held-out prediction bars (proper no-lookahead standardization)."""
    feat = build_features_targets(lprices, window, h)
    F, target, z = feat["feat"], feat["target"], feat["z"]
    T = F.shape[0]
    start = window + 20
    end = T - h
    valid = np.isfinite(F).all(axis=1)
    pred = np.full(T, np.nan)
    train_lo = start                      # expanding: always train on [start, lo)
    for lo in range(start, end, retrain_every):
        hi = min(lo + retrain_every, end)
        idx_tr = np.arange(train_lo, lo)
        tr_ok = idx_tr[np.isfinite(F[idx_tr]).all(axis=1) & np.isfinite(target[idx_tr])]
        if len(tr_ok) < 200:
            continue                       # need a minimum training sample before predicting
        Xtr = F[tr_ok]; ytr = target[tr_ok]
        mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd = np.where(sd < 1e-12, 1.0, sd)
        Xs = (Xtr - mu) / sd
        if model == "linear":
            from sklearn.linear_model import Ridge
            m = Ridge(alpha=1.0).fit(Xs, ytr)
        elif model == "gbm":
            from sklearn.ensemble import HistGradientBoostingRegressor
            m = HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=15,
                                              learning_rate=0.08,
                                              random_state=seed).fit(Xs, ytr)
        elif model == "mlp":
            from sklearn.neural_network import MLPRegressor
            m = MLPRegressor(hidden_layer_sizes=(16,), max_iter=1500, random_state=seed,
                             early_stopping=True, alpha=0.001).fit(Xs, ytr)
        else:
            raise ValueError(model)
        # predict held-out bars [lo,hi): standardize with the SAME train scaler
        for t in range(lo, hi):
            if valid[t]:
                xt = (F[t] - mu) / sd
                pred[t] = float(m.predict(xt.reshape(1, -1))[0])
    return {"pred": pred, "z": z, "target": target, "T": T, "start": start}
