#!/usr/bin/env python3
"""kalman.py — Kalman-filtered time-varying hedge for the spread stat-arb (Part 8).

Mirrors the causal, one-bar-delay discipline of baseline_models.walk_forward so the
resulting signal can be dropped straight into baseline_models.simulate and compared
net-of-cost against the fixed-window rolling OLS/Ridge hedges of Parts 5/7.

State-space model (log prices, target y, basket x):
    beta_t = beta_{t-1} + eta_t,            eta_t ~ N(0, Q)      (random-walk drift)
    y_t    = [1, x_t]^T beta_t + eps_t,     eps_t ~ N(0, R)      (observation)

State is (intercept, beta_1..beta_k); the design row is (1, x_t). The recursion is
the standard linear Gaussian filter:
    predict   beta_{t|t-1} = beta_{t-1}            ,  P_{t|t-1} = P_{t-1} + Q
    innovation v_t = y_t - [1,x_t] beta_{t|t-1}    ,  S_t = [1,x_t] P [1,x_t]^T + R
    update    K = P [1,x_t]^T / S_t ; beta_t = beta_{t|t-1} + K v_t ; P_t = (I-K[1,x_t])P

No-lookahead: the beta used to score bar t is the *predictive* state beta_{t|t-1},
which uses only information through bar t-1 -- exactly the timing of the rolling
window that ends at t-1 in walk_forward. z_t is the standardized innovation
v_t/sqrt(S_t). After scoring, bar t is fed in to update the state for future bars.

R (observation variance) is estimated from a warm-up OLS fit; Q (state drift) is set
as Q_ii = q^2 * R, a single tuneable dimensionless hyper-parameter q chosen on a
validation split. Output is API-compatible with walk_forward (z, beta, intercept).
"""
from __future__ import annotations

import time

import numpy as np

BARS_PER_YEAR = 252.0


def _stable_cov_update(P, x, S, K):
    """P_t = (I - K x^T) P_{t|t-1} in Joseph-stabilised form (keeps P symmetric)."""
    I_Kx = np.eye(len(P)) - np.outer(K, x)
    return I_Kx @ P @ I_Kx.T + np.outer(K, K) * max(S, 1e-18)


def kalman_hedge(lprices, q=1e-2, warm=120, R_scale=1.0):
    """Run the Kalman filter for the target-vs-basket hedge.

    lprices: (n_assets, T) log prices, row 0 = target, rows 1.. = basket.
    q: state-noise std (dimensionless, Q_ii = q^2 * R).
    warm: bars used for the initial OLS warm-up (also the first bar scored).
    R_scale: multiplier on the warm-up residual variance used for R.

    Returns dict API-compatible with baseline_models.walk_forward.
    """
    n_assets, T = lprices.shape
    k = n_assets - 1
    X = lprices[1:, :]            # (k, T) basket log prices
    y = lprices[0, :]             # target log price
    d = k + 1                     # intercept + k betas

    # ---- warm-up OLS to initialise state and R ----
    A0 = np.column_stack([np.ones(warm), X[:, :warm].T])   # (warm, d)
    y0 = y[:warm]
    beta0, *_ = np.linalg.lstsq(A0, y0, rcond=None)
    resid0 = y0 - A0 @ beta0
    R = float(np.dot(resid0, resid0)) / max(warm - d, 1) * R_scale
    try:
        P = R * np.linalg.inv(A0.T @ A0)      # OLS posterior covariance
    except np.linalg.LinAlgError:
        P = R * np.eye(d)
    Q = (q ** 2) * R * np.eye(d)              # diagonal state drift

    z = np.full(T, np.nan)
    beta_hist = np.full((T, k), np.nan)
    inter_hist = np.full(T, np.nan)
    v_hist = np.full(T, np.nan)

    t0 = time.perf_counter()
    state = beta0.copy()
    n_fits = 0
    for t in range(warm, T):
        x_t = np.concatenate([[1.0], X[:, t]])
        # predict (random walk: mean unchanged, add Q)
        P_p = P + Q
        # innovation against predictive state (causal, data through t-1)
        pred = float(state @ x_t)
        v = float(y[t] - pred)
        S = float(x_t @ P_p @ x_t) + R
        z[t] = v / np.sqrt(S) if S > 0 else 0.0
        v_hist[t] = v
        beta_hist[t] = state[1:]
        inter_hist[t] = state[0]
        # update with y_t (feed forward for future bars)
        K = P_p @ x_t / S
        state = state + K * v
        P = _stable_cov_update(P_p, x_t, S, K)
        n_fits += 1

    el = time.perf_counter() - t0
    return {
        "z": z, "beta": beta_hist, "intercept": inter_hist,
        "innovation": v_hist, "R": R, "q": q, "warm": warm,
        "n_fits": n_fits, "ms_per_fit": 1000.0 * el / max(n_fits, 1),
        "k": k, "T": T, "window": warm,
    }
