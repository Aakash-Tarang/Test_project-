"""Ornstein-Uhlenbeck process fit and implied half-life.

The OU process
    dX_t = theta (mu - X_t) dt + sigma dW_t
has the discrete (Euler) representation, for sampling interval dt (here dt=1):
    X_{t+1} - X_t = theta*mu*dt - theta*X_t*dt + sigma*sqrt(dt) eps_{t+1}
So we regress  dX = X_{t+1} - X_t  on a constant and X_t:
    dX = a + b X_t + e,
which gives
    theta = -b / dt ,   mu = a / (theta*dt) ,   sigma = std(e) / sqrt(dt).
The mean-reversion half-life is  ln(2)/theta  (in sampling-period units).
This is the classic method (e.g. Ornstein-Uhlenbeck as used in pairs/stat-arb
papers) for estimating the speed of residual reversion; it is compared to the
strategy holding period in the report (Section on diagnostics).
"""
import numpy as np

__all__ = ["fit_ou", "half_life"]


def fit_ou(x, dt=1.0):
    """Fit an OU process to a 1-D series via discretized regression.

    Returns theta (speed), mu (long-run mean), sigma (vol), halflife, residual
    std, and the OLS coefficient of determination of the auxiliary regression.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[~np.isnan(x)]
    if len(x) < 3:
        raise ValueError("OU fit requires at least 3 observations")
    dx = np.diff(x)
    x_prev = x[:-1]
    # regress dx on [1, x_prev]
    A = np.column_stack([np.ones(len(dx)), x_prev])
    beta, *_ = np.linalg.lstsq(A, dx, rcond=None)
    a, b = beta[0], beta[1]
    resid = dx - A @ beta
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((dx - dx.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    theta = -b / dt
    # a = theta*mu*dt  =>  mu = a / (theta*dt)   [if theta ~> 0]
    mu = a / (theta * dt) if abs(theta) > 1e-12 else float("nan")
    sigma = float(np.std(resid, ddof=1) / np.sqrt(dt))
    hl = half_life(theta)
    return {
        "theta": float(theta),
        "mu": float(mu),
        "sigma": float(sigma),
        "half_life": hl,          # in dt units
        "r2_aux": float(r2),
        "n_obs": int(len(x)),
    }


def half_life(theta):
    """Half-life ln(2)/theta of an OU process with speed theta."""
    theta = float(theta)
    if theta <= 1e-12:
        return float("inf")
    return float(np.log(2.0) / theta)
