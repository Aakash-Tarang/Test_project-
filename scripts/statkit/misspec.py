"""Model-misspecification tests: the Ramsey RESET test.

The Ramsey regression specification-error test (RESET, Ramsey 1969) asks whether a
linear model has *omitted nonlinearity*. Run the base regression
    y = X b + e
obtain fitted values yhat = X b, then run the auxiliary regression of y on X plus
powers yhat^2, yhat^3, ... and test (F or LM) whether the added powers are jointly
zero:

    H0 : y = X b  (linear specification adequate)          -> no omitted nonlinearity
    H1 : powers of yhat add explanatory power              -> misspecified / nonlinear

If H0 is rejected the linear model leaves systematic (nonlinear) structure in the
residuals, which motivates a nonlinear extension; if it is not rejected, a nonlinear
model has little to add on top of the linear fit.
"""
import numpy as np

__all__ = ["ramsey_reset"]


def ramsey_reset(y, X, powers=(2, 3), test="f"):
    """Ramsey RESET test for omitted nonlinearity.

    Parameters
    ----------
    y : array-like (n,)
    X : array-like (n, k) — exog WITHOUT a constant column (a constant is added here).
    powers : tuple of fitted-value powers to include (default (2,3)).
    test : 'f' (F-test) or 'lm' (LM/n*R^2).

    Returns dict with the statistic, pvalue, number of added terms, and the
    restricted/unrestricted residual sum of squares used for the F form.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(y)
    Xc = np.column_stack([np.ones(n), X])          # unrestricted will add powers
    # restricted (linear) fit
    br, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid_r = y - Xc @ br
    yhat = Xc @ br
    ssr_r = float(np.sum(resid_r ** 2))
    # add powers of yhat (centered to reduce collinearity)
    ph = yhat - yhat.mean()
    add = np.column_stack([ph ** p for p in powers])
    Xu = np.column_stack([Xc, add])
    bu, *_ = np.linalg.lstsq(Xu, y, rcond=None)
    resid_u = y - Xu @ bu
    ssr_u = float(np.sum(resid_u ** 2))
    q = add.shape[1]                               # number of added regressors
    df1 = q
    df2 = n - Xc.shape[1] - q
    if test == "lm":
        # LM = n R^2 from the unrestricted regression
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2u = 1.0 - ssr_u / ss_tot if ss_tot > 0 else 0.0
        stat = n * r2u
        # asymptotic chi2(df1)
        from scipy.stats import chi2
        pvalue = float(chi2.sf(stat, df1))
    else:  # F
        if df2 <= 0:
            stat, pvalue = float("nan"), float("nan")
        else:
            stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2) if ssr_u > 0 else float("inf")
            from scipy.stats import f
            pvalue = float(f.sf(stat, df1, df2))
    return {"statistic": float(stat), "pvalue": pvalue,
            "n_added": int(q), "powers": list(powers), "test": test,
            "ssr_restricted": ssr_r, "ssr_unrestricted": ssr_u,
            "reject_linear_at_5pct": bool(pvalue < 0.05)}
