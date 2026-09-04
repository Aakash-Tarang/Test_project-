#!/usr/bin/env python3
"""compare_nonlinear.py — Part 9 nonlinear extension + RESET test.

Asks, honestly, whether nonlinear signal extraction adds value over the linear
spread hedges of Parts 4-8 on the same no-lookahead, cost-aware footing.

Forecast task (scripts/model/nonlinear_models.py): predict R_t^h, the realized
h-bar (h=5) *short-side* market-neutral return of fading a rich causal spread on
target-vs-basket (log-price rolling-OLS hedge). Models compared:
    linear Ridge   (the linear baseline on the same causal features)
    GBM            HistGradientBoostingRegressor (nonlinear)
    MLP            shallow MLPRegressor (nonlinear)
Forecasters are fit on a causal *expanding* window and predict only held-out future
bars (no lookahead in features, target, or standardization).

Deliverables here:
    * out-of-sample (2022+) IC and R^2 of each forecaster per basket;
    * Ramsey RESET on the linear regression (does it leave omitted nonlinearity?);
    * a net-of-cost (10 bps) gated trade comparison: the Part-4 z-threshold book,
      optionally gated so entries require the forecaster to confirm reversion;
    * GBM learning-curve data (OOS error vs training sample size) for the renderer.

Writes (git-ignored): results/tables/nonlinear_summary.csv,
results/tables/nonlinear_reset.csv, results/tables/nonlinear_trading.csv,
results/tables/nonlinear_learning.csv.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "model"))
import nonlinear_models as nm  # noqa: E402
from baseline_models import walk_forward, simulate, metrics_from_nav  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "scripts", "statkit"))
import statkit  # noqa: E402

ENGINE = os.path.join(ROOT, "data", "processed", "engine")
TAB = os.path.join(ROOT, "results", "tables")
os.makedirs(TAB, exist_ok=True)

WINDOW, ENTRY, EXIT, COST, H = 120, 2.0, 0.5, 0.001, 5
TEST_START = "2022-01-01"
BASKETS = [
    ("ADBE", "ADBE.csv", ["CRM", "ADSK", "INTU"]),
    ("AVGO", "AVGO.csv", ["ADI", "INTC", "AAPL"]),
    ("GS", "GS.csv", ["BAC"]),
    ("CAT", "CAT.csv", ["HON"]),
]
MODELS = ["linear", "gbm", "mlp"]
COST_STR = "10 bps"


def load_lp(csvf):
    raw = pd.read_csv(os.path.join(ENGINE, csvf))
    cols = [c for c in raw.columns if c != "date"]
    dates = pd.to_datetime(raw["date"]).values
    lp = np.log(raw[cols].values.astype(float)).T
    return lp, dates


def oos_metrics(pred, target, dates, a):
    """IC and R^2 over bars >= a where both pred & target are finite."""
    ok = np.isfinite(pred) & np.isfinite(target) & (dates >= a)
    ix = np.where(ok)[0]
    if len(ix) < 30:
        return None
    p = pred[ix]; t = target[ix]
    ic = float(np.corrcoef(p, t)[0, 1]) if p.std() > 0 and t.std() > 0 else 0.0
    denom = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - float(np.sum((t - p) ** 2)) / denom if denom > 0 else 0.0
    return {"n": len(ix), "ic": ic, "r2": r2}


def gated_simulate(z, beta_hist, lp, a, b, pred, entry=ENTRY, exit_z=EXIT, cost=COST):
    """Part-4 book with optional reversion-confirmation gate from a forecaster.

    When pos==0 and |z_t|>=entry: open SHORT (z>0) only if pred[t]>0 (forecaster
    expects short-side reversion profit), LONG (z<0) only if pred[t]<0; if pred is NaN
    the gate is ignored (entry as baseline). Returns gross/net nav increments in [a,b).
    """
    n_assets, T = lp.shape
    k = n_assets - 1
    lr = np.diff(lp, axis=1)
    pos = 0
    beta = np.zeros(k)
    g = 1.0
    lg = 0.0
    cost_log = 0.0
    gross = np.zeros(b - a)
    net = np.zeros(b - a)
    n_trades = 0
    for idx, t in enumerate(range(a, b)):
        ret = 0.0
        if t >= 1 and pos != 0:
            s = -lr[0, t - 1] + float(beta @ lr[1:, t - 1])
            ret = (float(pos) / g) * s
        lg += ret
        if t >= 1 and t < b - H and np.isfinite(z[t]):
            zt = z[t]
            pv = pred[t] if np.isfinite(pred[t]) else (1.0 if zt > 0 else -1.0)
            if pos == 0:
                want = 0
                if zt >= entry and pv > 0:
                    want = 1
                elif zt <= -entry and pv < 0:
                    want = -1
                if want != 0:
                    pos = want
                    beta = beta_hist[t].copy()
                    g = 1.0 + float(np.abs(beta).sum())
                    cost_log += cost * g
                    n_trades += 1
            else:
                do = False
                if pos > 0 and zt <= exit_z:
                    do = True
                elif pos < 0 and zt >= -exit_z:
                    do = True
                if do:
                    cost_log += cost * g
                    pos = 0
                    beta = np.zeros(k)
                    g = 1.0
        gross[idx] = lg
        net[idx] = lg - cost_log
    return {"gross": gross, "net": net, "n_trades": n_trades}


def main():
    dates0 = None
    for key, csvf, _ in BASKETS:
        lp, dates = load_lp(csvf)
        dates0 = dates if dates0 is None else dates
        break
    test0 = int(np.searchsorted(dates0.astype("datetime64[D]"),
                                np.datetime64(TEST_START)))
    test0 = max(test0, WINDOW + 40)

    summ_rows = []
    reset_rows = []
    # ---------- 1. OOS IC / R^2 per basket and model ----------
    for key, csvf, _ in BASKETS:
        lp, dates = load_lp(csvf)
        for m in MODELS:
            if m == "mlp" and key != "ADBE":
                continue          # keep runtime sane; MLP reported on ADBE only
            wf = nm.walk_forecast(lp, m, WINDOW, H, 63, 0)
            om = oos_metrics(wf["pred"], wf["target"], dates, dates[test0])
            if om:
                summ_rows.append({"basket": key, "model": m,
                                  "oos_ic": round(om["ic"], 4),
                                  "oos_r2": round(om["r2"], 4),
                                  "n": int(om["n"])})
    # ---------- 2. RESET on the linear forecast regression (training rows only) ----
    for key, csvf, _ in BASKETS:
        lp, dates = load_lp(csvf)
        bf = nm.build_features_targets(lp, WINDOW, H)
        F, y = bf["feat"], bf["target"]
        ok = np.isfinite(F).all(axis=1) & np.isfinite(y) & (dates < dates[test0])
        ix = np.where(ok)[0]
        if len(ix) < 300:
            continue
        # subsample to keep OLS/aux manageable and representative
        step = max(1, len(ix) // 2000)
        X = F[ix[::step]]; yy = y[ix[::step]]
        # drop features that are constant to avoid singular design
        Xc = X[:, np.std(X, axis=0) > 1e-12]
        r = statkit.misspec.ramsey_reset(yy, Xc, powers=(2, 3), test="f")
        reset_rows.append({"basket": key, "reset_p": round(r["pvalue"], 4),
                           "reject_linear_at_5pct": r["reject_linear_at_5pct"],
                           "n_train": len(ix)})
    # ---------- 3. Gated net-of-cost trading on ADBE ----------
    key = "ADBE"
    csvf = [c for k, c, _ in BASKETS if k == key][0]
    lp, dates = load_lp(csvf)
    wf = walk_forward(lp, WINDOW, "OLS", {})
    preds = {m: nm.walk_forecast(lp, m, WINDOW, H, 63, 0)["pred"] for m in MODELS}
    trade_rows = []
    full_a = WINDOW
    # baseline (no gate)
    s = simulate(wf["z"], wf["beta"], lp, full_a, lp.shape[1], WINDOW,
                 ENTRY, EXIT, COST)
    mf = metrics_from_nav(s["net"])
    trade_rows.append({"config": "z-threshold (Part-4 baseline)",
                       "scope": "full 2010-2026", "net_sharpe": round(mf["sharpe"], 3),
                       "net_ann": round(mf["ann_ret"], 4), "maxdd": round(mf["maxdd"], 3),
                       "trades": int(s["n_trades"])})
    # gated variants
    for m in MODELS:
        sg = gated_simulate(wf["z"], wf["beta"], lp, full_a, lp.shape[1], preds[m])
        mg = metrics_from_nav(sg["net"])
        trade_rows.append({"config": f"z + {m}-gate", "scope": "full 2010-2026",
                           "net_sharpe": round(mg["sharpe"], 3),
                           "net_ann": round(mg["ann_ret"], 4),
                           "maxdd": round(mg["maxdd"], 3),
                           "trades": int(sg["n_trades"])})
    # test-only (2022+)
    st = simulate(wf["z"], wf["beta"], lp, test0, lp.shape[1], WINDOW,
                  ENTRY, EXIT, COST)
    mt = metrics_from_nav(st["net"])
    trade_rows.append({"config": "z-threshold (Part-4 baseline)",
                       "scope": "test 2022+", "net_sharpe": round(mt["sharpe"], 3),
                       "net_ann": round(mt["ann_ret"], 4), "maxdd": round(mt["maxdd"], 3),
                       "trades": int(st["n_trades"])})
    for m in MODELS:
        sgt = gated_simulate(wf["z"], wf["beta"], lp, test0, lp.shape[1], preds[m])
        mgt = metrics_from_nav(sgt["net"])
        trade_rows.append({"config": f"z + {m}-gate", "scope": "test 2022+",
                           "net_sharpe": round(mgt["sharpe"], 3),
                           "net_ann": round(mgt["ann_ret"], 4),
                           "maxdd": round(mgt["maxdd"], 3),
                           "trades": int(sgt["n_trades"])})
    # ---------- 4. GBM learning curve on ADBE ----------
    lc_rows = []
    bf = nm.build_features_targets(lp, WINDOW, H)
    F, y = bf["feat"], bf["target"]
    dates_ix = dates
    # fixed validation chunk 2019-2020 for the learning-curve OOS error
    val_a = int(np.searchsorted(dates.astype("datetime64[D]"),
                                np.datetime64("2019-01-01")))
    val_b = test0
    tr_start = WINDOW + 20
    val_ok = np.isfinite(F).all(axis=1) & np.isfinite(y) & (np.arange(len(F)) >= val_a) \
        & (np.arange(len(F)) < val_b)
    val_ix = np.where(val_ok)[0]
    for K in (400, 800, 1200, 1600, 2000, 2400, 2800):
        train_ok = np.isfinite(F).all(axis=1) & np.isfinite(y) \
            & (np.arange(len(F)) >= tr_start) & (np.arange(len(F)) < tr_start + K)
        tr_ix = np.where(train_ok)[0]
        if len(tr_ix) < 150 or len(val_ix) < 100:
            continue
        Xtr = F[tr_ix]; ytr = y[tr_ix]
        mu = Xtr.mean(0); sd = Xtr.std(0); sd = np.where(sd < 1e-12, 1.0, sd)
        from sklearn.ensemble import HistGradientBoostingRegressor
        g = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15,
                                          learning_rate=0.08, random_state=0)
        g.fit((Xtr - mu) / sd, ytr)
        Xv = (F[val_ix] - mu) / sd
        pv = g.predict(Xv)
        tv = y[val_ix]
        mse = float(np.mean((tv - pv) ** 2))
        denom = float(np.var(tv))
        lc_rows.append({"train_bars": int(len(tr_ix)),
                        "oos_mse": round(mse, 8),
                        "oos_r2": round(1.0 - mse / denom, 4) if denom > 0 else 0.0})

    pd.DataFrame(summ_rows).to_csv(os.path.join(TAB, "nonlinear_summary.csv"), index=False)
    pd.DataFrame(reset_rows).to_csv(os.path.join(TAB, "nonlinear_reset.csv"), index=False)
    pd.DataFrame(trade_rows).to_csv(os.path.join(TAB, "nonlinear_trading.csv"), index=False)
    pd.DataFrame(lc_rows).to_csv(os.path.join(TAB, "nonlinear_learning.csv"), index=False)

    print("== OOS IC / R^2 by basket & model (test 2022+) ==")
    print(pd.DataFrame(summ_rows).to_string(index=False))
    print("\n== Ramsey RESET on linear forecast model ==")
    print(pd.DataFrame(reset_rows).to_string(index=False))
    print("\n== Net-of-cost gated trading (ADBE) ==")
    print(pd.DataFrame(trade_rows).to_string(index=False))
    print("\n== GBM learning curve (ADBE) ==")
    print(pd.DataFrame(lc_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
