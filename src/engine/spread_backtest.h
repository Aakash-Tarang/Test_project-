// spread_backtest.h — causal, cost-aware, walk-forward spread backtest for a
// single target vs. a basket of regressors (cointegration-flavoured stat-arb).
//
// No-lookahead conventions:
//   1. Rolling regression of the target's LOG price on the basket features' LOG
//      prices over a window W. The window used to score bar t ends at bar t-1
//      (bar t is fed only AFTER it is scored), so the fitted beta_t is causal and
//      the target's own bar-t price never influences its own fitted value.
//   2. z_t = standardized residual of bar t against that past fit.
//        z_t >= +entry  => target rich  => SHORT target / LONG basket  (pos = +1)
//        z_t <= -entry  => target cheap => LONG  target / SHORT basket (pos = -1)
//      Close when |z_t| reverts below exit, or after > half_life_cap bars.
//   3. A decision made with data through close(t) starts earning return on bar
//      t+1 only (position active for bar d is decided at d-1). Standard
//      trade-on-close, no-lookahead convention (unit-tested).
//   4. Dollar-normalized market-neutral book: short $1 of target, long
//      beta_i dollars of feature i, rescaled by g = (1 + sum_i |beta_i|) so gross
//      notional is $1. Weights (raw beta, g) are fixed at entry (no intra-trade
//      rebalancing). Daily log P&L per active bar:
//            ret = (pos/g) * ( -lr_target + sum_i beta_i * lr_i )
//      and NAV = exp(cumulative log P&L).  pos=+1 short target, pos=-1 long target.
//   5. Cost: one-way proportional cost charged on the traded gross (g) at each
//      open and each close (both legs, both sides). Reported gross AND net.
#ifndef STATARB_SPREAD_BACKTEST_H
#define STATARB_SPREAD_BACKTEST_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "../model/rolling_regression.h"

namespace statarb {

struct SpreadConfig {
    size_t window = 120;
    double entry_z = 2.0;
    double exit_z = 0.5;
    double half_life_cap = 0.0;   // bars; <=0 disables
    double ridge_lambda = 0.0;
    double cost_one_way = 0.001;  // fraction of gross traded, each side
};

struct SpreadResult {
    std::vector<double> lognav_gross;
    std::vector<double> lognav_net;
    std::vector<int> pos;                 // +1 short-target, -1 long-target, 0 flat
    std::vector<double> z;
    size_t n_bars = 0, n_trades = 0;
    double cost_paid = 0.0;
    double ann_ret_net = 0.0, sharpe_net = 0.0, maxdd_net = 0.0;
    double sharpe_gross = 0.0, ann_ret_gross = 0.0;
    double avg_holding_bars = 0.0;
};

inline SpreadResult run_spread_backtest(const std::vector<std::vector<double>>& price,
                                        const SpreadConfig& cfg) {
    const size_t n_assets = price.size();
    const size_t T = n_assets ? price[0].size() : 0;
    if (n_assets < 2 || T == 0) return {};
    for (size_t a = 0; a < n_assets; ++a) if (price[a].size() != T) return {};
    const size_t k = n_assets - 1;
    const size_t warm = std::min(cfg.window, T);

    SpreadResult res;
    res.lognav_gross.assign(T, 0.0);
    res.lognav_net.assign(T, 0.0);
    res.pos.assign(T, 0);
    res.z.assign(T, 0.0);
    res.n_bars = T;

    // ---- log prices, log returns ----
    std::vector<std::vector<double>> lp(n_assets, std::vector<double>(T));
    std::vector<std::vector<double>> lr(n_assets, std::vector<double>(T, 0.0));
    for (size_t a = 0; a < n_assets; ++a)
        for (size_t t = 0; t < T; ++t) lp[a][t] = std::log(std::max(price[a][t], 1e-9));
    for (size_t a = 0; a < n_assets; ++a)
        for (size_t t = 1; t < T; ++t) lr[a][t] = lp[a][t] - lp[a][t - 1];

    // ---- causal z[t] and hedge betas ----
    std::vector<std::vector<double>> hedge_beta(T);
    std::vector<double> z(T, 0.0);
    {
        RollingRegression reg(cfg.window, k, cfg.ridge_lambda);
        std::vector<double> row(k);
        for (size_t t = 0; t < warm; ++t) {
            for (size_t i = 0; i < k; ++i) row[i] = lp[1 + i][t];
            reg.update(row, lp[0][t]);
        }
        for (size_t t = warm; t < T; ++t) {
            for (size_t i = 0; i < k; ++i) row[i] = lp[1 + i][t];
            if (reg.ready()) {
                reg.compute();
                if (reg.betaValid()) {
                    z[t] = reg.residualZScore(row, lp[0][t]);
                    hedge_beta[t] = reg.beta();
                }
            }
            reg.update(row, lp[0][t]);
        }
    }
    res.z = z;

    // ---- single stateful book; decisions take effect next bar ----
    int pos = 0;
    std::vector<double> beta(k, 0.0);
    double g = 1.0;
    long open_bar = -1;
    long total_holding = 0, n_opens = 0;

    double lg = 0.0, cost_log = 0.0, ln = 0.0;
    size_t n_trades = 0;

    for (size_t t = 0; t < T; ++t) {
        // --- P&L over bar t from the book active during bar t ---
        double ret = 0.0;
        if (t >= 1 && pos != 0) {
            double s = -lr[0][t];
            for (size_t i = 0; i < k; ++i) s += beta[i] * lr[1 + i][t];
            ret = ((double)pos / g) * s;
            ++total_holding;
        }
        lg += ret;
        res.pos[t] = pos;

        // --- decision for bar t+1 from causal z[t] ---
        if (t >= warm) {
            double zt = z[t];
            if (pos == 0) {
                int newpos = 0;
                if (zt >= cfg.entry_z) newpos = 1;
                else if (zt <= -cfg.entry_z) newpos = -1;
                if (newpos != 0) {
                    pos = newpos;
                    beta = (hedge_beta[t].size() == k) ? hedge_beta[t]
                                                       : std::vector<double>(k, 1.0 / (double)k);
                    g = 1.0;
                    for (double v : beta) g += std::fabs(v);
                    open_bar = (long)t;
                    ++n_opens;
                    ++n_trades;
                    cost_log += cfg.cost_one_way * g;      // open cost (both legs)
                }
            } else {
                bool exit = false;
                if (cfg.half_life_cap > 0 && open_bar >= 0 && (long)t - open_bar > (long)cfg.half_life_cap)
                    exit = true;
                if (pos > 0 && zt <= cfg.exit_z) exit = true;
                if (pos < 0 && zt >= -cfg.exit_z) exit = true;
                if (exit) {
                    cost_log += cfg.cost_one_way * g;      // close cost
                    ++n_trades;
                    pos = 0;
                    std::fill(beta.begin(), beta.end(), 0.0);
                    g = 1.0;
                }
            }
        }

        ln = lg - cost_log;
        res.lognav_gross[t] = lg;
        res.lognav_net[t] = ln;
    }
    res.n_trades = n_trades;
    res.cost_paid = cost_log;
    res.avg_holding_bars = n_opens ? (double)total_holding / (double)n_opens : 0.0;

    // ---- summary ----
    auto sharpe = [&](const std::vector<double>& nav, double& mean_out) {
        double m = 0, m2 = 0; long c = 0;
        for (size_t t = 1; t < T; ++t) {
            double r = nav[t] - nav[t - 1];
            c++; double d = r - m; m += d / c; double d2 = r - m; m2 += d * d2;
        }
        double sd = c > 1 ? std::sqrt(m2 / (c - 1)) : 0.0;
        mean_out = m;
        // annualize daily Sharpe by sqrt(252)
        return sd > 0 ? (m / sd) * std::sqrt(252.0) : 0.0;
    };
    double mn = 0, mg = 0;
    res.sharpe_net = sharpe(res.lognav_net, mn);
    res.sharpe_gross = sharpe(res.lognav_gross, mg);
    if (T > 1) {
        double years = (double)(T - 1) / 252.0;
        res.ann_ret_net = years > 0 ? res.lognav_net[T - 1] / years : 0.0;
        res.ann_ret_gross = years > 0 ? res.lognav_gross[T - 1] / years : 0.0;
    }
    double peak = res.lognav_net[0], mdd = 0.0;
    for (size_t t = 1; t < T; ++t) {
        double nav = res.lognav_net[t];
        peak = std::max(peak, nav);
        mdd = std::max(mdd, peak - nav);
    }
    res.maxdd_net = mdd;
    return res;
}

}  // namespace statarb
#endif  // STATARB_SPREAD_BACKTEST_H
