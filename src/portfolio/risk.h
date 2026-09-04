// risk.h — dependency-free portfolio-level risk controls (Part 7).
//
// Layered on top of the flat PortfolioBook (src/portfolio/portfolio_book.h) when the
// book spans multiple concurrent basket trades. Provides the two primitive controls
// that the higher-level (Python) portfolio run exercises and verifies per bar:
//
//   * inverse-vol allocation weights   w_i = (1/sigma_i) / sum_j (1/sigma_j)
//     sigma_i = trailing volatility of strategy i's daily P&L. Lower-vol strategies
//     get more weight (a light risk-parity flavour); weights are normalized to sum 1.
//
//   * gross / net exposure limits. A *book* of market-neutral spread strategies has
//     near-zero net dollar exposure by construction; we still verify it. weight-based
//     limits are expressed as a cap on the L1 norm of the (positive) weight vector
//     (gross) and on the absolute L1 norm of signed exposure (net), and as an
//     optional per-name concentration cap.
//
// These are pure, allocation-free helpers so the hot path stays predictable and the
// enforcement can be unit-tested independently of any market data.
#ifndef STATARB_RISK_H
#define STATARB_RISK_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace statarb::risk {

// Inverse-vol weights from per-strategy daily-P&L volatilities. vol<=0 (undefined /
// zero-history) entries get zero weight. Returns weights summing to 1 (empty if none).
inline std::vector<double> inverse_vol_weights(const std::vector<double>& sigma) {
    std::vector<double> inv(sigma.size(), 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < sigma.size(); ++i) {
        if (sigma[i] > 0.0) {
            inv[i] = 1.0 / sigma[i];
            sum += inv[i];
        }
    }
    if (sum <= 0.0) return std::vector<double>(sigma.size(), 0.0);
    for (double& w : inv) w /= sum;
    return inv;
}

// Weighted gross (L1) and net (absolute of sum) exposure of a signed weight vector.
inline double gross_exposure(const std::vector<double>& w) {
    double g = 0.0;
    for (double v : w) g += std::fabs(v);
    return g;
}
inline double net_exposure(const std::vector<double>& w) {
    double n = 0.0;
    for (double v : w) n += v;
    return std::fabs(n);
}

// Verify a signed weight vector against a gross cap (>= sum|w|), a net cap
// (>= |sum w|) and an optional per-name concentration cap. Returns true if all
// constraints hold; also reports the achieved gross/net via out-params.
inline bool within_limits(const std::vector<double>& w, double gross_cap,
                          double net_cap, double conc_cap, double& gross_out,
                          double& net_out) {
    gross_out = gross_exposure(w);
    net_out = net_exposure(w);
    if (gross_out > gross_cap + 1e-12) return false;
    if (net_out > net_cap + 1e-12) return false;
    if (conc_cap > 0.0) {
        for (double v : w)
            if (std::fabs(v) > conc_cap + 1e-12) return false;
    }
    return true;
}

}  // namespace statarb::risk
#endif  // STATARB_RISK_H
