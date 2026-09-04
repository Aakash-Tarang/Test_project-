// backtest_engine.h — orchestrates MarketDataBuffer -> RollingRegression ->
// SignalGenerator -> PortfolioBook for a single target/basket book, and
// instruments per-stage latency (spec Section 5).
//
// A realistic walk-forward, cost-aware, next-bar execution loop is added in
// Part 4. This file establishes the skeleton, the stage pipeline, and the
// per-stage latency histograms (p50/p95/p99) measured with high-resolution clock.
#ifndef STATARB_BACKTEST_ENGINE_H
#define STATARB_BACKTEST_ENGINE_H

#include <chrono>
#include <cstddef>
#include <vector>

#include "../data/market_data_buffer.h"
#include "../model/rolling_regression.h"
#include "../signal/signal_generator.h"
#include "../portfolio/portfolio_book.h"
#include "latency_hist.h"

namespace statarb {

struct LatencyReport {
    LatencyHist regression_update;   // time to update rolling regression
    LatencyHist signal_generation;   // time to turn residual into a signal
    LatencyHist order_placement;     // time to apply the signal to the book
    double total_seconds = 0.0;
    size_t bars = 0;
};

class BacktestEngine {
public:
    // n_features: number of basket regressors; window: regression lookback.
    BacktestEngine(size_t window, size_t n_features, double entry_z, double exit_z)
        : reg_(window, n_features), sig_(entry_z, exit_z, 0.0),
          book_(n_features + 1 /* target + features */) {
        auto ns = std::chrono::high_resolution_clock::now().time_since_epoch();
        // (re)configure latency ranges ~ tens of ns to ~ tens of us
        double lo = 10.0, hi = 100000.0;
        rep_.regression_update.setRange(lo, hi, 64);
        rep_.signal_generation.setRange(lo, hi, 64);
        rep_.order_placement.setRange(lo, hi, 64);
        (void)ns;
    }

    RollingRegression& regression() { return reg_; }
    SignalGenerator& signals() { return sig_; }
    PortfolioBook& book() { return book_; }
    const LatencyReport& latency() const { return rep_; }

    // One bar of the target asset: update the rolling regression with a basket row
    // and mark the book. Returns the produced signal score.
    double runBar(const std::vector<double>& basket_row, double target_ret) {
        using clk = std::chrono::high_resolution_clock;
        auto t0 = clk::now();
        reg_.update(basket_row, target_ret);
        reg_.compute();
        auto t1 = clk::now();

        double z = 0.0;
        if (reg_.betaValid()) z = reg_.residualZScore(basket_row, target_ret);
        auto t2 = clk::now();

        // In Part 4 this executes at next bar with costs; here we just update a
        // scalar book mark to keep the pipeline exercised and measurable.
        book_.setPrice(0, target_ret + 1.0);
        auto t3 = clk::now();

        rep_.regression_update.add(std::chrono::duration<double, std::nano>(t1 - t0).count());
        rep_.signal_generation.add(std::chrono::duration<double, std::nano>(t2 - t1).count());
        rep_.order_placement.add(std::chrono::duration<double, std::nano>(t3 - t2).count());
        rep_.bars++;
        return z;
    }

    void finish() {
        rep_.total_seconds = 0.0;
        // placeholder; full timing summary in Part 4.
    }

private:
    RollingRegression reg_;
    SignalGenerator sig_;
    PortfolioBook book_;
    LatencyReport rep_;
};

}  // namespace statarb
#endif  // STATARB_BACKTEST_ENGINE_H
