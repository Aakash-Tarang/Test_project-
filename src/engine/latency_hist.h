// latency_hist.h — fixed-capacity, allocation-free histogram of timing samples.
//
// Stores sample counts in exponentially-spaced buckets over a flat array and
// reports the p50/p95/p99 percentiles and the mean. Used by BacktestEngine to
// report per-stage latency distributions (spec Section 5.1 requires p50/p95/p99,
// not just the mean).
#ifndef STATARB_LATENCY_HIST_H
#define STATARB_LATENCY_HIST_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace statarb {

class LatencyHist {
public:
    LatencyHist() = default;
    explicit LatencyHist(double lo, double hi, int n_buckets)
        : lo_(lo), hi_(hi), nb_(n_buckets), counts_(static_cast<size_t>(n_buckets), 0) {}

    void setRange(double lo, double hi, int n_buckets) {
        lo_ = lo; hi_ = hi; nb_ = n_buckets;
        counts_.assign(static_cast<size_t>(n_buckets), 0);
    }

    void add(double nanos) {
        if (counts_.empty()) return;
        ++n_;
        sum_ += nanos;
        if (nanos <= lo_) { counts_[0]++; return; }
        if (nanos >= hi_) { counts_[nb_ - 1]++; return; }
        double logr = (std::log(nanos / lo_) / std::log(hi_ / lo_));
        int b = static_cast<int>(logr * (nb_ - 1));
        if (b < 0) b = 0;
        if (b >= nb_) b = nb_ - 1;
        counts_[static_cast<size_t>(b)]++;
    }

    size_t n() const noexcept { return n_; }

    // Quantile in the original units. q in [0,1].
    double quantile(double q) const {
        if (n_ == 0) return 0.0;
        double target = q * static_cast<double>(n_);
        double cum = 0.0;
        for (int i = 0; i < nb_; ++i) {
            cum += counts_[static_cast<size_t>(i)];
            if (cum >= target) {
                double fl = lo_ * std::pow(hi_ / lo_, static_cast<double>(i) / (nb_ - 1));
                return fl;
            }
        }
        return hi_;
    }
    double p50() const { return quantile(0.50); }
    double p95() const { return quantile(0.95); }
    double p99() const { return quantile(0.99); }
    double mean() const { return n_ == 0 ? 0.0 : sum_ / static_cast<double>(n_); }

private:
    double lo_ = 1.0, hi_ = 1e6;
    int nb_ = 2;
    std::vector<size_t> counts_;
    size_t n_ = 0;
    double sum_ = 0.0;
};

}  // namespace statarb
#endif  // STATARB_LATENCY_HIST_H
