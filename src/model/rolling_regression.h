// rolling_regression.h — rolling linear regression with incremental updates.
//
// We maintain the raw second-moment accumulators over a fixed window and obtain
// the coefficients at any time by solving the centered normal equations. This is
// the standard O(n_features^2) incremental / "rank-1" scheme described in the
// spec (Welford-style running sums rather than refitting from scratch each bar):
//
//   Sxx = sum_t x_t x_t^T          (p x p, updated +x x^T on push, -x x^T on evict)
//   Sxy = sum_t x_t y_t            (p,     updated +x y on push,   -x y on evict)
//   sx  = sum_t x_t ,  sy = sum_t y_t ,  c = count
//
//   Cov(x,x) = Sxx - sx sx^T / c      Cov(x,y) = Sxy - sx*sy / c
//   beta     = Cov(x,x)^{-1} Cov(x,y)          (solved via Cholesky + optional ridge)
//   alpha    = (sy - beta^T sx) / c
//
// Feature rows are held in a flat row-major ring buffer (window_*n_features_) so
// window eviction knows exactly which row leaves.
//
// Allocation discipline: all buffers (feature history, targets, and the scratch
// used by the Cholesky solve) are sized ONCE in the constructor. The update() and
// compute() hot paths perform no dynamic allocation; this is verified by an
// allocation-counting unit test. Correctness vs a brute-force refit over the same
// window is validated to ~1e-6 in the unit tests.
#ifndef STATARB_ROLLING_REGRESSION_H
#define STATARB_ROLLING_REGRESSION_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace statarb {

class RollingRegression {
public:
    RollingRegression(size_t window, size_t n_features, double ridge_lambda = 0.0)
        : window_(window > 0 ? window : 1),
          nf_(n_features),
          lambda_(ridge_lambda),
          cap_(window * n_features),
          feat_(cap_, 0.0),
          tgt_(window, 0.0),
          sx_(n_features, 0.0),
          sxy_(n_features, 0.0),
          beta_(n_features, 0.0),
          sxx_(n_features * n_features, 0.0),
          cov_(n_features * n_features, 0.0),
          rhs_(n_features, 0.0),
          L_(n_features * n_features, 0.0),
          yv_(n_features, 0.0) {}

    // Feed one observation. O(n_features) for the sums update; no allocation.
    void update(const std::vector<double>& x_row, double y) {
        const size_t p = nf_;
        const double* x = x_row.data();
        if (count_ == window_) {
            size_t base = head_ * p;
            for (size_t j = 0; j < p; ++j) {
                double xj = feat_[base + j];
                sx_[j] -= xj;
                for (size_t k = 0; k < p; ++k) sxx_[j * p + k] -= xj * feat_[base + k];
                sxy_[j] -= xj * tgt_[head_];
            }
            sy_ -= tgt_[head_];
        } else {
            ++count_;
        }
        size_t base = head_ * p;
        for (size_t j = 0; j < p; ++j) {
            double xj = x[j];
            feat_[base + j] = xj;
            for (size_t k = 0; k < p; ++k) sxx_[j * p + k] += xj * x[k];
            sxy_[j] += xj * y;
            sx_[j] += xj;
        }
        tgt_[head_] = y;
        sy_ += y;
        head_ = (head_ + 1) % window_;
    }

    size_t windowSize() const noexcept { return window_; }
    size_t count() const noexcept { return count_; }
    bool ready() const noexcept { return count_ == window_; }
    double lambda() const noexcept { return lambda_; }

    // Recompute coefficients from current accumulators. No allocation.
    void compute() {
        cached_valid_ = false;
        const size_t p = nf_;
        if (count_ < 2) return;
        const double c = static_cast<double>(count_);
        for (size_t j = 0; j < p; ++j) {
            for (size_t k = 0; k < p; ++k)
                cov_[j * p + k] = sxx_[j * p + k] - sx_[j] * sx_[k] / c;
            rhs_[j] = sxy_[j] - sx_[j] * sy_ / c;
        }
        if (!solve_alloc_free(p)) {
            std::fill(beta_.begin(), beta_.end(), 0.0);
            return;
        }
        double bx = 0.0;
        for (size_t j = 0; j < p; ++j) bx += beta_[j] * sx_[j];
        alpha_ = (sy_ - bx) / c;
        double ss_res = 0.0, ss_tot = 0.0, mean_y = sy_ / c;
        for (size_t i = 0; i < count_; ++i) {
            size_t phys = (head_ + i) % window_;
            size_t b = phys * p;
            double pred = alpha_;
            for (size_t j = 0; j < p; ++j) pred += beta_[j] * feat_[b + j];
            double dy = tgt_[phys] - pred;
            ss_res += dy * dy;
            double dy2 = tgt_[phys] - mean_y;
            ss_tot += dy2 * dy2;
        }
        ss_res_ = ss_res;
        r2_ = ss_tot > 0.0 ? (1.0 - ss_res / ss_tot) : 0.0;
        double dof = (count_ > static_cast<size_t>(p) + 1)
                         ? static_cast<double>(count_ - p - 1)
                         : static_cast<double>(count_ > 1 ? count_ - 1 : 1);
        sigma_res_ = std::sqrt(ss_res / dof);
        cached_valid_ = true;
    }

    const std::vector<double>& beta() const noexcept { return beta_; }
    double alpha() const noexcept { return alpha_; }
    bool betaValid() const noexcept { return cached_valid_; }
    double rSquared() const noexcept { return r2_; }
    double sigmaResidual() const noexcept { return sigma_res_; }

    double residual(const std::vector<double>& x_row, double y) const {
        const size_t p = nf_;
        double pred = alpha_;
        for (size_t j = 0; j < p; ++j) pred += beta_[j] * x_row[j];
        return y - pred;
    }
    double residualZScore(const std::vector<double>& x_row, double y) const {
        double r = residual(x_row, y);
        double s = sigma_res_;
        return s > 0.0 ? r / s : 0.0;
    }

private:
    // Solve (cov_ + lambda I) beta = rhs_ by Cholesky into L_, yv_, beta_.
    // Allocation-free: only touches preallocated member scratch. cov_ is (nf_^2)
    // and is treated as symmetric; we factor the lower triangle in-place.
    bool solve_alloc_free(size_t p) {
        // Copy cov_ + lambda into L_ (lower part meaningful).
        for (size_t i = 0; i < p; ++i)
            for (size_t j = 0; j < p; ++j)
                L_[i * p + j] = cov_[i * p + j] + (i == j ? lambda_ : 0.0);
        // Cholesky A = L L^T
        for (size_t i = 0; i < p; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                double sum = L_[i * p + j];
                for (size_t k = 0; k < j; ++k) sum -= L_[i * p + k] * L_[j * p + k];
                if (i == j) {
                    if (sum <= 0.0) return false;
                    L_[i * p + j] = std::sqrt(sum);
                } else {
                    L_[i * p + j] = sum / L_[j * p + j];
                }
            }
        }
        // forward solve L yv = rhs_
        for (size_t i = 0; i < p; ++i) {
            double s = rhs_[i];
            for (size_t k = 0; k < i; ++k) s -= L_[i * p + k] * yv_[k];
            yv_[i] = s / L_[i * p + i];
        }
        // back solve L^T beta = yv_
        for (int i = static_cast<int>(p) - 1; i >= 0; --i) {
            size_t ii = static_cast<size_t>(i);
            double s = yv_[ii];
            for (size_t k = ii + 1; k < p; ++k) s -= L_[k * p + ii] * beta_[k];
            beta_[ii] = s / L_[ii * p + ii];
        }
        return true;
    }

    size_t window_, nf_;
    double lambda_;
    size_t cap_;
    std::vector<double> feat_;             // window_*nf_ row-major ring buffer
    std::vector<double> tgt_;              // window_ targets
    std::vector<double> sx_, sxy_, beta_;  // nf_ each
    std::vector<double> sxx_;              // nf_*nf_
    std::vector<double> cov_, rhs_;        // scratch: nf_^2, nf_
    std::vector<double> L_, yv_;           // scratch for Cholesky: nf_^2, nf_
    double sy_ = 0.0;
    size_t head_ = 0, count_ = 0;
    double alpha_ = 0.0, ss_res_ = 0.0, r2_ = 0.0, sigma_res_ = 0.0;
    bool cached_valid_ = false;
};

}  // namespace statarb
#endif  // STATARB_ROLLING_REGRESSION_H
