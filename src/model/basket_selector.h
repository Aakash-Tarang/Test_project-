// basket_selector.h — basket construction / asset-selection interface.
//
// Implements the five baseline linear estimators compared in Part 5:
//   OLS, Ridge, Lasso, ElasticNet, Principal-Component Regression (PCR, "PCA").
// Each estimates y ~ X (a target vs. a basket of candidates) and returns the
// fitted per-feature slope weights, so a consumer can form the tradeable spread
// residual e_t = y_t - (intercept + sum_i beta_i x_it).
//
// Conventions (all methods agree here):
//   * A free intercept is fit by centering every feature column and y internally.
//     BasketResult::intercept holds alpha, weights the slopes in the ORIGINAL
//     feature scale (so pred = intercept + sum_i weights_i x_i reproduces the fit).
//   * OLS / Ridge: solve the centered normal equations (X^T X + lambda I) beta =
//     X^T y with lambda >= 0 (Ridge). Closed form via Cholesky (dense_la.h).
//   * Lasso / ElasticNet: coordinate descent on unit-norm (column-standardized)
//     centered design; objective
//         min 1/2 ||y_c - U b||^2 + alpha * sum_j [ rho |b_j| + (1-rho)/2 b_j^2 ]
//     with rho = l1_ratio (1 => Lasso, in (0,1) => ElasticNet). Coefficients are
//     mapped back to the original feature scale before returning.
//   * PCA (PCR): eigendecompose the centered feature covariance (cyclic Jacobi,
//     dense_la.h), keep the top_k principal components, regress y on them, and
//     return the effective per-feature slope in the original space.
//
// The reporting/hyper-parameter sweep over real data is done in Python/sklearn
// (scripts/model/ + scripts/analysis/compare_models.py); this header is the
// dependency-free in-engine implementation, exercised by unit tests that assert
// Ridge->OLS as lambda->0, Lasso sparsification, EN between, and PCR(k=n)=OLS.
#ifndef STATARB_BASKET_SELECTOR_H
#define STATARB_BASKET_SELECTOR_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "dense_la.h"

namespace statarb {

enum class SelectionMethod { OLS, Ridge, Lasso, ElasticNet, PCA };

// Hyper-parameters for the penalized methods. Unused entries are ignored by a
// given method (OLS ignores all; PCA uses top_k only; Lasso uses alpha with an
// implicit l1_ratio of 1).
struct BasketParams {
    double alpha = 0.0;      // Ridge penalty (Ridge) or L1/L2 strength (Lasso/EN)
    double l1_ratio = 1.0;   // elastic-net mixing in [0,1]; 1 = Lasso, 0 = Ridge
    size_t top_k = 3;        // number of components for PCR
    double tol = 1e-8;
    size_t max_iter = 200000;
};

struct BasketResult {
    std::vector<double> weights;  // fitted per-feature slope (original scale)
    double intercept = 0.0;       // fitted intercept (centering constant)
    size_t nonzero = 0;           // number of |weight_j| > 0 (sparsity report)
    bool ok = false;
};

class BasketSelector {
public:
    explicit BasketSelector(SelectionMethod method) : method_(method) {}

    // X: (n_obs x n_feat) row-major features, y: (n_obs) target. Regression with
    // intercept. weights return in the original feature space (zero for dropped
    // / shrunk features).
    BasketResult select(const std::vector<double>& X, size_t n_obs, size_t n_feat,
                        const std::vector<double>& y,
                        const BasketParams& params = {}) const {
        BasketResult r;
        if (n_obs == 0 || n_feat == 0 || X.size() != n_obs * n_feat || y.size() != n_obs) return r;
        switch (method_) {
            case SelectionMethod::OLS:
                fit_ols(X, n_obs, n_feat, y, 0.0, r);
                break;
            case SelectionMethod::Ridge:
                fit_ols(X, n_obs, n_feat, y, params.alpha, r);
                break;
            case SelectionMethod::Lasso:
                fit_en(X, n_obs, n_feat, y, params.alpha, 1.0,
                       params.tol, params.max_iter, r);
                break;
            case SelectionMethod::ElasticNet: {
                double rho = std::max(0.0, std::min(1.0, params.l1_ratio));
                fit_en(X, n_obs, n_feat, y, params.alpha, rho,
                       params.tol, params.max_iter, r);
                break;
            }
            case SelectionMethod::PCA:
                fit_pcr(X, n_obs, n_feat, y, params.top_k, r);
                break;
        }
        size_t nz = 0;
        for (double w : r.weights)
            if (std::fabs(w) > 1e-14) ++nz;
        r.nonzero = nz;
        r.ok = r.weights.size() == n_feat;
        return r;
    }

private:
    // Column means / centered design helpers. Returns per-column standard-deviation
    // scale s_j (0 for a constant column) and centered feature matrix z (n_obs*n).
    static void center(const std::vector<double>& X, size_t n, size_t p,
                       const std::vector<double>& y, double& my,
                       std::vector<double>& mx, std::vector<double>& yc,
                       std::vector<double>& z, std::vector<double>& s) {
        mx.assign(p, 0.0);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < p; ++j) mx[j] += X[i * p + j];
        for (size_t j = 0; j < p; ++j) mx[j] /= (double)n;
        my = 0.0;
        for (size_t i = 0; i < n; ++i) my += y[i];
        my /= (double)n;
        z.resize(n * p);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < p; ++j) z[i * p + j] = X[i * p + j] - mx[j];
        yc.resize(n);
        for (size_t i = 0; i < n; ++i) yc[i] = y[i] - my;
        s.assign(p, 0.0);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < p; ++j) s[j] += z[i * p + j] * z[i * p + j];
        for (size_t j = 0; j < p; ++j) s[j] = std::sqrt(s[j]);  // L2 norm of column
    }

    // Set intercept from beta given original column means.
    static void finish(BasketResult& r, const std::vector<double>& mx, double my,
                       const std::vector<double>& beta, size_t p) {
        double inter = my;
        for (size_t j = 0; j < p; ++j) inter -= beta[j] * mx[j];
        r.weights = beta;
        r.intercept = inter;
    }

    // OLS / Ridge on centered normal equations (closed form).
    static void fit_ols(const std::vector<double>& X, size_t n, size_t p,
                        const std::vector<double>& y, double lambda, BasketResult& r) {
        double my; std::vector<double> mx, yc, z, s;
        center(X, n, p, y, my, mx, yc, z, s);
        la::Mat G(p, p, 0.0);
        std::vector<double> c(p, 0.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t a = 0; a < p; ++a) {
                double za = z[i * p + a];
                c[a] += za * yc[i];
                for (size_t b = 0; b < p; ++b) G(a, b) += za * z[i * p + b];
            }
        }
        std::vector<double> beta(p, 0.0);
        bool ok = la::solve_spd_ridge(G, c, beta, lambda);
        if (!ok) { r.ok = false; return; }
        finish(r, mx, my, beta, p);
    }

    // Lasso / ElasticNet via coordinate descent on the centered design with
    // unit-norm columns. rho in [0,1], alpha >= 0.
    static void fit_en(const std::vector<double>& X, size_t n, size_t p,
                       const std::vector<double>& y, double alpha, double rho,
                       double tol, size_t max_iter, BasketResult& r) {
        double my; std::vector<double> mx, yc, z, s;
        center(X, n, p, y, my, mx, yc, z, s);
        // column-normalized design u_ij = z_ij / s_j
        std::vector<double> u(n * p, 0.0);
        std::vector<double> G(p * p, 0.0), c(p, 0.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t a = 0; a < p; ++a) {
                double sj = s[a];
                double ua = sj > 0.0 ? z[i * p + a] / sj : 0.0;
                u[i * p + a] = ua;
                c[a] += ua * yc[i];
            }
        }
        for (size_t a = 0; a < p; ++a)
            for (size_t b = 0; b < p; ++b) {
                double sum = 0.0;
                for (size_t i = 0; i < n; ++i) sum += u[i * p + a] * u[i * p + b];
                G[a * p + b] = sum;  // G_aa = 1 (unit norm)
            }
        std::vector<double> b(p, 0.0);
        const double denom = 1.0 + alpha * (1.0 - rho);
        if (alpha <= 0.0) {
            // no penalty => ridge-like OLS on normalized design; solve (G) b = c
            la::solve_spd_ridge(la_mat(p, G), c, b, 0.0);
        } else {
            for (size_t it = 0; it < max_iter; ++it) {
                double max_delta = 0.0;
                for (size_t j = 0; j < p; ++j) {
                    double acc = 0.0;
                    for (size_t k = 0; k < p; ++k) acc += G[j * p + k] * b[k];
                    double pj = c[j] - acc + b[j];  // partial residual (G_jj=1)
                    double bnew = 0.0;
                    double thr = alpha * rho;
                    if (pj > thr) bnew = (pj - thr) / denom;
                    else if (pj < -thr) bnew = (pj + thr) / denom;
                    double delta = std::fabs(bnew - b[j]);
                    if (delta > max_delta) max_delta = delta;
                    b[j] = bnew;
                }
                if (max_delta < tol) break;
            }
        }
        // map back to original feature scale: beta_j = b_j / s_j
        std::vector<double> beta(p, 0.0);
        for (size_t j = 0; j < p; ++j) beta[j] = s[j] > 0.0 ? b[j] / s[j] : 0.0;
        finish(r, mx, my, beta, p);
    }

    // Principal component regression: regress y on top_k PCs of the centered X.
    static void fit_pcr(const std::vector<double>& X, size_t n, size_t p,
                        const std::vector<double>& y, size_t top_k, BasketResult& r) {
        double my; std::vector<double> mx, yc, z, s;
        center(X, n, p, y, my, mx, yc, z, s);
        size_t k = std::min(top_k, p);
        if (k == 0) k = p;
        // covariance S = Z^T Z (centered)
        la::Mat S(p, p, 0.0);
        for (size_t a = 0; a < p; ++a)
            for (size_t b = 0; b < p; ++b) {
                double sum = 0.0;
                for (size_t i = 0; i < n; ++i) sum += z[i * p + a] * z[i * p + b];
                S(a, b) = sum;
            }
        std::vector<double> w; la::Mat V;
        if (!la::symmetric_eig(S, w, V)) { r.ok = false; return; }
        // score regression in component space: beta_pc_j = (v_j . (Z^T y_c)) / w_j
        //   (components orthonormal, diag = eigenvalues w_j)
        std::vector<double> bpc(p, 0.0), zty(p, 0.0);
        for (size_t a = 0; a < p; ++a) {
            double acc = 0.0;
            for (size_t i = 0; i < n; ++i) acc += z[i * p + a] * yc[i];
            zty[a] = acc;
        }
        for (size_t j = 0; j < p; ++j) {
            double acc = 0.0;
            for (size_t a = 0; a < p; ++a) acc += V(a, j) * zty[a];
            if (w[j] > 1e-12) bpc[j] = acc / w[j];
        }
        // effective per-feature slope = V_{:k} * beta_pc_{:k}
        std::vector<double> beta(p, 0.0);
        for (size_t a = 0; a < p; ++a) {
            double acc = 0.0;
            for (size_t j = 0; j < k; ++j) acc += V(a, j) * bpc[j];
            beta[a] = acc;
        }
        finish(r, mx, my, beta, p);
    }

    static la::Mat la_mat(size_t p, const std::vector<double>& v) {
        la::Mat M(p, p);
        for (size_t i = 0; i < p * p; ++i) M.d[i] = v[i];
        return M;
    }

    SelectionMethod method_;
};

}  // namespace statarb
#endif  // STATARB_BASKET_SELECTOR_H
