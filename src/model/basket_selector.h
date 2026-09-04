// basket_selector.h — basket construction / asset-selection interface.
//
// Full OLS/Ridge/Lasso/ElasticNet/PCA fitting is implemented in Part 5 (and the
// regression engine is shared with RollingRegression). Here we fix the stable
// public API and provide the OLS closed-form path so the pipeline can be wired
// end-to-end now. Selection returns the chosen candidate indices and fitted
// weights as contiguous vectors (cache-friendly throughout).
#ifndef STATARB_BASKET_SELECTOR_H
#define STATARB_BASKET_SELECTOR_H

#include <cstddef>
#include <vector>

#include "dense_la.h"

namespace statarb {

enum class SelectionMethod { OLS, Ridge, Lasso, ElasticNet, PCA };

struct BasketResult {
    std::vector<size_t> selected;  // candidate indices (subset of candidates)
    std::vector<double> weights;   // fitted beta per selected index (incl. no intercept)
    bool ok = false;
};

class BasketSelector {
public:
    explicit BasketSelector(SelectionMethod method) : method_(method) {}

    // X: (n_obs x n_feat) row-major features (baskets, already aligned), y: returns.
    // top_k: for PCA the number of components to retain. weights returned map back
    // to the ORIGINAL feature space for OLS/Ridge; for PCA they are the loadings in
    // the reduced space (Part 5 maps them back to per-asset weights).
    BasketResult select(const std::vector<double>& X, size_t n_obs, size_t n_feat,
                        const std::vector<double>& y, double lambda = 0.0,
                        size_t top_k = 3) const {
        BasketResult r;
        r.selected.resize(n_feat);
        for (size_t i = 0; i < n_feat; ++i) r.selected[i] = i;
        r.weights.assign(n_feat, 0.0);
        switch (method_) {
            case SelectionMethod::OLS:
            case SelectionMethod::Ridge:
                _ols_or_ridge(X, n_obs, n_feat, y, lambda, r.weights);
                break;
            case SelectionMethod::Lasso:
            case SelectionMethod::ElasticNet:
            case SelectionMethod::PCA:
                // Stub — full implementation lands in Part 5 (coordinate descent,
                // PCA via la::symmetric_eig). For now returns OLS as a placeholder so
                // the API/compilation is complete; clearly marked TODO.
                _ols_or_ridge(X, n_obs, n_feat, y, 0.0, r.weights);
                r.ok = false;
                return r;
        }
        r.ok = true;
        return r;
    }

private:
    // Fit y = X beta (no intercept; caller centers) by solving (X^T X + lambda I)
    // beta = X^T y.
    static void _ols_or_ridge(const std::vector<double>& X, size_t n_obs, size_t n_feat,
                              const std::vector<double>& y, double lambda,
                              std::vector<double>& beta) {
        la::Mat A(n_feat, n_feat, 0.0);
        std::vector<double> rhs(n_feat, 0.0);
        for (size_t i = 0; i < n_obs; ++i) {
            for (size_t a = 0; a < n_feat; ++a) {
                double xa = X[i * n_feat + a];
                rhs[a] += xa * y[i];
                for (size_t b = 0; b < n_feat; ++b) A(a, b) += xa * X[i * n_feat + b];
            }
        }
        la::solve_spd_ridge(A, rhs, beta, lambda);
    }

    SelectionMethod method_;
};

}  // namespace statarb
#endif  // STATARB_BASKET_SELECTOR_H
