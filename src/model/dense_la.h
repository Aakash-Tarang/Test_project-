// dense_la.h — minimal, dependency-free, cache-friendly dense linear algebra.
//
// Environment note: Eigen is not installable in this sandbox (apt/conda blocked;
// only pypi and GitHub are reachable). Per the project rule that the C++ engine
// be dependency-free at build time, we provide a small, contiguous, row-major
// dense layer sufficient for the regressions in this report: a symmetric positive
// definite (SPD) solver via Cholesky (with an optional ridge on the diagonal) and
// a symmetric eigen-decomposition via cyclic Jacobi (used for PCA in Part 5).
//
// Contiguity: matrices are stored row-major in a single std::vector<double>, so
// cache behavior is good and operations vectorize. The matrix sizes here are
// tiny (basket size n_features ~ 4-15), so these O(n^3) routines are far cheaper
// than the O(T n^2) rolling-sum updates they accompany.
#ifndef STATARB_DENSE_LA_H
#define STATARB_DENSE_LA_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace statarb::la {

// Dense row-major matrix.
struct Mat {
    size_t rows = 0, cols = 0;
    std::vector<double> d;  // rows*cols contiguous

    Mat() = default;
    Mat(size_t r, size_t c, double fill = 0.0) : rows(r), cols(c), d(r * c, fill) {}
    double& operator()(size_t i, size_t j) { return d[i * cols + j]; }
    double operator()(size_t i, size_t j) const { return d[i * cols + j]; }
};

// Solve the SPD system (A + lambda*I) x = b by Cholesky factorization.
// lambda >= 0 adds ridge (also makes A invertible if only PSD). Returns false if
// factorization fails (not SPD), in which case x is left unchanged.
inline bool solve_spd_ridge(const Mat& A, const std::vector<double>& b,
                            std::vector<double>& x, double lambda = 0.0) {
    const size_t n = A.rows;
    if (A.cols != n || b.size() != n) return false;
    // Copy A + lambda*I into lower-triangular work matrix L (store lower).
    Mat L(n, n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            L(i, j) = A(i, j) + (i == j ? lambda : 0.0);

    // Cholesky: A = L L^T (lower).
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = L(i, j);
            for (size_t k = 0; k < j; ++k) sum -= L(i, k) * L(j, k);
            if (i == j) {
                if (sum <= 0.0) return false;  // not SPD
                L(i, j) = std::sqrt(sum);
            } else {
                L(i, j) = sum / L(j, j);
            }
        }
    }
    // Solve L y = b, then L^T x = y.
    std::vector<double> y(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double s = b[i];
        for (size_t k = 0; k < i; ++k) s -= L(i, k) * y[k];
        y[i] = s / L(i, i);
    }
    x.assign(n, 0.0);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        double s = y[static_cast<size_t>(i)];
        for (size_t k = static_cast<size_t>(i) + 1; k < n; ++k)
            s -= L(k, static_cast<size_t>(i)) * x[k];
        x[static_cast<size_t>(i)] = s / L(static_cast<size_t>(i), static_cast<size_t>(i));
    }
    return true;
}

// Symmetric eigen-decomposition A = Q diag(w) Q^T via cyclic Jacobi rotations.
// A is symmetric, n x n. Returns eigenvalues in descending order.
inline bool symmetric_eig(const Mat& A, std::vector<double>& w, Mat& Q) {
    const size_t n = A.rows;
    if (A.cols != n) return false;
    Mat a = A;
    Q = Mat(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) Q(i, i) = 1.0;
    const int max_sweeps = 50;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double off = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j) off += a(i, j) * a(i, j);
        if (off < 1e-24) break;
        for (size_t p = 0; p < n; ++p) {
            for (size_t q = p + 1; q < n; ++q) {
                if (std::abs(a(p, q)) < 1e-300) continue;
                double app = a(p, p), aqq = a(q, q), apq = a(p, q);
                double theta = (aqq - app) / (2.0 * apq);
                double t = (theta >= 0 ? 1.0 : -1.0) /
                           (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                double c = 1.0 / std::sqrt(t * t + 1.0);
                double s = t * c;
                for (size_t k = 0; k < n; ++k) {
                    double akp = a(k, p), akq = a(k, q);
                    a(k, p) = c * akp - s * akq;
                    a(k, q) = s * akp + c * akq;
                }
                for (size_t k = 0; k < n; ++k) {
                    double apk = a(p, k), aqk = a(q, k);
                    a(p, k) = c * apk - s * aqk;
                    a(q, k) = s * apk + c * aqk;
                }
                for (size_t k = 0; k < n; ++k) {
                    double qkp = Q(k, p), qkq = Q(k, q);
                    Q(k, p) = c * qkp - s * qkq;
                    Q(k, q) = s * qkp + c * qkq;
                }
            }
        }
    }
    w.resize(n);
    for (size_t i = 0; i < n; ++i) w[i] = a(i, i);
    // Order descending via a permutation on eigenvectors too.
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t i, size_t j) { return w[i] > w[j]; });
    std::vector<double> ww(n);
    Mat qq(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        ww[i] = w[idx[i]];
        for (size_t k = 0; k < n; ++k) qq(k, i) = Q(k, idx[i]);
    }
    w.swap(ww);
    Q = qq;
    return true;
}

}  // namespace statarb::la
#endif  // STATARB_DENSE_LA_H
