// engine_tests.cpp — unit tests for the C++ engine classes.
// Run via: build/statarbsim --test   (harness in test/minimal_test.hpp)
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <random>
#include <vector>

#include "minimal_test.hpp"
#include "../src/data/market_data_buffer.h"
#include "../src/model/dense_la.h"
#include "../src/model/rolling_regression.h"
#include "../src/model/basket_selector.h"
#include "../src/signal/signal_generator.h"
#include "../src/portfolio/portfolio_book.h"
#include "../src/engine/latency_hist.h"
#include "../src/engine/backtest_engine.h"
#include "../src/engine/spread_backtest.h"

using namespace statarb;

// ---------------------------------------------------------------------------
// Brute-force reference OLS for comparison.
// ---------------------------------------------------------------------------
static void brute_force_ols(const std::vector<double>& X, size_t n, size_t p,
                            const std::vector<double>& y, std::vector<double>& beta,
                            double& alpha) {
    la::Mat A(p, p, 0.0);
    std::vector<double> rhs(p, 0.0), sx(p, 0.0);
    double sy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j) sx[j] += X[i * p + j];
        sy += y[i];
    }
    double c = (double)n;
    for (size_t i = 0; i < n; ++i) {
        for (size_t a = 0; a < p; ++a)
            for (size_t b = 0; b < p; ++b) A(a, b) += X[i * p + a] * X[i * p + b];
    }
    for (size_t a = 0; a < p; ++a) {
        for (size_t b = 0; b < p; ++b) A(a, b) -= sx[a] * sx[b] / c;
        rhs[a] = 0.0;
    }
    for (size_t i = 0; i < n; ++i)
        for (size_t a = 0; a < p; ++a) rhs[a] += X[i * p + a] * y[i];
    for (size_t a = 0; a < p; ++a) rhs[a] -= sx[a] * sy / c;
    la::solve_spd_ridge(A, rhs, beta, 0.0);
    alpha = 0.0;
    for (size_t a = 0; a < p; ++a) alpha -= beta[a] * sx[a] / c;
    alpha += sy / c;
}

// ---------------------------------------------------------------------------
// MarketDataBuffer tests
// ---------------------------------------------------------------------------
static void test_buffer_basic_and_contiguous() {
    MarketDataBuffer b(100);
    for (size_t i = 0; i < 50; ++i)
        b.push((std::int64_t)i, (double)i, 1000.0 + i);
    CHECK(b.size() == 50);
    CHECK(b.priceAt(0) == 0.0);
    CHECK(b.priceAt(49) == 49.0);
    CHECK(b.volumeAt(5) == 1005.0);
    CHECK(b.timestampAt(10) == 10);
}

// --- global allocation counter to verify no dynamic allocation in hot path ---
namespace {
std::size_t g_alloc_count = 0;
bool g_counting = false;
}
void* operator new(std::size_t n) {
    if (g_counting) ++g_alloc_count;
    if (void* p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void* operator new[](std::size_t n) {
    if (g_counting) ++g_alloc_count;
    if (void* p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete[](void* p) noexcept { std::free(p); }

struct AllocGuard {
    std::size_t before;
    AllocGuard() { g_counting = true; before = g_alloc_count; }
    ~AllocGuard() { g_counting = false; }
    std::size_t during() const { return g_alloc_count - before; }
};

static void test_no_allocation_in_hot_push_path() {
    // After construction (reserve done), pushing within capacity must not allocate.
    {
        MarketDataBuffer b(1000);
        b.push(1, 1.0, 1.0);  // warm
        AllocGuard g;
        for (int i = 0; i < 500; ++i) b.push((std::int64_t)i, 2.0, 2.0);
        CHECK(g.during() == 0);
    }
    {
        // RollingRegression pre-allocates all buffers in constructor; update path
        // must not allocate.
        RollingRegression rr(250, 3);
        std::vector<double> row = {1.0, 2.0, 3.0};
        rr.update(row, 1.0); rr.update(row, 2.0);  // warm (std::vector growth settles)
        AllocGuard g;
        for (int i = 0; i < 1000; ++i) rr.update(row, (double)i);
        rr.compute();
        CHECK(g.during() == 0);
    }
    // overflow guard still enforced
    {
        MarketDataBuffer b(4);
        for (int i = 0; i < 4; ++i) b.push(i, 1.0, 1.0);
        bool threw = false;
        try { b.push(9, 1.0, 1.0); } catch (const std::overflow_error&) { threw = true; }
        CHECK(threw);
    }
}

// ---------------------------------------------------------------------------
// Dense LA tests
// ---------------------------------------------------------------------------
static void test_spd_solve() {
    // A = [[3,1],[1,2]] positive definite; A x = b with x=[1,1] => b=[4,3]
    la::Mat A(2, 2);
    A(0,0)=3; A(0,1)=1; A(1,0)=1; A(1,1)=2;
    std::vector<double> b = {4.0, 3.0}, x;
    CHECK(la::solve_spd_ridge(A, b, x, 0.0));
    CHECK_NEAR(x[0], 1.0, 1e-9);
    CHECK_NEAR(x[1], 1.0, 1e-9);
}

static void test_ridge_inverts_singular() {
    // Singular A = ones (rank 1): raw solve fails, ridge makes it solvable.
    la::Mat A(3, 3, 1.0);
    std::vector<double> b = {1.0, 2.0, 3.0}, x;
    CHECK(!la::solve_spd_ridge(A, b, x, 0.0));
    CHECK(la::solve_spd_ridge(A, b, x, 1e-3));
}

static void test_symmetric_eig() {
    // Diagonal matrix: exact eigenvalues 2, 1.
    la::Mat A(2, 2, 0.0); A(0,0)=2; A(1,1)=1;
    std::vector<double> w; la::Mat Q;
    CHECK(la::symmetric_eig(A, w, Q));
    CHECK_NEAR(w[0], 2.0, 1e-6);
    CHECK_NEAR(w[1], 1.0, 1e-6);
    // With an off-diagonal term, trace invariance must hold: sum(eig) == trace(A).
    A(0,1) = 0.2; A(1,0) = 0.2;
    CHECK(la::symmetric_eig(A, w, Q));
    CHECK_NEAR(w[0] + w[1], 3.0, 1e-6);   // trace invariant
    // Q is orthogonal-ish and reconstructs A: tr(Q^T diag(w) Q - A) ~ 0
    double err = 0.0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            double recon = 0.0;
            for (int k = 0; k < 2; ++k) recon += Q(i,k)*w[k]*Q(j,k);
            err += (recon - (i==j ? (i==0? (A(0,0)) : A(1,1)) : A(i,j))) *
                   (recon - (i==j ? (i==0? A(0,0) : A(1,1)) : A(i,j)));
        }
    CHECK_NEAR(err, 0.0, 1e-6);
}

// ---------------------------------------------------------------------------
// RollingRegression incremental == brute force
// ---------------------------------------------------------------------------
static void test_rolling_regression_matches_bruteforce() {
    const size_t window = 120, p = 3;
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0, 1);
    // true beta
    std::vector<double> beta_true = {1.2, -0.7, 0.4};
    // generate x and y = x*beta + noise
    std::vector<double> all_x(window * p), all_y(window);
    for (size_t i = 0; i < window; ++i) {
        double yv = 0.5;
        for (size_t j = 0; j < p; ++j) { all_x[i*p+j] = nd(rng); yv += all_x[i*p+j]*beta_true[j]; }
        all_y[i] = yv + 0.05 * nd(rng);
    }
    RollingRegression rr(window, p);
    std::vector<double> row(p);
    for (size_t i = 0; i < window; ++i) {
        for (size_t j = 0; j < p; ++j) row[j] = all_x[i*p+j];
        rr.update(row, all_y[i]);
    }
    rr.compute();
    CHECK(rr.ready());
    std::vector<double> beta_ref; double alpha_ref;
    brute_force_ols(all_x, window, p, all_y, beta_ref, alpha_ref);
    CHECK_NEAR(rr.alpha(), alpha_ref, 1e-6);
    for (size_t j = 0; j < p; ++j)
        CHECK_NEAR(rr.beta()[j], beta_ref[j], 1e-6);
    // R^2 finite in [0,1]-ish
    CHECK(rr.rSquared() <= 1.0001);
}

static void test_rolling_eviction_matches_recomputed_sliding() {
    const size_t window = 80, p = 2;
    std::mt19937 rng(7);
    std::normal_distribution<double> nd(0, 1);
    // stream 400 rows
    const size_t T = 400;
    std::vector<double> xs(T * p), ys(T);
    for (size_t i = 0; i < T; ++i) {
        xs[i*p] = nd(rng); xs[i*p+1] = nd(rng);
        ys[i] = 0.3 + 1.0*xs[i*p] - 0.5*xs[i*p+1] + 0.1*nd(rng);
    }
    RollingRegression rr(window, p);
    std::vector<double> row(p);
    for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < p; ++j) row[j] = xs[i*p+j];
        rr.update(row, ys[i]);
        if (i >= window - 1) {
            rr.compute();
            // brute force on last `window` rows
            std::vector<double> X(window*p), y(window);
            for (size_t k = 0; k < window; ++k) {
                size_t t = i - (window - 1) + k;
                for (size_t j = 0; j < p; ++j) X[k*p+j] = xs[t*p+j];
                y[k] = ys[t];
            }
            std::vector<double> bref(p); double aref;
            brute_force_ols(X, window, p, y, bref, aref);
            CHECK_NEAR(rr.alpha(), aref, 1e-6);
            for (size_t j = 0; j < p; ++j) CHECK_NEAR(rr.beta()[j], bref[j], 1e-6);
        }
    }
}

static void test_window_not_yet_full() {
    RollingRegression rr(100, 2);
    std::vector<double> row = {1.0, 2.0};
    rr.update(row, 3.0);
    CHECK(!rr.ready());
    rr.compute();
    CHECK(!rr.betaValid());   // too few points => not valid
}

// ---------------------------------------------------------------------------
// SignalGenerator
// ---------------------------------------------------------------------------
static void test_signal_bands() {
    SignalGenerator sg(2.0, 0.5, 0.0);
    CHECK(sg.evaluate(2.5, 0.0).action == SignalAction::SHORT_TARGET);
    CHECK(sg.evaluate(-2.5, 0.0).action == SignalAction::LONG_TARGET);
    CHECK(sg.evaluate(0.1, 0.0).action == SignalAction::FLAT);
}

static void test_signal_half_life_cap() {
    SignalGenerator sg(1.0, 0.5, 10.0);
    // even if z still beyond entry, after > half_life_cap bars => forced flat
    CHECK(sg.evaluate(3.0, 11.0).action == SignalAction::FLAT);
    CHECK(sg.evaluate(3.0, 3.0).action == SignalAction::SHORT_TARGET);
}

// ---------------------------------------------------------------------------
// PortfolioBook
// ---------------------------------------------------------------------------
static void test_book_exposures() {
    PortfolioBook book(4);
    book.setPosition(0, 100.0);
    book.setPosition(1, -50.0);
    book.setPosition(2, 0.0);
    CHECK_NEAR(book.grossExposure(), 150.0, 1e-9);
    CHECK_NEAR(book.netExposure(), 50.0, 1e-9);
}

// ---------------------------------------------------------------------------
// LatencyHist percentiles
// ---------------------------------------------------------------------------
static void test_latency_percentiles() {
    LatencyHist h;
    h.setRange(1.0, 1e6, 1000);
    // 100 samples from 100..199 (all >= lo, < hi, roughly uniform in log)
    for (int i = 0; i < 100; ++i) h.add(100.0 + i);
    CHECK(h.n() == 100);
    double p50 = h.p50();
    CHECK(p50 >= 100.0 && p50 <= 200.0);
    CHECK(h.p99() >= h.p50());
    CHECK(h.mean() >= 100.0 && h.mean() <= 200.0);
}

// ---------------------------------------------------------------------------
// BacktestEngine end-to-end pipeline
// ---------------------------------------------------------------------------
static void test_engine_runs_and_measures_latency() {
    BacktestEngine eng(60, 2, 2.0, 0.5);
    std::mt19937 rng(3);
    std::normal_distribution<double> nd(0, 1);
    std::vector<double> row(2);
    for (int i = 0; i < 2000; ++i) {
        row[0] = nd(rng); row[1] = nd(rng);
        double target = 0.5*row[0] - 0.3*row[1] + 0.1*nd(rng);
        eng.runBar(row, target);
    }
    CHECK(eng.latency().bars == 2000);
    CHECK(eng.latency().regression_update.n() == 2000);
    CHECK(eng.latency().signal_generation.n() == 2000);
    CHECK(eng.latency().regression_update.p95() >= eng.latency().regression_update.p50());
    CHECK(eng.latency().order_placement.n() == 2000);
}

// ---------------------------------------------------------------------------
// Spread backtest correctness (Part 4)
// ---------------------------------------------------------------------------
// Build cointegrated-like synthetic prices: features geometric random walks,
// target log-price = 0.7 F1 + 0.5 F2 + a controlled spread term `sp`.
static std::vector<std::vector<double>> synth_prices(size_t T, size_t k,
                                                     const std::vector<double>& sp,
                                                     unsigned seed = 1) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<std::vector<double>> P(1 + k, std::vector<double>(T));
    std::vector<double> F1(T), F2(T), lgF(k, 0.0);
    // one shared-ish factor + idiosyncratic
    double flog = 0.0;
    for (size_t t = 0; t < T; ++t) {
        flog += 0.0005 + 0.01 * nd(rng);
        double lf = 0.8 * flog;
        std::vector<double> feat(k);
        for (size_t i = 0; i < k; ++i) feat[i] = lf + 0.01 * nd(rng) * std::sqrt((double)t + 1);
        // target log price
        double tl = 0.7 * feat[0] + 0.5 * (k > 1 ? feat[1] : feat[0]) + sp[t];
        P[0][t] = std::exp(tl);
        for (size_t i = 0; i < k; ++i) P[1 + i][t] = std::exp(feat[i]);
    }
    (void)lgF; (void)F1; (void)F2;
    return P;
}

// Deterministic series: features grow smoothly, target = 0.7 f1 + 0.5 f2 + sp.
// No noise => the only source of a large z is the injected spike, so no spurious
// trades occur before it.
static std::vector<std::vector<double>> det_prices(size_t T, size_t k,
                                                   const std::vector<double>& sp) {
    std::vector<std::vector<double>> P(1 + k, std::vector<double>(T));
    for (size_t t = 0; t < T; ++t) {
        std::vector<double> feat(k);
        double lf = 0.001 * (double)t;
        for (size_t i = 0; i < k; ++i) feat[i] = lf * (1.0 + 0.1 * (double)i);
        P[0][t] = std::exp(0.7 * feat[0] + 0.5 * feat[1] + sp[t]);
        for (size_t i = 0; i < k; ++i) P[1 + i][t] = std::exp(feat[i]);
    }
    return P;
}

static void test_backtest_one_bar_delay_no_lookahead() {
    const size_t T = 500, k = 2, window = 60;
    std::vector<double> sp(T, 0.0);   // zero spread => z ~ 0, flat
    auto P = det_prices(T, k, sp);
    size_t t0 = 250;
    P[0][t0] *= 2.5;                 // huge single-day target spike => z[t0] large
    statarb::SpreadConfig cfg;
    cfg.window = window; cfg.entry_z = 3.0; cfg.exit_z = 0.5; cfg.cost_one_way = 0.0005;
    auto r = statarb::run_spread_backtest(P, cfg);
    // before t0 there must be no position (deterministic, z~0)
    for (size_t t = window + 1; t < t0; ++t) CHECK(r.pos[t] == 0);
    // The spike is only visible at bar t0, so a position can open at the earliest
    // bar t0+1. pos[t0] (active during bar t0) must still be 0.
    CHECK(r.pos[t0] == 0);
    bool opened = (r.pos[t0 + 1] != 0);
    // If it opened (spike z>entry), it opened with the correct sign (short target).
    if (opened) CHECK(r.pos[t0 + 1] == 1);
    // never trade before the window is warm
    for (size_t t = 0; t <= window; ++t) CHECK(r.pos[t] == 0);
}

static void test_backtest_no_trade_when_too_short() {
    const size_t T = 40, k = 2;   // shorter than window => no warm window => no trades
    std::vector<double> sp(T, 0.0);
    auto P = synth_prices(T, k, sp, 3);
    statarb::SpreadConfig cfg;
    cfg.window = 120;
    auto r = statarb::run_spread_backtest(P, cfg);
    CHECK(r.n_trades == 0);
    for (auto p : r.pos) CHECK(p == 0);
}

static void test_backtest_costs_never_improve_net() {
    // oscillating square-wave spread => many trades
    const size_t T = 1500, k = 2, window = 60;
    std::vector<double> sp(T, 0.0);
    for (size_t t = 0; t < T; ++t) sp[t] = (t / 40) % 2 == 0 ? 0.8 : -0.8;
    auto P = synth_prices(T, k, sp, 11);
    statarb::SpreadConfig cfg;
    cfg.window = window; cfg.entry_z = 0.3; cfg.exit_z = 0.05; cfg.cost_one_way = 0.002;
    auto r = statarb::run_spread_backtest(P, cfg);
    CHECK(r.n_trades > 0);
    // net log NAV must never exceed gross (costs only subtract)
    for (size_t t = 0; t < T; ++t)
        CHECK(r.lognav_net[t] <= r.lognav_gross[t] + 1e-12);
    CHECK(r.cost_paid > 0.0);
    // gross lognav should beat net by (at least) the cost
    CHECK(r.lognav_net[T - 1] < r.lognav_gross[T - 1]);
}

// ---------------------------------------------------------------------------
int run_engine_tests() {
    RUN(test_backtest_one_bar_delay_no_lookahead);
    RUN(test_backtest_no_trade_when_too_short);
    RUN(test_backtest_costs_never_improve_net);
    RUN(test_buffer_basic_and_contiguous);
    RUN(test_no_allocation_in_hot_push_path);
    RUN(test_spd_solve);
    RUN(test_ridge_inverts_singular);
    RUN(test_symmetric_eig);
    RUN(test_rolling_regression_matches_bruteforce);
    RUN(test_rolling_eviction_matches_recomputed_sliding);
    RUN(test_window_not_yet_full);
    RUN(test_signal_bands);
    RUN(test_signal_half_life_cap);
    RUN(test_book_exposures);
    RUN(test_latency_percentiles);
    RUN(test_engine_runs_and_measures_latency);
    return test::summary();
}
