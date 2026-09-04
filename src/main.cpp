// statarbsim — low-latency statistical arbitrage backtest engine (entry point).
//
// CLI:
//   statarbsim --version   print version
//   statarbsim --test      run the unit-test suite (test/engine_tests.cpp)
//   statarbsim --demo      run a latency demo over synthetic basket data and print
//                          per-stage p50/p95/p99 histograms (feeds report figures)
int run_engine_tests();

#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "engine/backtest_engine.h"

namespace {
void run_demo() {
    using statarb::BacktestEngine;
    BacktestEngine eng(120, 4, 2.0, 0.5);
    std::mt19937 rng(1234);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> row(4);
    const int T = 30000;
    for (int i = 0; i < T; ++i) {
        double target = 1.0;
        for (auto& r : row) r = nd(rng);
        target = 0.9 * row[0] - 0.4 * row[1] + 0.2 * row[2] - 0.1 * row[3] + 0.3 * nd(rng);
        eng.runBar(row, target);
    }
    eng.finish();
    const auto& rep = eng.latency();
    auto ns = [](double v) { return v; };  // values already in ns
    std::printf("bars simulated: %zu\n", rep.bars);
    std::printf("stage                   p50(ns)   p95(ns)   p99(ns)   mean(ns)\n");
    std::printf("regression_update  %10.1f %9.1f %9.1f %9.1f\n",
                ns(rep.regression_update.p50()), ns(rep.regression_update.p95()),
                ns(rep.regression_update.p99()), ns(rep.regression_update.mean()));
    std::printf("signal_generation   %10.1f %9.1f %9.1f %9.1f\n",
                ns(rep.signal_generation.p50()), ns(rep.signal_generation.p95()),
                ns(rep.signal_generation.p99()), ns(rep.signal_generation.mean()));
    std::printf("order_placement     %10.1f %9.1f %9.1f %9.1f\n",
                ns(rep.order_placement.p50()), ns(rep.order_placement.p95()),
                ns(rep.order_placement.p99()), ns(rep.order_placement.mean()));
}
}  // namespace

int main(int argc, char** argv) {
    std::string cmd = argc > 1 ? argv[1] : "";
    if (cmd == "--version") {
        std::printf("statarbsim 0.3.0 (C++ engine skeleton, Part 3)\n");
        return 0;
    }
    if (cmd == "--test") {
        return run_engine_tests();
    }
    if (cmd == "--demo") {
        run_demo();
        return 0;
    }
    std::printf("statarbsim 0.3.0 (Part 3 skeleton). Usage:\n"
                "  statarbsim --version | --test | --demo\n");
    return 0;
}
