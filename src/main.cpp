// statarbsim — low-latency statistical arbitrage backtest engine (entry point).
//
// CLI:
//   statarbsim --version            print version
//   statarbsim --test               run the unit-test suite (test/engine_tests.cpp)
//   statarbsim --demo               latency demo over synthetic data (Part 3)
//   statarbsim --backtest <csv> <out> [window entry_z exit_z cost_way]
//                                   run the causal spread backtest on an exported
//                                   target+basket CSV; write per-bar series to <out>
int run_engine_tests();

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "data/csv_loader.h"
#include "engine/backtest_engine.h"
#include "engine/spread_backtest.h"

namespace {

void run_demo() {
    using statarb::BacktestEngine;
    BacktestEngine eng(120, 4, 2.0, 0.5);
    std::mt19937 rng(1234);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> row(4);
    const int T = 30000;
    for (int i = 0; i < T; ++i) {
        for (auto& r : row) r = nd(rng);
        double target = 0.9 * row[0] - 0.4 * row[1] + 0.2 * row[2] - 0.1 * row[3] + 0.3 * nd(rng);
        eng.runBar(row, target);
    }
    eng.finish();
    const auto& rep = eng.latency();
    std::printf("bars simulated: %zu\n", rep.bars);
    std::printf("stage                   p50(ns)   p95(ns)   p99(ns)   mean(ns)\n");
    std::printf("regression_update  %10.1f %9.1f %9.1f %9.1f\n",
                rep.regression_update.p50(), rep.regression_update.p95(),
                rep.regression_update.p99(), rep.regression_update.mean());
    std::printf("signal_generation   %10.1f %9.1f %9.1f %9.1f\n",
                rep.signal_generation.p50(), rep.signal_generation.p95(),
                rep.signal_generation.p99(), rep.signal_generation.mean());
    std::printf("order_placement     %10.1f %9.1f %9.1f %9.1f\n",
                rep.order_placement.p50(), rep.order_placement.p95(),
                rep.order_placement.p99(), rep.order_placement.mean());
}

int run_backtest(const char* csv, const char* out, statarb::SpreadConfig cfg) {
    using statarb::SpreadResult;
    std::vector<std::vector<double>> cols;   // cols[i] over rows
    if (!statarb::load_csv_columns(csv, cols, 2)) {
        std::printf("ERROR: could not load %s\n", csv);
        return 1;
    }
    size_t n_assets = cols.size();
    size_t T = n_assets ? cols[0].size() : 0;
    // convert column-major to price[a][t]
    std::vector<std::vector<double>> price(n_assets, std::vector<double>(T));
    for (size_t a = 0; a < n_assets; ++a)
        for (size_t t = 0; t < T; ++t) price[a][t] = cols[a][t];

    SpreadResult r = statarb::run_spread_backtest(price, cfg);
    std::printf("target basket backtest: %zu assets x %zu bars\n", n_assets, T);
    std::printf("trades: %zu | cost(log): %.4f | avg holding: %.1f bars\n",
                r.n_trades, r.cost_paid, r.avg_holding_bars);
    std::printf("gross: ann_ret %.4f sharpe %.3f\n", r.ann_ret_gross, r.sharpe_gross);
    std::printf("net  : ann_ret %.4f sharpe %.3f maxdd %.4f\n",
                r.ann_ret_net, r.sharpe_net, r.maxdd_net);

    FILE* f = std::fopen(out, "w");
    if (f) {
        std::fprintf(f, "row,lognav_gross,lognav_net,pos,z\n");
        for (size_t t = 0; t < T; ++t)
            std::fprintf(f, "%zu,%.6f,%.6f,%d,%.4f\n", t, r.lognav_gross[t], r.lognav_net[t],
                         r.pos[t], r.z[t]);
        std::fclose(f);
        std::printf("wrote %s\n", out);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string cmd = argc > 1 ? argv[1] : "";
    if (cmd == "--version") { std::printf("statarbsim 0.4.0 (C++ engine, Part 4)\n"); return 0; }
    if (cmd == "--test") return run_engine_tests();
    if (cmd == "--demo") { run_demo(); return 0; }
    if (cmd == "--backtest") {
        if (argc < 4) {
            std::printf("usage: statarbsim --backtest <csv> <out> [window entry_z exit_z cost_way]\n");
            return 1;
        }
        statarb::SpreadConfig cfg;
        if (argc >= 5) cfg.window = (size_t)std::atoi(argv[4]);
        if (argc >= 6) cfg.entry_z = std::atof(argv[5]);
        if (argc >= 7) cfg.exit_z = std::atof(argv[6]);
        if (argc >= 8) cfg.cost_one_way = std::atof(argv[7]);
        return run_backtest(argv[2], argv[3], cfg);
    }
    std::printf("statarbsim 0.4.0 (Part 4). Usage:\n"
                "  --version | --test | --demo | --backtest <csv> <out> [window entry exit cost]\n");
    return 0;
}
