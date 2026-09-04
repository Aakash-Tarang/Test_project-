// microbench.cpp — cache-efficiency microbenchmarks for report Section 9.
//
//   microbench vector_vs_deque : maintain a rolling-window sum of the last W
//                                values under continuous push+evict. Both do the
//                                SAME O(1) arithmetic (new - leaving); the only
//                                difference is container: contiguous std::vector
//                                ring vs std::deque. Measures the cost of
//                                non-contiguous (deque) storage.
//   microbench soa_vs_aos      : column reduction (acc += price[i]*vol[i]) over a
//                                large array in Structure-of-Arrays layout vs
//                                Array-of-Structures layout. Memory-bound, so it
//                                exposes cache/prefetch differences.
//
// Reported in ns/op via steady_clock with a consumed accumulator.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
struct Timings { double v = 0, d = 0; double ratio = 0; };
volatile double g_sink = 0.0;

template <typename F>
double time_it(F&& f, int iters) {
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) g_sink += f();
    auto t1 = Clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
}

Timings bench_vector_vs_deque(int iters, int window) {
    Timings r;
    const long total = (long)iters;
    // Rolling window stored contiguously (std::vector). Each step pushes one new
    // value, evicts one, and iterates the whole window (like the residual-stat
    // pass of RollingRegression::compute). Contiguous iteration vectorizes and
    // prefetches well.
    {
        std::vector<double> buf(window);
        for (int i = 0; i < window; ++i) buf[i] = 0.5 * i;
        int head = 0; long n = 0; double out = 0.0;
        r.v = time_it([&]() {
            double newv = 1.0 + 0.001 * (n & 1023);
            buf[head] = newv;
            head = (head + 1) % window;
            double s = 0.0;
            for (double x : buf) s += x;         // full contiguous sweep
            ++n; out = s;
            return out;
        }, total);
    }
    // Same operation on std::deque (non-contiguous storage).
    {
        std::deque<double> d(window);
        for (int i = 0; i < window; ++i) d[i] = 0.5 * i;
        long n = 0; double out = 0.0;
        r.d = time_it([&]() {
            d.push_back(1.0 + 0.001 * (n & 1023));
            d.pop_front();
            double s = 0.0;
            for (double x : d) s += x;           // full deque sweep (scattered blocks)
            ++n; out = s;
            return out;
        }, total);
    }
    r.ratio = r.d / (r.v > 0 ? r.v : 1.0);
    return r;
}

Timings bench_soa_vs_aos(int iters) {
    Timings r;
    const int n = 1 << 20;  // 1M elements -> memory-bound
    // SoA
    std::vector<double> price(n), vol(n);
    for (int i = 0; i < n; ++i) { price[i] = 0.01 + (i % 10000) * 0.01; vol[i] = 1e6 + (i % 777); }
    // AoS
    struct Tick { double price; double vol; };
    std::vector<Tick> ticks(n);
    for (int i = 0; i < n; ++i) ticks[i] = {price[i], vol[i]};

    r.v = time_it([&]() {
        double acc = 0.0;
        for (int i = 0; i < n; ++i) acc += price[i] * vol[i];
        return acc;
    }, iters);
    r.d = time_it([&]() {
        double acc = 0.0;
        for (int i = 0; i < n; ++i) acc += ticks[i].price * ticks[i].vol;
        return acc;
    }, iters);
    // lower ratio (aos/soa) < 1 => soa faster
    r.ratio = r.d / (r.v > 0 ? r.v : 1.0);
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    std::string which = argc > 1 ? argv[1] : "all";
    int iters = argc > 2 ? std::atoi(argv[2]) : 50;
    int window = argc > 3 ? std::atoi(argv[3]) : 250;
    if (which == "vector_vs_deque" || which == "all") {
        auto t = bench_vector_vs_deque(iters, window);
        std::printf("vector_vs_deque(window=%d): vector %.1f ns/op | deque %.1f ns/op | deque/vector %.3fx\n",
                    window, t.v, t.d, t.ratio);
    }
    if (which == "soa_vs_aos" || which == "all") {
        auto t = bench_soa_vs_aos(iters);
        std::printf("soa_vs_aos(n=1M):           soa %.1f ns/op | aos %.1f ns/op | aos/soa %.3fx\n",
                    t.v, t.d, t.ratio);
    }
    (void)g_sink;
    return 0;
}
