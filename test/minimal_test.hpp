// minimal_test.hpp — tiny dependency-free test harness.
//
// The C++ engine must be dependency-free at build time (no external test
// framework fetchable in this sandbox), so we provide a small assertion-based
// harness. Usage:
//     #include "minimal_test.hpp"
//     static void test_foo() { ... CHECK(cond); ... }
//     int main(){ RUN(test_foo); return test::summary(); }
#ifndef STATARB_MINIMAL_TEST_HPP
#define STATARB_MINIMAL_TEST_HPP

#include <cmath>
#include <cstdio>
#include <string>

namespace test {
struct Registry {
    int checks = 0;
    int failed_checks = 0;
    int tests = 0;
    int failed_tests = 0;
    bool in_test = false;
    std::string current;
};
inline Registry& registry() { static Registry r; return r; }

#define CHECK(cond)                                                          \
    do {                                                                     \
        ++::test::registry().checks;                                         \
        if (!(cond)) {                                                       \
            ++::test::registry().failed_checks;                              \
            std::printf("    CHECK FAILED %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        }                                                                    \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                \
    do {                                                                     \
        ++::test::registry().checks;                                         \
        double _a = (a), _b = (b), _t = (tol);                               \
        if (!(std::abs(_a - _b) <= _t)) {                                    \
            ++::test::registry().failed_checks;                              \
            std::printf("    CHECK_NEAR FAILED %s:%d  %s=%g != %s=%g (tol %g)\n", \
                        __FILE__, __LINE__, #a, _a, #b, _b, _t);             \
        }                                                                    \
    } while (0)

#define RUN(fn)                                                              \
    do {                                                                     \
        ++::test::registry().tests;                                          \
        ::test::registry().current = #fn;                                    \
        int _before = ::test::registry().failed_checks;                      \
        std::printf("  [run] %s\n", #fn);                                    \
        fn();                                                                \
        if (::test::registry().failed_checks > _before) ++::test::registry().failed_tests; \
    } while (0)

inline int summary() {
    Registry& r = registry();
    std::printf("\n== tests: %d run, %d failed; checks: %d, failed %d ==\n",
                r.tests, r.failed_tests, r.checks, r.failed_checks);
    return r.failed_checks == 0 ? 0 : 1;
}
}  // namespace test

#endif  // STATARB_MINIMAL_TEST_HPP
