#!/usr/bin/env python3
"""Run the statkit unit tests (synthetic-data checks of the math).

Discovery: any function named test_* in any module under scripts/statkit/tests/.
Prints per-test PASS/FAIL and a summary; exits non-zero if any test fails.

Usage:  python3 scripts/run_statkit_tests.py
"""
import importlib
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                      # so `scripts.statkit...` imports
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def discover():
    tests_dir = os.path.join(ROOT, "scripts", "statkit", "tests")
    mods = []
    for f in sorted(os.listdir(tests_dir)):
        if f.startswith("test_") and f.endswith(".py"):
            mods.append("statkit.tests." + f[:-3])
    return mods


def main():
    passed, failed = [], []
    for mod in discover():
        m = importlib.import_module(mod)
        for name in sorted(dir(m)):
            if name.startswith("test_"):
                fn = getattr(m, name)
                if callable(fn):
                    try:
                        fn()
                        passed.append(f"{mod}:{name}")
                        print(f"  [PASS] {mod}.{name}")
                    except Exception as e:
                        failed.append(f"{mod}:{name}")
                        print(f"  [FAIL] {mod}.{name}: {e}")
                        traceback.print_exc()
    print(f"\n== statkit tests: {len(passed)} passed, {len(failed)} failed ==")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
