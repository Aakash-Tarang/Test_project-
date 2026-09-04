#!/usr/bin/env python3
"""Capture real latency + microbenchmark numbers from the C++ engine.

Ensures the C++ engine is built (CMake), then runs:
    build/statarbsim --demo       -> per-stage p50/p95/p99 latency histogram data
    build/microbench all ...      -> vector-vs-deque and SoA-vs-AoS throughput

Writes:
    results/tables/engine_latency.csv   (stage, p50_ns, p95_ns, p99_ns, mean_ns, samples)
    results/tables/engine_bench.csv     (bench, value_a, value_b, ratio)

The numbers come straight from real runs (no hand-typing), so the report figures
regenerated from these are reproducible.
"""
import os
import re
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD = os.path.join(ROOT, "build")
TAB = os.path.join(ROOT, "results", "tables")
os.makedirs(TAB, exist_ok=True)


def ensure_build():
    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + env.get("PATH", "")
    if not os.path.exists(os.path.join(BUILD, "statarbsim")):
        subprocess.run(["cmake", "-S", os.path.join(ROOT, "src"), "-B", BUILD], check=True, env=env)
    subprocess.run(["cmake", "--build", BUILD, "-j2"], check=True, env=env)


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout


def capture_latency():
    out = run([os.path.join(BUILD, "statarbsim"), "--demo"])
    rows = []
    # demo prints e.g.:
    #   regression_update        599.5      1075.8 ...
    for line in out.splitlines():
        m = re.match(r"\s*([a-z_]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            rows.append({
                "stage": m.group(1),
                "p50_ns": float(m.group(2)),
                "p95_ns": float(m.group(3)),
                "p99_ns": float(m.group(4)),
                "mean_ns": float(m.group(5)),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "engine_latency.csv"), index=False)
    print("wrote results/tables/engine_latency.csv")
    return out


def capture_bench():
    out_vd = run([os.path.join(BUILD, "microbench"), "vector_vs_deque", "2000", "250"])
    out_sa = run([os.path.join(BUILD, "microbench"), "soa_vs_aos", "30"])
    rows = []
    m = re.search(r"vector_vs_deque\(window=\d+\): vector ([\d.]+) ns/op \| deque ([\d.]+) ns/op \| deque/vector ([\d.]+)x", out_vd)
    if m:
        rows.append({"bench": "vector_vs_deque", "a": float(m.group(1)), "b": float(m.group(2)),
                     "ratio": float(m.group(3)), "a_unit": "ns/op", "b_unit": "ns/op"})
    m = re.search(r"soa_vs_aos\(n=1M\): +soa ([\d.]+) ns/op \| aos ([\d.]+) ns/op \| aos/soa ([\d.]+)x", out_sa)
    if m:
        rows.append({"bench": "soa_vs_aos", "a": float(m.group(1)), "b": float(m.group(2)),
                     "ratio": float(m.group(3)), "a_unit": "ns/op", "b_unit": "ns/op"})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "engine_bench.csv"), index=False)
    print("wrote results/tables/engine_bench.csv")


def main():
    ensure_build()
    capture_latency()
    capture_bench()


if __name__ == "__main__":
    main()
