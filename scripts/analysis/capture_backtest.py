#!/usr/bin/env python3
"""Run the C++ causal spread backtest on the real exported target+basket data at
several cost levels and record the results to results/tables/backtest_summary.csv.

Also writes the main per-bar series (cost = 10 bps one-way) to
results/backtests/<target>_main.csv for the equity-curve figure.
"""
import argparse
import os
import re
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD = os.path.join(ROOT, "build")
TAB = os.path.join(ROOT, "results", "tables")
BKT = os.path.join(ROOT, "results", "backtests")
ENGINE = os.path.join(ROOT, "data", "processed", "engine")
os.makedirs(TAB, exist_ok=True)
os.makedirs(BKT, exist_ok=True)

CFG = argparse.ArgumentParser()
CFG.add_argument("--target", default="ADBE")
CFG.add_argument("--basket", nargs="+", default=["CRM", "ADSK", "INTU"])
CFG.add_argument("--window", type=int, default=120)
CFG.add_argument("--entry", type=float, default=2.0)
CFG.add_argument("--exit", dest="exit_z", type=float, default=0.5)


def ensure_build():
    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + env.get("PATH", "")
    if not os.path.exists(os.path.join(BUILD, "statarbsim")):
        subprocess.run(["cmake", "-S", os.path.join(ROOT, "src"), "-B", BUILD], check=True, env=env)
    subprocess.run(["cmake", "--build", BUILD, "-j2"], check=True, env=env)


def run_backtest(csv, out, window, entry, exit_, cost):
    cmd = [os.path.join(BUILD, "statarbsim"), "--backtest", csv, out,
           str(window), str(entry), str(exit_), str(cost)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout


def parse(text):
    d = {}
    for m in re.finditer(r"trades: (\d+) \| cost\(log\): ([0-9.eE+-]+) \| avg holding: ([0-9.]+) bars", text):
        d["trades"] = int(m.group(1)); d["cost_log"] = float(m.group(2)); d["avg_hold"] = float(m.group(3))
    m = re.search(r"gross: ann_ret ([0-9.eE+-]+) sharpe ([0-9.eE+-]+)", text)
    if m: d["gross_ann_ret"] = float(m.group(1)); d["gross_sharpe"] = float(m.group(2))
    m = re.search(r"net  : ann_ret ([0-9.eE+-]+) sharpe ([0-9.eE+-]+) maxdd ([0-9.eE+-]+)", text)
    if m: d["net_ann_ret"] = float(m.group(1)); d["net_sharpe"] = float(m.group(2)); d["maxdd"] = float(m.group(3))
    return d


def main():
    args = CFG.parse_args()
    ensure_build()
    csv = os.path.join(ENGINE, f"{args.target}.csv")
    if not os.path.exists(csv):
        # export it first
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "data", "export_engine_input.py"),
                        "--target", args.target] + ["--basket"] + args.basket, check=True, cwd=ROOT)
    costs = [0.0, 0.0005, 0.001, 0.002, 0.005]
    rows = []
    main_series = None
    for c in costs:
        out = os.path.join(BKT, f"{args.target}_cost{int(c * 1e4):04d}.csv")
        text = run_backtest(csv, out, args.window, args.entry, args.exit_z, c)
        d = {"cost_way_bps": c * 1e4}
        d.update(parse(text))
        rows.append(d)
        if abs(c - 0.001) < 1e-12:
            main_series = out
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "backtest_summary.csv"), index=False)
    print("wrote results/tables/backtest_summary.csv")
    print(df.to_string(index=False))
    # copy main series to a canonical name
    if main_series:
        dest = os.path.join(BKT, f"{args.target}_main.csv")
        subprocess.run(["cp", main_series, dest])
        print("main series ->", dest)


if __name__ == "__main__":
    main()
