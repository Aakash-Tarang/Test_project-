#!/usr/bin/env python3
"""Report build driver -- the single entry point for building the book.

Does four things, in order:

  1. regenerate every figure and table from ``results/`` by running the checked-in
     scripts under ``scripts/`` (so no number in the book is ever hand-typed);
  2. compile ``report/report.tex`` -> ``report/report.pdf`` when a TeX engine is
     available (the canonical deliverable);
  3. ALWAYS render ``report/report.html`` -- the full book as a self-contained HTML
     preview, produced by ``report/latex_to_html.py`` (math rasterised locally with
     matplotlib's mathtext; no TeX engine and no CDN required);
  4. optionally serve ``report/`` over HTTP for a browser preview (``--serve``).

The LaTeX source is the source of truth; the HTML is a preview of it.

Usage:
    python3 report/build_report.py [--pdf] [--no-preview] [--no-regenerate]
                                   [--serve [PORT]] [--dpi N] [--check-math]
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "report")
sys.path.insert(0, REPORT)


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


# ---------------------------------------------------------------------------
# 1. regeneration: capture (results/tables/*.csv) then render (figures + .tex)
# ---------------------------------------------------------------------------
ANALYSIS = [
    "make_data_summary.py",          # Part 1: data panel summary tables
    "make_statkit_validation.py",    # Part 2: synthetic validation of the toolkit
    "capture_engine_bench.py",       # Part 3: C++ engine latency / cache benchmarks
    "capture_backtest.py",           # Part 4: spread backtest + cost sweep
    "compare_models.py",             # Part 5: OLS / ridge / lasso / EN / PCR
    "run_diagnostics.py",            # Part 6: stationarity, cointegration, HAC
    "capture_portfolio.py",          # Part 7: multi-basket book, risk, regimes
    "compare_kalman.py",             # Part 8: Kalman vs rolling OLS
    "compare_nonlinear.py",          # Part 9: nonlinear signals, RESET, gates
    "compare_multtest.py",           # Part 10: M-strategy audit, RC/SPA, DM
    "make_book_tables.py",           # Part 12: extra book tables + power/DSR
]
PLOTTING = [
    "render_report_data.py",
    "render_method_validation.py",
    "render_engine_results.py",
    "render_backtest.py",
    "render_model_comparison.py",
    "render_diagnostics.py",
    "render_portfolio.py",
    "render_kalman.py",
    "render_nonlinear.py",
    "render_multtest.py",
]


def regenerate(skip_missing_ok=True):
    """(Re)generate every table and figure from results/ using scripts/."""
    scripts = ([os.path.join(ROOT, "scripts", "analysis", s) for s in ANALYSIS]
               + [os.path.join(ROOT, "scripts", "plotting", s) for s in PLOTTING])
    ok = warn = 0
    for s in scripts:
        if not os.path.exists(s):
            if not skip_missing_ok:
                print("  [warn] missing script %s" % s)
            warn += 1
            continue
        rc, out, err = run([sys.executable, s])
        if rc != 0:
            print("  [FAIL] %s:\n%s" % (os.path.basename(s), err.strip()[-600:]))
            warn += 1
        else:
            ok += 1
    print("      %d scripts ok, %d skipped/failed" % (ok, warn))


def have_tex():
    for exe in ("latexmk", "pdflatex", "xelatex", "lualatex", "tectonic"):
        if shutil.which(exe):
            return exe
    return None


def compile_latex(engine):
    tex = os.path.join(REPORT, "report.tex")
    out = os.path.join(REPORT, "report.pdf")
    if engine == "latexmk":
        return run([engine, "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                    os.path.basename(tex)], cwd=REPORT)[0] == 0
    if engine == "tectonic":
        return run([engine, "-X", "compile", os.path.basename(tex)], cwd=REPORT)[0] == 0
    rc, _, err = run([engine, "-interaction=nonstopmode", "-halt-on-error",
                      "-jobname=report", os.path.basename(tex)], cwd=REPORT)
    if rc != 0:
        print("      ! %s failed:\n%s" % (engine, err.strip()[-800:]))
        return False
    if shutil.which("biber"):
        run(["biber", "report"], cwd=REPORT)
        for _ in range(2):
            run([engine, "-interaction=nonstopmode", "-jobname=report",
                 os.path.basename(tex)], cwd=REPORT)
    return os.path.exists(out)


def copy_figures_to_report():
    """Copy generated figures from results/figures into report/figures so that both the
    LaTeX (\\includegraphics{figures/...}) and the HTML preview resolve them."""
    src = os.path.join(ROOT, "results", "figures")
    dst = os.path.join(REPORT, "figures")
    if not os.path.isdir(src):
        return 0
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(src)):
        if f.endswith((".png", ".pdf", ".svg")):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
            n += 1
    return n


def render_preview(dpi=170, embed_figures=False, embed_math=False):
    """Render the whole book to report/report.html via latex_to_html.py."""
    try:
        import latex_to_html
    except Exception as e:
        print("  [preview] cannot import latex_to_html: %s" % e)
        return None
    n = copy_figures_to_report()
    print("      copied %d figures from results/figures -> report/figures" % n)
    return latex_to_html.render_book(REPORT, dpi=dpi, embed_figures=embed_figures,
                                     embed_math=embed_math)


def serve(directory, port):
    """Serve report/ on 0.0.0.0 so the book can be read in a browser preview."""
    import http.server
    import functools
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), handler)
    print("[serve] http://0.0.0.0:%d/report.html  (Ctrl-C to stop)" % port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pdf", action="store_true", help="force a LaTeX compilation attempt")
    ap.add_argument("--no-preview", action="store_true", help="skip the HTML book")
    ap.add_argument("--no-regenerate", action="store_true", help="skip step 1")
    ap.add_argument("--dpi", type=int, default=170, help="math rasterisation dpi")
    ap.add_argument("--embed-figures", action="store_true",
                    help="inline figure PNGs into report.html (fully portable, larger)")
    ap.add_argument("--portable", action="store_true",
                    help="single self-contained report.html: embed figures AND math")
    ap.add_argument("--serve", nargs="?", const=8010, type=int, default=None, metavar="PORT",
                    help="after building, serve report/ over HTTP (default port 8010)")
    ap.add_argument("--check-math", action="store_true",
                    help="only verify that every math expression is mathtext-renderable")
    args = ap.parse_args()

    if args.check_math:
        import latex_to_html
        return latex_to_html.check_math(REPORT)

    if not args.no_regenerate:
        print("[1/3] Regenerating figures/tables from results/ ...")
        regenerate()

    pdf_ok = False
    engine = have_tex()
    if engine:
        print("[2/3] TeX engine found: %s. Compiling report.pdf ..." % engine)
        pdf_ok = compile_latex(engine)
        print("      -> report/report.pdf built" if pdf_ok
              else "      ! LaTeX compilation failed; the HTML book is still rendered.")
    else:
        print("[2/3] No TeX engine in this sandbox (see environment/env_map.md).")
        print("      report/report.tex is canonical; compile it with latexmk anywhere")
        print("      a TeX distribution exists. The HTML book below is the preview.")
        if args.pdf:
            print("      (--pdf requested but no engine is installed)")

    if not args.no_preview:
        print("[3/3] Rendering the HTML book (math rasterised locally) ...")
        render_preview(dpi=args.dpi, embed_figures=args.embed_figures or args.portable,
                       embed_math=args.portable)

    if args.serve:
        return serve(REPORT, args.serve)
    return 0 if (pdf_ok or not args.pdf) else 1


if __name__ == "__main__":
    sys.exit(main())
