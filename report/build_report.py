#!/usr/bin/env python3
"""Report build driver.

Single entry point for building the report. Does three things, in order:
  1. regenerate figures/tables from results/ (via scripts/; no-op until Parts 1+);
  2. compile report.tex -> report.pdf if a TeX engine is present (canonical deliverable);
  3. always render a self-contained report.html sandbox preview (no CDN: math as images).

The LaTeX source (report.tex) is the source of truth. The HTML is a preview only.

Usage:
    python3 report/build_report.py [--pdf] [--no-preview] [--no-regenerate]
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "report")


def run(cmd, cwd=None, silent=True):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def regenerate():
    """(Re)generate figures/tables from results/ using scripts/."""
    scripts = [
        os.path.join(ROOT, "scripts", "analysis", "make_data_summary.py"),
        os.path.join(ROOT, "scripts", "analysis", "make_statkit_validation.py"),
        os.path.join(ROOT, "scripts", "plotting", "render_report_data.py"),
        os.path.join(ROOT, "scripts", "plotting", "render_method_validation.py"),
        # C++ engine benchmarks (Part 3): capture then render.
        os.path.join(ROOT, "scripts", "analysis", "capture_engine_bench.py"),
        os.path.join(ROOT, "scripts", "plotting", "render_engine_results.py"),
        # Spread backtest (Part 4): capture then render.
        os.path.join(ROOT, "scripts", "analysis", "capture_backtest.py"),
        os.path.join(ROOT, "scripts", "plotting", "render_backtest.py"),
        # Baseline model comparison (Part 5): capture then render.
        os.path.join(ROOT, "scripts", "analysis", "compare_models.py"),
        os.path.join(ROOT, "scripts", "plotting", "render_model_comparison.py"),
        # Statistical diagnostics (Part 6): capture then render.
        os.path.join(ROOT, "scripts", "analysis", "run_diagnostics.py"),
        os.path.join(ROOT, "scripts", "plotting", "render_diagnostics.py"),
    ]
    for s in scripts:
        if os.path.exists(s):
            rc, out, err = run(["python3", s])
            if rc != 0:
                print(f"  [warn] {os.path.basename(s)} failed:\n{err[-500:]}")
            elif out.strip():
                print(f"  [ok]   {os.path.basename(s)}")
    return


def have_tex():
    for exe in ("latexmk", "pdflatex", "xelatex", "lualatex", "tectonic"):
        if shutil.which(exe):
            return exe
    return None


def compile_latex(engine):
    tex = os.path.join(REPORT, "report.tex")
    out = os.path.join(REPORT, "report.pdf")
    os.makedirs(REPORT, exist_ok=True)
    if engine == "latexmk":
        return run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                    os.path.basename(tex)], cwd=REPORT)[0] == 0
    if engine == "tectonic":
        return run(["tectonic", "-X", "compile", os.path.basename(tex)], cwd=REPORT)[0] == 0
    # pdflatex -> biber -> pdflatex x2
    rc, _, _ = run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-jobname=report", os.path.basename(tex)], cwd=REPORT)
    if rc != 0:
        return False
    biber = shutil.which("biber")
    if biber:
        run([biber, "report"], cwd=REPORT)
        run(["pdflatex", "-interaction=nonstopmode", "-jobname=report", os.path.basename(tex)], cwd=REPORT)
        run(["pdflatex", "-interaction=nonstopmode", "-jobname=report", os.path.basename(tex)], cwd=REPORT)
    return os.path.exists(out)


def copy_figures_to_report():
    """Copy generated figures from results/figures into report/figures so both the
    LaTeX (\includegraphics{figures/...}) and the HTML preview resolve them."""
    src = os.path.join(ROOT, "results", "figures")
    dst = os.path.join(REPORT, "figures")
    if not os.path.isdir(src):
        return
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith((".png", ".pdf")):
            import shutil
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))


def render_html_preview():
    """Self-contained HTML preview. Math rendered to SVG images via matplotlib
    (so no CDN / network is needed). In Part 0 the content is the report skeleton."""
    copy_figures_to_report()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  [preview] matplotlib unavailable; skipping math rendering")
        plt = None

    tex = os.path.join(REPORT, "report.tex")
    if not os.path.exists(tex):
        print("  [preview] no report.tex; skipping preview")
        return False

    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Multi-Asset Statistical Arbitrage — Report</title>",
        "<style>",
        "body{font-family:Georgia,serif;max-width:900px;margin:24px auto;padding:0 20px;line-height:1.5;color:#222}",
        "h1{font-size:1.6em;border-bottom:2px solid #333;padding-bottom:6px}",
        "h2{font-size:1.25em;margin-top:2em;color:#111}",
        "h3{font-size:1.05em}",
        ".meta{color:#666;font-size:0.9em}",
        "code{background:#f4f4f4;padding:1px 4px;border-radius:3px}",
        "table{border-collapse:collapse}",
        "</style></head><body>",
    ]
    html.append("<h1>Multi-Asset Statistical Arbitrage</h1>")
    html.append("<p class='meta'>Sandbox HTML preview of the canonical LaTeX report "
                "(<code>report/report.tex</code>). The PDF is the canonical deliverable and "
                "compiles wherever a TeX engine is available. This preview is regenerated "
                "every Part from the latest report content and results.</p>")

    # Show the report skeleton structure (section headers pulled from report.tex).
    import re
    with open(tex) as f:
        content = f.read()
    headers = re.findall(r"\\section\*?\{(.*?)\}", content)
    subheaders = re.findall(r"\\subsection\*?\{(.*?)\}", content)
    html.append("<h2>Sections in this build</h2><ol>")
    for h in headers:
        html.append(f"<li>{h}</li>")
    html.append("</ol>")
    html.append("<p>Subsections: " + ", ".join(subheaders) + "</p>")

    # Embed generated data tables + figures (Section 3) so the preview is substantive.
    import glob
    data_tab = os.path.join(ROOT, "results", "tables", "data_summary.csv")
    if os.path.exists(data_tab):
        html.append("<h2>Section 3 — Data (generated)</h2>")
        try:
            import csv, io
            with open(data_tab) as f:
                rows = list(csv.reader(f))[1:]
            html.append("<table><tr><th>Metric</th><th>Value</th></tr>")
            for r in rows:
                html.append(f"<tr><td>{r[0]}</td><td>{r[1]}</td></tr>")
            html.append("</table>")
        except Exception:
            pass
        for png in ("universe_by_year.png", "sector_comp.png"):
            fp = os.path.join(ROOT, "results", "figures", png)
            if os.path.exists(fp):
                html.append(f"<figure><img src='figures/{png}' style='max-width:100%'><figcaption>{png}</figcaption></figure>")

    # Embed Part 3 C++ engine latency + cache figures (System Architecture).
    for png in ("latency_hist.png", "cache_bench.png"):
        fp = os.path.join(ROOT, "results", "figures", png)
        if os.path.exists(fp):
            html.append(f"<figure><img src='figures/{png}' style='max-width:100%'><figcaption>{png}</figcaption></figure>")

    # Embed Part 4 backtest equity curve + cost sensitivity (Section 7).
    for png in ("equity_curve.png", "cost_sensitivity.png"):
        fp = os.path.join(ROOT, "results", "figures", png)
        if os.path.exists(fp):
            html.append(f"<figure><img src='figures/{png}' style='max-width:100%'><figcaption>{png}</figcaption></figure>")

    # Embed Part 5 model-comparison figures (Section 5).
    for png in ("model_compare_sharpe.png", "model_compare_ic.png",
                "model_compare_stability.png"):
        fp = os.path.join(ROOT, "results", "figures", png)
        if os.path.exists(fp):
            html.append(f"<figure><img src='figures/{png}' style='max-width:100%'><figcaption>{png}</figcaption></figure>")

    # Embed Part 6 diagnostics figures (Robustness section).
    for png in ("diagnostics_residual.png", "diagnostics_halflife.png"):
        fp = os.path.join(ROOT, "results", "figures", png)
        if os.path.exists(fp):
            html.append(f"<figure><img src='figures/{png}' style='max-width:100%'><figcaption>{png}</figcaption></figure>")
    dt = os.path.join(ROOT, "results", "tables", "diagnostics_summary.csv")
    if os.path.exists(dt):
        try:
            import csv as _csv
            with open(dt) as f:
                rws = list(_csv.reader(f))
            html.append("<h2>Diagnostics battery (Robustness / Part 6)</h2><table><tr>")
            for h in rws[0]:
                html.append(f"<th>{h}</th>")
            html.append("</tr>")
            for r in rws[1:]:
                html.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
            html.append("</table>")
        except Exception:
            pass

    # Embed Part 2 synthetic-validation table (Methodology / appendix).
    vt = os.path.join(ROOT, "results", "tables", "statkit_validation.csv")
    if os.path.exists(vt):
        html.append("<h2>Statistical toolkit — synthetic validation (Methodology appendix)</h2>")
        try:
            import csv
            with open(vt) as f:
                rows = list(csv.reader(f))[1:]
            html.append("<table><tr><th>Check</th><th>Value</th></tr>")
            for r in rows:
                html.append(f"<tr><td>{r[0]}</td><td>{r[1]}</td></tr>")
            html.append("</table>")
        except Exception:
            pass

    html.append("<hr><p class='meta'>Build time: " + __import__("datetime").datetime.now().isoformat() + "</p>")
    html.append("</body></html>")

    out = os.path.join(REPORT, "report.html")
    with open(out, "w") as f:
        f.write("\n".join(html))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true", help="force LaTeX compilation")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--no-regenerate", action="store_true")
    args = ap.parse_args()

    if not args.no_regenerate:
        print("[1/3] Regenerating figures/tables from results/ ...")
        regenerate()

    pdf_ok = False
    engine = have_tex()
    if engine:
        print(f"[2/3] TeX engine found: {engine}. Compiling report.pdf ...")
        pdf_ok = compile_latex(engine)
        if pdf_ok:
            print(f"      -> report/report.pdf built")
        else:
            print("      ! LaTeX compilation failed (report.pdf not produced); preview still generated.")
    else:
        print("[2/3] No TeX engine available in this sandbox (see environment/env_map.md).")
        print("      report.tex is canonical; compile it anywhere a TeX engine exists.")
        if not args.pdf:
            print("      Use --pdf to force an attempt (will fail without a TeX engine).")

    if not args.no_preview:
        print("[3/3] Rendering self-contained HTML preview ...")
        render_html_preview()
        print("      -> report/report.html (open in browser)")

    return 0 if (pdf_ok or not args.pdf) else 1


if __name__ == "__main__":
    sys.exit(main())
