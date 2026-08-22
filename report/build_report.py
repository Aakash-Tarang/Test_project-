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
    # Wired up in Part 1+ (scripts/plot_*.py, scripts/make_tables.py). Currently a no-op.
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


def render_html_preview():
    """Self-contained HTML preview. Math rendered to SVG images via matplotlib
    (so no CDN / network is needed). In Part 0 the content is the report skeleton."""
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
