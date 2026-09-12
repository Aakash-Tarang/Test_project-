# HANDOFF — All 14 parts complete; the book is the deliverable

> Read this entire file first. It is the state of the repository, what was verified, and
> what a future session should do next.

## 1. Context

Incremental **Multi-Asset Statistical Arbitrage** project. **Parts 0–13 are all done**:
plan/env, data (Yahoo 195-name, 2010–2026 total-return panel), `statkit`, the C++ engine,
the correctness core, baseline model comparison, diagnostics, the portfolio book, the
Kalman hedge, nonlinear extensions, the multiple-testing audit, architecture/latency,
report assembly, and finalization.

The last session did two things beyond Part 13 as originally scoped:

1. **Rewrote the report as a book.** `report/report.tex` is now a root file that
   `\input`s 29 chapter files from `report/chapters/` (front matter, Part I theory from
   first principles, Part II phase-by-phase, Part III synthesis, four appendices). The old
   single-file report prose is recoverable with `git show a9e9f4a:report/report.tex`.
2. **Wrote a LaTeX→HTML book renderer** (`report/latex_to_html.py`) so the whole book is
   readable in this sandbox, which has no TeX engine. `report/build_report.py` drives it.

Work happens on the session branch (currently `arena/01a09676-test-project`; earlier
sessions used `arena/01a070da-test-project`). Never push to a different branch than the
one the session is tracking.

> **Sandbox recovery** (used repeatedly): the workspace can be re-cloned between sessions.
> `git fetch origin '+refs/heads/*:refs/remotes/origin/*'` then
> `git reset --hard origin/<session-branch>`; then `bash environment/setup.sh`;
> `export PATH="$HOME/.local/bin:$PATH"`; `bash scripts/data/run_pipeline.sh`;
> `bash environment/check.sh` regenerates all git-ignored `results/`, `data/processed/` and
> `build/` outputs (~14 min). Re-export basket CSVs if `data/processed/engine/*.csv` is
> missing: `python3 scripts/data/export_engine_input.py --target <T> --basket ...` for
> ADBE / AVGO / GS / CAT.

## 2. Environment facts that keep biting

- **No TeX engine** is installable (apt is blocked). `report/report.tex` is canonical;
  `report/report.html` is the in-sandbox preview. Do not try `apt install texlive`.
- **pip is PEP-668**: always `pip install --user --break-system-packages …` (what
  `environment/setup.sh` does).
- **`scripts/data/download.py --source github` needs network**; the raw Yahoo CSVs are
  already committed under `data/raw/yfinance/` and `clean.py` prefers them, so the whole
  pipeline runs offline. Skip `download.py`.
- **Latency numbers move between sessions** because they are measured on a shared 2-core
  sandbox. This build: vector/deque 1.084×, SoA/AoS 1.033×, regression update p50
  599.5 ns. An earlier session recorded 1.264× / 1.152× / 518 ns. Both are real
  measurements; the book reports the live ones and says so (Chapter 32, "Honesty remark").
- matplotlib's **mathtext** is not LaTeX: no `\begin{cases}`/`matrix`, no `\tfrac`, `\le`,
  `\bigl`, `\lvert`, `\bmod`, `\underbrace`, `\boxed`, `\texttt`. `latex_to_html.py`
  translates or decomposes all of them; run `python3 report/latex_to_html.py --check-math`
  after editing chapters (currently **3,375 expressions, 0 failures**).

## 3. Verified state (regenerate with `bash environment/check.sh`)

| Gate | Result |
|---|---|
| `environment/check.sh` | 52 checks passed, 0 failed |
| C++ `./build/statarbsim --test` | 24 tests, 2,852 assertions, 0 failures |
| `scripts/run_statkit_tests.py` | 35 tests passed |
| `report/build_report.py` | 24 figures, 23 tables, 352 labels, 18 citations, 0 unresolved refs, 0 unrenderable math |
| Data panel | 764,896 rows, 195 tickers, 2010-01-04 → 2026-09-01 (16.66 yr), 11 sectors |

### The honest empirical summary

*Established:* the causal rolling spread reverts (ADF *p* ≈ 0, half-life ≈ 9.4 bars vs.
≈ 171 for a static 16-year hedge); out-of-sample IC 0.12–0.21; ridge and a low-noise Kalman
hedge beat rolling OLS net of cost (0.09–0.11 vs 0.05) mainly by cutting turnover; linear
beats GBM/MLP out of sample; RESET does not reject linearity on the main basket
(*p* = 0.376); HAC inflates the naive *t*-statistics 3–4× (CRM 29.5 → 7.3).

*Not established:* any tradeable magnitude. At 10 bps the single-basket strategy's cost
drag is 2.39 %/yr against a 3.02 %/yr gross return (87 % consumed; break-even ≈ 11 bps);
the four-basket book is net-negative (−0.40 to −0.46); with *M* = 44 audited strategies,
Bonferroni survivors = 0, White RC *p* = 0.868, Hansen SPA *p* = 0.638, best HAC *t* = 1.57
against E[max *t*] = 2.75 under independence (2.18 at the observed *M*_eff = 10.7),
DSR = 0.305, and the required sample for 95 % power is 22.6 years uncorrected / 77.8 years
Bonferroni-corrected. No pairwise Diebold–Mariano test shows superiority (|DM| < 1.3,
*p* > 0.22).

## 4. File map of what the last session added or changed

```
report/report.tex                          root: preamble, macros, \input of 29 chapters
report/chapters/00_frontmatter.tex         abstract, how to read, notation
report/chapters/01..11_*.tex               Part I: theory from first principles
report/chapters/20_phase_map.tex           Part II opener: the phase map
report/chapters/21..33_*.tex               Parts 0-13, one chapter each (2-3 parts per ch.)
report/chapters/40,41,42_*.tex             Part III: synthesis, non-goals, conclusion
report/chapters/90_appendix_derivations.tex        12 extended derivations
report/chapters/91_appendix_statkit_validation.tex toolkit validation + all 35 tests
report/chapters/92_appendix_reproducibility.tex    commands, expected output, artifact map
report/chapters/93_appendix_glossary.tex           symbol index, glossary, LoF/LoT
report/latex_to_html.py     LaTeX -> HTML book renderer (2-pass, labels/refs/TOC, mathtext)
report/build_report.py      build driver: regenerate -> PDF (if TeX) -> HTML -> --serve
report/references.bib       + 9 entries added for Part I (Granger-Newbold, Lo, Politis-
                            Romano, Newey-West 1994, Harvey, Artzner, Bailey, Lopez de
                            Prado, Dickey-Fuller 1979b)
scripts/analysis/make_book_tables.py       6 extra LaTeX tables + power/DSR from
                                           results/tables/multtest_pnl.csv
scripts/analysis/compare_multtest.py       patched: also dumps multtest_pnl.csv (T x M)
results/tables/multtest_pnl.csv, power_dsr.csv     new generated artifacts
.gitignore                                 + /report/math/  (HTML math rasterisations)
README.md                                  rewritten as the final deliverable README
```

`report/tables/*.tex`, `report/figures/*`, `report/report.html`, `report/math/` and
`results/**` are git-ignored and regenerated; `report/chapters/*.tex`,
`report/latex_to_html.py` and `report/build_report.py` are tracked source.

## 5. If you continue: prioritised next steps

The book's Chapter 42 has the full table with effort/payoff. In order:

1. **Breadth, not depth.** Pool all 195 names' baskets into one cross-sectional signal
   (the transfer coefficient is what is missing, not the IC). This is the only change that
   can move the *statistical* verdict, because E[max *t*] grows like √(2 log M) while the
   signal grows like √breadth.
2. **Higher frequency.** Daily bars give 3,979 observations; the half-life is ≈ 9 bars, so
   intraday data would raise power by an order of magnitude — but costs then dominate, so
   the cost model must become an impact model first.
3. **A real cost/impact model** (Almgren–Chriss style, calibrated to ADV) instead of the
   current fixed-bps + slippage multiplier.
4. **Regime-conditional allocation** — the book's regime table shows the sign of the
   strategy's return flips across volatility regimes; nothing currently exploits that.
5. **Portfolio-level risk** beyond inverse-vol + caps: a proper covariance estimate
   (Ledoit–Wolf), CVaR optimisation, and the VaR subadditivity failure documented in
   Chapter 9 needs a coherent measure in the book, not just in the appendix.
6. **Port the Kalman filter to C++** — the Python research layer proved the value
   (q = 10⁻⁵ net Sharpe 0.113 vs 0.076 rolling OLS at w=60) but the engine still runs the
   rolling regression.
7. **Fix the resolution artifact** in the latency histogram (Chapter 32): 600 ns p50 with a
   clock resolution that makes the measurement marginal; batch N updates per timing.
8. **Pre-register** the next experiment (hypothesis, M, α, stopping rule) before running it,
   so the multiplicity correction is a formality rather than a post-mortem.
9. **Delisting/suspension edge cases** in the panel (currently: drop and continue, disclosed
   in Chapter 22).
10. **Compile the PDF.** `latexmk -pdf report/report.tex` on a machine with TeX Live; the
    `.tex` has never been compiled by an engine, so expect a first-pass of package-level
    fixes (the HTML renderer is deliberately more forgiving than LaTeX).

### Known cosmetic gaps in the HTML preview

* Floats are placed where they are written, not floated; page breaks are ignored.
* `\listoffigures`/`\listoftables` are generated by the renderer from the build's own
  float list, so they always match the HTML (they will differ slightly from the PDF's).
* Theorem environments are boxes, not amsthm-styled; `proof` ends with □ but has no
  QED-hanging-after-lists logic.
* The sidebar shows parts + sections; the in-page Contents shows subsections too.
