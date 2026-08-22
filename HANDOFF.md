# HANDOFF — Next session: **Part 1 — Data Acquisition & Universe Construction**

> Read this entire file before doing anything. This is the briefing for the next autonomous session.

---

## 1. Who you are / what you're doing
You are continuing an incremental project: **Multi-Asset Statistical Arbitrage** (basket stat-arb via regularized regression, nonlinear extensions, and a low-latency C++ engine). **Part 0 is complete** (this session). Your job is to complete **Part 1: Data Acquisition & Universe Construction**, then write a new `HANDOFF.md` for Part 2.

Work **only** on branch `arena/01a029cc-test-project`. Commit and push **only** to that branch.

---

## 2. The full spec & master plan
- **`PLAN.md`** — the master plan: 13 parts, cross-cutting rules, exit gates. Re-read it.
- The full problem statement (13 sections of requirements) is not in a file — it was given in the conversation. The master plan encodes it. If in doubt about a requirement, consult `PLAN.md` and the report skeleton `report/report.tex` (which mirrors all required report sections).

---

## 3. What Part 0 delivered (verified working)
- **Master plan** (`PLAN.md`, 13 parts).
- **Environment recon + tooling** (`environment/`):
  - `setup.sh` (idempotent bootstrap), `check.sh` (exit-gate check, **currently 10/10 PASS**), `requirements.txt`, `env_map.md` (verified reachability matrix), `data_candidates.md` (data-source candidates).
- **C++ scaffold** (`src/CMakeLists.txt`, `src/main.cpp`) — compiles, runs `statarbsim 0.1.0`.
- **Report skeleton** (`report/report.tex` = canonical LaTeX with all 13 required sections; `report/references.bib` with real refs; `report/build_report.py` = single build driver that regenerates figures, compiles LaTeX when possible, always emits `report/report.html` preview).
- **Directory structure** per spec §5.2 with `.gitkeep` placeholders.
- **Top-level `README.md` + `Makefile` + `.gitignore`.**

### Verified environment constraints (do NOT waste time re-testing blocked hosts)
- **Reachable:** pypi.org, files.pythonhosted.org, github.com, api.github.com (5000 req/hr), codeload.github.com (tarballs).
- **BLOCKED:** all market-data APIs (Yahoo query1, stooq, api.nasdaq.com, google), Debian apt mirrors, conda, rust/crates.io, GitHub release-assets, raw.githubusercontent.com.
- **No TeX engine installable** → report is canonical LaTeX; in-sandbox deliverable is the HTML preview. `report.pdf` compiles wherever a TeX engine exists.
- pip is PEP-668: **always** use `python3 -m pip install --user --break-system-packages`.
- Toolchain: g++ 12.2 + make + cmake 4.4.2 (in `~/.local/bin`; add to PATH or use full path).

---

## 4. YOUR TASK — Part 1: Data Acquisition & Universe Construction

### 4.1 Goal
Produce a **real, documented, QA'd equity dataset** + a defined universe, enough to support a basket stat-arb study (spec §4.1): ≥ 200 liquid US equities, ≥ 10 years of daily data, corporate-action-adjusted, survivorship bias addressed or disclosed.

### 4.2 Data source (this is the crux)
**Real market data must come from a GitHub-hosted dataset** (Yahoo/stooq/Nasdaq are blocked at runtime). Download as a repo tarball:
```
curl -L "https://api.github.com/repos/{owner}/{repo}/tarball/{ref}" -o /tmp/src.tar.gz
```
This mechanism is verified working.

**Recommended plan (from `environment/data_candidates.md`):**
1. **Constituent membership (survivorship-free):** `riazarbi/sp500-scraper` has Wikipedia + BlackRock-iShares yearly S&P 500 constituent lists. Use these to define the *historical* universe per year (avoids using only today's members).
2. **Price panel:** find a GitHub repo with per-ticker daily OHLCV CSVs covering ≥ 10 years for most of the S&P 500 (see candidates: `KrishnachaithanyaThummala/S-P-500-Stock-Market-Analysis`, `perjmi/spawithdata`, `Nahomkel/S-P-500-Stocks-Daily`, and search for more — e.g. query the GitHub search API for "SP500 OHLC daily CSV", "S&P 500 daily data", "yahoo finance historical stock data"). QA for span, coverage, and split/dividend adjustment (Adj Close).
3. **Merge** membership × prices into a clean universe panel.
4. **Disclose honestly** whatever limitations you find. If no survivorship-free price panel exists, use a today-constituent panel and **quantify** the survivorship bias (e.g., how many current members have < 10 yr of history; contrast with historical membership counts). The spec explicitly permits a survivorship-biased dataset **if quantified and disclosed**.

**Do NOT fabricate data.** If you cannot find a ≥ 10-yr panel on GitHub, use the best documented available and make the coverage/span limitation explicit in the report — a clear negative is a success; a fake is a failure.

### 4.3 Deliverables for this Part
1. `scripts/data/download.py` — downloads the chosen source(s) to `data/raw/` (git-ignored), records a **manifest** (source URL, git SHA of source repo, retrieval date, file checksums) in `data/manifest/`.
2. `scripts/data/clean.py` — cleans to a uniform schema (columns: `date, open, high, low, close, adj_close, volume, ticker`), handles corporate actions (use Adj Close; document methodology), builds the universe membership by year.
3. Output: parquet/CSV in `data/processed/` (git-ignored).
4. `results/tables/data_summary.csv` (or JSON) — universe size by year, date span per name, price/return summary stats, missing-data stats.
5. **Report update:** fill in `report/report.tex` **§Data** (Section 3) with sources, cleaning, adjustment methodology, universe description, survivorship analysis, and a summary-statistics table **regenerated from results** (wire it through `report/build_report.py` → `scripts/plotting`/`scripts/data`). Update the HTML preview via `report/build_report.py`.
6. Document the full dataset provenance in `data/README.md`.

### 4.4 Exit gate (must all pass before you're done)
- [ ] ≥ 200 tickers and ≥ 10 years of daily data (or an explicit, documented lesser span with reasons).
- [ ] `data/manifest/` has a deterministic provenance manifest (source, SHA, date, checksums).
- [ ] `data/processed/` is loadable by a small Python check; a `scripts/data/qa.py` validates no negative prices, sane volumes, no duplicate dates, no gaps beyond documented rules.
- [ ] Corporate-action adjustment methodology documented.
- [ ] Survivorship bias **addressed or disclosed+quantified**.
- [ ] `report/report.tex` §Data updated with real content; `report/build_report.py` runs and regenerates the preview; figures/tables come from `results/`, not hand-typed.
- [ ] `environment/check.sh` still passes (all 10 checks) — and any new checks you add also pass.
- [ ] Raw data NOT committed (git-ignored); only scripts + manifest + small tracked summaries committed.
- [ ] `git status` clean after commit; new `HANDOFF.md` written for Part 2.

### 4.5 Hints / pitfalls
- **GitHub API rate limit** is 5000 req/hr unauthenticated — plenty, but don't hammer the contents API for thousands of files. Prefer one tarball download over N API calls.
- Big repos: `riazarbi/sp500-scraper` is ~114 MB tarball; a full price-panel repo may be large. Watch the 2-core/3.8 GB RAM. Download to `data/raw/`, process with pandas in chunks if needed.
- Keep processed data lean (daily OHLCV for ~500 × ~3000 days is trivially small — ~1.5M rows).
- **Splits/dividends:** use `adj_close` if present; if only raw OHLCV, document that prices are unadjusted and compute total-return-free analysis, or derive a simple adjustment from `adj_close`/`close` ratios. Be explicit.
- The report's **§Data table** must be auto-generated. Set up a tiny convention now: `scripts/plotting/` writes figures to `results/figures/`, `scripts/analysis/` writes tables to `results/tables/`, and `report/build_report.py` calls them. Keep this convention for all future Parts.
- **Future-proof the schema:** a consistent `date, ticker, OHLCV, adj_close` long/wide format now saves pain in Parts 3–13.

---

## 5. How to verify / handoff ritual (same every Part)
1. Implement → run your new checks → run `environment/check.sh` (must pass) → update report → commit → push to `arena/01a029cc-test-project`.
2. Rewrite `HANDOFF.md` (overwrite this file) for **Part 2: Statistical Foundation Toolkit**, including: what you did, decisions/assumptions you made, any data caveats, the exact environment state, and the next Part's task + exit gate.
3. Do **not** delete or restructure existing files unless the plan calls for it; keep the conventions (paths, build driver, report regeneration).

Good luck — take the data-quality section seriously; it is the foundation every later number rests on.
