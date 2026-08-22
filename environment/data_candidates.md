# Real Market Data — Candidate GitHub Sources

**Constraint:** only `github.com` / `api.github.com` / `codeload.github.com` are reachable, so real daily equity OHLCV must come from a GitHub-hosted dataset (downloaded as a repo tarball). Yahoo/stooq/Nasdaq APIs are blocked at runtime.

**Part 1 must:** (1) select one primary source after a QA pass, (2) download via tarball, (3) clean to a uniform schema, (4) document source + caveats + retrieval date, (5) commit only the *pipeline + manifest*, never the raw data.

## Candidate sources (found via GitHub search, 2026-08-22)
| Repo | size | What it offers | Caveats |
|---|---|---|---|
| `muchcreative/SP500-OHLC-Database-From-YF` | 5 MB | **Methodology** to build a 15-yr survivorship-bias-free S&P 500 OHLC DB (collect historical constituents, pull Yahoo history, handle missing) | It is scripts/tutorial, **not** a data dump — use as methodology reference; we cannot run Yahoo here |
| `riazarbi/sp500-scraper` | 114 MB | **S&P 500 constituent history** from Wikipedia/BlackRock-iShares (survivorship-free constituent sets per year) | Constituent *membership*, not price data — combine with a price source |
| `perjmi/spawithdata` | 44 MB | Multi-source OHLC data (`data/` dir; SP500/DOW/NDX/DAX/FTSE) | Index-level data, may lack per-stock panels |
| `Nahomkel/S-P-500-Stocks-Daily` | 1.4 MB | Daily S&P 500 **close** prices, 2011–2014 | Only ~3 yr, close-only, small — likely insufficient alone |
| `KrishnachaithanyaThummala/S-P-500-Stock-Market-Analysis` | 8.3 MB | Daily OHLCV for S&P 500 companies (Yahoo-sourced) + sector info | Span/coverage must be QA'd |

## Recommended strategy (to be executed & finalized in Part 1)
1. **Constituent membership (survivorship-free):** `riazarbi/sp500-scraper` `wikipedia/` + `ishares/` yearly constituent lists → defines the *historical* universe per year (avoids survivorship bias from a today-only list).
2. **Price panel:** pick a GitHub repo containing per-ticker daily OHLCV CSV(s) covering ≥ 10 years for the bulk of the S&P 500; QA for span, coverage, split/dividend adjustment (Adj Close).
3. Merge membership×prices to build the universe panel in `data/processed/`.
4. If a survivorship-bias-free price panel is not available, **use a today-constituent panel and quantify the bias explicitly** (compare index membership count over time; state the survivorship caveat in the report §Data) — the spec permits this if disclosed.

**QA gates for the chosen source:** ≥ 200 tickers; ≥ 10 years daily; no obviously bad prices (negatives/zero-volume spikes); a deterministic checksum manifest committed to `data/manifest/`; retrieval date + git SHA of source recorded in `results/data_summary.csv`.

If no single repo suffices, Part 1 should combine two repos (membership + prices) and document the merge. If GitHub-hosted price panels prove too sparse for 10 yr, fall back to a documented shorter panel + an explicit survivorship/coverage limitation discussion — **do not fabricate data.**
