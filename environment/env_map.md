# Verified Environment Map (confirmed 2026-08-22)

Results of live probes. **Re-verify with `environment/check.sh` each session.** Hosts are reachable only if listed below — everything else is assumed blocked.

## Reachable
| Host / tool | Use |
|---|---|
| `pypi.org`, `files.pythonhosted.org` | pip installs (Python stack + cmake) |
| `github.com`, `api.github.com` (5000 req/hr unauthenticated), `codeload.github.com` | **Real market data via repo tarballs** |
| `g++` 12.2, `make`, `git`, `python3` 3.11 | build & scripting |

## Blocked (do not waste time retrying)
- **Market data APIs:** `query1.finance.yahoo.com`, `stooq.com`, `api.nasdaq.com`, `google.com` → **all blocked.** yfinance is installed but its CLI cannot fetch data here.
- **Package infra:** Debian `apt` mirrors, `conda.anaconda.org`, `repo.anaconda.com`, `static.rust-lang.org`, `crates.io`, `codeload.github.com/release-assets`, `raw.githubusercontent.com`, `objects.githubusercontent.com`.
- **Consequence:** no TeX engine (texlive/tectonic) can be installed; no raw Yahoo/stooq pulls at runtime.

## Installed (user-level, in `~/.local` / `~/.local/bin`)
- Python: numpy 2.4.6, pandas 3.0.5, scipy 1.17.1, statsmodels 0.14.6, matplotlib 3.11.1, seaborn 0.13.2, sklearn 1.9.0, joblib, yfinance 1.6.0, reportlab 5.0.1
- Build: cmake 4.4.2, g++ 12.2, make 4.3

## Consequence for design decisions
1. **Real market data ⇒ GitHub-hosted datasets** (see `data_candidates.md`). Verified: `api.github.com/repos/{o}/{r}/tarball/{ref}` tarballs download and extract correctly.
2. **C++ engine must be dependency-free at build time** (Eigen must be vendored, or linear algebra hand-rolled). No `#include` outside repo + toolchain.
3. **Report is canonical LaTeX** that compiles in a normal TeX environment; in-sandbox we ship a self-contained HTML preview (no CDN: math rendered as images via matplotlib). PDF builds wherever a TeX engine exists.
