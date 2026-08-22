# Data

**Placeholder — populated in Part 1.** This directory holds raw and processed market data.

- `raw/` — original downloads from the chosen source (git-ignored).
- `processed/` — cleaned, uniform-schema panels (git-ignored).
- `manifest/` — provenance manifests (source URL, source repo SHA, retrieval date, file checksums). `*.md` is tracked; `*.json` is git-ignored (regenerable).

## Provenance / reproducibility rule
Every raw file must be traceable to its source via a manifest entry. The download and cleaning pipeline lives in `scripts/data/`. Never commit raw market data or large binaries.
