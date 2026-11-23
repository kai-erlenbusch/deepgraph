---
log_id: 2026-11-22_04
date: 2026-11-22
type: ADR
status: approved
scope: [all]
source_artifact: walkthrough.md
conversation_id: e91adbce-084e-445b-98b1-a434c428c86d
---

# ADR (`D12b` Amended): Split Open-Format (`Apache-2.0`) & Non-Commercial Engine (`CC-BY-NC-4.0` + Commercial) Licensing

## 1. Decision

Before publishing `v0.1.0` to PyPI or npm, lock the split ecosystem licensing model across the two sibling repositories (copyright **Kai Erlenbusch**):

1. **`arrowtiles` (`kai-erlenbusch/arrowtiles`, PyPI `arrowtiles`, `docs/SPEC.md`):**
   - Remains permissively licensed under **Apache License 2.0 (`Apache-2.0`)** (`Copyright 2026 Kai Erlenbusch`).
   - Keeping the `.arrowtiles` `v0.1` wire-format specification, DuckDB + Rust pyramid builder, reader, and validator permissive removes all friction for researchers, open datasets, and engineering teams adopting `.arrowtiles`, while preserving `Apache-2.0`'s explicit patent-retaliation shield and `NOTICE` attribution.
2. **`deepgraph` (`kai-erlenbusch/deepgraph`, npm `deepgraph`):**
   - Licensed under **Creative Commons Attribution-NonCommercial 4.0 International (`CC-BY-NC-4.0`)** (`LICENSE`, `"license": "CC-BY-NC-4.0"` in `package.json`) free of charge for non-commercial use (students, academic research, nonprofits, and personal projects).
   - **Commercial License Required:** Any use primarily intended for commercial advantage or monetary compensation—including SaaS products, paid applications, consulting deliverables, and internal business tools at for-profit organizations—requires a separate commercial license via **`erlenbusch@uchicago.edu`** (`NOTICE`, `README.md`, `docs/API.md`).
3. **Clean Pre-Release History on `deepgraph`:**
   - Replace `kai-erlenbusch/deepgraph` `main` prior to `v0.1.0` with a clean `CC-BY-NC-4.0` initial commit (and remove stale pre-release branches/runs) so no pre-release `Apache-2.0` commit remains publicly check-outable.
   - Maintain strict clean-room isolation in both repositories (100% original code by Kai Erlenbusch plus permissive `MIT` / `Apache-2.0` / `BSD` runtime dependencies; zero third-party `CC BY-NC-SA` code) so Kai Erlenbusch holds sole copyright ownership for commercial licensing.
