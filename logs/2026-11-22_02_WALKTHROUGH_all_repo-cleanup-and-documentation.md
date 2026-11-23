---
log_id: 2026-11-22_02
date: 2026-11-22
type: WALKTHROUGH
status: executed
scope: [all]
source_artifact: walkthrough.md
conversation_id: e91adbce-084e-445b-98b1-a434c428c86d
---

# Walkthrough: Repository File/Folder Cleanup & `v0.1.0` Documentation + `AGENTS.md` (`deepgraph` & `arrowtiles`)

## 1. Summary of Changes

### Phase A — Cleanup, Disk Organization, Structured `logs/` Bootstrap & CI
- **`deepgraph`:**
  - Removed obsolete tracked files: `task.md`, `START_HERE.md`, `src/assets/{hero.png,typescript.svg,vite.svg}`, `tests/memory-leak.spec.ts` (superseded by `tests/gpu-vram-lifecycle.spec.ts`), and `.github/workflows/perf-ci.yml`.
  - Split `package.json` scripts into fast `< 5 MiB` sample-only `npm run test:e2e` (`library-embed.spec.ts`, `phase8b-hdr-density.spec.ts`, `picking-layer.spec.ts`) and explicit `npm run test:e2e:heavy` (`gpu-vram-lifecycle.spec.ts`).
  - Upgraded `.gitignore` (`logs/*`, `!logs/README.md`, `!logs/20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]_*.md`) keeping `logs/oct/` and `logs/old/` gitignored, created `logs/README.md`, and added Node 22 `.github/workflows/ci.yml` (`main` / `master`).
  - Reclaimed `~30.3 GB` on disk (`public/gaia.arrowtiles` duplicate + `public/test*.arrowtiles` + ephemeral caches removed while preserving `public/gaia_full.arrowtiles` `31.0 GB` and `public/lmsys_full.arrowtiles` `841 MB`) and relocated `archive/` + `research/` to `<workspace>/archive/deepgraph-2026-07/`.
- **`arrowtiles` (`fe386af`):**
  - Removed obsolete tracked files: `requirements.txt` (cleaned `!requirements.txt` from `.gitignore`), `START_HERE.md`, and `tests/test_legacy_duckdb_extension.py`.
  - Added `[project.optional-dependencies] dev = ["pytest>=8.0"]` in `pyproject.toml` and multi-OS (`ubuntu-latest`, `windows-latest`) Rust + Python `.github/workflows/ci.yml` (`main` / `master`).
  - Reclaimed `~891 MB` of unused scratch/benchmark files while preserving `samples/lmsys_d7_proportional_25k.arrowtiles` (`291 MB`), `gaia_sample.arrowtiles`, `lmsys_sample.arrowtiles`, and `lmsys_full_v01.arrowtiles`; relocated `archive/` to `<workspace>/archive/arrowtiles-2026-07/`; and moved 27 loose June/July raw logs into `logs/old/`.

### Phase B — `AGENTS.md` & `v0.1.0` Documentation Overhaul
- Created `AGENTS.md` in both `deepgraph` and `arrowtiles` covering sibling repository boundaries, clean-room licensing rules, module maps, WebGPU/spec/determinism invariants, dual-bundle worker/WASM rules, sample fixture & GPU coexistence rules, and chronological `logs/` conventions.
- Rewrote `arrowtiles/README.md` with the deterministic 2-stage `run_bucketer` + Rayon Zstd PMTiles v3 architecture (removing all stale `INSTALL lindel`, `requirements.txt`, and `bucketed_temp.parquet` instructions), full CLI docs (`build`, `validate`, `inspect`), and verified `ArrowTilesReader` Python API examples (`header`, `metadata`, `schema`, `list_tiles`, `read_tile`, `iter_tiles`).
- Updated `deepgraph/README.md` and `docs/API.md` with `npm run test:e2e` vs `npm run test:e2e:heavy` commands, the sibling `../arrowtiles/samples/` prerequisite, and links to `AGENTS.md`.

---

## 2. Verification Results

- **`arrowtiles` Pytest Suite (CPU-Only):** **`53 passed in 24.55s`** (`0 skipped`).
- **`arrowtiles/README.md` Python API Smoke Test (`python -c`):** **`README_PYTHON_API_SMOKE_OK`** (`tiles = 146`, `iter_count = 146`, `root_rows = 19,768`, `cols = ['row_id', 'model', 'num_of_tokens', 'x_u16', 'y_u16', 'z', 'x_decoded', 'y_decoded']`).
- **`arrowtiles` CLI (`validate` & `inspect`):** `[PASS] samples\lmsys_sample.arrowtiles`.
- **`deepgraph` Typecheck, Builds & Consumer Verification:** `npm run typecheck`, `npm run build:lib` (`46.62 kB` gzip bundler / `99.97 kB` gzip standalone), `npm run verify:types`, and `npm run build:demo` all exited `0`.
- **`deepgraph` Sample-Only Playwright E2E (`npm run test:e2e`, Intel `xe-lpg` iGPU):** **`3 passed in 1.1m`** (`liveDevices = 0`, `liveBuffers = 0`, `liveBytes = 0`, `uncapturedErrors = []`).
- **Stale-Reference Grep Audit:** `0` stale hits across `README.md`, `AGENTS.md`, `docs/API.md`, `package.json`, and `pyproject.toml`.
