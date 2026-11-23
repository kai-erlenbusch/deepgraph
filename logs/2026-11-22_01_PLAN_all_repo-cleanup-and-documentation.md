---
log_id: 2026-11-22_01
date: 2026-11-22
type: PLAN
status: executed
scope: [all]
source_artifact: repo_cleanup_and_docs_plan.md
conversation_id: e91adbce-084e-445b-98b1-a434c428c86d
---

# Plan: Repository File/Folder Cleanup & `v0.1.0` Documentation + `AGENTS.md` (`deepgraph` & `arrowtiles`)

## 1. Approved Scope & Review Amendments

1. **Agent Onboarding (`AGENTS.md` Only):** Delete `START_HERE.md` in both repositories and consolidate agent guidance into `AGENTS.md` (no `CLAUDE.md`).
2. **Untracked Disk Cleanup & Sample Preservation:**
   - Remove `deepgraph/public/gaia.arrowtiles` (`30.3 GB` duplicate), `public/test.arrowtiles`, `public/test_categorical.arrowtiles`, and ephemeral caches while preserving `public/gaia_full.arrowtiles` (`31.0 GB`) and `public/lmsys_full.arrowtiles` (`841 MB`).
   - In `arrowtiles/samples/`, preserve `gaia_sample.arrowtiles`, `lmsys_sample.arrowtiles`, `lmsys_full_v01.arrowtiles`, and `lmsys_d7_proportional_25k.arrowtiles` (used by `gpu-vram-lifecycle.spec.ts`, `verify_phase5b.mjs`, and `build_phase4_datasets.py`), removing only the 3 unused `lmsys_d7_*` variants (`~883 MB`).
   - Move `deepgraph/{archive,research}` and `arrowtiles/archive` to `<workspace>/archive/`, and move loose June/July raw logs in `arrowtiles/logs/` into `arrowtiles/logs/old/`.
3. **Phase A (Cleanup, Structured `logs/` Bootstrap & CI):**
   - `deepgraph`: Delete `task.md`, `START_HERE.md`, `src/assets/*`, `tests/memory-leak.spec.ts`, and `.github/workflows/perf-ci.yml`; split `npm run test:e2e` vs `npm run test:e2e:heavy`; bootstrap `logs/README.md`; add Node 22 `.github/workflows/ci.yml` (`main`/`master`).
   - `arrowtiles`: Delete `requirements.txt`, `START_HERE.md`, and `tests/test_legacy_duckdb_extension.py`; remove `!requirements.txt` from `.gitignore`; add `[project.optional-dependencies] dev = ["pytest>=8.0"]`; add Rust + Python `.github/workflows/ci.yml` (`main`/`master`).
4. **Phase B (`AGENTS.md` & `v0.1.0` Documentation):**
   - Author `AGENTS.md` in both repositories.
   - Overhaul `arrowtiles/README.md` with the real 2-stage `run_bucketer` + Rayon Zstd PMTiles v3 architecture, CLI reference (`build`, `validate`, `inspect`), and verified `ArrowTilesReader` methods (`header`, `metadata`, `schema`, `list_tiles`, `read_tile`, `iter_tiles`).
   - Polish `deepgraph/README.md` and `docs/API.md` (documenting `test:e2e` vs `test:e2e:heavy` and the sibling `../arrowtiles/samples/` prerequisite).
