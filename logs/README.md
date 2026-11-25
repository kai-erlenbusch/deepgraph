# Deepgraph Engineering, Research & Audit Logs (`logs/`)

Institutional context, architecture decisions, read-only audits, implementation plans, and post-execution walkthroughs for **Deepgraph (`deepgraph`)** are tracked here in Git while binary/runtime drill artifacts and pre-revival raw transcripts (`logs/old/`, `logs/oct/`, `logs/assets/`) stay gitignored.

Cross-repository revival logs covering both `arrowtiles` and `deepgraph` (`2026-11-19_01` through `2026-11-21_08`) are archived in the sibling [`arrowtiles/logs/`](https://github.com/kai-erlenbusch/arrowtiles/tree/main/logs) index; standalone `deepgraph` logs are indexed chronologically below.

---

## 1. File Naming Convention

```text
YYYY-MM-DD_<seq>_<TYPE>_<scope>_<topic-slug>.md
```

- `YYYY-MM-DD`: ISO-8601 date.
- `<seq>`: Two-digit daily sequence (`01`, `02`, ...) ordered chronologically within that date.
- `<TYPE>`: Controlled uppercase enum: `RESEARCH` | `AUDIT` | `PLAN` | `ADR` | `WALKTHROUGH`.
- `<scope>`: Lowercase target slice (e.g., `all`, `viewer`, `shaders`, `picking`, `packaging`, `graph`).
- `<topic-slug>`: Concise 3–6 word `kebab-case` summary (no spaces).

---

## 2. Mandatory YAML Frontmatter Header

Prepend every `logs/YYYY-MM-DD_*.md` file with:

```yaml
---
log_id: YYYY-MM-DD_01
date: YYYY-MM-DD
type: RESEARCH | AUDIT | PLAN | ADR | WALKTHROUGH
status: draft | pending-approval | approved | executed | superseded
scope: [all]
source_artifact: <original_artifact_filename_if_exported>
conversation_id: <agent_conversation_uuid_if_applicable>
---
```

---

## 3. Chronological Log Index

| Date & Seq | Type | Scope | File | Summary |
| :--- | :---: | :---: | :--- | :--- |
| `2026-11-22_01` | `PLAN` | `all` | [`2026-11-22_01_PLAN_all_repo-cleanup-and-documentation.md`](2026-11-22_01_PLAN_all_repo-cleanup-and-documentation.md) | Two-phase pre-release plan covering dead-file & disk cleanup, chronological `logs/` bootstrap, CI alignment, `AGENTS.md`, and `v0.1.0` documentation overhaul across `deepgraph` and `arrowtiles`. |
| `2026-11-22_02` | `WALKTHROUGH` | `all` | [`2026-11-22_02_WALKTHROUGH_all_repo-cleanup-and-documentation.md`](2026-11-22_02_WALKTHROUGH_all_repo-cleanup-and-documentation.md) | Executed Phase A & B repo cleanup (~32.5 GB disk reclaimed, `logs/` & CI aligned) and `v0.1.0` `AGENTS.md` + `README.md` overhaul (`53 passed` pytest, `3 passed` Intel-iGPU E2E). |
| `2026-11-22_03` | `WALKTHROUGH` | `all` | [`2026-11-22_03_WALKTHROUGH_all_stage-2a-github-renames-ci-and-duckdb2.md`](2026-11-22_03_WALKTHROUGH_all_stage-2a-github-renames-ci-and-duckdb2.md) | Completed `D12c` Stage 2a GitHub renames (`arrowtiles` + `deepgraph`), 5-OS `abi3` wheel/sdist dry-run, `pmtiles` crate pruning, and DuckDB `2.0.1` canonical `NULL`-slot IPC determinism fix (`53 passed`, `0` warnings). |
| `2026-11-22_04` | `ADR` | `all` | [`2026-11-22_04_ADR_all_d12b-split-licensing-cc-by-nc-and-apache2.md`](2026-11-22_04_ADR_all_d12b-split-licensing-cc-by-nc-and-apache2.md) | Amended `D12b` to split ecosystem licensing (`arrowtiles` = `Apache-2.0`, `deepgraph` = `CC-BY-NC-4.0` + commercial license via `erlenbusch@uchicago.edu`, copyright Kai Erlenbusch). |
| `2026-11-22_05` | `WALKTHROUGH` | `all` | [`2026-11-22_05_WALKTHROUGH_all_stage-2b-v0-1-0-public-release.md`](2026-11-22_05_WALKTHROUGH_all_stage-2b-v0-1-0-public-release.md) | Published `arrowtiles` `0.1.0` (`Apache-2.0`, 5 `cp39-abi3` wheels + `sdist`) to PyPI and `deepgraph` `0.1.0` (`CC-BY-NC-4.0`, `10` files) to npm alongside tagged GitHub Releases (`v0.1.0`). |
| `2026-11-23_01` | `RESEARCH` | `all` | [`2026-11-23_01_RESEARCH_all_apple-embedding-atlas-architectural-review.md`](2026-11-23_01_RESEARCH_all_apple-embedding-atlas-architectural-review.md) | Deep architectural review of `apple/embedding-atlas` (`arXiv:2505.06386`, `2504.07285`, `2607.08746`): confirmed `deepgraph`'s `300×` scale & GPU-edge moat and extracted 8 clean-room `v0.2` takeaways (offline watershed/`c-TF-IDF` `metadata.labels`, Deriche IIR KDE + Sobel contours, `in_degree_kcore`, `narwhals`/`anywidget`). |
| `2026-11-23_02` | `AUDIT` | `all` | [`2026-11-23_02_AUDIT_all_post-v0-1-0-audit-verification-and-mosaic-roadmap.md`](2026-11-23_02_AUDIT_all_post-v0-1-0-audit-verification-and-mosaic-roadmap.md) | Live empirical verification of the 25-finding post-`v0.1.0` audit (`23/25` reproduced, `2` corrected on `65 KB` PyPI `sdist` & native PMTiles `2^41` round-trip, narrow-span `Float64` promoted to `P1`) + unified `v0.1.1 → 1.35B Stress Test → pydeepgraph/Marimo/Mosaic → v0.2` roadmap. |
| `2026-11-23_03` | `PLAN` | `all` | [`2026-11-23_03_PLAN_all_step1-v011-and-step2-crucible.md`](2026-11-23_03_PLAN_all_step1-v011-and-step2-crucible.md) | Approved amended Step 1 (`v0.1.1` hardening incorporating all 16 technical review fixes) & Step 2 (`1.35B` Gaia hardware crucible & dual-GPU/WAN benchmark) execution plan. |
| `2026-11-23_04` | `WALKTHROUGH` | `all` | [`2026-11-23_04_WALKTHROUGH_all_v011-hardening-and-1-35b-crucible.md`](2026-11-23_04_WALKTHROUGH_all_v011-hardening-and-1-35b-crucible.md) | Completed `v0.1.1` hardening (`67 passed` pytest, `10 passed` Node unit, Flechette `51.00 kB` gz + Blob `standalone.js`, `3,840`-ticket optical flux gate) and `1,348,176,623`-star Gaia DR3 crucible (`31.81 GiB`, `485.95s` validate, `0.69–0.73s` WAN TTFF, `174.8–178.9 FPS` up to `7.22M` stars, `0.0000%` shimmer on Intel `xe-lpg` & RTX 5090). |
| `2026-11-24_01` | `PLAN` | `all` | [`2026-11-24_01_PLAN_all_step3-pydeepgraph-mosaic-and-ego-arcs.md`](2026-11-24_01_PLAN_all_step3-pydeepgraph-mosaic-and-ego-arcs.md) | Approved gated Step 3A (`deepgraph@0.1.2`: `@binding(10,11)` GPU brush/lasso/cross-filter, 2-pass `unselectedOpacity`, `100k` instanced Bézier `EdgeOverlay`, optional `deepgraph/mosaic` subpath) & Step 3B (`pydeepgraph==0.1.0` + Marimo reactive workflows) implementation plan. |
| `2026-11-24_02` | `WALKTHROUGH` | `all` | [`2026-11-24_02_WALKTHROUGH_all_step3a-v012-filter-mosaic-and-ego-arcs.md`](2026-11-24_02_WALKTHROUGH_all_step3a-v012-filter-mosaic-and-ego-arcs.md) | Completed Step 3A (`deepgraph@0.1.2`) and paused at Review Gate: `0`-storage-buffer `@binding(10,11)` GPU filter/brush/spotlight/lasso, 2-pass context dim, `100k` Bézier `EdgeOverlay`, and `deepgraph/mosaic` (`19/19` unit, `6/6` Intel `xe-lpg` E2E, `0.0000%` shimmer, `0 B` leaked). |
| `2026-11-24_03` | `WALKTHROUGH` | `all` | [`2026-11-24_03_WALKTHROUGH_all_step3b-pydeepgraph-and-marimo-showcase.md`](2026-11-24_03_WALKTHROUGH_all_step3b-pydeepgraph-and-marimo-showcase.md) | Completed & Goldfish-hardened Step 3B (`pydeepgraph==0.1.0` + `deepgraph@0.1.2`): `228.60 kB` gz `widget.js` ESM, reader-refcounted `mmap` HTTP `206` + `<= 512 KiB` `CommRangeSource`, server-validated `selection_sql`, `HUGEINT` hash sum, and `2.84M` LMSys 10-NN + Gaia `1.35B` Marimo showcases (`20/20` Node unit, `12/12` pytest, `7/7` Intel `xe-lpg` WebGPU E2E). |
