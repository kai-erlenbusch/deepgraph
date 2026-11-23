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
