---
log_id: 2026-11-23_04
date: 2026-11-24
type: WALKTHROUGH
status: executed
scope: [all]
source_artifact: BENCHMARK_v011_crucible.md
conversation_id: 08f564f2-95c1-4b5f-b446-5fb13d4d48ab
---

# Step 1 (`v0.1.1` Hardening) & Step 2 (`1.35B` Gaia Hardware Crucible) Walkthrough

## 1. Summary of Executed Changes

### A. `arrowtiles` (`0.1.1`)
1. **Rust Engine (`arrowtiles-engine/src/lib.rs`, `Cargo.toml`):**
   - Reserved `"tiling"` and `"required_features"` in `RESERVED_METADATA_KEYS`, enforced the `64 MiB` compressed-tile `SPEC.md` ceiling, cast `Utf8View -> Utf8`, `BinaryView -> Binary`, and negative-scale `Decimal128/256 -> Float64` *before* `RecordBatch::try_new`, wrapped both Pass 1 Stage A and Pass 2 tile packing in a per-instance scoped `rayon::ThreadPool`, extracted `take_ffi_stream` prior to fallible argument validation, added `py.check_signals()` + `py.allow_threads(...)`, and switched Pass 2 output replacement to atomic `std::fs::rename` (`-10.92 MB` duplicate `assets/screenshot{1,2,3}.png` removed; PEP 639 `pyproject.toml` + `[tool.maturin] exclude` + isolated `test-wheels` `CWD` guard hardened).
2. **DuckDB Builder & Validator (`arrowtiles/builder.py`, `arrowtiles/validator.py`):**
   - Pinned session `SET TimeZone = 'UTC'`, replaced `duckdb_tmp` directory reuse with per-build `tempfile.TemporaryDirectory`, supported `BOOLEAN` and numeric `categorical_cols` (`ORDER BY MIN({qc}) ASC, lbl ASC`), enforced `COUNT(DISTINCT id_col)` in `build()`, renamed internal sort column to `"__at_sort_src"` while preserving `_priority_column_name`, guarded narrow-span (`span < max(|min|, |max|) * 1e-4`) and constant `> 2^24` inexact `Float64` columns against `Float32` downcast, emitted UTC epoch-ms `stats` for temporal columns, pushed `row_limit` into `WHERE (off.base_offset + src.file_row_number) < N`, released `reader_pass1`/`reader_pass2` handles before cleanup, and hardened `validator.py` (`bool` exclusion, mandatory `elif` validation, cross-tile `null_count`, `with mmap.mmap(...)`).
3. **Verification Gates:**
   - `cargo clippy --locked -- -D warnings`, `maturin develop --release --locked`, `maturin sdist` (`67.6 KB`), and `pytest -v -rs -p no:cacheprovider -W error::UserWarning` (`67 passed, 0 skipped`).

### B. `deepgraph` (`0.1.1`)
1. **Flechette Worker Migration, Self-Contained `standalone.js` & 1D RTO (`src/workerCodec.ts`, `src/pmtiles.worker.ts`, `src/PMTilesClient.ts`, `vite.standalone.config.ts`):**
   - Replaced `apache-arrow` with `@uwdata/flechette@^2.8.1` (`0 B` main-thread Arrow parser weight; `dist/index.js` `51.00 kB` gzip; single-file `dist/standalone.js` `198.96 kB` gzip with `Blob`-inlined worker verified in a `blob:` `anywidget` harness with `/assets/*` blocked `404`).
   - Added `Utf8View`, `Decimal128` (including negative scale `< 0`), `Date32`/`Date64`, `Timestamp(s/ms/us/ns)`, `Float16`, `Dictionary`, `BigInt > 2^53 - 1`, 3-row padded `Uint16` alignment, `1D RTO` `colOrigins` (`BigInt` subtraction prior to `Float32`), RFC 8878 `256 MiB` Zstd bomb header guard, `CustomRangeSource`, `FastFetchSource.abortPrefetch()`, `DatasetSession.open()` `AbortSignal` supersession teardown, `maxTileCache >= 32`, and immediate reclamation of off-screen un-uploaded CPU column buffers (`retainedTileCpuBytes === 0`).
2. **Renderer, Shaders, Picking & Overlay (`src/core/*`, `src/lib/*`, `src/demo/gaiaPreset.ts`):**
   - Fixed `transform: 'literal'` using zero-cancellation `Float64` RTC `(effVal - camAnchorData) * signedWorldScale` + relative `baseZoom` LOD/sizing, fixed spatial `flipY: true` `dataYSpan`, preserved `'optical'` pipeline across domain updates, merged partial `resolveRowAttributes` sidecars, wired `GPUDevice.lost`, validated `PickingLayer` slot identity across both `mapAsync` awaits, fixed `HighlightOverlay` categorical axes + `rowIdBigInt` deduplication, upgraded Categorical Mode 1 to `Q = 12.0` (`64/64` unique palette colors, `wRgbSum < 5120u`), upgraded Optical Mode 0/3 `splat_optical_pixel` to flux-proportional `ticket = ceil(cMax / 16) <= 3840u` (`40×` faint-star capacity, `0.0000%` shimmer on 80+ SM discrete GPUs), guarded all 6 `dummyStorageBuffer` bindings, moved Gaia `teff <= 0` filter to `metricType: 6`, and added an idle-aware occluded-window progression guard in `RenderLoop.ts`.

---

## 2. Step 2 Progressive `1.35B` Gaia Hardware Crucible & Dual-GPU Results

Full benchmark tables and reproduction commands are archived in sibling `arrowtiles` (`docs/BENCHMARK_v011_crucible.md`):
- **Progressive Out-of-Core Build (`memory_limit="24GB"`, `0 B` spilled to OS drive):**
  - Rung 1 (`2,835,490` LMSys rows): `9.35s` (`151.32 MiB`, `57` tiles, `validate_file` `0.85s` `PASS`).
  - Rung 2 (`100,000,000` Gaia stars): `155.46s` (`643,273 rows/s`, `2.44 GiB`, `3.77 GiB` peak scratch, `validate_file` `25.15s` `PASS`).
  - Rung 3 (`500,000,000` Gaia stars): `2,191.65s` (`228,138 rows/s`, `11.98 GiB`, `18.18 GiB` peak scratch, `validate_file` `148.02s` `PASS`).
  - Rung 4 (`1,348,176,623` full local Gaia DR3 stars -> promoted to `public/gaia_full.arrowtiles`): **`15,132.05s` (`4h 12m`, `89,094 rows/s`, `31.81 GiB`, `36,931` tiles, `48.27 GiB` peak scratch, `27.59 GiB` peak process commit, `validate_file` `485.95s` `PASS`)**.
- **Dual-GPU WebGPU Crucible (`Intel xe-lpg` iGPU + `NVIDIA RTX 5090` `blackwell` dGPU):**
  - Cold Loopback TTFF: `389.4 ms` (iGPU) / `742.0 ms` (dGPU); Simulated `50 ms / 50 Mbps` WAN Cold TTFF: **`687.5 ms` (iGPU) / `734.1 ms` (dGPU)** across `4` HTTP `206` Range requests (`2.53 MiB`, rendering `78,959` `mag <= 14.0` stars from the `79,032`-row `Z=0/0/0` root tile).
  - Steady-State `1280×720` FPS: `67.4–179.6 FPS` (`0.153–0.234 ms` GPU dispatch) on Intel `xe-lpg` iGPU (`67.4 FPS` at `7.22M` LMC stars); **`174.8–178.9 FPS` (`0.097–0.266 ms` GPU dispatch) across all 5 regimes up to `7,224,142` active stars on NVIDIA RTX 5090**.
  - Stationary 2-Frame Shimmer: **`0` changed pixels (`0.0000%`, `Δ=0`)** across Modes `0`, `1`, and `2` (`845k–850k` lit pixels) on **both** GPUs.
  - `row_id > 10^9` GPU `pickAt()` Parity: Exact `rowId` and attribute parity against raw `s3_cache/batch_{027,032,033}.parquet` (`1,202,742,788`, `1,228,887,815`, `1,060,650,830`; `11.5 ms` iGPU / `5.8 ms` dGPU warm LRU pick).
  - 5-Gate VRAM Teardown: **`liveDevices = 0`, `liveBuffers = 0`, `liveBytes = 0`** (`349.4 MB / 349.4 MB` in `crucible-1b` and `835.6 MB / 835.6 MB` in the 5-gate VRAM lifecycle suite).
