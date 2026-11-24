---
log_id: 2026-11-23_03
date: 2026-11-23
type: PLAN
status: executed
scope: [all]
source_artifact: step1_v011_and_step2_crucible_plan.md
conversation_id: 81e56273-50a1-47ea-a76f-69cec1757cbc
---

# Amended Step 1 (`v0.1.1` Hardening) & Step 2 (`1.35B` Gaia Hardware Crucible) Implementation Plan

## 1. Objective & Gate Sequencing

Execute a strict two-stage stabilization and hardware validation pass across **`arrowtiles`** (`0.1.1`) and **`deepgraph`** (`0.1.1`) with **zero `docs/SPEC.md` wire-format breaks** before advancing to `pydeepgraph`, Marimo, UW Mosaic, or `.arrowtiles v0.2`:

1. **Step 1 — `v0.1.1` Hardening (`arrowtiles 0.1.1` & `deepgraph 0.1.1`):**
   - Resolve all confirmed `P1` and high-leverage `P2` findings from `2026-11-23_02` while incorporating all **16 mandatory technical corrections** identified during engineering review (`review_of_step1_v011_and_step2_crucible_plan.md`).
2. **Step 2 — `1.35B` Real Gaia DR3 Hardware Crucible & Dual-GPU / WAN Benchmark Suite:**
   - Run the progressive `2.84M` LMSys → `100M` Gaia → `500M` Gaia → **`1,348,176,623` (`1.35B`) full real Gaia DR3** build & validation ladder with all temporary scratch pinned to `../_scratch_crucible` on the data drive (`memory_limit="24GB"`, `0 B` spilled to the OS drive), followed by dual-GPU (`Intel xe-lpg` iGPU + `NVIDIA RTX 5090` dGPU when idle) and `50 ms / 50 Mbps` WAN benchmarks.

---

## 2. Step 1 Scope & Incorporated Engineering Corrections (`16 / 16`)

### A. `arrowtiles` (`0.1.1`)
1. **Rust Engine (`arrowtiles-engine/src/lib.rs`, `Cargo.toml`):**
   - Reserve `"tiling"` and `"required_features"` in `RESERVED_METADATA_KEYS`; enforce `MAX_COMPRESSED_TILE_BYTES = 64 MiB` in `flush_chunk_buffer`.
   - Map `Utf8View -> Utf8` and `BinaryView -> Binary` in `ArrowTilesPacker::new` (preserving field metadata) and cast *before* `RecordBatch::try_new` (`Pushback #1`).
   - Replace global Rayon pool mutation with per-instance scoped `rayon::ThreadPool` (`rayon_threads: Option<usize> = None` appended at the end of `run_bucketer` and `ArrowTilesPacker::new`, wrapping both Pass 1 and Pass 2 `into_par_iter()` blocks; `Pushback #8`).
   - Take ownership of `ArrowArrayStream` before writer checks in `process_batch`, add `py.check_signals()`, release GIL via `py.allow_threads(...)` around DuckDB batch pulls and Parquet partition writes, use atomic `std::fs::rename` in `finalize()`, and pin `pyo3-log = "0.12"`.
2. **Builder, Validator, Packaging & CI (`arrowtiles/builder.py`, `validator.py`, `pyproject.toml`, `.github/workflows/*`):**
   - Fix `BOOLEAN` / non-string `categorical_cols` (`CAST({qc} AS VARCHAR)` in preflight & lookup table ordered by `MIN({qc}) ASC, lbl ASC`).
   - Enforce `COUNT(DISTINCT id_col)` uniqueness in `build()`, prevent unrequested `sort_col` schema leakage in `build_generic()` via internal `"__at_sort_src"` + `_priority_column_name` (`Pushback #10`), guard narrow-span `Float64 -> Float32` downcasts (`span < maxAbs * 1e-4`), set `SET TimeZone = 'UTC'` + UTC epoch-ms `stats` (`epoch(qc) * 1000.0`) on temporal columns (`Pushback #9`), isolate `temp_dir=None` via per-instance `TemporaryDirectory`, verify partition sizes in `manifest.json`, free `reader_pass1` immediately after FFI (`Pushback #15`), and push down `row_limit` via `WHERE (off.base_offset + src.file_row_number) < N` (`Pushback #14`).
   - Enforce strict integer/number checks (`not isinstance(v, bool)`), mandatory `elif` error branches, cross-tile `null_count` verification, `with mmap.mmap(...)`, and guarded `all_tiles` in `validator.py`.
   - Migrate `pyproject.toml` to PEP 639 (`license = "Apache-2.0"`, `license-files = ["LICENSE", "NOTICE"]`), remove duplicate `assets/screenshot{1..3}.png`, guard `tests/conftest.py` wheel testing + deterministic synthetic CI fixtures (`0 skipped`), and add `test-wheels` gates across Linux/Windows/macOS (`Pushback #7`).

### B. `deepgraph` (`0.1.1`)
1. **`@uwdata/flechette@^2.8.1` Worker Codec & Zero Main-Thread Arrow (`src/workerCodec.ts`, `src/pmtiles.worker.ts`, `src/PMTilesClient.ts`, `src/lib/DatasetSession.ts`):**
   - Replace `apache-arrow` with `@uwdata/flechette` (`{ useBigInt: true, useDate: false }`), supporting `Utf8View`, `BinaryView`, `Decimal128/256`, `Timestamp`, `Date32/64`, `Float16`, legacy `v0.0` `Dictionary` integer codes (`chunk.values` + `chunk.validity`), 8-byte-padded `Uint16` subarray slicing, `BigInt`-first `colOrigins` subtraction, and RFC 8878 Zstd `> 256 MiB` frame-header rejection (`Pushbacks #2, #4`).
   - Delegate `PMTilesClient.getSchema()` to the worker pool (`0 B` main-thread Arrow), require valid `arrow_schema` on `v0.1`, clamp `maxTileCache >= 32`, fix `setRequestedColumns` initial-load check, and export `CustomRangeSource` / `TileSource` (`Pushback #5`).
2. **Single-File Standalone Worker & Packaging (`vite.standalone.config.ts`, `scripts/emit_dts.mjs`, `scripts/verify_consumer_dts.mjs`):**
   - Inline `pmtiles.worker-*.js` as a lazy cached `Blob` URL inside `dist/standalone.js` (`enforce: 'post'`, `generateBundle: { order: 'post' }`, deleting only from the in-memory bundle so `dist/index.js`'s CSP-friendly external worker remains intact; `Pushbacks #3, #6`).
3. **Coordinates, `1D RTO`, Shaders & WebGPU Lifecycle (`ViewState.ts`, `ComputeRasterizer.ts`, `PickingLayer.ts`, `HighlightOverlay.ts`, `TileGpuStore.ts`, `Renderer.ts`, `RenderLoop.ts`, `ViewerLayer.ts`):**
   - Fix `transform: 'literal'` (`isXSpatial`/`isYSpatial` strictly `x_u16`/`y_u16`, unclamped `Float64`-anchored RTC `(effVal - camAnchorData) * worldScale`, relative `baseZoom` LOD/sizing; `Pushbacks #12, #13`).
   - Fix spatial `flipY: true` `dataYSpan`, preserve `'optical'` pipeline on domain updates, merge `{ ...decoded.attributes, ...enriched }` in `ViewerLayer`, handle `GPUDevice.lost`, validate `PickingLayer` slot snapshots across both `mapAsync` awaits, support categorical chart axes + `rowIdBigInt` deduplication in `HighlightOverlay`, gate Optical `splat_optical_pixel` at `< 96` atomic hits, upgrade Categorical Mode 1 to `Q = 12.0` (`64/64` unique palette colors, `wRgbSum < 5120u` = `2,560` center hits, `0.0000%` shimmer), guard all 6 `dummyStorageBuffer` bindings, move Gaia `teff <= 0` to `metricType: 6` (`Pushback #11`), and add an occluded-window `RenderLoop` progression guard.
4. **CI & npm OIDC Release Workflows (`.github/workflows/ci.yml`, `.github/workflows/release.yml`):**
   - Deterministic `npm ci`, `npm run typecheck`, `npm run build:lib`, `npm run verify:types`, `npm run test:unit`, and tag-triggered `npm publish --provenance --access public` (`id-token: write`).

---

## 3. Step 2 Crucible & Verification Protocol

1. **Progressive Build & Validation Ladder (`scripts/bench/run_v011_crucible.py`):**
   - Rung 1 (`2.84M` Wide LMSys) → Rung 2 (`100M` Gaia) → Rung 3 (`500M` Gaia + `< 220 GiB` extrapolated scratch guard) → Rung 4 (`1,348,176,623` full real Gaia DR3 → `public/gaia_v011_full.arrowtiles`, preserving `public/gaia_full.arrowtiles` untouched), plus `row_id > 1,000,000,000` ground-truth Parquet cross-check.
2. **Dual-GPU & WAN Playwright Crucible (`tests/gpu-vram-lifecycle.spec.ts` + `tests/crucible-1b.spec.ts`; `Pushback #16`):**
   - Pass 1 on Intel `xe-lpg` iGPU (`--force_low_power_gpu`); Pass 2 on NVIDIA RTX 5090 (`DEEPGRAPH_ALLOW_DGPU=1`) strictly gated on `nvidia-smi` idle check (`memory.used <= 512 MiB`, `utilization.gpu <= 5%`).
   - Record verified results in `docs/BENCHMARK_v011_crucible.md` and align `README.md` in both repositories.
