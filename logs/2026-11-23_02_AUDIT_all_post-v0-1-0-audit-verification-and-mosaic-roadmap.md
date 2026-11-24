---
log_id: 2026-11-23_02
date: 2026-11-23
type: AUDIT
status: approved
scope: [all]
source_artifact: audit_verification_and_roadmap_assessment.md
conversation_id: e91adbce-084e-445b-98b1-a434c428c86d
---

# Independent Verification & Assessment of Post-`v0.1.0` Audit & Strategic Pivot

## 1. Summary of Empirical Verification

Independently tested all 25 findings (`14` in `deepgraph`, `11` in `arrowtiles`) and Section 5 architectural constraints from `post_v0_1_0_audit_and_strategic_pivot.md` via live Node (`apache-arrow@21.1.0`, `pmtiles`), Python (`arrowtiles`, `pyarrow`, `duckdb`), Rust (`arrowtiles._core`), and PyPI tarball inspection:

- **Confirmed Verbatim (`23 / 25`):**
  - **`deepgraph`:** `transform: 'literal'` ignoring custom `field` (`isXSpatial = true`), spatial `flipY: true` double-negating `dataYSpan` + incomplete `RenderLoop` `camHash`, `Timestamp_us` (`1,000×`) / `Decimal128` / `Utf8View` worker mismatches, missing `GPUDevice.lost` + `PickingLayer` `await mapAsync` slot-reuse race, `dist/standalone.js` external relative `Worker` URL throwing `TypeError: Invalid URL` under `blob:` (`anywidget`/Marimo), `apache-arrow` 3-row `Uint16` 8-byte padding (`valuesLength = 4` vs `chunkLength = 3`), `resolveRowAttributes` overwriting retained attributes, Mode 2 `rawC <= 0 && colorMin > 100` heuristic leak, `setRequestedColumns` initial-load abort, `8/8` storage-buffer ceiling, and `256/256 B` `SlotUniforms` saturation.
  - **`arrowtiles`:** `BOOLEAN` `categorical_cols` 100% silent `NULL` wipeout (`[None, None, None, None]` passing `validate_file`), `ArrowTilesBuilder.build()` missing `id_col` uniqueness check + `build_generic` `sort_col` schema leakage, `validator.py` missing `else:`/`not isinstance(v, bool)` guards on 6 integer metadata keys + unguarded `archive_mmap` cleanup, `ArrowTilesPacker` missing `64 MiB` tile guard + `"tiling"`/`"required_features"` in `RESERVED_METADATA_KEYS`, shared `./duckdb_temp` CWD race, and narrow-span `Float64 -> Float32` Unix-epoch collapse (`[1767312000..+9]` collapsing to 1 value — promoted from `P2` to `P1`).
- **Empirical Corrections (`2`):**
  1. **PyPI `sdist` Size (`arrowtiles` P2 #8):** Live `arrowtiles-0.1.0.tar.gz` on PyPI is `64,977 bytes` (`16` files, `0` PNGs), not `21.8 MB` (`assets/*.png` duplicates only bloat Git, not `sdist`).
  2. **PMTiles `g * 2^41 + tid_2d` Compatibility (Section 5 Track 3 — Reconciled Across Peer Review):** Rust `_core.ArrowTilesPacker(..., max_z=2)` and Python `pmtiles` (`Reader.get`) preserve `header.max_zoom = 2` on `tile_id >= 2^41`, whereas JS `PMTiles.getZxy` rejects `z > header.maxZoom` (`21 > 2` -> `undefined`) and `tileIdToZxy` caps at `g < 2731`. Using `pmtiles`'s already-exported `findTile` + `SharedPromiseCache` in a 15-line `PMTilesClient.getTileById(tileId)` helper (paired with per-subspace schemas in `ArrowTilesPacker`) supports all `g = 0..4095` subspaces with zero upstream forks.

---

## 2. Reconciled Three-Agent Architectural Guardrails

1. **`@uwdata/flechette` (`BSD-3-Clause`, `~16 kB` gz) Replaces `apache-arrow` JS (`DG-P1-3`):** Fixes `Utf8View`, `Decimal128`, `Timestamp`, `Date32`, `Float16`, and 8-byte-padded `Uint16` (`DG-P2-4`) while deduplicating with `@uwdata/mosaic-core`; **must** pass `{ useBigInt: true, useDate: false }` (for `row_id > 2^53 - 1`) and map `f.type.typeId` explicitly (since `String(f.type)` is `"[object Object]"`), and delegate `PMTilesClient.getSchema()` to `pmtiles.worker.ts` (`0 B` main-thread Arrow).
2. **End-to-End Narrow-Span `Float64` & Temporal Precision (`AT-P2-3` + `DG-P1-3`):** Emit `"type": "numeric"` epoch-ms `metadata.stats` for `TIMESTAMP`/`DATE` in `builder.py` and apply internal 1D Relative-to-Origin (`RTO`: subtract `colOrigin64 = stats[col].min` in `Float64` before `Float32Array` upload, keeping public `ViewState` / `getViewport()` / `PointHitEvent` domains unshifted).
3. **Enterprise CSP & Async Module Worker Safety (`DG-P1-5`):** Inline `pmtiles.worker` as a cached module-level `Blob` URL **only** in `dist/standalone.js` (never synchronously calling `URL.revokeObjectURL`, and keeping `dist/index.js`'s separate worker chunk for `worker-src 'self'` CSP).
4. **Baseline-Compliant Mosaic Filter Uniforms (`8 / 8` Storage Buffers):** Use a shared `256 B` `@binding(10) var<uniform>` buffer for 60 FPS range/category/lasso predicates (`0.25 KiB` VRAM), allocating per-slot `vec4<u32>` (`16 B`-aligned per W3C WGSL §14.4.3, up to `64 KiB` / `522,240` bits) uniform bitmasks lazily only when a DuckDB `row_id` SQL escape-hatch filter is active.
5. **Dual-Transport `pydeepgraph` (`CustomRangeSource`):** Auto-probe loopback/proxied HTTP `206 Range` (`150 ms` timeout) for full 6-worker parallel NVMe streaming on local Marimo/Jupyter workstations, with automatic fallback to **`<= 512 KiB`-chunked** binary `CommRangeSource` (safe under Marimo/`websockets`/Cloudflare `1 MiB` and Tornado `10 MiB` WebSocket frame caps) on Colab, SSH Remote, and JupyterHub.

---

## 3. Unified Post-`v0.1.0` Sequencing

1. **Step 1 — `v0.1.1` Hardening Sprint (`deepgraph 0.1.1` & `arrowtiles 0.1.1`):** Fix confirmed `P1`s + narrow-span `Float64` guard + `Blob`-inlined standalone worker + `temp_dir` isolation + fast CPU unit tests & CI/release gates.
2. **Step 2 — `1.35B` Gaia Hardware Stress Test & Benchmark Publication:** Run `100M → 500M → 1.35B` (`s3_cache`) on `v0.1.1` with `D:` scratch isolation and `24 GB` DuckDB memory cap, benchmarking both Intel `xe-lpg` iGPU and RTX 5090 + `50 ms / 50 Mbps` WAN.
3. **Step 3 — `pydeepgraph` + Marimo + Hybrid UW Mosaic (`MosaicClient`) + Tier-1 Ego-Graph Arcs (Zero Format Break):** Ship `mo.ui.anywidget` + loopback `206 Range` daemon, `@binding(10) var<uniform>` WGSL brush/lasso cross-filtering, two-path `DeepgraphMosaicClient` (`< 0.3 ms` GPU uniform fast-path + DuckDB `row_id` SQL escape hatch), and curved 1-hop hover/click ego-arcs.
4. **Step 4 — Unified `.arrowtiles v0.2` Container Release:** Bundle `g * 2^41` column-group subspaces, `"tile_stats"`, offline `metadata.labels` (`2D watershed + c-TF-IDF`), and `Subspace 64/65` tiled edges + Rust `in_degree_kcore`/`pagerank` into one single spec bump.
