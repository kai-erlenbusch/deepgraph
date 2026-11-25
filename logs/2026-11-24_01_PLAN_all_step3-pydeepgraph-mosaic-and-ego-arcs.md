---
log_id: 2026-11-24_01
date: 2026-11-24
type: PLAN
status: approved
scope: [all]
source_artifact: step3_pydeepgraph_mosaic_and_ego_arcs_plan.md
conversation_id: 08f564f2-95c1-4b5f-b446-5fb13d4d48ab
---

# Step 3 Implementation Plan: GPU Brush/Lasso, Hybrid UW Mosaic, Tier-1 Ego-Graph Arcs & `pydeepgraph` + Marimo

## 1. Goal & Scope

Close the Python-to-WebGPU analytical loop and deliver first-class interactive graph and cross-filtering capabilities **on 100% unmodified `.arrowtiles` `v0.1` archives** (zero wire-format churn ahead of Step 4's `.arrowtiles v0.2`), executed across two review-gated sub-phases:

1. **Step 3A — `deepgraph@0.1.2` (Core WebGPU Selection, Tier-1 Graph Arcs & `deepgraph/mosaic`):**
   - **Baseline-Safe GPU Cross-Filtering & 2D Brush/Lasso (`0` extra storage buffers):** Add a static `1,024 B` `@binding(10)` global filter/lasso uniform buffer + a shared, lazily allocated `32,768 B`-stride (`262,144`-bit, `vec4<u32>`-aligned) `@binding(11)` dynamic slot bitmask uniform buffer across `ComputeRasterizer` (`4/12` uniform, `2/8` dynamic uniform, `8/8` storage) and `PickingLayer` (`@binding(9)` + `@binding(10)`, `3/12` uniform, `2/8` dynamic uniform, `8/8` storage), paired with a plain-WebGPU `SelectionOverlay` (including draggable spotlight-lens rect translation) and off-main-thread worker `row_id` bitmask compilation.
   - **Tier-1 Curved Ego-Graph & kNN Arcs (`EdgeOverlay`):** Render up to `100,000` anti-aliased quadratic Bézier arcs via an instanced `GPUBufferUsage.VERTEX` render pass (`0` storage buffers, `1D RTO` dataset anchor + `128 B` `Float64` camera-relative `RTC` uniforms/frame, 2-second axis-transition interpolation, and complementary bidirectional curvature) driven by both static `plot.setEdges(...)` and async on-hover/on-click `resolveEdges` hooks.
   - **Hybrid UW Mosaic Subpath (`deepgraph/mosaic`):** Ship an optional peer-dependency subpath (`@uwdata/mosaic-core`, `@uwdata/mosaic-sql`) implementing `DeepgraphMosaicClient` with a **Two-Path Engine**: **Path A (`0 ms` GPU Fast Path)** gated strictly by `plot.canApplyFilterOnGpu(filter)` (`<= 6` GPU-resident range slots including idle `@binding(4)`/`@binding(5)` when `isXSpatial && !auxField1` / `isYSpatial && !auxField2`, and `<= 512` category codes); **Path B (DuckDB `row_id` SQL Escape Hatch)** executes `SELECT row_id FROM <table> WHERE <filter>` for cold/text/joined/high-cardinality predicates and compiles `@binding(11)` slot bitmasks in workers.
2. **Step 3B — `pydeepgraph==0.1.0` (`deepgraph/python/`) & Reactive Marimo Workflows:**
   - Package `pydeepgraph` (`pip install pydeepgraph`, universal `py3-none-any.whl`) inside `kai-erlenbusch/deepgraph/python/` (`CC-BY-NC-4.0` + Commercial), bundling `dist/standalone.js` offline.
   - Dual-transport streaming (`transport="auto"`): auto-probes `_http_range.py` (`127.0.0.1:<port>` HTTP `206` with `Access-Control-Allow-Private-Network: true`, `150 ms` timeout) for 6-worker parallel NVMe throughput locally, falling back automatically to `<= 512 KiB`-chunked (`maxLeafPrefetchBytes = 256 KiB`) binary `CommRangeSource` over `anywidget` custom messages on firewalled SSH/Colab/JupyterHub/Marimo kernels, plus `narwhals` + content-hashed (`SUM(hash(*))`), size-bounded `PYDEEPGRAPH_CACHE_DIR` DataFrame/Parquet ingestion.
   - **Marimo Reactive Discipline** (`mo.ui.anywidget`): high-frequency tile fetches, cold-column click enrichment, and hover ego-arcs travel over non-reactive custom messages (`model.send`), while `model.save_changes()` fires **strictly on committed selections** (`click`, `brush-end`, `lasso-end`, `clear`), publishing `u64`-safe `selected_row_ids` (`Union([Int(), Unicode()])`), `clicked_row`, and a ready-to-run DuckDB **`selection_sql`** predicate over all `N` backing rows.

---

## 2. Locked Technical Amendments (Peer Review + Shader Audit)

1. **Extra Spatial Filter Slots & Active Column Sync (Step 3A):**
   - Bind up to `2` extra unmapped numeric/categorical filter columns into `@binding(4)` (`xBuf`) and `@binding(5)` (`yBuf`) **only** when `isXSpatial && !tileStore.auxField1` and `isYSpatial && !tileStore.auxField2` (preserving `teff` / `pmra` / `pmdec` in Modes `2`/`3`/`4`), sync newly referenced filter columns through `TileGpuStore.getActiveColumns()` + `session.setRequestedColumns()`, and gate Mosaic Path A strictly via `plot.canApplyFilterOnGpu(filter)`.
2. **Structural `.d.ts` Isolation for `deepgraph/mosaic` (Step 3A):**
   - Place entry at `src/mosaic.ts` (`"include"`d in `tsconfig.build.json`) and type its public API against self-contained structural interfaces (`MosaicCoordinatorLike`, `MosaicSelectionLike`, `MosaicClientLike`, `MosaicSelectionClauseLike`) with **zero `.d.ts` imports** from `@uwdata/mosaic-core` so `npm run verify:types` (`skipLibCheck: false` under `bundler` and `NodeNext`) passes without leaking `@uwdata/flechette` (`TS7008`) or `@duckdb/duckdb-wasm` types.
3. **2-Pass Commutative `unselectedOpacity > 0` & Oversize-Leaf `rowBaseOffset` (Step 3A):**
   - When `unselectedOpacity > 0` and a filter is active in Mode `0`/`1`, run 2 subpasses on one `GPUCommandEncoder` reusing `accumBuffer`: Pass 1 dispatches `main_unselected` and resolves desaturated at `unselectedOpacity` (`loadOp: 'clear'`); `clearBuffer(accumBuffer)`; Pass 2 dispatches `main_selected` and composites on top (`loadOp: 'load'`) via a premultiplied-alpha foreground resolve pipeline (`one` / `one-minus-src-alpha`, outputting `vec4(0.0)` on empty pixels so background context is never wiped with opaque `bg` or starved of the `3,840` optical flux tickets). Mode `2` always hard-culls unselected points so mean fields stay unbiased.
   - Repurpose `SlotUniforms._pad63` (`offset 252 B`) as `rowBaseOffset: u32`; apply the `2,048`-element Hilbert scramble to `localI` (`0 .. chunkCount - 1`, `chunkCount <= 262,144`) before `i = u.rowBaseOffset + localI`; write `> 262,144`-row overflow chunk uniforms to tail slots `(maxSlots + extraIdx) * 256` and distinct `32 KiB` `slotMaskBuffer` slices via `[uniformSlot * 256, maskSlice * 32768]`; and hide `HighlightOverlay` rings on filtered-out rows.
4. **WebSocket `<= 512 KiB` Chunk Cap & Chrome PNA Headers (Step 3B):**
   - Keep `CommRangeSource` binary chunks `<= 512 KiB` (`maxLeafPrefetchBytes = 256 * 1024`) to stay under `python-websockets` / Marimo's `1 MiB` frame cap, paired with `transport="auto"` `127.0.0.1` HTTP `206` (`Access-Control-Allow-Private-Network: true`).
5. **`u64`-Safe `selected_row_ids` & Content-Hashed `narwhals` Cache (Step 3B):**
   - Use `traitlets.List(traitlets.Union([traitlets.Int(), traitlets.Unicode()]))` for `row_id > 2^53 - 1` (`Gaia` `source_id`) and content-hash (`SUM(hash(*))`) + LRU-bound `PYDEEPGRAPH_CACHE_DIR`.

---

## 3. Execution Gate & Verification Matrix

- **Step 3A Gate (`deepgraph@0.1.2` — Pause Before Step 3B):**
  - `npm run typecheck`, `npm run build:lib`, `npm run verify:types`, `npm run test:unit`
  - Playwright Intel `xe-lpg` iGPU (`--force_low_power_gpu`, `0 B` on discrete RTX 5090): `npm run test:e2e` + `tests/brush-mosaic.spec.ts` + `tests/ego-edges.spec.ts` (`liveDevices === 0, liveBytes === 0`, `0.0000%` stationary shimmer).
- **Step 3B Gate (`pydeepgraph==0.1.0` — Pause Before Any Push / Publish):**
  - `npm run build:widget`, `pytest python/tests -v -W error::UserWarning`, and live `marimo run --headless` Playwright WebGPU E2E (`tests/marimo-anywidget.spec.ts`).
