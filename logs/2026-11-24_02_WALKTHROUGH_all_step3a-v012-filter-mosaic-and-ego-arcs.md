---
log_id: 2026-11-24_02
date: 2026-11-24
type: WALKTHROUGH
status: executed
scope: [all]
source_artifact: step3a_v012_walkthrough.md
conversation_id: 08f564f2-95c1-4b5f-b446-5fb13d4d48ab
---

# Step 3A Walkthrough — `deepgraph@0.1.2` GPU Cross-Filtering, 2D Brush/Spotlight/Lasso, Curved Bézier Ego-Arcs & `deepgraph/mosaic`

## 1. Executive Summary

Completed **Step 3A (`deepgraph@0.1.2`)** in `deepgraph` (`baseline HEAD 8194b4a`) while keeping sibling `arrowtiles` (`43e89b6`, `.arrowtiles v0.1`) **100% untouched**, and paused at the **Step 3A Review Gate** prior to Step 3B (`pydeepgraph==0.1.0`) and prior to `git commit` / `git push` / `npm publish`.

### Verification Matrix Summary

| Gate | Command | Result |
| :--- | :--- | :--- |
| TypeScript (`src` + `tests/unit`) | `npm run typecheck` | **PASS (`0` errors)** |
| Library + Standalone + Mosaic + `.d.ts` | `npm run build:lib` | **PASS** (`dist/index.js` `302.92 kB` / `72.20 kB` gz; `dist/standalone.js` `736.95 kB` / `222.15 kB` gz; `dist/mosaic.js` `12.75 kB` / `3.85 kB` gz; `dist/assets/pmtiles.worker-DS-0jLfz.js` `393.44 kB`; `3` `.d.ts` files, `0` leaked symbols) |
| External Consumer `.d.ts` + Vite Build | `npm run verify:types` | **PASS** (`tsc --noEmit` under `bundler` + `NodeNext` with `skipLibCheck: false`, `types: []`, verifying `.`, `./standalone`, and `./mosaic` without requiring `@uwdata/mosaic-*` installed + downstream `vite build`) |
| Standalone Demo App Build | `npm run build:demo` | **PASS** (`dist-demo/` built in `225 ms`) |
| Node Unit Suite (`node:test`) | `npm run test:unit` | **PASS (`19 / 19` tests in `538 ms`)** |
| Intel `xe-lpg` iGPU Playwright E2E | `npm run test:e2e` | **PASS (`6 / 6` tests across `5` specs in `50.6s`, `0 B` on discrete RTX 5090, `0.0000%` stationary shimmer, `liveDevices === 0, liveBuffers === 0, liveBytes === 0`)** |
| Sibling `arrowtiles` Boundary | `git -C ../duckdb-arrowtiles status` | **CLEAN (`0` modified/untracked files)** |

---

## 2. Delivered Components & Invariants Verified

### 2.1 Baseline-Safe GPU Cross-Filtering & 2D Brush / Spotlight-Lens / Lasso (`FilterGpuResources.ts`, `SelectionOverlay.ts`, `ComputeRasterizer.ts`, `PickingLayer.ts`, `HighlightOverlay.ts`)
- **Zero Extra Storage Buffers (`8 / 8` Baseline Limit Preserved):**
  - `ComputeRasterizer`: `@binding(10)` static `1,024 B` `GlobalFilterUniforms` + `@binding(11)` dynamic `32,768 B`-stride (`2,048 × vec4<u32>` = `262,144` rows/slice) `SlotVisibilityMask` (`4 / 12` uniform buffers, `2 / 8` dynamic uniform buffers, `8 / 8` storage buffers).
  - `PickingLayer`: `@binding(9)` `GlobalFilterUniforms` + `@binding(10)` dynamic `SlotVisibilityMask` (`3 / 12` uniform buffers, `2 / 8` dynamic uniform buffers, `8 / 8` storage buffers).
- **Idle Spatial Axis Slot Reuse (`planGpuFilterExecution`):**
  - When `isXSpatial && !auxField1` and/or `isYSpatial && !auxField2`, up to **2 unmapped archive columns** (e.g. `num_of_tokens` + `language` on a 2D UMAP) are automatically bound into idle `@binding(4)` / `@binding(5)` slots for `0 ms` Path A GPU evaluation alongside `color`, `size`, and spatial `x_u16` / `y_u16` (`<= 6` simultaneous numeric `1D RTO` ranges + `2 × 512`-code category bitsets).
- **Zero-Shimmer Two-Subpass Context Dimming (`unselectedOpacity > 0`):**
  - Mode `0` (Optical) and Mode `1` (Points) execute `main_unselected` → `fs_dim_bg` (desaturated at `unselectedOpacity`, `loadOp: 'clear'`) → `encoder.clearBuffer(accumBuffer)` → `main_selected` (full `3,840`-ticket optical / `5,120`-weight Mode 1 budget) → `fs_fg_overlay` (`srcFactor: 'one'`, `dstFactor: 'one-minus-src-alpha'`, `loadOp: 'load'`). Verified `0.0000%` stationary shimmer (`changedPixels === 0`) on both LMSys Mode 1 (`121,928` lit pixels) and Gaia Mode 0 (`226,094` lit pixels).
  - Mode `2` (`aggregation: 'mean_field'`) and Modes `3`/`4` always hard-cull unselected rows so per-pixel averages remain unbiased.
  - `PickingLayer` always skips unselected/dimmed rows (`dimmedPicked40064 === false`), and `HighlightOverlay` hides hover/selection rings on culled rows (`markerVisibleWhenCulled === false`).
- **Selection UX (`SelectionOverlay.ts`, `Renderer.ts`, `ViewerLayer.ts`):**
  - Fragment-stage even-odd concave polygon fill + analytical segment SDF (`576 B` uniform, `0` storage buffers, Ramer–Douglas–Peucker `<= 64` GPU vertices) with `0.0000%` stationary shimmer (`62,235` lit pixels, `changedPixels === 0`).
  - Supports `'pan'` (with `Shift+drag` temporary brush), `'brush'` (draw + **draggable spotlight-lens translation** when dragging inside an active brush + `< 4 CSS px` click-to-clear suppressing accidental point `click`), and `'lasso'`, while keeping middle/right-drag pan and wheel zoom active in all modes.
- **Lazy VRAM & Worker Row-ID Indexing (`workerCodec.ts`, `pmtiles.worker.ts`):**
  - Idle filter GPU footprint is `33,792 B` (`33 KiB`: `1 KiB` `GlobalFilterUniforms` + `32 KiB` all-ones dummy slice); `slotMaskBuffer` allocates lazily on `setFilteredRowIds(...)` (`8,750,080 B` across `266` slots) and `.destroy()`s immediately back to `33,792 B` on `setFilteredRowIds(null)`.
  - Worker row-ID index uses `Uint32Array` (`<= 2^32 - 1`) and promotes automatically to `BigUint64Array` (`> 2^32 - 1`, preserving full 64-bit Gaia `source_id > 2^53 - 1`), chunking `> 262,144`-row oversize leaf tiles via `SlotUniforms.rowBaseOffset` (`offset 252 B`).

### 2.2 Curved Quadratic Bézier Graph & Ego-Network Arcs (`EdgeOverlay.ts`)
- Instanced `GPUBufferUsage.VERTEX` pipeline (`48 B/instance`, `96` vertices / `16` quadratic Bézier segments per arc, `<= 100,000` arcs/layer across `staticLayer` + `egoLayer`, `0` storage buffers, `2 × 256 B` dynamic uniform buffer so both layers draw in one `GPUCommandEncoder` without uniform overwrite).
- `Float64` dataset-anchor `1D RTO` endpoint packing + `128 B` per-frame `Float64` camera-relative `RTC` uniforms + smooth 2-second axis-transition interpolation (`sourceAttributes` / `targetAttributes`).
- Right-handed perpendicular bow separates `A -> B` and `B -> A` bidirectional edges (`chordCenterDiff === 0`, `upperArcHits === 4`, `lowerArcHits === 4`), with `0.0000%` stationary shimmer (`225,009` lit pixels, `changedPixels === 0` across `25,000` static arcs) and `AbortSignal`-cancelled async `resolveEdges` (`hover` supersession + `click` pinning).

### 2.3 Optional `deepgraph/mosaic` Subpath (`src/mosaic.ts`)
- Built separately via `vite.mosaic.config.ts` (`dist/mosaic.js` `12.75 kB` / `3.85 kB` gz + self-contained structural `dist/mosaic.d.ts` with `0` `@uwdata/mosaic-*` / `@duckdb/duckdb-wasm` `.d.ts` imports and `0 B` added to `dist/index.js` or `dist/standalone.js`).
- `DeepgraphMosaicClient` subclasses `@uwdata/mosaic-core`'s `MosaicClient` (`instanceof MosaicClient === true`), passes `super(undefined)` to avoid Mosaic's `Coordinator.updateSelection` `sql = "null"` trap on 0-SQL updates, evaluates GPU-compatible cross-filters synchronously via **Path A (`sqlQueryCount === 0`)**, and routes `ILIKE` / cold-column / `OR`-union predicates through **Path B (`SELECT "row_id"` IPC → `plot.setFilteredRowIds`)**, alongside division-free `buildLassoDuckDbSql`.

---

## 3. Runtime Bugs Caught & Fixed During Step 3A Verification

1. **`GPUDevice.limits.maxBufferSize >= 2 GiB` Signed 32-Bit Bitwise Overflow (`FilterGpuResources.ts`, `ComputeRasterizer.ts`):**
   - Live Intel `xe-lpg` WebGPU reported `maxBufferSize = 2,147,483,648` (`2^31`); JS `maxBuf & ~255` coerced to signed `Int32` `-2,147,483,648` and threw `RangeError` in `createBuffer`. Replaced bitwise alignment masks on GPU limit numbers with `Float64`-safe `Math.floor(maxBuf / 256) * 256` (and `Math.floor(Number(maxBind) / 8) * 8` in `ComputeRasterizer.ts`).
2. **UW Mosaic `AsyncDispatch` Coalesced Queue Settling & 1-Field `clauseIntervals` (`src/mosaic.ts`):**
   - Handled both `clauseInterval(col, [lo, hi])` (`[lo, hi]`) and 1-field `clauseIntervals([col], [[lo, hi]])` (`[[lo, hi]]`) on Path A, and wired `filterSettled` to drain `Selection`'s `AsyncDispatch` in-flight promise chain before resolving.
3. **Immediate Hover Ego-Arc Cancellation (`src/lib/ViewerLayer.ts`):**
   - Aborted `egoHoverAbortController` immediately when `PickingLayer` switches `activeHoverWinnerKey` (prior to worker `decodeTileRow`), preventing stale hover edge resolvers from running across rapid pointer sweeps.

---

## 4. Independent Review Gate Polish Pass (Pre-Step 3B Checkpoint)

Folded in all findings from both independent Step 3A reviews prior to creating the local `v0.1.2` checkpoint commit:
- **Streaming & Column-Switch `row_id` Mask Preservation (`TileGpuStore.ts`, `ViewerLayer.ts`):** Split `onSlotBindingsInvalidated` (called on upload to invalidate `GPUBindGroup`s) from `onSlotDisposed` (strictly on eviction/unload/clear) so freshly streamed or column-switched tiles retain their `32 KiB` slot bitmasks; verified live via column-switch + zoom under active `setFilteredRowIds([40064])` in `tests/brush-mosaic.spec.ts`.
- **Zero Main-Thread $O(N)$ `Set<bigint>` & Sort (`FilterGpuResources.ts`, `PMTilesClient.ts`, `pmtiles.worker.ts`, `workerCodec.ts`):** Replaced `activeFilteredRowIdSet` with $O(1)$ CPU-mirrored `32 KiB` slot mask bit-checks (`isSlotRowSelected`) + worker `testProbeRowIdsInFilter` for `<= 64` evicted-tile markers, moved typed-array `.sort()` into `pmtiles.worker.ts`, added `freeExtraSliceIndices` recycling, and intersected `queryTileSpatialSelection` (`getSelectedRowIds()`) with active `row_id` filters.
- **RDP SQL Parity, `EdgeOverlay` Chart-Null Culling & Ground-Truth Node Pixel Proof (`mosaic.ts`, `EdgeOverlay.ts`, `types.ts`, `ComputeRasterizer.ts`, `PickingLayer.ts`):** Shared `simplifyRdpPolygon` inside `buildLassoDuckDbSql`, widened `sourceAttributes`/`targetAttributes` to `Record<string, unknown>`, culled non-spatial chart edges with `null`/`NaN` attributes, replaced eager WGSL `select(0.0, *Buf[i], has*Buf)` reads with guarded `if (has*Buf)` branches, added ground-truth LMSys `40064 -> 38479` endpoint/apex pixel assertions + `transitionT = 0.5` pixel assertions in `tests/ego-edges.spec.ts`, added the `Selection & Graph` folder to `src/demo/ControlsPanel.ts`, updated `docs/API.md`, and added `.gitattributes`.

