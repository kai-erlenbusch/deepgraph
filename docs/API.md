# Deepgraph Public JavaScript / TypeScript API (`v0.1 — Phase 8 / D11`)

> **Status:** Complete & Verified (`v0.1.0` — `D12a` npm package: `deepgraph`; `D12b` license: `CC-BY-NC-4.0` + commercial license via `erlenbusch@uchicago.edu`; held `"private": true` pending Owner Release Gate `⛔ D12c`).

`Deepgraph` is an embeddable WebGPU compute scatterplot & spatial-pyramid library for streaming multi-million to billion-point **`.arrowtiles` (`v0.1`)** archives over HTTP Range requests or local `File` handles.

The core library exports **only** `Deepgraph` (`create`, `isSupported`) and pure TypeScript interfaces from `src/index.ts`. It has **zero dependency on `lil-gui`**, never reads `window.location` or URL query parameters, never mutates DOM elements outside the host `container` element passed to `Deepgraph.create(...)`, and keeps all domain-specific extensions cleanly isolated behind optional `preset` / `experimental` configuration.

---

## 1. Quick Start

```ts
import { Deepgraph } from 'deepgraph';

if (!Deepgraph.isSupported()) {
  throw new Error('WebGPU is required');
}

// Host container should have explicit non-zero CSS dimensions (e.g. width: 100%; height: 600px)
const container = document.getElementById('plot')!;

const plot = await Deepgraph.create(container, {
  url: 'https://cdn.example.com/embeddings.arrowtiles',
  encoding: {
    // Omitting x/y uses the archive's spatial quadtree positions automatically
    color: { field: 'clusters_14', scale: 'viridis' },
    size: { field: 'num_of_tokens', domain: [0, 4096], range: [0.5, 3.0] },
  },
});

// Switch to custom 2D chart axes (smooth 2-second WGSL transition), then reset back to spatial positions:
plot.setEncoding({ x: 'num_of_tokens', y: 'turn', color: 'model' });
plot.setEncoding({ x: null, y: null });

// Subscribe to point interactions (GPU PickingLayer lands in Phase 7) & frame telemetry
const offClick = plot.on('click', (hit) => {
  if (!hit) return;
  console.log('Clicked row_id:', hit.rowId, hit.attributes);
});

// Clean up all WebGPU buffers (accumBuffer, slot GPUBuffers, uniforms), Web Workers, and DOM canvas
plot.destroy();
offClick();
```

---

## 2. Static Methods & Factory Options

### `Deepgraph.isSupported() => boolean`

Synchronously checks whether `navigator.gpu` is available in the current browser environment before creating a plot.

### `Deepgraph.create(container, options?) => Promise<DeepgraphPlot>`

Creates a WebGPU canvas inside `container` (which **should have explicit non-zero CSS `width` and `height`**, observed automatically via `ResizeObserver`), initializes the Web Worker pool, and optionally loads an initial `.arrowtiles` source. If `options.url` fails validation, `create()` automatically cleans up all allocated resources before rejecting.

```ts
type DeepgraphPresetResolver = (
  rawMetadata: Record<string, unknown>,
  source: string | File
) => DeepgraphPreset | undefined;

interface DeepgraphOptions {
  /** Optional initial `.arrowtiles` URL (HTTP Range server / CDN) or browser `File` object. */
  url?: string | File;
  /** Initial visual channel encoding (overrides archive `metadata.view` defaults). */
  encoding?: DeepgraphEncoding;
  /** Optional domain preset object OR `(rawMetadata, source) => DeepgraphPreset | undefined` resolver. */
  preset?: DeepgraphPreset | DeepgraphPresetResolver;
  /** Quadtree Z-level overfetch offset relative to `floor(log2(zoom))` (default: `2`). */
  detailOffset?: number;
  /** Maximum decoded tiles retained in the client LRU cache (default: `250`). */
  maxTileCache?: number;
  /** WebGPU adapter power preference (`'high-performance'` default, or `'low-power'` to prefer integrated GPUs). */
  powerPreference?: 'high-performance' | 'low-power';
  /**
   * Optional async resolver (`D8(2)`) invoked on `'click'` and `plot.pickAt(...)` by default
   * to enrich `{ rowId, attributes }` with cold or dropped free-text columns (e.g. querying a
   * `row_id`-sorted Parquet sidecar). Receives an `AbortSignal` that aborts if superseded by a
   * newer hover, `pointerleave`, `load()`, or `destroy()`.
   */
  resolveRowAttributes?: (
    rowId: number,
    retainedAttributes: Record<string, unknown>,
    rowIdBigInt?: bigint,
    signal?: AbortSignal
  ) => Promise<Record<string, unknown>> | Record<string, unknown>;
  /**
   * Whether `resolveRowAttributes` should also run on `'hover'` picks (`false` default).
   * Keep `false` when `resolveRowAttributes` queries a remote backend or sidecar file so rapid
   * pointer movement never triggers sidecar query storms.
   */
  resolveRowAttributesOnHover?: boolean;
}
```

---

## 3. Declarative Encoding (`DeepgraphEncoding`)

- **Spatial Coordinates by Default:** Omitting `x` and `y` on `create()` / `load()` uses the archive's built-in spatial quadtree coordinates. Calling `plot.setEncoding({ x: null, y: null })` transitions any custom chart axis back to spatial positions without exposing internal tile column names.
- **Multiscale Zoom-Adaptive Density & Optical RGB / Log-Density Resolve (`Phase 8b`):**
  - **Optical Sky & Trails (`pipeline: 'optical'` / `'trails'`):** Evaluates per-star optical RGB (`three_aces_filmic` + `linear_to_srgb`) and accumulates 16-bit integer channels (`word0 = R16 | (G16 << 16)`, `word1 = B16`, scale `1.0 = 1024.0`, `64×` headroom before `65535` wrap) resolved via `min(ch, 1024u) / 1024.0` (`0.0000%` stationary shimmer, natural additive blue+gold star mixing, and incandescent white-gold bulge blooming).
  - **Log-Density Points (`'points'` / `D10`):** Both categorical and generic numeric `'points'` modes accumulate into the same `2 × u32` compute buffer and resolve via `log2(1 + count)` brightness with a continuous zoom-adaptive `refCount` (`96 → 1.15` across `Z=0..12+`): categorical Mode 1 averages the RGB of the first 64 weighted hits per pixel (`64 × 15 < 1024`), while numeric Mode 1 accumulates an order-independent `1,024`-level normalized value sum (`word1 = Σ(tU * wVal)`, `prevW < 1,048,576`) and evaluates the colormap on the exact mean scalar in Resolve (`0.0000%` stationary shimmer, `1,024` levels, preventing dense cores from saturating to white).
  - **Continuous Log-Zoom Point Sizing & `'auto'` Exposure/Opacity Semantics:** Rendered point diameter scales smoothly past `zoom = 64` via `pointSize` (`1.0` default) and `zoomBalance` (`0.22` default). When `exposure: 'auto'` or `pointOpacity: 'auto'` is active, gain follows a deterministic `log2(zoom)` curve — preserving the baseline optical exposure/opacity across `Z=0..7` (`zoom <= 36`) and ramping exposure smoothly above `zoom > 36` so deep-zoom starfields (`Z >= 8`) and faint extragalactic overdensities remain crisp without pan-induced brightness pumping.

```ts
type ColorScaleName =
  | 'viridis' | 'plasma' | 'magma' | 'inferno'
  | 'rdbu' | 'coolwarm' | 'jet';

type AggregationStrategy = 'points' | 'mean_field';

interface DeepgraphEncoding {
  /** Column name for chart mode, `null` to reset to archive spatial X, or full axis descriptor. */
  x?: string | null | {
    field?: string | null;
    transform?: 'literal' | 'linear';
    domain?: [number, number];
    range?: [number, number];
  };
  /** Column name for chart mode, `null` to reset to archive spatial Y, or full axis descriptor. */
  y?: string | null | {
    field?: string | null;
    transform?: 'literal' | 'linear';
    domain?: [number, number];
    range?: [number, number];
  };
  color?: string | {
    field: string;
    scale?: ColorScaleName;
    domain?: [number, number];
    invert?: boolean;
  };
  size?: string | {
    field: string;
    domain?: [number, number];
    range?: [number, number];
  };
  /** Invert the Y axis (`true` for inverted-magnitude / HR diagrams). */
  flipY?: boolean;
  /** Spatial/pixel aggregation mode (`'points'` log-density or `'mean_field'` per-pixel mean). */
  aggregation?: AggregationStrategy;
  /** Switch canvas background + blending to high-contrast light mode (`false` default). */
  lightBackground?: boolean;
  /** Pixel kernel radius (`0..2`) for `'mean_field'` aggregation. */
  smoothingKernel?: number;
  /** User multiplier on rendered point diameter (`1.0` default). */
  pointSize?: number;
  /** Continuous log-zoom point growth exponent in `pow(zoom, zoomBalance)` (`0.22` default). */
  zoomBalance?: number;
  /** Per-point opacity override (`0..1`), or `'auto'` (`'auto'` default) for zoom-adaptive opacity. */
  pointOpacity?: 'auto' | number;
  /** Tone-mapping exposure multiplier, or `'auto'` (`'auto'` default). */
  exposure?: 'auto' | number;
}
```

---

## 4. Instance Methods (`DeepgraphPlot`)

| Method | Signature | Description |
| :--- | :--- | :--- |
| **`load`** | `load(source: string \| File, opts?: LoadDatasetOptions): Promise<DatasetMetadataSummary>` | Disposes previous dataset GPU buffers, validates the `.arrowtiles` (`v0.1` or legacy `v0.0`) archive using a single `PMTiles` instance and monotonic load-generation token (rejecting superseded in-flight loads with `DOMException(..., 'AbortError')`), resolves `opts.preset`, and streams active columns lazily. |
| **`setEncoding`** | `setEncoding(encoding: DeepgraphEncoding): void` | Updates visual channel mappings, axis/color/size domains, `flipY`, or aggregation mode. Axis changes animate smoothly over 2 seconds in WGSL; domain/scale tweaks apply in `0 ms` via GPU uniforms. |
| **`getEncoding`** | `getEncoding(): DeepgraphEncoding` | Returns a copy of the active declarative encoding. |
| **`setViewOptions`** | `setViewOptions(patch: ViewOptionsPatch): void` | Low-level shader, multiscale (`pointSize`, `zoomBalance`, `pointOpacity`, `exposure`), `detailOffset`, and `experimental` controls. |
| **`getViewState`** | `getViewState(): ViewStateSnapshot` | Returns a read-only snapshot of active view parameters (including canonical `effectiveFlipY`, `pointSize`, `zoomBalance`, `pointOpacity`, and `exposure`). |
| **`getMetadata`** | `getMetadata(): DatasetMetadataSummary \| null` | Returns parsed archive metadata (`format`, `version`, `tiling`, `archiveMaxZoom`, `totalRows`, `droppedColumns`, `schemaFields`, `stats`, `categories`, `dataBounds`). |
| **`getCategories`** | `getCategories(field: string): string[] \| undefined` | Returns ordered category labels (`0..K-1`) for a categorical column. |
| **`getCategoryColorHex`** | `getCategoryColorHex(index: number): number` | Returns the deterministic 24-bit RGB hex color (`0xRRGGBB`) for category `index` (`0..63` palette + golden-ratio HSV above `64`). |
| **`getViewport`** | `getViewport(): { x: number; y: number; zoom: number }` | Returns current viewport center `(x, y)` in **data coordinates** (`dataBounds` when spatial, active axis `domain` on chart axes) and camera `zoom`. |
| **`setViewport`** | `setViewport(vp: { x?: number; y?: number; zoom?: number }): void` | Pans/zooms the viewport using **data coordinates** `(x, y)` and magnification `zoom`. |
| **`fitView`** | `fitView(): void` | Resets camera center and zoom to frame the full active dataset bounds. |
| **`pickAt`** | `pickAt(cssX: number, cssY: number): Promise<PointHitEvent \| null>` | Dispatches a `9×9 CSS px` GPU picking pass around `(cssX, cssY)` (`1 KB` `slotBestHit` buffer, breaking ties by coarser zoom `z` then `rawRowIdx`) and decodes `row_id` + retained columns directly from `.arrowtiles` on demand. |
| **`on` / `off`** | `on(event, handler): () => void` / `off(event, handler): void` | Subscribes or unsubscribes from typed events (`click`, `hover`, `select`, `telemetry`, `load`, `error`). `on(...)` returns an unsubscribe function. |
| **`destroy`** | `destroy(): void` | Cancels any in-flight `load()`, terminates Web Workers, destroys all `TileGpuStore`, `ComputeRasterizer`, `PickingLayer`, and `HighlightOverlay` `GPUBuffer`s, unconfigures the WebGPU swapchain context, calls `device.destroy()`, removes `ResizeObserver` & pointer/wheel listeners, and removes `<canvas>` from `container`. |

---

## 5. Events (`plot.on(...)`)

| Event | Payload Type | Semantics |
| :--- | :--- | :--- |
| `'load'` | `DatasetMetadataSummary` | Fired when `load(...)` completes (and refreshed internally if a legacy `v0.0` tile provides lazy category dictionaries). |
| `'telemetry'` | `TelemetrySnapshot` | Fired every `~250 ms` with frame timings, active tile/point counts, live tile VRAM bytes, `coldBootMs`, and steady-state `sloP95Ms`. |
| `'error'` | `Error` | Fired on uncaptured WebGPU device errors, point-decode / `resolveRowAttributes` exceptions, or fatal render-loop errors. |
| `'click'` | `PointHitEvent \| null` | Triggered on `< 4 CSS px` pointer clicks via the `9×9 CSS px` GPU `PickingLayer`, returning `{ rowId: number, rowIdBigInt?: bigint, screenX, screenY, dataX, dataY, attributes }` decoded directly from `.arrowtiles` (plus optional `resolveRowAttributes`), or `null` on empty space. |
| `'hover'` | `PointHitEvent \| null` | Throttled (`45 ms`), single-in-flight pointer-hover hit event (dispatched only when at least one `'hover'` listener is attached). Renders the cyan hover ring immediately from GPU winner coordinates (`0 ms` network wait), deduplicates repeated hovers over the same point, reuses a byte-capped (`<= 8` tiles / `<= 24 MiB`) compressed-tile LRU + 1-tile worker Arrow `Table` cache (`O(1)` consecutive intra-tile lookups), cancels stale decodes via `AbortSignal`, and clears automatically (`hover: null`) on `pointerleave`. |
| `'select'` | `PointSelectEvent` | Emitted alongside `'click'` with `{ points: PointHitEvent[] }` (`[hit]` on hit, `[]` on background click) + selected accent ring overlay. |

---

## 6. Domain Presets & Experimental Extensions (`DeepgraphPreset`)

Domain-specific pipelines (such as astronomical Hammer sky projections, custom `esa_*` colormaps, or 6D proper-motion star trails/streamlines) are configured via `DeepgraphPreset` + `experimental` so core types stay clean:

```ts
const skySurveyPreset: DeepgraphPreset = {
  name: 'sky-survey',
  fallbackRootBounds: { minX: -2.0, maxX: 2.0, minY: -1.0, maxY: 1.0 },
  renderPipeline: 'optical',
  defaultEncoding: {
    color: { field: 'bp_rp', scale: 'viridis', domain: [-0.5, 4.0] },
    size: { field: 'phot_g_mean_mag', domain: [-2.0, 18.0] },
  },
  experimental: {
    magnitudeFields: ['phot_g_mean_mag', 'abs_m'],
    opticalColorField: 'bp_rp',
    legacyFlipY: false,
  },
};
```

---

## 7. Packaging, Runtime Dependencies & Worker / WASM Resolution (`Phase 8`)

### Package Entry Points (`package.json`)

- **`"exports"`:**
  - **`"."`** (`import { Deepgraph } from 'deepgraph'`): `{ "types": "./dist/index.d.ts", "import": "./dist/index.js" }` — **Bundler entry** externalizing main-thread runtime dependencies (`pmtiles`, `p-queue`, `fflate`, `apache-arrow`, `@bokuweb/zstd-wasm`) so Vite / Rollup / Webpack / esbuild consumers deduplicate shared packages cleanly while keeping `dist/assets/pmtiles.worker-*.js` + base64 `zstd.wasm` self-contained.
  - **`"./standalone"`** (`import { Deepgraph } from 'deepgraph/standalone'`): `{ "types": "./dist/index.d.ts", "import": "./dist/standalone.js" }` — **Zero-build standalone entry** bundling all runtime dependencies for direct `<script type="module">` HTML pages (see [`examples/embed.html`](../examples/embed.html)).
- **`"main"` / `"module"`:** `"./dist/index.js"` (`"type": "module"`, `"sideEffects": false`)
- **`"types"`:** `"./dist/index.d.ts"` (emitted with `"rootDir": "./src"` and `"stripInternal": true`, rewritten to `.js` relative specifiers so standard downstream TypeScript projects pass `tsc --noEmit` under both `bundler` and `nodenext` without `allowImportingTsExtensions` or `@webgpu/types`)
- **`"license"`:** `"CC-BY-NC-4.0"` (free for non-commercial use; commercial use requires a separate license via `erlenbusch@uchicago.edu` — see `LICENSE` and `NOTICE`; `"name": "deepgraph"`, `"files": ["dist", "docs/API.md", "LICENSE", "NOTICE"]`; `"private": true` preserved until owner publish gate `⛔ D12c`)

### Build & Verification Commands

| Script | Output / Check |
| :--- | :--- |
| `npm run build` / `npm run build:lib` | Builds both the externalized bundler entry (`dist/index.js`) and the self-contained browser entry (`dist/standalone.js`) plus self-contained `dist/assets/pmtiles.worker-*.js` (`copyPublicDir: false`), and emits clean TypeScript declarations (`dist/index.d.ts`, `dist/lib/types.d.ts`). |
| `npm run verify:types` | Runs external consumer `tsc --noEmit` under both `"moduleResolution": "bundler"` and `"moduleResolution": "nodenext"` (`allowImportingTsExtensions` disabled, `skipLibCheck: false`, `types: []`, testing both `"."` and `"./standalone"`) **plus** a real `vite build` consumer application importing `dist/index.js` alongside `apache-arrow`. |
| `npm run build:demo` | Builds the standalone interactive `lil-gui` demo application into `dist-demo/` (`copyPublicDir: false`). |
| `npm run test:e2e` | Runs the 3 fast Playwright WebGPU specs (`library-embed`, `phase8b-hdr-density`, `picking-layer`) against `< 5 MiB` sample archives in sibling `../arrowtiles/samples/`. |
| `npm run test:e2e:heavy` | Runs the 5-gate VRAM & LRU lifecycle stress suite (`gpu-vram-lifecycle.spec.ts`) against local `public/gaia_full.arrowtiles` and `../arrowtiles/samples/lmsys_d7_proportional_25k.arrowtiles`. |

### Bundled vs. Externalized Dependencies & Zero-Config Worker / WASM Loading

- **Zero Demo Dependencies in `dist/`:** `lil-gui` (`MIT`) is a `devDependency` used exclusively by `src/demo/` (`0 B` in `dist/`).
- **Clean Bundler Deduplication (`dist/index.js`) + Zero-Build Standalone (`dist/standalone.js`):** Downstream bundlers consume `dist/index.js` and share `pmtiles` (`BSD-3-Clause`), `p-queue` (`MIT`), `fflate` (`MIT`), `apache-arrow` (`Apache-2.0`), and `@bokuweb/zstd-wasm` (`MIT`) from `node_modules` without duplicating `apache-arrow` on the main thread, while zero-build HTML pages import `dist/standalone.js`.
- **Self-Contained Relative Worker + Inlined `zstd.wasm` (Both Builds):**
  1. Both `dist/index.js` and `dist/standalone.js` instantiate their Web Worker pool via `new Worker(new URL("assets/pmtiles.worker-*.js", import.meta.url), { type: "module" })` — always resolved relative to `dist/` rather than the host page's root URL.
  2. `dist/assets/pmtiles.worker-*.js` stays 100% self-contained in both builds (`0` bare module specifiers) and embeds `@bokuweb/zstd-wasm`'s `zstd.wasm` binary as an inline `data:application/wasm;base64,...` URI so consumers never hit broken `/assets/*.wasm` 404s or need custom worker/WASM bundler plugins.
