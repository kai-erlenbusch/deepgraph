# AGENTS.md — Deepgraph (`deepgraph`)

Engineering context, architectural invariants, and verification commands for AI coding agents working in **Deepgraph**.

---

## 1. Repository Scope & Sibling Boundary

- **This repository (`deepgraph`):** Pure-WebGPU billion-point and graph visualization library (`npm install deepgraph`) + standalone demo application.
  - Target GitHub repository: `https://github.com/kai-erlenbusch/deepgraph`
- **Sibling repository (`arrowtiles`):** Python + Rust (`PyO3`) `.arrowtiles` `v0.1` spatial pyramid builder, reference reader, validator, and wire-format specification (`docs/SPEC.md`).
  - Located locally at `../arrowtiles` (or `../duckdb-arrowtiles` prior to local folder rename).
  - Target GitHub repository: `https://github.com/kai-erlenbusch/arrowtiles`
- If a task modifies `.arrowtiles` archive generation, schema/metadata encoding, or `SPEC.md`, work in the sibling `arrowtiles` repository.

---

## 2. Clean-Room & Dual-Licensing Rule (`CC-BY-NC-4.0` + Commercial)

- Deepgraph is copyrighted by **Kai Erlenbusch** (`NOTICE`) and distributed under **CC BY-NC 4.0** (`LICENSE`) for non-commercial use, with separate commercial licensing via `erlenbusch@uchicago.edu`.
- **NEVER** read, inspect, port, or copy code from third-party `CC BY-NC-SA 4.0` or copyleft repositories (`deepscatter`, `deepscatter-source`, `Quadfeather`, `gaia-deepscatter-study`). All shaders, picking algorithms, and streaming structures must remain 100% original clean-room implementations (using only permissive `MIT` / `Apache-2.0` / `BSD`-compatible runtime dependencies) so Kai Erlenbusch retains sole copyright ownership and unrestricted commercial licensing rights.
- Do **not** prepend custom headers inside `LICENSE` (which must remain verbatim `CC BY-NC 4.0` text) and do **not** add per-file license headers unless explicitly requested; root `LICENSE` and `NOTICE` are sufficient.

---

## 3. Source Layout & Strict Library Isolation

```text
src/
├── index.ts                 # Public library entry (`export { Deepgraph }` + public TS types ONLY)
├── PMTilesClient.ts         # HTTP Range / File PMTiles v3 client, worker pool & compressed pick LRU
├── pmtiles.worker.ts        # Off-main-thread Zstd WASM (`@bokuweb/zstd-wasm`) + Arrow IPC decoder
├── core/
│   ├── Renderer.ts          # Plain-WebGPU adapter/device/swapchain, Camera2D & pointer/wheel input
│   ├── TileGpuStore.ts      # Bounded LRU GPU storage-buffer slots + Float64 RTC coordinate uniforms
│   ├── ComputeRasterizer.ts # Commutative 2×u32 atomic WGSL compute rasterizer + per-pixel resolve
│   ├── PickingLayer.ts      # 9×9 CSS px GPU compute hit-testing (<= 8 storage buffers/stage)
│   └── HighlightOverlay.ts  # Plain-WebGPU hover & selection ring pass
├── lib/
│   ├── ViewerLayer.ts       # Public `Deepgraph` facade (`create`, `isSupported`, plot lifecycle)
│   ├── DatasetSession.ts    # SPEC v0.1 validation (`0.x` minor <= 1, `tiling`, `required_features`)
│   ├── ViewState.ts         # Declarative channel encoding, presets & 2-second axis transitions
│   ├── RenderLoop.ts        # rAF frame loop & library-owned GPU/CPU byte telemetry
│   ├── diagnostics.ts       # `@internal` symbol hook for Playwright tests (stripped from `.d.ts`)
│   └── types.ts             # Public TypeScript interfaces
├── demo/                    # Demo-only UI (`ControlsPanel.ts`, `gaiaPreset.ts`, `gpuStats.ts`)
└── main.ts                  # Demo app entry (`index.html`), uses ONLY public `src/index.ts` API
```

### Mandatory Library Boundaries
1. **Zero Demo Leakage in Library Code:** Files under `src/index.ts`, `src/lib/*`, `src/core/*`, `src/PMTilesClient.ts`, and `src/pmtiles.worker.ts` **MUST NOT** import `lil-gui` or `src/demo/*`, read `window.location`, mutate DOM elements outside the host `container` passed to `Deepgraph.create(...)`, or monkey-patch WebGPU prototypes (`window.__gpuStats`).
2. **Zero Three.js / WebGL:** All rendering runs through plain WebGPU (`ComputeRasterizer` + `HighlightOverlay`). Do not introduce Three.js or WebGL dependencies.
3. **No Main-Thread $O(N)$ Data Loops:** Tile decompression, Arrow IPC column extraction, and per-row pick decoding happen inside `pmtiles.worker.ts` or WGSL compute shaders.

---

## 4. WebGPU, Shader & Packaging Invariants

1. **Zero-Shimmer Commutative Accumulation (`ComputeRasterizer.ts`):**
   - Optical (`Mode 0`/`3`), Categorical/Numeric Points (`Mode 1`), and Density/KDE (`Mode 2`) accumulate into `2 × u32` per-pixel storage buffers using order-independent `atomicAdd` operations so stationary frames exhibit `0.0000%` pixel shimmer.
2. **Universal WebGPU Storage-Buffer Limit (`PickingLayer.ts`):**
   - Keep `PickingLayer` compute pipelines at **`<= 8` storage buffers per shader stage** (`1` dynamic uniform buffer + `8` storage buffers) so picking works on baseline WebGPU adapters without elevated limits.
3. **Deterministic GPU Memory Teardown:**
   - Every `GPUBuffer` allocated by `TileGpuStore`, `ComputeRasterizer`, `PickingLayer`, and `HighlightOverlay` must be explicitly `.destroy()`ed upon LRU eviction and `plot.destroy()`, followed by `context.unconfigure()` and `device.destroy()` (`liveDevices === 0`, `liveBytes === 0`).
4. **Dual Library Build & Self-Contained Worker (`npm run build:lib`):**
   - `dist/index.js` (`"."` export) externalizes main-thread runtime packages (`pmtiles`, `p-queue`, `fflate`, `apache-arrow`, `@bokuweb/zstd-wasm`) so downstream bundlers deduplicate `apache-arrow`.
   - `dist/standalone.js` (`"./standalone"` export) bundles all runtime dependencies for zero-build `<script type="module">` HTML embeds (`examples/embed.html`).
   - In **both** builds, `dist/assets/pmtiles.worker-*.js` **must** remain 100% self-contained (`0` bare specifiers) with `zstd.wasm` inlined as a `data:application/wasm;base64,...` URI. Always run `npm run verify:types` after build changes.

---

## 5. Build, Verification & GPU Coexistence Commands

```bash
npm run typecheck        # Strict TypeScript check (`tsc --noEmit`)
npm run build:lib        # Build `dist/index.js`, `dist/standalone.js`, worker & clean `.d.ts`
npm run verify:types     # External `tsc` (`bundler` + `nodenext`) + real downstream `vite build`
npm run build:demo       # Build standalone demo app into `dist-demo/`
npm run test:e2e         # Fast Playwright E2E suite (< 5 MiB samples in `../arrowtiles/samples/`)
npm run test:e2e:heavy   # Heavy 5-gate VRAM/LRU suite (requires local `public/gaia_full.arrowtiles` + `25k` sample)
```

- **Sample Archives for Local `test:e2e`:** Vite's range server (`vite.config.ts`) resolves `/samples/*.arrowtiles` from `../arrowtiles/samples/` (falling back to `../duckdb-arrowtiles/samples/`). Generate them from the sibling `arrowtiles` repo via `python scripts/build_samples.py` if missing.
- **Discrete GPU Coexistence:** Playwright uses `--force_low_power_gpu` (`playwright.config.ts`) and `powerPreference: 'low-power'` so automated browser tests run on the integrated GPU without contending with discrete CUDA/ML workloads.
- **Release Gate:** Do **not** run `git push` or `npm publish` unless explicitly instructed by the repository owner.

---

## 6. Engineering, Research & Audit Logs (`logs/`)

Persist substantive `RESEARCH`, read-only `AUDIT`, approved `PLAN`, locked `ADR`, and post-execution `WALKTHROUGH` documents in `logs/` following [`logs/README.md`](logs/README.md) (`YYYY-MM-DD_<seq>_<TYPE>_<scope>_<topic-slug>.md` with YAML frontmatter, indexed chronologically in `logs/README.md`). Never commit credentials, `.env` values, or raw binary drill artifacts (`logs/assets/`, `logs/old/`, `logs/oct/` remain gitignored).
