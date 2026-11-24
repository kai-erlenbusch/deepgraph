---
log_id: 2026-11-23_01
date: 2026-11-23
type: RESEARCH
status: approved
scope: [all]
source_artifact: apple_embedding_atlas_deep_review.md
conversation_id: e91adbce-084e-445b-98b1-a434c428c86d
---

# Deep Research & Architectural Review: `apple/embedding-atlas` vs. `deepgraph` & `arrowtiles`

> **Clean-Room & License Note (`AGENTS.md` §2):** `apple/embedding-atlas` is MIT-licensed (`Copyright (c) 2025 Apple Inc.`). This memo records high-level algorithmic, mathematical, and architectural findings from `arXiv:2505.06386`, `arXiv:2504.07285`, `arXiv:2607.08746`, and system inspection only; zero third-party source code is copied into `deepgraph` or `arrowtiles`.

---

## 1. Key Competitive & Architectural Findings

1. **Scale, Precision & Picking Advantage Confirmed for `deepgraph` + `arrowtiles`:**
   - `apple/embedding-atlas` uploads full `Float32Array` `x, y` arrays into browser RAM/VRAM upfront (`downsampleMaxPoints = 4,000,000` stochastic drop cap on instanced 4-vertex quads), requires optional WebGPU `shader-f16` (`enable f16;`), uses global `Float32` transforms, and executes DuckDB SQL queries on mouse hover (`ORDER BY (x - px)^2 + (y - py)^2 LIMIT 1`).
   - `deepgraph` + `.arrowtiles` streams bounded spatial pyramids (`1.8B+` points) over static HTTP Range inside a `128 MB` VRAM LRU with `Float64` Relative-to-Center (`RTC`) deep zoom, `100%` baseline WebGPU (`2 × u32` atomics, `<= 8` storage buffers/stage), and `0.1 ms` GPU compute hit-testing (`PickingLayer.ts`).
2. **Wide-Open Graph Visualization Moat Validated (`arXiv:2607.08746`):**
   - Even with Apple & Georgia Tech's July 2026 kNN network-science paper (`graph_metrics.py`), `embedding-atlas` only precomputes scalar node columns (`pagerank`, `in_degree`, `in_degree_core`, `clustering_coefficient`) and lists neighbor rows—it renders **zero** edge segments, curved ego-arcs, or bundled network flows on the GPU canvas.

---

## 2. Eight Clean-Room Takeaways Ranked by ROI

1. **Offline Multiscale Semantic Landmarks (`arrowtiles build --auto-labels --text-col` + `deepgraph` Label Layer):**
   - Run 2D KDE watershed + saddle-border persistence merging + tilted cutoff plane (`arXiv:2504.07285`), stem-grouped `c-TF-IDF` keyword ranking, and Been–Daiches–Yap (`IEEE TVCG 2006`) continuous `[minZoom, maxZoom]` zoom-interval placement **offline during `arrowtiles build`**, storing `~20 KB` of `metadata.labels` so `100M–1B`-row archives display instant, zero-flicker semantic cluster labels on Frame 1 without downloading raw text columns.
2. **Constant-Time Separable Recursive Gaussian KDE + Topographic Sobel Contours (`Mode 2`):**
   - Splat bilinear weights into a `1/4`-res `f32` grid (avoiding `shader-f16`) with a `3σ` viewport border margin, run a 2-pass separable 4th-order Deriche IIR filter (`O(1)` per pixel regardless of `σ`), quantize into stepped bands, and apply a `3×3` Sobel filter for live per-category topographic contours weighted by `rows_represented`.
3. **Directed `in_degree_core` & Hubness Node Metrics (`arrowtiles build --edges --graph-metrics`):**
   - Compute `O(E)` Batagelj–Zaveršnik directed in-degree `k`-core, `in_degree`, and `pagerank` in Rust when ingesting `--edges`, exposing them both as visual columns and as a structural hub priority option (`--sort-by in_degree_kcore DESC`).
4. **Area-Conserving Sub-Pixel Footprint & Beer–Lambert Optical Depth (`τ = -∑ coverage · ln(1 - α)`):**
   - Conserve exact pixel-overlap area across `r = 0.1 px → 8 px` to eliminate `1 px` zoom brightness shifts, and support optical-depth transmittance (`1 - exp(-τ)`) for order-independent categorical/edge overlays.
5. **`pydeepgraph` (`anywidget`) Ergonomics:**
   - Use `narwhals` for zero-dependency Pandas/Polars/Arrow/DuckDB inputs, guard VS Code notebook wheel-scroll hijacking (`vscIPyWidgets`), auto-sync Jupyter/VS Code dark/light CSS variables via `MutationObserver`, and expose `widget.selection()`.
6. **Draggable Range-Brush Lens + Simplified Polygon Lasso (`v0.2a`):**
   - Pair Ramer–Douglas–Peucker simplified lassos (`vec4` bbox early-out before WGSL parity test) with a draggable/resizable 2D rectangle brush for continuous `60 FPS` subspace sweeping.
7. **Auto-`opacity` & Auto-`pointSize` from `Z=0..2` Peak `256×256` Bin Density:**
   - Auto-calibrate initial `pointSize` and `opacity` from coarse root-tile peak cell density (ignoring out-of-grid outliers rather than clamping).
8. **Declarative Agent / MCP Inspection Surface (`v0.2+`):**
   - Expose schema, `ViewState`, filter/camera controls, and canvas screenshots cleanly for AI research/debugging agents.
