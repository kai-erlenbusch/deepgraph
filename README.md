# Deepgraph WebGPU 🌌

Deepgraph is a high-performance, WebGPU-accelerated static embedding engine designed for massive point cloud datasets. Acting as the parallel 2D/2.5D counterpart to the `deepgraph-3d` DuckDB engine, this repository leverages modern browser hardware to render millions of data points at 60 FPS using an out-of-core Apache Arrow streaming pipeline.

## 🚀 Key Features

### The WebGPU TSL Pipeline
- **Hardware-level Anti-Aliasing (`fwidth`):** Guarantees perfectly smooth 1-pixel soft edges regardless of zoom level, entirely eliminating pixelation.
- **Stochastic Sub-Pixel Dithering:** Employs TSL `hash()` and probabilistic `Discard` to simulate sub-8-bit opacities, allowing delicate topological clusters to be rendered without "brightness blowout".
- **Pre-multiplied Alpha Blending:** Maps density mathematically to avoid standard Z-sorting bottlenecks.

### Advanced 2.5D Interaction
- **True Frustum Culling:** Implements mathematically precise `THREE.Frustum` intersections against an infinitely tall `THREE.Box3`. The camera can be seamlessly tilted into a 2.5D extruded view without horizon tiles randomly disappearing.
- **Micro-Picking (1x1 Offset Pass):** Hover tooltips use an optimized camera `setViewOffset` to render exactly one pixel per frame, protecting the 60 FPS target regardless of dataset size.

### Zero-Copy Data Streaming
- **Apache Arrow Interleaving:** Dedicated Web Workers fetch and decode columnar binary data, streaming raw `Float32Array` vectors directly into GPU buffers.
- **LOD Overdraw Prevention:** The Quadtree `TileManager` calculates dynamic Level of Detail (LOD), guaranteeing seamless visual transitions by strictly hiding parent tiles the moment their higher-resolution children are fully loaded into VRAM.

## 💻 Local Development

### Prerequisites
- Node.js (v18+)
- A modern browser with **WebGPU enabled** (Chrome 113+, Edge 113+, Firefox Nightly, or Safari 18+).

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kai-erlenbusch/deepgraph.git
   cd deepgraph
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```
   The application will run the optimized Vite pipeline at `http://localhost:5178`.

### Building for Production
```bash
npm run build
```

## 🗺️ Architecture Overview
Deepgraph is split into a modular, multi-threaded pipeline:

1. **`main.ts` (TSL Engine):** The heart of the visualization. An `InstancedMesh` with a `MeshBasicNodeMaterial` rewritten entirely in Three.js Shading Language to handle custom Pre-multiplied Blending, fwidth derivatives, and stochastic sub-pixel dithering.
2. **`Renderer.ts`:** Manages the WebGPU context, continuous render loops, and 2D/2.5D camera orbital mechanics. It broadcasts real-time physical screen metrics (Device Pixel Ratio, World-Units-per-Pixel) to the shaders.
3. **`TileManager.ts`:** Handles the spatial Quadtree indexing and Frustum intersection logic. It strictly manages the GPU cache through an LRU eviction policy.
4. **`ArrowWorker.ts`:** A dedicated Web Worker that parses the Apache Arrow binary files from the network, mapping vectors into WebGPU-ready 16-byte interleaved buffers to prevent main-thread UI stalling.

## 📄 License
MIT License
