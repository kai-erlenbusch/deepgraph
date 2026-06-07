# Deepgraph WebGPU 🌌

Deepgraph is a high-performance, WebGPU-accelerated static embedding engine designed for massive point cloud datasets. Built as the WebGPU-first successor to Nomic AI's [deepscatter](https://github.com/nomic-ai/deepscatter), this repository leverages modern browser hardware to render billions of data points (such as the 1.8-billion star ESA GAIA dataset) at 60 FPS using an out-of-core Apache Arrow streaming pipeline.

## 🚀 Key Features

### The WebGPU TSL Pipeline
- **Hardware-level Anti-Aliasing (`fwidth`):** Guarantees perfectly smooth 1-pixel soft edges regardless of zoom level, entirely eliminating pixelation.
- **Stochastic Sub-Pixel Dithering:** Employs TSL `hash()` and probabilistic `Discard` to simulate sub-8-bit opacities, allowing delicate topological clusters to be rendered without "brightness blowout".
- **Pre-multiplied Alpha Blending:** Maps density mathematically to avoid standard Z-sorting bottlenecks.

### Advanced 2.5D Interaction
- **True Frustum Culling:** Implements mathematically precise `THREE.Frustum` intersections against an infinitely tall `THREE.Box3`. The camera can be seamlessly tilted into a 2.5D extruded view without horizon tiles randomly disappearing.
- **Micro-Picking (1x1 Offset Pass):** Hover tooltips use an optimized camera `setViewOffset` to render exactly one pixel per frame, protecting the 60 FPS target regardless of dataset size.

### Zero-Copy Data Streaming & Additive LOD
- **Apache Arrow Interleaving:** Dedicated Web Workers fetch and decode columnar binary data (Feather format), streaming raw `Float32Array` vectors directly into GPU buffers with zero parsing overhead.
- **Progressive Subsampling (Additive LOD):** By leveraging a spatial `ix` index (via Quadtree partitioning and Morton ordering/reservoir sampling), Deepgraph natively supports Additive LOD. Parent tiles smoothly layer detail exactly where the user is looking without density blowouts.

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
2. **`Renderer.ts`:** Manages the WebGPU context, continuous render loops, and 2D/2.5D camera orbital mechanics. It broadcasts real-time physical screen metrics to the shaders.
3. **`TileManager.ts`:** Handles the spatial Quadtree indexing and Frustum intersection logic. It strictly manages the GPU cache through an LRU eviction policy.
4. **`ArrowWorker.ts`:** A dedicated Web Worker that parses the Apache Arrow binary files from the network, mapping vectors into WebGPU-ready 16-byte interleaved buffers to prevent main-thread UI stalling.

## 📄 License
MIT License
