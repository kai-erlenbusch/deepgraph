# Deepgraph WebGPU 🌌

Deepgraph is a high-performance, WebGPU-accelerated static embedding engine designed for massive point cloud datasets. Built as the WebGPU-first successor to Nomic AI's [deepscatter](https://github.com/nomic-ai/deepscatter), this repository leverages modern browser hardware to render billions of data points—such as the 1.8-billion star ESA GAIA dataset—at a silky-smooth 60 FPS using an out-of-core Apache Arrow streaming pipeline.

Our "North Star" is to build the ultimate 2D scatterplot rendering engine capable of handling web-scale data visual analytics directly in the browser, completely bypassing the V8 garbage collector for rendering data.

## 🚀 Core Technologies & Key Features

### The WebGPU TSL Pipeline
The heart of Deepgraph is written entirely using Three.js Shading Language (TSL) to interface directly with WebGPU. This unlocks mathematical precision not previously possible in standard WebGL workflows:
- **Hardware-level Anti-Aliasing (`fwidth`):** By computing screen-space derivatives with `fwidth()`, we guarantee perfectly smooth, mathematically precise 1-pixel soft edges regardless of your camera's zoom level. This entirely eliminates the jagged pixelation common in scaling point clouds.
- **Stochastic Sub-Pixel Dithering:** We employ a custom `hash()` function paired with probabilistic `Discard` operations in the fragment shader. This simulates sub-8-bit opacities, allowing delicate, ultra-dense topological clusters to be rendered without "brightness blowout" or white-washing.
- **Pre-multiplied Alpha Blending:** Color handling maps density mathematically to avoid standard Z-sorting bottlenecks, making depth-sorting irrelevant for performance.

### Zero-Copy Data Streaming & Additive LOD
Handling 1.8 billion points requires a complete rethink of data transit:
- **Apache Arrow Interleaving:** Deepgraph completely abandons JSON and CSV. Dedicated Web Workers fetch and decode columnar binary data (Feather format), streaming raw `Float32Array` vectors directly into GPU buffers. This creates a "zero-parse" pipeline.
- **Progressive Subsampling (Additive LOD):** By leveraging a spatial `ix` index—using Quadtree partitioning and Morton ordering/reservoir sampling—Deepgraph natively supports Additive LOD. Parent tiles (e.g. `0/0/0.feather`) perfectly represent a random, uniform geographic sample. As you zoom, child tiles smoothly layer additional detail exactly where the user is looking.

### Advanced 2.5D Interaction *(Experimental)*
> ⚠️ **Note:** The primary focus and North Star of this project is high-performance 2D data visualization. The 2.5D camera mechanics are purely experimental and not recommended for production use.

- **True Frustum Culling:** Implements mathematically precise `THREE.Frustum` intersections against an infinitely tall `THREE.Box3`. In experimental mode, the camera can be seamlessly tilted into a 2.5D extruded view without horizon tiles randomly disappearing.
- **Micro-Picking (1x1 Offset Pass):** Hover tooltips use an optimized camera `setViewOffset` to render exactly one pixel per frame, protecting the 60 FPS target regardless of dataset size.

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
To bundle the application and workers into static assets:
```bash
npm run build
```

## 🗺️ Architecture Overview
Deepgraph is split into a modular, multi-threaded pipeline designed to prevent main-thread UI stalling:

1. **`main.ts` (TSL Engine):** The heart of the visualization. An `InstancedMesh` with a `MeshBasicNodeMaterial` rewritten entirely in Three.js Shading Language to handle custom Pre-multiplied Blending, `fwidth` derivatives, and stochastic sub-pixel dithering.
2. **`Renderer.ts`:** Manages the WebGPU context, continuous render loops, and camera orbital mechanics. It broadcasts real-time physical screen metrics (Device Pixel Ratio, World-Units-per-Pixel) to the shaders to maintain point clarity across devices.
3. **`TileManager.ts`:** Handles the spatial Quadtree indexing and Frustum intersection logic. It strictly manages the GPU cache through an LRU eviction policy to prevent VRAM overflow.
4. **`ArrowWorker.ts`:** A dedicated Web Worker that parses the Apache Arrow binary files from the network, mapping vectors into WebGPU-ready 16-byte interleaved buffers.

## 📄 License
MIT License
