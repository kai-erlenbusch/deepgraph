# Deepgraph WebGPU 🌌

Deepgraph is a high-performance, WebGPU-accelerated point cloud and scatterplot rendering engine designed for massive datasets. Inspired by architectures like Nomic Deepscatter and the PubMed Landscape, Deepgraph uses cutting-edge hardware-accelerated shaders and streaming binary formats to render hundreds of thousands of data points at 60 FPS directly in the browser.

## 🚀 Features

- **WebGPU Powered:** Built on the cutting-edge Three.js Shading Language (TSL) and WebGPU backend, offloading all calculation and blending math directly to the GPU.
- **Hardware-level Anti-Aliasing:** Utilizes GPU hardware derivatives (`fwidth`) to guarantee perfect 1-pixel soft edges regardless of zoom level, entirely eliminating pixelation and blockiness.
- **Deepscatter Architecture ("North Star"):** 
  - **Pre-multiplied Alpha Blending:** Prevents massive point clusters from blowing out to pure white.
  - **Microscopic Point Sizing:** Dynamically locks point sizes to exact physical screen pixels (e.g. 1.4px).
  - **Stochastic Sub-pixel Dithering:** Employs TSL `hash()` and probabilistic `Discard` to simulate sub-8-bit opacities, allowing you to perfectly resolve delicate topological "cracks" and voids in dense UMAP embeddings.
- **Quadtree Streaming:** Engineered to stream and decode Apache Arrow binary files over the network asynchronously using Web Workers.
- **Interactive Spatial Indexing:** Uses bounding boxes and quadtree logic to only load and render tiles that intersect with the current camera view.

## 🛠️ Technology Stack

- **Framework:** Vite + TypeScript
- **Rendering:** Three.js (`three/webgpu` + `three/tsl`)
- **Data Format:** Apache Arrow (IPC format)
- **Concurrency:** Dedicated Web Workers for off-main-thread Arrow decoding

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
   The application will be available at `http://localhost:5173`.

### Building for Production
```bash
npm run build
```

## 🗺️ Architecture Overview
Deepgraph is split into a modular, multi-threaded pipeline:

1. **`Renderer.ts`:** Manages the WebGPU context, orthographic camera scaling, and continuous render loops. It broadcasts real-time physical screen metrics (Device Pixel Ratio, World-Units-per-Pixel) to the shaders.
2. **`TileManager.ts`:** Handles the spatial Quadtree indexing. It detects when the camera pan/zooms into new bounding box territories and requests raw Arrow binary data.
3. **`ArrowWorker.ts`:** A dedicated Web Worker that parses the Apache Arrow binary files from the network and extracts them into typed `Float32Array` buffers (Positions, Colors, etc.) to prevent main-thread UI stalling.
4. **`main.ts` (TSL Engine):** The heart of the visualization. An `InstancedMesh` with a `MeshBasicNodeMaterial` entirely rewritten in Three.js Shading Language to handle custom Pre-multiplied Blending, fwidth derivatives, and stochastic sub-pixel dithering.

## 📄 License
MIT License
