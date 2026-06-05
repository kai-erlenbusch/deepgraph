import * as THREE from 'three';
import { Renderer } from './core/Renderer';
import { TileManager, BoundingBox } from './data/TileManager';
import { Table } from 'apache-arrow';

const TILE_SERVER_URL = '/data';

class Scatterplot {
  private scene: THREE.Scene;
  private pointsMaterial: THREE.PointsMaterial;
  private tileMeshes: Map<Table, THREE.Points> = new Map();

  constructor(scene: THREE.Scene) {
    this.scene = scene;
    this.pointsMaterial = new THREE.PointsMaterial({
      size: 2,
      sizeAttenuation: false,
      color: 0xffffff, // White base to let vertex colors show through
      vertexColors: true, // Enable per-point coloring
      transparent: true,
      opacity: 0.8
    });
  }

  public updateTiles(tables: Table[]) {
    // Determine which tables are no longer needed
    const currentTables = new Set(tables);
    for (const [table, mesh] of this.tileMeshes.entries()) {
      if (!currentTables.has(table)) {
        this.scene.remove(mesh);
        mesh.geometry.dispose();
        this.tileMeshes.delete(table);
      }
    }

    // Add new tables
    for (const table of tables) {
      if (!this.tileMeshes.has(table)) {
        this.addTile(table);
      }
    }
  }

  private addTile(table: Table) {
    const numRows = table.numRows;
    

    // Use vector access directly to avoid toArray() crashes on some types
    const xCol = table.getChild('x');
    const yCol = table.getChild('y');

    if (!xCol || !yCol) {
      console.warn('Table is missing x or y columns');
      return;
    }

    // Create interleaved or separated position buffer
    const positions = new Float32Array(numRows * 3);
    for (let i = 0; i < numRows; i++) {
      positions[i * 3 + 0] = xCol.get(i) as number;
      positions[i * 3 + 1] = yCol.get(i) as number;
      positions[i * 3 + 2] = 0; // z is 0
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    // Generate colors from model_id
    const modelIdCol = table.getChild('model_id');
    if (modelIdCol) {
      const colors = new Float32Array(numRows * 3);
      
      // Categorical palette
      const palette = [
        [0.2, 0.6, 1.0], // gpt-4 (Blue)
        [1.0, 0.4, 0.2], // claude-3 (Orange)
        [0.4, 0.8, 0.4], // gemini-1.5 (Green)
        [0.8, 0.2, 0.6], // llama-3 (Pink)
        [0.9, 0.8, 0.2]  // mixtral (Yellow)
      ];
      
      for (let i = 0; i < numRows; i++) {
        // modelIdCol might be a string or a dictionary. 
        // We'll just hash the string or grab its length to assign a deterministic color
        const val = modelIdCol.get(i);
        let id = 0;
        if (typeof val === 'string') {
          id = val.length % 5; // Hacky hash just to visualize different models
        } else if (typeof val === 'number') {
          id = Math.abs(Math.floor(val)) % 5;
        }

        const c = palette[id];
        colors[i * 3 + 0] = c[0];
        colors[i * 3 + 1] = c[1];
        colors[i * 3 + 2] = c[2];
      }
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    }

    const points = new THREE.Points(geometry, this.pointsMaterial);
    points.frustumCulled = false; // We handle culling manually via TileManager
    
    this.scene.add(points);
    this.tileMeshes.set(table, points);
  }
}

async function init() {
  const container = document.getElementById('app')!;
  const uiText = document.querySelector('#ui p')!;

  if (!navigator.gpu) {
    uiText.textContent = 'WebGPU is not supported by your browser.';
    return;
  }

  const rendererWrapper = new Renderer(container);
  await rendererWrapper.init();

  // Real LMSys dataset bounds calculated by build_quadtree.py
  const rootBounds: BoundingBox = { 
    minX: -13.004743576049805, 
    maxX: 27.21806526184082, 
    minY: -18.281795501708984, 
    maxY: 21.94101333618164 
  };
  const tileManager = new TileManager(TILE_SERVER_URL, rootBounds);
  
  uiText.innerHTML = `WebGPU is supported!<br/>Streaming Quadtree Tiles...`;

  const scatterplot = new Scatterplot(rendererWrapper.scene);

  let lastBoundsString = "";

  rendererWrapper.renderer.setAnimationLoop(async () => {
    try {
      // 1. Get viewport bounds
      const bounds = rendererWrapper.getViewportBounds();
      const zoomLevel = Math.max(0, Math.floor(Math.log2(rendererWrapper.camera.zoom)));
      
      // 2. Fetch visible tiles
      const visibleTables = await tileManager.getVisibleTiles(bounds, zoomLevel);
      
      // 3. Update scatterplot geometry
      scatterplot.updateTiles(visibleTables);
      
      let totalPoints = 0;
      for (const t of visibleTables) totalPoints += t.numRows;
      uiText.innerHTML = `Streaming Quadtree<br/>Tiles rendered: ${visibleTables.length}<br/>Points: ${totalPoints}<br/>Zoom Level: ${zoomLevel}`;

      // 4. Render
      rendererWrapper.render();
    } catch (err) {
      console.error("Animation loop crash:", err);
      rendererWrapper.renderer.setAnimationLoop(null); // Stop loop to avoid 3000 errors
    }
  });
}

init().catch(console.error);
