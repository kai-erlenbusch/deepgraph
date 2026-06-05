import * as THREE from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import { 
  attribute, float, positionLocal, vec3, vec2, uv, distance, smoothstep,
  fwidth, hash, instanceIndex, Discard, max
} from 'three/tsl';
import { Renderer } from './core/Renderer';
import { TileManager, BoundingBox, TileData } from './data/TileManager';

const TILE_SERVER_URL = '/data';

class Scatterplot {
  private scene: THREE.Scene;
  private material: MeshBasicNodeMaterial;
  private tileMeshes: Map<string, THREE.Mesh> = new Map();
  private quadGeometry = new THREE.PlaneGeometry(1, 1);

  private pickingScene: THREE.Scene;
  private pickingMaterial: MeshBasicNodeMaterial;
  private pickingMeshes: Map<string, THREE.Mesh> = new Map();
  private globalPickingId = 0;
  public pickingMap: Map<number, { tileKey: string, rowIndex: number }> = new Map();

  constructor(scene: THREE.Scene, rendererWrapper: Renderer) {
    this.scene = scene;
    
    this.material = new MeshBasicNodeMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.CustomBlending,
      blendSrc: THREE.OneFactor, // Pre-multiplying in shader
      blendDst: THREE.OneMinusSrcAlphaFactor,
      blendEquation: THREE.AddEquation,
    });

    // Dynamic quad scaling locked to exact microscopic physical pixels
    // Deepscatter typically uses 1.2 - 1.4 pixels for these massive datasets
    const targetPixels = float(1.4); 
    const size = targetPixels.mul(rendererWrapper.worldUnitsPerPixelUniform);
    
    // Calculate distance from the quad's center (0.5, 0.5)
    const dist = distance(uv(), vec2(0.5));
    
    // Mathematically perfect 1-pixel anti-aliasing via hardware derivatives
    const delta = fwidth(dist);
    const alphaEdge = smoothstep(float(0.5).add(delta), float(0.5).sub(delta), dist);
    
    // Density-Targeted Alpha (Low opacity for deep blending accumulation)
    const baseOpacity = float(0.04); 
    const finalAlpha = alphaEdge.mul(baseOpacity);

    // Stochastic Sub-pixel Dithering (The Secret Weapon)
    const threshold = float(1.0 / 255.0);
    const randomVal = hash(instanceIndex);
    const isSubPixelOpacity = finalAlpha.lessThan(threshold);
    const probDiscard = randomVal.greaterThan(finalAlpha.mul(255.0));
    
    // Discard entire instances probabilistically if they fall below monitor capabilities
    Discard(isSubPixelOpacity.and(probDiscard));
    
    // If it survives the discard, ensure it acts as at least 1/255 opacity
    const safeAlpha = max(finalAlpha, threshold);
    
    // Pre-multiply color by alpha before outputting to blending hardware
    const baseColor = attribute('instanceColor', 'vec3');
    this.material.colorNode = baseColor.mul(safeAlpha);
    this.material.opacityNode = safeAlpha;
    
    // World position: instance offset + local vertex position * size
    this.material.positionNode = attribute('offset', 'vec3').add(positionLocal.mul(size));

    // Picking Setup
    this.pickingScene = new THREE.Scene();
    this.pickingScene.background = new THREE.Color(0x000000); // 0 is null id
    this.pickingMaterial = new MeshBasicNodeMaterial({
      depthWrite: false,
      blending: THREE.NoBlending,
    });
    this.pickingMaterial.colorNode = attribute('pickingColor', 'vec3');
    this.pickingMaterial.positionNode = this.material.positionNode;
  }

  public updateTiles(tiles: TileData[]) {
    // Determine which tiles are no longer needed
    const currentKeys = new Set(tiles.map(t => t.key));
    for (const [key, mesh] of this.tileMeshes.entries()) {
      if (!currentKeys.has(key)) {
        this.scene.remove(mesh);
        mesh.geometry.dispose();
        this.tileMeshes.delete(key);
        
        const pickingMesh = this.pickingMeshes.get(key);
        if (pickingMesh) {
          this.pickingScene.remove(pickingMesh);
          this.pickingMeshes.delete(key);
        }
      }
    }

    // Add new tiles
    for (const tile of tiles) {
      if (!this.tileMeshes.has(tile.key)) {
        this.addTile(tile);
      }
    }
  }

  private addTile(tile: TileData) {
    const instancedGeometry = new THREE.InstancedBufferGeometry();
    instancedGeometry.index = this.quadGeometry.index;
    instancedGeometry.instanceCount = tile.numRows;
    instancedGeometry.attributes.position = this.quadGeometry.attributes.position;
    instancedGeometry.attributes.uv = this.quadGeometry.attributes.uv;

    instancedGeometry.setAttribute('offset', new THREE.InstancedBufferAttribute(tile.positions, 3));
    instancedGeometry.setAttribute('instanceColor', new THREE.InstancedBufferAttribute(tile.colors, 3));

    const sizes = new Float32Array(tile.numRows);
    for (let i = 0; i < tile.numRows; i++) {
      sizes[i] = 0.5 + Math.random() * 2.0; 
    }
    instancedGeometry.setAttribute('instanceSize', new THREE.InstancedBufferAttribute(sizes, 1));

    const pickingColors = new Float32Array(tile.numRows * 3);
    for (let i = 0; i < tile.numRows; i++) {
      const id = ++this.globalPickingId; // starts at 1
      this.pickingMap.set(id, { tileKey: tile.key, rowIndex: i });
      pickingColors[i * 3 + 0] = ((id >> 16) & 255) / 255;
      pickingColors[i * 3 + 1] = ((id >> 8) & 255) / 255;
      pickingColors[i * 3 + 2] = (id & 255) / 255;
    }
    instancedGeometry.setAttribute('pickingColor', new THREE.InstancedBufferAttribute(pickingColors, 3));

    const mesh = new THREE.Mesh(instancedGeometry, this.material);
    mesh.frustumCulled = false; // We handle culling manually via TileManager
    
    this.scene.add(mesh);
    this.tileMeshes.set(tile.key, mesh);

    const pickingMesh = new THREE.Mesh(instancedGeometry, this.pickingMaterial);
    pickingMesh.frustumCulled = false;
    this.pickingScene.add(pickingMesh);
    this.pickingMeshes.set(tile.key, pickingMesh);
  }

  public getPickingScene() {
    return this.pickingScene;
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

  const scatterplot = new Scatterplot(rendererWrapper.scene, rendererWrapper);

  const pickingTexture = new THREE.RenderTarget(window.innerWidth, window.innerHeight, {
    colorSpace: THREE.NoColorSpace
  });
  window.addEventListener('resize', () => {
    pickingTexture.setSize(window.innerWidth, window.innerHeight);
  });

  const mouse = new THREE.Vector2();
  let mouseMoved = false;

  window.addEventListener('mousemove', (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
    mouseMoved = true;
  });

  const tooltip = document.createElement('div');
  tooltip.style.position = 'absolute';
  tooltip.style.background = 'rgba(0,0,0,0.8)';
  tooltip.style.color = 'white';
  tooltip.style.padding = '5px';
  tooltip.style.borderRadius = '5px';
  tooltip.style.display = 'none';
  tooltip.style.pointerEvents = 'none';
  tooltip.style.zIndex = '1000';
  document.body.appendChild(tooltip);

  let lastBoundsString = "";

  rendererWrapper.renderer.setAnimationLoop(async () => {
    try {
      // 1. Get viewport bounds
      const bounds = rendererWrapper.getViewportBounds();
      const zoomLevel = Math.max(0, Math.floor(Math.log2(rendererWrapper.camera.zoom)));
      
      // 2. Fetch visible tiles
      const visibleTiles = await tileManager.getVisibleTiles(bounds, zoomLevel);
      
      // 3. Update scatterplot geometry
      scatterplot.updateTiles(visibleTiles);
      
      let totalPoints = 0;
      for (const t of visibleTiles) totalPoints += t.numRows;
      uiText.innerHTML = `Streaming Quadtree<br/>Tiles rendered: ${visibleTiles.length}<br/>Points: ${totalPoints}<br/>Zoom Level: ${zoomLevel}`;

      // 4. Render
      rendererWrapper.render();

      // 5. Picking Pass
      if (mouseMoved) {
        mouseMoved = false;
        
        rendererWrapper.renderer.setRenderTarget(pickingTexture);
        rendererWrapper.renderer.render(scatterplot.getPickingScene(), rendererWrapper.camera);
        rendererWrapper.renderer.setRenderTarget(null);
        
        // readRenderTargetPixelsAsync coordinates are usually bottom-left origin
        const pickY = window.innerHeight - mouse.y;
        const pixelBuffer = await rendererWrapper.renderer.readRenderTargetPixelsAsync(pickingTexture, mouse.x, pickY, 1, 1);
        
        const id = (pixelBuffer[0] << 16) | (pixelBuffer[1] << 8) | pixelBuffer[2];
        if (id > 0 && scatterplot.pickingMap.has(id)) {
          const data = scatterplot.pickingMap.get(id)!;
          tooltip.style.display = 'block';
          tooltip.style.left = mouse.x + 15 + 'px';
          tooltip.style.top = mouse.y + 15 + 'px';
          tooltip.style.fontFamily = 'monospace';
          tooltip.innerHTML = `Tile: ${data.tileKey}<br/>Row: ${data.rowIndex}<br/>Global ID: ${id}`;
        } else {
          tooltip.style.display = 'none';
        }
      }
    } catch (err) {
      console.error("Animation loop crash:", err);
      rendererWrapper.renderer.setAnimationLoop(null); // Stop loop to avoid 3000 errors
    }
  });
}

init().catch(console.error);
