import * as THREE from 'three';

export interface BoundingBox {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export interface TileData {
  key: string;
  interleavedBuffer: ArrayBuffer;
  numRows: number;
}

export class TileNode {
  z: number;
  x: number;
  y: number;
  bounds: BoundingBox;
  box3: THREE.Box3;
  tileData: TileData | null = null;
  lastAccessFrame: number = 0;
  children: TileNode[] | null = null;

  constructor(z: number, x: number, y: number, bounds: BoundingBox) {
    this.z = z;
    this.x = x;
    this.y = y;
    this.bounds = bounds;
    this.box3 = new THREE.Box3(
      new THREE.Vector3(bounds.minX, bounds.minY, -1000),
      new THREE.Vector3(bounds.maxX, bounds.maxY, 1000)
    );
  }

  // Check if this tile intersects with the camera viewport
  intersects(frustum: THREE.Frustum): boolean {
    return frustum.intersectsBox(this.box3);
  }
}

export class TileManager {
  private baseUrl: string;
  public root: TileNode | null = null;
  public activeTiles: TileData[] = [];
  
  // Track fetching to avoid duplicate requests
  private fetchCache: Map<string, Promise<TileData | null>> = new Map();
  private pendingRequests: Map<string, (data: TileData | null) => void> = new Map();
  private worker: Worker;

  // LRU Cache tracking
  private currentFrame = 0;
  public maxCacheSize = 50; // Maximum number of loaded tiles in memory

  constructor(baseUrl: string, rootBounds: BoundingBox = { minX: 0, minY: 0, maxX: 100, maxY: 100 }) {
    this.baseUrl = baseUrl;
    this.root = new TileNode(0, 0, 0, rootBounds);

    // Initialize Web Worker
    this.worker = new Worker(new URL('./ArrowWorker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e) => {
      const { key, interleavedBuffer, numRows, error } = e.data;
      const resolve = this.pendingRequests.get(key);
      if (resolve) {
        if (error || !interleavedBuffer) {
          if (error !== '404') console.warn(`Worker error for ${key}:`, error);
          resolve(null);
        } else {
          resolve({ key, interleavedBuffer, numRows });
        }
        this.pendingRequests.delete(key);
      }
    };
  }

  public async init() {
    await this.loadTile(this.root!);
  }

  private getTileUrl(z: number, x: number, y: number): string {
    return `${this.baseUrl}/${z}/${x}/${y}.feather`;
  }

  public async loadTile(node: TileNode): Promise<TileData | null> {
    const key = `${node.z}/${node.x}/${node.y}`;
    if (this.fetchCache.has(key)) {
      return this.fetchCache.get(key)!;
    }

    const promise = new Promise<TileData | null>((resolve) => {
      this.pendingRequests.set(key, resolve);
      this.worker.postMessage({ url: this.getTileUrl(node.z, node.x, node.y), key });
    }).then(data => {
      node.tileData = data;
      if (data) {
        console.log(`Loaded tile ${key} with ${data.numRows} rows via Worker.`);
      }
      return data;
    });

    this.fetchCache.set(key, promise);
    return promise;
  }

  // Traverse the quadtree and collect tiles that should be rendered
  // Returns an array of TileData
  public async getVisibleTiles(frustum: THREE.Frustum, maxZoom: number): Promise<TileData[]> {
    this.currentFrame++;
    if (!this.root) return [];
    
    const visibleTiles: TileData[] = [];
    const queue: TileNode[] = [this.root];

    while (queue.length > 0) {
      const node = queue.shift()!;

      if (!node.intersects(frustum)) {
        continue;
      }

      // If we don't have the table yet, start loading it
      if (!node.tileData && !this.fetchCache.has(`${node.z}/${node.x}/${node.y}`)) {
        console.log(`Starting background load for ${node.z}/${node.x}/${node.y}`);
        this.loadTile(node); 
      }

      // If we have data, we can render this tile's base points
      if (node.tileData) {
        node.lastAccessFrame = this.currentFrame;
        
        let allIntersectingChildrenLoaded = false;
        if (node.z < maxZoom) {
          if (!node.children) {
            this.createChildren(node);
          }
          
          const intersectingChildren = node.children!.filter(c => c.intersects(frustum));
          if (intersectingChildren.length > 0) {
            allIntersectingChildrenLoaded = intersectingChildren.every(c => c.tileData !== null);
            queue.push(...intersectingChildren);
          }
        }
        
        // Prevent LOD Overdraw: Only render parent if children aren't fully ready
        if (!allIntersectingChildrenLoaded) {
          visibleTiles.push(node.tileData);
        }
      }
    }

    this.activeTiles = visibleTiles;
    this.evictStaleTiles();
    return visibleTiles;
  }

  private evictStaleTiles() {
    // We count how many nodes actually have tileData
    let loadedCount = 0;
    const loadedNodes: TileNode[] = [];
    
    const traverse = (n: TileNode) => {
      if (n.tileData) {
        loadedCount++;
        loadedNodes.push(n);
      }
      if (n.children) n.children.forEach(traverse);
    };
    
    if (this.root) traverse(this.root);

    if (loadedCount <= this.maxCacheSize) return;

    // Sort by oldest access frame first
    loadedNodes.sort((a, b) => a.lastAccessFrame - b.lastAccessFrame);
    
    const excess = loadedCount - this.maxCacheSize;
    let evicted = 0;
    
    for (const node of loadedNodes) {
      if (evicted >= excess) break;
      // Never evict tiles that were accessed THIS frame
      if (node.lastAccessFrame === this.currentFrame) continue;
      
      const key = `${node.z}/${node.x}/${node.y}`;
      this.fetchCache.delete(key);
      node.tileData = null; // Drop reference so garbage collector can clean up
      evicted++;
      console.log(`Evicted tile ${key} from LRU Cache`);
    }
  }

  private createChildren(node: TileNode) {
    const { minX, minY, maxX, maxY } = node.bounds;
    const midX = (minX + maxX) / 2;
    const midY = (minY + maxY) / 2;

    const z = node.z + 1;
    const x = node.x * 2;
    const y = node.y * 2;

    node.children = [
      new TileNode(z, x, y, { minX, minY, maxX: midX, maxY: midY }),             // NW
      new TileNode(z, x + 1, y, { minX: midX, minY, maxX, maxY: midY }),         // NE
      new TileNode(z, x, y + 1, { minX, minY: midY, maxX: midX, maxY }),         // SW
      new TileNode(z, x + 1, y + 1, { minX: midX, minY: midY, maxX, maxY })      // SE
    ];
  }
}
