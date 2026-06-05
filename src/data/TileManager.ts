import { tableFromIPC, Table } from 'apache-arrow';

export interface BoundingBox {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export class TileNode {
  z: number;
  x: number;
  y: number;
  bounds: BoundingBox;
  table: Table | null = null;
  loading: boolean = false;
  children: TileNode[] | null = null;

  constructor(z: number, x: number, y: number, bounds: BoundingBox) {
    this.z = z;
    this.x = x;
    this.y = y;
    this.bounds = bounds;
  }

  // Check if this tile intersects with the camera viewport
  intersects(viewport: BoundingBox): boolean {
    return !(
      viewport.maxX < this.bounds.minX ||
      viewport.minX > this.bounds.maxX ||
      viewport.maxY < this.bounds.minY ||
      viewport.minY > this.bounds.maxY
    );
  }
}

export class TileManager {
  private baseUrl: string;
  public root: TileNode | null = null;
  public activeTables: Table[] = [];
  
  // Track fetching to avoid duplicate requests
  private fetchCache: Map<string, Promise<Table | null>> = new Map();

  constructor(baseUrl: string, rootBounds: BoundingBox = { minX: 0, minY: 0, maxX: 100, maxY: 100 }) {
    this.baseUrl = baseUrl;
    this.root = new TileNode(0, 0, 0, rootBounds);
  }

  public async init() {
    await this.loadTile(this.root!);
  }

  private getTileUrl(z: number, x: number, y: number): string {
    return `${this.baseUrl}/${z}/${x}/${y}.feather`;
  }

  public async loadTile(node: TileNode): Promise<Table | null> {
    const key = `${node.z}/${node.x}/${node.y}`;
    if (this.fetchCache.has(key)) {
      return this.fetchCache.get(key)!;
    }

    const promise = (async () => {
      try {
        const url = this.getTileUrl(node.z, node.x, node.y);
        const response = await fetch(url, { cache: 'no-cache' });
        if (!response.ok) {
          // Normal for non-existent children
          return null;
        }
        const buffer = await response.arrayBuffer();
        const table = tableFromIPC(buffer);
        node.table = table;
        console.log(`Loaded tile ${key} with ${table.numRows} rows.`);
        return table;
      } catch (e) {
        console.warn(`Failed to fetch tile ${key}:`, e);
        return null;
      }
    })();

    this.fetchCache.set(key, promise);
    return promise;
  }

  // Traverse the quadtree and collect tiles that should be rendered
  // Returns an array of Arrow Tables
  public async getVisibleTiles(viewport: BoundingBox, maxZoom: number): Promise<Table[]> {
    if (!this.root) return [];
    
    const visibleTables: Table[] = [];
    const queue: TileNode[] = [this.root];

    while (queue.length > 0) {
      const node = queue.shift()!;
      // console.log(`Checking intersection for node ${node.z}/${node.x}/${node.y} bounds:`, node.bounds, `against viewport:`, viewport);

      if (!node.intersects(viewport)) {
        // console.log(`Node ${node.z}/${node.x}/${node.y} DOES NOT intersect viewport`);
        continue;
      }
      
      // console.log(`Node ${node.z}/${node.x}/${node.y} INTERSECTS viewport`);

      // If we don't have the table yet, start loading it
      if (!node.table && !this.fetchCache.has(`${node.z}/${node.x}/${node.y}`)) {
        console.log(`Starting load for ${node.z}/${node.x}/${node.y}`);
        this.loadTile(node); // Background load
      }

      // If we have data, we can render this tile's base points
      if (node.table) {
        visibleTables.push(node.table);
      }

      // If we are below maxZoom, we should traverse children
      if (node.z < maxZoom) {
        if (!node.children) {
          this.createChildren(node);
        }
        queue.push(...node.children!);
      }
    }

    this.activeTables = visibleTables;
    return visibleTables;
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
