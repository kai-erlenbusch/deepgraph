import { tableFromIPC } from 'apache-arrow';

// The categorical palette
const hexPalette = [0x173F5F, 0x20639B, 0x3CAEA3, 0xF6D55C, 0xED553B];
const palette = hexPalette.map(h => {
  return [
    ((h >> 16) & 255) / 255,
    ((h >> 8) & 255) / 255,
    (h & 255) / 255
  ];
});

self.onmessage = async (e: MessageEvent) => {
  const { url, key } = e.data;
  
  try {
    const response = await fetch(url, { cache: 'no-cache' });
    
    if (!response.ok) {
        throw new Error(`HTTP ${response.status} - Tile not found or server error.`);
    }
    
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('text/html')) {
        throw new Error('Received HTML fallback instead of binary data (Tile missing).');
    }

    const buffer = await response.arrayBuffer();
    const table = tableFromIPC(buffer);
    const numRows = table.numRows;
    
    const xCol = table.getChild('x');
    const yCol = table.getChild('y');
    const modelIdCol = table.getChild('model_id');
    
    if (!xCol || !yCol) {
      self.postMessage({ key, error: 'Missing columns' });
      return;
    }

    const positions = new Float32Array(numRows * 3);
    const colors = new Float32Array(numRows * 3);
    
    for (let i = 0; i < numRows; i++) {
      positions[i * 3 + 0] = xCol.get(i) as number;
      positions[i * 3 + 1] = yCol.get(i) as number;
      positions[i * 3 + 2] = 0;

      let id = 0;
      if (modelIdCol) {
        const val = modelIdCol.get(i);
        if (typeof val === 'number' || typeof val === 'bigint') {
          id = Number(Math.abs(Number(val))) % 5;
        }
      }
      
      const c = palette[id];
      colors[i * 3 + 0] = c[0];
      colors[i * 3 + 1] = c[1];
      colors[i * 3 + 2] = c[2];
    }
    
    // Transfer the typed arrays back to the main thread (zero-copy)
    self.postMessage(
      { key, positions, colors, numRows }, 
      { transfer: [positions.buffer, colors.buffer] }
    );
    
  } catch (err) {
    self.postMessage({ key, error: String(err) });
  }
};
