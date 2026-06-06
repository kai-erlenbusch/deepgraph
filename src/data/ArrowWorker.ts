import { tableFromIPC } from 'apache-arrow';

// The categorical palette
const hexPalette = [0x173F5F, 0x20639B, 0x3CAEA3, 0xF6D55C, 0xED553B];
const palette = hexPalette.map(h => {
  return [
    (h >> 16) & 255,
    (h >> 8) & 255,
    h & 255
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

    const interleavedBuffer = new ArrayBuffer(numRows * 16);
    const floatView = new Float32Array(interleavedBuffer);
    const byteView = new Uint8Array(interleavedBuffer);
    
    for (let i = 0; i < numRows; i++) {
      floatView[i * 4 + 0] = xCol.get(i) as number;
      floatView[i * 4 + 1] = yCol.get(i) as number;
      floatView[i * 4 + 2] = 0.5 + Math.random() * 2.0; // Size
      // bytes 12-15 = Color RGBA

      let id = 0;
      if (modelIdCol) {
        const val = modelIdCol.get(i);
        if (typeof val === 'number' || typeof val === 'bigint') {
          id = Number(Math.abs(Number(val))) % 5;
        }
      }
      
      const c = palette[id];
      byteView[i * 16 + 12] = c[0]; // R
      byteView[i * 16 + 13] = c[1]; // G
      byteView[i * 16 + 14] = c[2]; // B
      byteView[i * 16 + 15] = 255;  // A
    }
    
    // Transfer the single buffer back to the main thread (zero-copy)
    self.postMessage(
      { key, interleavedBuffer, numRows }, 
      { transfer: [interleavedBuffer] }
    );
    
  } catch (err) {
    self.postMessage({ key, error: String(err) });
  }
};
