import { tableToIPC, vectorFromArray, Table } from 'apache-arrow';
import fs from 'fs';

const NUM_NODES = 10000;
const x = new Float32Array(NUM_NODES);
const y = new Float32Array(NUM_NODES);
const vx = new Float32Array(NUM_NODES);
const vy = new Float32Array(NUM_NODES);

for (let i = 0; i < NUM_NODES; i++) {
    x[i] = (Math.random() - 0.5) * 100;
    y[i] = (Math.random() - 0.5) * 100;
    vx[i] = 0;
    vy[i] = 0;
}

const table = new Table({
    x: vectorFromArray(x),
    y: vectorFromArray(y),
    vx: vectorFromArray(vx),
    vy: vectorFromArray(vy),
});

const ipcBuffer = tableToIPC(table);
// Create public directory if it doesn't exist
if (!fs.existsSync('./public')) {
    fs.mkdirSync('./public');
}
fs.writeFileSync('./public/test.feather', ipcBuffer);
console.log(`Generated test.feather with ${NUM_NODES} nodes in /public`);
