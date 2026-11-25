import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const pythonDir = path.resolve(__dirname, '..');
const rootDir = path.resolve(pythonDir, '..');

// 1. Sync root CC-BY-NC-4.0 LICENSE and NOTICE into python/ for PyPI sdist & wheel packaging
for (const fname of ['LICENSE', 'NOTICE']) {
    const src = path.join(rootDir, fname);
    const dst = path.join(pythonDir, fname);
    if (!fs.existsSync(src)) {
        throw new Error(`[sync_assets] Missing root ${fname} at ${src}`);
    }
    fs.copyFileSync(src, dst);
}

// 2. Verify self-contained offline widget.js bundle
const widgetJs = path.join(pythonDir, 'pydeepgraph', '_static', 'widget.js');
if (!fs.existsSync(widgetJs)) {
    throw new Error(`[sync_assets] Missing built widget bundle at ${widgetJs}`);
}
const code = fs.readFileSync(widgetJs, 'utf8');
const staticImportRe = /(?:^|[;\n])\s*import\s+(?:[^"']*?\s+from\s+)?["']([^"']+)["']/g;
const imports = [...code.matchAll(staticImportRe)].map((m) => m[1]);
if (imports.length > 0) {
    throw new Error(
        `[sync_assets] widget.js must be 100% self-contained (0 external imports), found: ${JSON.stringify(imports)}`
    );
}
if (!code.includes('createObjectURL(new Blob(')) {
    throw new Error('[sync_assets] widget.js is missing inlined Blob worker URL');
}
if (code.includes('@uwdata/mosaic') || code.includes('duckdb-wasm')) {
    throw new Error('[sync_assets] widget.js leaked Mosaic/DuckDB-WASM dependencies');
}
const sizeKiB = (Buffer.byteLength(code, 'utf8') / 1024).toFixed(1);
console.log(
    `[sync_assets] Verified self-contained pydeepgraph/_static/widget.js (${sizeKiB} KiB, 0 external imports) & synced LICENSE/NOTICE.`
);
