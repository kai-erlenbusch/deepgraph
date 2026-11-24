import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const distDir = path.join(rootDir, 'dist');
const tscCli = path.join(rootDir, 'node_modules', 'typescript', 'bin', 'tsc');

// 1. Emit declarations via TypeScript 6.0 (`rootDir: "./src"`, `stripInternal: true`)
execFileSync(process.execPath, [tscCli, '-p', path.join(rootDir, 'tsconfig.build.json')], {
    cwd: rootDir,
    stdio: 'inherit',
});

// 2. Prune internal unreferenced .d.ts files so only the sealed public declaration graph ships
const allowedDtsRel = new Set(['index.d.ts', path.join('lib', 'types.d.ts')]);

function walkFiles(dir) {
    const out = [];
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            out.push(...walkFiles(full));
        } else {
            out.push(full);
        }
    }
    return out;
}

for (const file of walkFiles(distDir)) {
    if (!file.endsWith('.d.ts')) continue;
    const rel = path.relative(distDir, file);
    if (!allowedDtsRel.has(rel)) {
        fs.unlinkSync(file);
    }
}

function pruneEmptyDirs(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        if (entry.isDirectory()) {
            const sub = path.join(dir, entry.name);
            pruneEmptyDirs(sub);
            if (fs.readdirSync(sub).length === 0) {
                fs.rmdirSync(sub);
            }
        }
    }
}
pruneEmptyDirs(distDir);

// 3. Rewrite TypeScript 6.0 `.ts` relative specifiers in emitted `.d.ts` files to `.js` (`TS5097` fix)
const dtsFiles = walkFiles(distDir).filter((f) => f.endsWith('.d.ts'));
if (dtsFiles.length === 0) {
    throw new Error('No .d.ts files found in dist/');
}

for (const dtsPath of dtsFiles) {
    const raw = fs.readFileSync(dtsPath, 'utf8');
    const rewritten = raw
        .replace(/(from\s+['"]\.\.?\/[^'"]+?)\.ts(['"])/g, '$1.js$2')
        .replace(/(import\(\s*['"]\.\.?\/[^'"]+?)\.ts(['"]\s*\))/g, '$1.js$2');
    fs.writeFileSync(dtsPath, rewritten, 'utf8');
}

// 4. Hard hygiene assertions on emitted .d.ts and .js bundles
for (const dtsPath of dtsFiles) {
    const content = fs.readFileSync(dtsPath, 'utf8');
    if (/['"]\.\.?\/[^'"]+\.ts['"]/.test(content)) {
        throw new Error(`Found lingering .ts specifier in ${dtsPath}`);
    }
    for (const forbidden of [
        'INTERNAL_DIAGNOSTICS',
        'ViewerLayer',
        'getRowAttributeResolver',
        'TileGpuStore',
        'PMTilesClient',
        'lil-gui',
        'three',
        'apache-arrow',
    ]) {
        if (content.includes(forbidden)) {
            throw new Error(`Forbidden internal/retired symbol '${forbidden}' leaked into ${dtsPath}`);
        }
    }
}

const distIndexPath = path.join(distDir, 'index.js');
const distStandalonePath = path.join(distDir, 'standalone.js');
if (!fs.existsSync(distIndexPath)) {
    throw new Error('Missing bundler entry dist/index.js');
}
if (!fs.existsSync(distStandalonePath)) {
    throw new Error('Missing self-contained entry dist/standalone.js');
}

function findBareImports(jsCode) {
    const matches = [];
    const importRe = /(?:^|[;\n])\s*import\s+(?:[^'"]+?\s+from\s+)?['"]([^'"]+)['"]/g;
    let m;
    while ((m = importRe.exec(jsCode)) !== null) {
        const spec = m[1];
        if (!spec.startsWith('./') && !spec.startsWith('../') && !spec.startsWith('data:')) {
            matches.push(spec);
        }
    }
    return matches;
}

const indexJs = fs.readFileSync(distIndexPath, 'utf8');
const indexBareImports = findBareImports(indexJs);
for (const expectedExt of ['pmtiles', 'p-queue', 'fflate']) {
    if (!indexBareImports.includes(expectedExt)) {
        throw new Error(
            `Expected dist/index.js to externalize '${expectedExt}', found imports: ${JSON.stringify(indexBareImports)}`
        );
    }
}
for (const forbiddenMainArrow of ['apache-arrow', '@uwdata/flechette']) {
    if (indexBareImports.includes(forbiddenMainArrow) || indexJs.includes(forbiddenMainArrow)) {
        throw new Error(
            `Main-thread dist/index.js must carry 0 B Arrow parser weight, found '${forbiddenMainArrow}'`
        );
    }
}
if (Buffer.byteLength(indexJs, 'utf8') > 220 * 1024) {
    throw new Error(
        `dist/index.js (${Buffer.byteLength(indexJs, 'utf8')} B) exceeds 220 KiB externalized bundler ceiling`
    );
}

const standaloneJs = fs.readFileSync(distStandalonePath, 'utf8');
const standaloneBareImports = findBareImports(standaloneJs);
if (standaloneBareImports.length > 0) {
    throw new Error(
        `dist/standalone.js must be 100% self-contained for <script type="module">, found bare imports: ${JSON.stringify(standaloneBareImports)}`
    );
}
if (standaloneJs.includes('pmtiles.worker-') || !standaloneJs.includes('URL.createObjectURL')) {
    throw new Error(
        'dist/standalone.js must inline its Web Worker via cached URL.createObjectURL(Blob) with zero external pmtiles.worker-*.js references'
    );
}

const jsFiles = walkFiles(distDir).filter((f) => f.endsWith('.js'));
const workerFiles = jsFiles.filter((f) => path.basename(f).startsWith('pmtiles.worker-'));
if (workerFiles.length !== 1) {
    throw new Error(
        `Expected exactly 1 dist/assets/pmtiles.worker-*.js chunk (for dist/index.js), found ${workerFiles.length}: ${workerFiles.map((f) => path.basename(f)).join(', ')}`
    );
}

for (const workerPath of workerFiles) {
    const workerJs = fs.readFileSync(workerPath, 'utf8');
    const bareWorkerImports = findBareImports(workerJs);
    if (bareWorkerImports.length > 0) {
        throw new Error(
            `Worker chunk ${path.relative(rootDir, workerPath)} must stay self-contained in both builds, found bare imports: ${JSON.stringify(bareWorkerImports)}`
        );
    }
    if (!workerJs.includes('data:application/wasm;base64,')) {
        throw new Error(`Worker chunk ${path.relative(rootDir, workerPath)} is missing inlined base64 zstd.wasm`);
    }
}

for (const jsPath of jsFiles) {
    const js = fs.readFileSync(jsPath, 'utf8');
    for (const forbiddenJs of ['lil-gui', '__gpuStats', 'installGpuStatsInstrumentation']) {
        if (js.includes(forbiddenJs)) {
            throw new Error(`Forbidden demo/test symbol '${forbiddenJs}' bundled into ${jsPath}`);
        }
    }
    if (/["']\/assets\/[^"']+\.wasm["']/.test(js)) {
        throw new Error(`Broken root-relative /assets/*.wasm URL found in ${jsPath}`);
    }
}

console.log(
    `[emit_dts] Emitted & verified ${dtsFiles.length} clean .d.ts files, externalized dist/index.js (${(
        Buffer.byteLength(indexJs, 'utf8') / 1024
    ).toFixed(1)} KiB, externals=${JSON.stringify(indexBareImports)}), self-contained dist/standalone.js (${(
        Buffer.byteLength(standaloneJs, 'utf8') / 1024
    ).toFixed(1)} KiB), and ${workerFiles.length} self-contained worker chunk(s).`
);
