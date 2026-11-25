import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const distIndexDts = path.join(rootDir, 'dist', 'index.d.ts');
const distStandaloneJs = path.join(rootDir, 'dist', 'standalone.js');
const tscCli = path.join(rootDir, 'node_modules', 'typescript', 'bin', 'tsc');
const viteCli = path.join(rootDir, 'node_modules', 'vite', 'bin', 'vite.js');

if (!fs.existsSync(distIndexDts) || !fs.existsSync(distStandaloneJs)) {
    throw new Error('dist/index.d.ts or dist/standalone.js not found — run `npm run build:lib` first.');
}

const tmpConsumerDir = fs.mkdtempSync(path.join(os.tmpdir(), 'deepgraph-consumer-dts-'));

function linkRootDep(nmRoot, depName) {
    const src = path.join(rootDir, 'node_modules', depName);
    const dst = path.join(nmRoot, depName);
    fs.mkdirSync(path.dirname(dst), { recursive: true });
    fs.symlinkSync(src, dst, 'junction');
}

function walkFiles(dir) {
    const out = [];
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) out.push(...walkFiles(full));
        else out.push(full);
    }
    return out;
}

try {
    const nmRoot = path.join(tmpConsumerDir, 'node_modules');
    const nmPkgDir = path.join(nmRoot, 'deepgraph');
    fs.mkdirSync(path.dirname(nmPkgDir), { recursive: true });
    fs.symlinkSync(rootDir, nmPkgDir, 'junction');

    // Link runtime dependencies so a real downstream Vite consumer resolves externalized imports cleanly
    for (const dep of [
        'pmtiles',
        'p-queue',
        'fflate',
        '@uwdata/flechette',
        '@uwdata/wk-geojson',
        '@bokuweb/zstd-wasm',
    ]) {
        if (fs.existsSync(path.join(rootDir, 'node_modules', dep))) {
            linkRootDep(nmRoot, dep);
        }
    }

    const consumerTs = `
import {
    Deepgraph,
    type CustomRangeSource,
    type TileSource,
    type DeepgraphOptions,
    type DeepgraphPlot,
    type DeepgraphEncoding,
    type DeepgraphFilter,
    type EdgeSpec,
    type InteractionMode,
    type PointHitEvent,
    type PointSelectEvent,
    type DatasetMetadataSummary,
} from 'deepgraph';
import { Deepgraph as StandaloneDeepgraph } from 'deepgraph/standalone';
import {
    DeepgraphMosaicClient,
    buildLassoDuckDbSql,
    type DeepgraphMosaicOptions,
} from 'deepgraph/mosaic';

export async function mountConsumer(el: HTMLElement): Promise<DeepgraphPlot> {
    if (!Deepgraph.isSupported() || !StandaloneDeepgraph.isSupported()) {
        throw new Error('No WebGPU');
    }
    const customSrc: CustomRangeSource = {
        key: 'mem://fixture',
        getBytes: async (_offset: number, length: number) => ({
            data: new ArrayBuffer(Math.max(0, length)),
        }),
    };
    const pmtilesCompatibleSrc: CustomRangeSource = {
        getKey: () => 'mem://pmtiles-compat',
        getBytes: customSrc.getBytes,
    };
    const sourceUnion: TileSource = pmtilesCompatibleSrc;
    void sourceUnion;
    const enc: DeepgraphEncoding = {
        x: { field: 'x_val', transform: 'literal' },
        color: { field: 'model', scale: 'viridis' },
        aggregation: 'points',
        pointSize: 1.1,
        zoomBalance: 0.22,
        pointOpacity: 'auto',
        exposure: 'auto',
    };
    const filt: DeepgraphFilter = {
        rect: { minX: -1, maxX: 1, minY: -1, maxY: 1 },
        ranges: { num_of_tokens: [10, 500] },
        unselectedOpacity: 0.08,
    };
    const edges: EdgeSpec[] = [
        { source: { x: 0, y: 0 }, target: { x: 1, y: 1 }, weight: 0.8, directed: true },
    ];
    const mode: InteractionMode = 'brush';
    const opts: DeepgraphOptions = {
        url: '/samples/lmsys_sample.arrowtiles',
        powerPreference: 'low-power',
        encoding: enc,
        interactionMode: mode,
        filter: filt,
        edges,
        resolveEdges: async (hit) => [
            {
                source: { x: hit.dataX, y: hit.dataY },
                target: { x: 0, y: 0 },
                sourceAttributes: hit.attributes,
                targetAttributes: { num_of_tokens: null },
            },
        ],
        resolveRowAttributesOnHover: true,
        resolveRowAttributes: async (rowId, attrs, rowIdBigInt, signal) => {
            if (signal?.aborted) return attrs;
            return {
                ...attrs,
                checkedId: rowIdBigInt ?? rowId,
            };
        },
    };
    const plot = await Deepgraph.create(el, opts);
    const meta: DatasetMetadataSummary | null = plot.getMetadata();
    void meta;
    const canGpu: boolean = plot.canApplyFilterOnGpu(filt);
    void canGpu;
    await plot.setFilteredRowIds([1n, 2n]);
    await plot.setFilteredRowIds(null);
    plot.on('click', (hit: PointHitEvent | null) => {
        if (hit) console.log(hit.rowId, hit.dataX, hit.dataY);
    });
    plot.on('select', (ev: PointSelectEvent) => {
        console.log(ev.clause?.kind, ev.points.length);
    });
    const sqlStr: string = buildLassoDuckDbSql('x', 'y', [
        [0, 0],
        [1, 0],
        [0, 1],
    ]);
    void sqlStr;
    const mosaicOpts: DeepgraphMosaicOptions = {
        plot,
        table: 'items',
        coordinator: null,
    };
    const mc = new DeepgraphMosaicClient(mosaicOpts);
    mc.destroy();
    const picked: PointHitEvent | null = await plot.pickAt(100, 100);
    void picked;
    return plot;
}
`;
    fs.writeFileSync(path.join(tmpConsumerDir, 'consumer.ts'), consumerTs, 'utf8');
    fs.writeFileSync(
        path.join(tmpConsumerDir, 'package.json'),
        JSON.stringify({ name: 'external-consumer-fixture', private: true, type: 'module' }, null, 2),
        'utf8'
    );

    // Check 1: Standard `bundler` moduleResolution WITHOUT `allowImportingTsExtensions`,
    //          WITH `skipLibCheck: false`, and WITHOUT `@webgpu/types`!
    const bundlerTsconfig = {
        compilerOptions: {
            target: 'ES2022',
            module: 'ESNext',
            moduleResolution: 'bundler',
            lib: ['ES2022', 'DOM'],
            types: [],
            strict: true,
            skipLibCheck: false,
            noEmit: true,
        },
        include: ['consumer.ts'],
    };
    fs.writeFileSync(
        path.join(tmpConsumerDir, 'tsconfig.bundler.json'),
        JSON.stringify(bundlerTsconfig, null, 2),
        'utf8'
    );
    execFileSync(process.execPath, [tscCli, '-p', path.join(tmpConsumerDir, 'tsconfig.bundler.json')], {
        cwd: tmpConsumerDir,
        stdio: 'inherit',
    });

    // Check 2: Strict `nodenext` moduleResolution WITHOUT `allowImportingTsExtensions` and WITH `skipLibCheck: false`!
    const nodeNextTsconfig = {
        compilerOptions: {
            target: 'ES2022',
            module: 'NodeNext',
            moduleResolution: 'NodeNext',
            lib: ['ES2022', 'DOM'],
            types: [],
            strict: true,
            skipLibCheck: false,
            noEmit: true,
        },
        include: ['consumer.ts'],
    };
    fs.writeFileSync(
        path.join(tmpConsumerDir, 'tsconfig.nodenext.json'),
        JSON.stringify(nodeNextTsconfig, null, 2),
        'utf8'
    );
    execFileSync(
        process.execPath,
        [tscCli, '-p', path.join(tmpConsumerDir, 'tsconfig.nodenext.json')],
        {
            cwd: tmpConsumerDir,
            stdio: 'inherit',
        }
    );

    // Check 3: Real `vite build` Consumer App importing externalized `dist/index.js`, `deepgraph/mosaic`, & `@uwdata/flechette`!
    for (const mosaicDep of ['@uwdata/mosaic-core', '@uwdata/mosaic-sql', '@duckdb/duckdb-wasm', 'apache-arrow', 'qs', 'tslib', 'flatbuffers']) {
        if (fs.existsSync(path.join(rootDir, 'node_modules', mosaicDep))) {
            linkRootDep(nmRoot, mosaicDep);
        }
    }
    const viteAppTs = `
import { Deepgraph } from 'deepgraph';
import { DeepgraphMosaicClient, buildLassoDuckDbSql } from 'deepgraph/mosaic';
import { Selection } from '@uwdata/mosaic-core';
import { tableFromArrays } from '@uwdata/flechette';

const sharedTable = tableFromArrays({ v: new Int32Array([1, 2, 3]) });
const stage = document.getElementById('stage');
const polySql = buildLassoDuckDbSql('x', 'y', [[0, 0], [1, 0], [0, 1]]);
if (stage && Deepgraph.isSupported() && sharedTable.numRows === 3 && polySql.length > 0) {
    const brush = Selection.crossfilter();
    void Deepgraph.create(stage, {
        url: '/samples/lmsys_sample.arrowtiles',
        powerPreference: 'low-power',
    }).then((plot) => {
        new DeepgraphMosaicClient({ plot, table: 'lmsys', as: brush, filterBy: brush, coordinator: null });
    });
}
`;
    fs.writeFileSync(path.join(tmpConsumerDir, 'app.ts'), viteAppTs, 'utf8');
    fs.writeFileSync(
        path.join(tmpConsumerDir, 'index.html'),
        '<!doctype html><html><body><div id="stage"></div><script type="module" src="/app.ts"></script></body></html>',
        'utf8'
    );

    execFileSync(process.execPath, [viteCli, 'build', '--outDir', 'dist-consumer', '--logLevel', 'warn'], {
        cwd: tmpConsumerDir,
        stdio: 'inherit',
    });

    const consumerFiles = walkFiles(path.join(tmpConsumerDir, 'dist-consumer'));
    const consumerJsFiles = consumerFiles.filter((f) => f.endsWith('.js'));
    const consumerWorker = consumerJsFiles.find((f) => path.basename(f).startsWith('pmtiles.worker-'));
    if (!consumerWorker) {
        throw new Error(
            `Consumer vite build failed to emit pmtiles.worker-*.js asset; emitted: ${consumerFiles.map((f) => path.basename(f)).join(', ')}`
        );
    }
    const consumerWorkerCode = fs.readFileSync(consumerWorker, 'utf8');
    if (!consumerWorkerCode.includes('data:application/wasm;base64,')) {
        throw new Error('Consumer vite build worker chunk lost inlined base64 zstd.wasm');
    }
    for (const jf of consumerJsFiles) {
        const code = fs.readFileSync(jf, 'utf8');
        if (/["']\/assets\/[^"']+\.wasm["']/.test(code)) {
            throw new Error(`Broken /assets/*.wasm path in consumer bundle ${path.basename(jf)}`);
        }
    }

    console.log(
        `[verify_consumer_dts] PASS: External consumer tsc --noEmit ('bundler' + 'nodenext', '.' + './standalone') AND real 'vite build' consumer app (${consumerJsFiles.length} JS chunks incl. self-contained worker) succeeded.`
    );
} finally {
    fs.rmSync(tmpConsumerDir, { recursive: true, force: true });
}
