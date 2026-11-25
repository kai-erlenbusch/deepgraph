import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
    Coordinator,
    MosaicClient,
    Selection,
    clauseInterval,
    clauseIntervals,
    clauseMatch,
    clausePoint,
} from '@uwdata/mosaic-core';
import {
    columnFromArray,
    tableFromArrays,
    tableFromColumns,
    tableToIPC,
} from '@uwdata/flechette';
import { Query, sql } from '@uwdata/mosaic-sql';
import {
    evaluatePointMatchesFilterCpu,
    FILTER_FLAG_HAS_CAT0,
    FILTER_FLAG_HAS_CAT1,
    FILTER_FLAG_HAS_GLOBAL,
    FILTER_FLAG_HAS_LASSO,
    FILTER_FLAG_HAS_RECT,
    FILTER_FLAG_HAS_SLOT_MASK,
    GLOBAL_FILTER_UNIFORM_BYTES,
    MAX_LASSO_GPU_VERTICES,
    packGlobalFilterUniforms,
    planGpuFilterExecution,
    pointInPolygon2D,
    projectDataToWorld64,
    simplifyRdpPolygon,
    SLOT_MASK_ROWS_PER_SLICE,
    SLOT_MASK_SLICE_BYTES,
    unprojectCssToData64,
    type AxisProjectionContext,
    type GpuFilterCapabilityContext,
} from '../../src/core/FilterGpuResources.ts';
import type { Camera2D } from '../../src/core/Renderer.ts';
import {
    compileSlotBitmaskSlices,
    extractTileRowIndexFromTables,
    queryTileSpatialSelection,
    testProbeRowIdsInFilter,
} from '../../src/workerCodec.ts';
import {
    buildLassoDuckDbSql,
    DeepgraphMosaicClient,
} from '../../src/mosaic.ts';
import type {
    DeepgraphFilter,
    DeepgraphPlot,
    PointSelectEvent,
    SelectionGeometryClause,
} from '../../src/lib/types.ts';

describe('FilterGpuResources — RDP Lasso, Float64 Projection, Idle-Slot Planner & Uniform Packing', () => {
    it('simplifies a 500-vertex concave 5-pointed star to <= 64 vertices while preserving interior & notch parity', () => {
        const star: [number, number][] = [];
        const cx = 50;
        const cy = 50;
        const outerR = 40;
        const innerR = 14;
        const totalPts = 500;
        for (let k = 0; k < totalPts; k++) {
            const frac = (k / totalPts) * 10; // 10 alternating outer/inner segments
            const seg = Math.floor(frac);
            const localT = frac - seg;
            const a0 = (seg * Math.PI) / 5 - Math.PI / 2;
            const a1 = ((seg + 1) * Math.PI) / 5 - Math.PI / 2;
            const r0 = seg % 2 === 0 ? outerR : innerR;
            const r1 = seg % 2 === 0 ? innerR : outerR;
            const x0 = cx + Math.cos(a0) * r0;
            const y0 = cy + Math.sin(a0) * r0;
            const x1 = cx + Math.cos(a1) * r1;
            const y1 = cy + Math.sin(a1) * r1;
            // Small deterministic jitter along each segment so RDP has real vertices to prune
            const jitter = Math.sin(k * 1.7) * 0.02;
            star.push([x0 + (x1 - x0) * localT + jitter, y0 + (y1 - y0) * localT - jitter]);
        }
        // Append explicit closing vertex to verify seam deduplication
        star.push([star[0][0], star[0][1]]);

        const simp = simplifyRdpPolygon(star, MAX_LASSO_GPU_VERTICES);
        assert.ok(simp.length >= 10 && simp.length <= MAX_LASSO_GPU_VERTICES);

        // Center is inside
        assert.equal(pointInPolygon2D(cx, cy, simp), true);
        // Top tip (50, ~15) is inside
        assert.equal(pointInPolygon2D(50, 18, simp), true);
        // Concave valley outside inner radius (angle -PI/2 + PI/5, radius 28 > innerR=14) is OUTSIDE!
        const valleyAngle = -Math.PI / 2 + Math.PI / 5;
        const vx = cx + Math.cos(valleyAngle) * 28;
        const vy = cy + Math.sin(valleyAngle) * 28;
        assert.equal(pointInPolygon2D(vx, vy, star), false);
        assert.equal(pointInPolygon2D(vx, vy, simp), false);
    });

    it('round-trips projectDataToWorld64 <-> unprojectCssToData64 in Float64 across spatial dataBounds, flipY, and 1D-RTO chart axes', () => {
        const cam: Camera2D = {
            position: {
                x: 0.37,
                y: 0.62,
                z: 0,
                set(x: number, y: number, z = 0) {
                    this.x = x;
                    this.y = y;
                    this.z = z;
                },
            },
            left: -1,
            right: 1,
            top: 1,
            bottom: -1,
            zoom: 64,
            updateProjectionMatrix() {},
            updateMatrixWorld() {},
        };

        // 1. Spatial dataBounds (Y-up world normalization)
        const spatialCtx: AxisProjectionContext = {
            xAxis: { field: 'x_u16', transform: 'linear', domain: [0, 1], range: [0, 1] },
            yAxis: { field: 'y_u16', transform: 'linear', domain: [0, 1], range: [0, 1] },
            rootBounds: { minX: 0, maxX: 1, minY: 0, maxY: 1 },
            dataBounds: { min_x: -18.5, max_x: 112.25, min_y: -42.0, max_y: 96.75 },
            effectiveFlipY: false,
            colOrigins: {},
        };
        const cssW = 960;
        const cssH = 600;
        const d0 = unprojectCssToData64(312.5, 184.25, cssW, cssH, cam, spatialCtx);
        const w0 = projectDataToWorld64(d0.dataX, d0.dataY, spatialCtx);
        const camLeft = cam.position.x + cam.left / cam.zoom;
        const camRight = cam.position.x + cam.right / cam.zoom;
        const camBottom = cam.position.y + cam.bottom / cam.zoom;
        const camTop = cam.position.y + cam.top / cam.zoom;
        const roundCssX = ((w0.worldX - camLeft) / (camRight - camLeft)) * cssW;
        const roundCssY = ((camTop - w0.worldY) / (camTop - camBottom)) * cssH;
        assert.ok(Math.abs(roundCssX - 312.5) < 1e-9);
        assert.ok(Math.abs(roundCssY - 184.25) < 1e-9);

        // 2. High-offset 1D-RTO chart axes (modern epoch-ms origin)
        const rtoOriginX = 1_710_000_000_000;
        const chartCtx: AxisProjectionContext = {
            xAxis: { field: 'created_ms', transform: 'literal', domain: [0, 86_400_000], range: [0, 1] },
            yAxis: { field: 'score', transform: 'literal', domain: [-5, 5], range: [0, 1] },
            rootBounds: { minX: 0, maxX: 1, minY: 0, maxY: 1 },
            effectiveFlipY: false,
            colOrigins: { created_ms: rtoOriginX, score: 100.0 },
        };
        const targetX = rtoOriginX + 12_345.678;
        const targetY = 102.375;
        const wc = projectDataToWorld64(targetX, targetY, chartCtx);
        const cssX2 = ((wc.worldX - camLeft) / (camRight - camLeft)) * cssW;
        const cssY2 = ((camTop - wc.worldY) / (camTop - camBottom)) * cssH;
        const back = unprojectCssToData64(cssX2, cssY2, cssW, cssH, cam, chartCtx);
        assert.ok(Math.abs(back.dataX - targetX) < 1e-4);
        assert.ok(Math.abs(back.dataY - targetY) < 1e-9);
    });

    it('plans up to 2 extra GPU filter columns on spatial maps and rejects cold/overflow predicates cleanly', () => {
        const baseCtx: GpuFilterCapabilityContext = {
            isXSpatial: true,
            isYSpatial: true,
            xField: 'x_u16',
            yField: 'y_u16',
            colorField: 'model',
            sizeField: '',
            auxField1: '',
            auxField2: '',
            schemaFields: ['row_id', 'x_u16', 'y_u16', 'model', 'num_of_tokens', 'language', 'turn'],
            categories: {
                model: ['gpt-4', 'mixtral', 'llama-3'],
                language: ['English', 'French', 'Spanish'],
                turn: ['1', '2', '3'],
            },
        };

        // Bound `model` + 2 unmapped spatial columns (`num_of_tokens` + `language`) -> supported!
        const planOk = planGpuFilterExecution(
            {
                ranges: { num_of_tokens: [50, 400] },
                categories: { model: ['mixtral'], language: ['English', 'French'] },
            },
            baseCtx
        );
        assert.equal(planOk.supported, true);
        assert.equal(planOk.extraFilterField1, 'num_of_tokens');
        assert.equal(planOk.extraFilterField2, 'language');

        // 3 unmapped columns (`num_of_tokens`, `language`, `turn` range) -> exceeds 2 idle spatial slots -> false
        const planTooManyCols = planGpuFilterExecution(
            {
                ranges: { num_of_tokens: [50, 400], turn: [1, 2] },
                categories: { language: ['English'] },
            },
            baseCtx
        );
        assert.equal(planTooManyCols.supported, false);

        // During a 2-second axis transition (`auxField1` & `auxField2` occupied), 0 extra slots free -> false
        const planBusyTransition = planGpuFilterExecution(
            { ranges: { num_of_tokens: [50, 400] } },
            { ...baseCtx, auxField1: 'turn', auxField2: 'language' }
        );
        assert.equal(planBusyTransition.supported, false);

        // Cold column not in archive schema -> false
        const planCold = planGpuFilterExecution(
            { ranges: { cold_parquet_col: [0, 10] } },
            baseCtx
        );
        assert.equal(planCold.supported, false);

        // > 512 category dictionary -> false
        const hugeDict = Array.from({ length: 513 }, (_, i) => `c_${i}`);
        const planHugeDict = planGpuFilterExecution(
            { categories: { model: ['c_0'] } },
            { ...baseCtx, categories: { ...baseCtx.categories, model: hugeDict } }
        );
        assert.equal(planHugeDict.supported, false);
    });

    it('packs 1,024-byte GlobalFilterUniforms with 1D RTO shifts and 512-code category bitmasks', () => {
        const buf = new ArrayBuffer(GLOBAL_FILTER_UNIFORM_BYTES);
        const projCtx: AxisProjectionContext = {
            xAxis: { field: 'x_u16', transform: 'linear', domain: [0, 1], range: [0, 1] },
            yAxis: { field: 'y_u16', transform: 'linear', domain: [0, 1], range: [0, 1] },
            rootBounds: { minX: 0, maxX: 100, minY: 0, maxY: 100 },
            effectiveFlipY: false,
            colOrigins: { ts_ms: 1_700_000_000_000 },
        };
        const bigAllowedDict = Array.from({ length: 512 }, (_, i) => `cat_${i}`);
        const filter: DeepgraphFilter = {
            rect: { minX: 10, maxX: 40, minY: 20, maxY: 80 },
            lasso: {
                points: [
                    [10, 20],
                    [40, 20],
                    [25, 80],
                ],
            },
            ranges: {
                ts_ms: [1_700_000_001_000, 1_700_000_005_500],
                x_u16: [25, 75],
            },
            categories: {
                cat_a: ['cat_0', 'cat_511'],
                cat_b: [33],
            },
            unselectedOpacity: 0.08,
        };

        const packed = packGlobalFilterUniforms(buf, {
            filter,
            hasSlotBitmask: true,
            camLeft: 0.1,
            camTop: 0.9,
            projCtx,
            boundChannelCols: ['cat_a', 'ts_ms', 'cat_b', ''],
            categories: { cat_a: bigAllowedDict, cat_b: ['x'] },
        });

        const u32 = new Uint32Array(buf);
        const f32 = new Float32Array(buf);
        const expectedFlags =
            FILTER_FLAG_HAS_GLOBAL |
            FILTER_FLAG_HAS_SLOT_MASK |
            FILTER_FLAG_HAS_RECT |
            FILTER_FLAG_HAS_LASSO |
            FILTER_FLAG_HAS_CAT0 |
            FILTER_FLAG_HAS_CAT1;
        assert.equal(packed.activeFlags, expectedFlags);
        assert.equal(u32[0], expectedFlags);
        assert.equal(u32[1], 3); // 3 lasso vertices
        assert.equal(u32[2], (1 << 1) | (1 << 4)); // channel 1 (ts_ms) + channel 4 (x_u16)
        assert.equal(u32[3], 0 | (2 << 8)); // cat0 -> ch 0, cat1 -> ch 2
        assert.ok(Math.abs(f32[12] - 0.08) < 1e-6);

        // 1D RTO shifted ts_ms in chanRanges[1] -> [1000, 5500]
        assert.equal(f32[16 + 1 * 4 + 0], 1000);
        assert.equal(f32[16 + 1 * 4 + 1], 5500);

        // catMask0: bit 0 (word 0, bit 0) and bit 511 (word 15, bit 31)
        assert.equal(u32[40 + 0], 1);
        assert.equal(u32[40 + 15], 0x80000000);
        // catMask1: bit 33 (word 1, bit 1)
        assert.equal(u32[56 + 1], 2);

        // CPU row predicate helper agrees on string/code categories & BigInt rowId set
        assert.equal(
            evaluatePointMatchesFilterCpu(
                25,
                40,
                { ts_ms: 1_700_000_002_000, x_u16: 50, cat_a: 'cat_511', cat_b: 33 },
                filter,
                { cat_a: bigAllowedDict, cat_b: [] },
                new Set([999n]),
                999n
            ),
            true
        );
        assert.equal(
            evaluatePointMatchesFilterCpu(
                25,
                40,
                { ts_ms: 1_700_000_002_000, x_u16: 50, cat_a: 'cat_511', cat_b: 33 },
                filter,
                { cat_a: bigAllowedDict, cat_b: [] },
                new Set([999n]),
                1000n
            ),
            false
        );
    });
});

describe('workerCodec — Row Index Extraction (u32 vs u64), 262k Multi-Slice Bitmasks & Spatial Query', () => {
    it('uses compact Uint32Array for <= 2^32-1 IDs and promotes to BigUint64Array for > 2^53-1 Gaia IDs', () => {
        const tbl32 = tableFromArrays({
            row_id: new Uint32Array([10, 40064, 0xffffffff]),
            x_u16: new Uint16Array([100, 32768, 65000]),
            y_u16: new Uint16Array([200, 16384, 60000]),
        });
        const idx32 = extractTileRowIndexFromTables([tbl32])!;
        assert.equal(idx32.isBigInt64, false);
        assert.ok(idx32.rowIds instanceof Uint32Array);
        assert.equal(idx32.byteLength, 3 * (4 + 2 + 2));

        const gaiaId = 5_853_498_713_190_525_696n; // > Number.MAX_SAFE_INTEGER
        const tbl64 = tableFromColumns({
            row_id: columnFromArray([42n, gaiaId, 9_007_199_254_740_995n]),
            x_u16: columnFromArray(new Uint16Array([0, 32768, 65535])),
            y_u16: columnFromArray(new Uint16Array([0, 32768, 65535])),
        });
        const idx64 = extractTileRowIndexFromTables([tbl64])!;
        assert.equal(idx64.isBigInt64, true);
        assert.ok(idx64.rowIds instanceof BigUint64Array);
        assert.equal(idx64.byteLength, 3 * (8 + 2 + 2));

        const slices64 = compileSlotBitmaskSlices(
            idx64,
            null,
            // Sorted ascending: 9.007e15 < 5.853e18
            new BigUint64Array([9_007_199_254_740_995n, gaiaId])
        );
        assert.equal(slices64.length, 1);
        assert.equal(slices64[0].byteLength, SLOT_MASK_SLICE_BYTES);
        const words = new Uint32Array(slices64[0]);
        // Rows 1 and 2 matched -> bits 1 and 2 set (0b110 = 6)
        assert.equal(words[0], 0b110);
    });

    it('chunks oversize leaf tiles (> 262,144 rows) into multiple 32,768-byte aligned bitmask slices', () => {
        const numRows = 300_000;
        const ids = new Uint32Array(numRows);
        for (let i = 0; i < numRows; i++) ids[i] = i + 1000;
        const idx = {
            rowIds: ids,
            isBigInt64: false,
            xU16: new Uint16Array(numRows),
            yU16: new Uint16Array(numRows),
            byteLength: numRows * 8,
        };

        // Match row 5 (slice 0, bit 5) and row 262,144 + 33 = 262,177 (slice 1, word 1, bit 1)
        const target0 = 1000 + 5;
        const target1 = 1000 + SLOT_MASK_ROWS_PER_SLICE + 33;
        const slices = compileSlotBitmaskSlices(idx, new Uint32Array([target0, target1]), null);
        assert.equal(slices.length, 2);
        assert.equal(slices[0].byteLength, SLOT_MASK_SLICE_BYTES);
        assert.equal(slices[1].byteLength, SLOT_MASK_SLICE_BYTES);

        const s0 = new Uint32Array(slices[0]);
        const s1 = new Uint32Array(slices[1]);
        assert.equal(s0[0], 1 << 5);
        assert.equal(s1[0], 0);
        assert.equal(s1[1], 1 << 1);
    });

    it('queries spatial rect & lasso row_ids accurately in worker normalized tile coordinates', () => {
        const idx = {
            rowIds: new BigUint64Array([101n, 202n, 303n]),
            isBigInt64: true,
            xU16: new Uint16Array([0, 32768, 65535]),
            yU16: new Uint16Array([65535, 32768, 0]),
            byteLength: 36,
        };
        // With dataBounds [0..100, 0..100], yU16=32768 -> normY ~ 0.5 -> dataY ~ 50, xU16=32768 -> dataX ~ 50
        const hits = queryTileSpatialSelection('0/0/0', idx, {
            rect: { minX: 40, maxX: 60, minY: 40, maxY: 60 },
            rootBounds: { minX: 0, maxX: 1, minY: 0, maxY: 1 },
            dataBounds: { min_x: 0, max_x: 100, min_y: 0, max_y: 100 },
            effectiveFlipY: false,
        });
        assert.deepEqual(Array.from(hits), [202n]);

        // Intersects spatial selection with active sorted row_id filter (`P2-1`)
        const hitsMaskedOut = queryTileSpatialSelection(
            '0/0/0',
            idx,
            {
                rect: { minX: -1, maxX: 101, minY: -1, maxY: 101 },
                rootBounds: { minX: 0, maxX: 1, minY: 0, maxY: 1 },
                dataBounds: { min_x: 0, max_x: 100, min_y: 0, max_y: 100 },
                effectiveFlipY: false,
            },
            true,
            null,
            new BigUint64Array([101n, 303n])
        );
        assert.deepEqual(Array.from(hitsMaskedOut), [101n, 303n]);

        // Probe vector binary search for <= 64 HighlightOverlay markers (`P1-1`)
        const probeRes = testProbeRowIdsInFilter(
            new BigUint64Array([101n, 202n, 9_007_199_254_740_995n]),
            null,
            new BigUint64Array([101n, 9_007_199_254_740_995n])
        );
        assert.deepEqual(Array.from(probeRes), [1, 0, 1]);
    });
});

describe('deepgraph/mosaic — Division-Free DuckDB Lasso SQL & Real UW Mosaic Coordinator (Path A + Path B)', () => {
    it('generates division-free DuckDB WHERE SQL with AABB zonemap pushdown, safe horizontal edges, and exact RDP parity on >64 vertices', () => {
        // Polygon containing explicit horizontal top & bottom edges (yi === yj)
        const poly: [number, number][] = [
            [10, 20],
            [50, 20], // horizontal bottom edge -> skipped safely without `/ 0`!
            [60, 45],
            [35, 70],
            [10, 70], // horizontal top edge -> skipped safely without `/ 0`!
        ];
        const sqlStr = buildLassoDuckDbSql('umap_x', 'umap_y', poly);
        assert.ok(sqlStr.includes('"umap_x" BETWEEN 10 AND 60'));
        assert.ok(sqlStr.includes('"umap_y" BETWEEN 20 AND 70'));
        assert.ok(sqlStr.includes('% 2) = 1'));
        assert.equal(sqlStr.includes('/'), false, 'Generated SQL must contain zero runtime divisions');

        // 500-vertex variable-density concave star -> buildLassoDuckDbSql uses identical RDP vertices as GPU!
        const denseStar: [number, number][] = [];
        for (let k = 0; k < 500; k++) {
            const seg = Math.floor((k / 500) * 10);
            const t = (k / 500) * 10 - seg;
            const a0 = (seg * Math.PI) / 5;
            const a1 = ((seg + 1) * Math.PI) / 5;
            const r0 = seg % 2 === 0 ? 40 : 12;
            const r1 = seg % 2 === 0 ? 12 : 40;
            denseStar.push([
                50 + Math.cos(a0) * r0 * (1 - t) + Math.cos(a1) * r1 * t,
                50 + Math.sin(a0) * r0 * (1 - t) + Math.sin(a1) * r1 * t,
            ]);
        }
        const gpuRdp = simplifyRdpPolygon(denseStar, 64);
        const starSql = buildLassoDuckDbSql('x', 'y', denseStar, 64);
        let minX = Infinity;
        let maxX = -Infinity;
        let nonHorizEdges = 0;
        for (let i = 0, j = gpuRdp.length - 1; i < gpuRdp.length; j = i++) {
            if (gpuRdp[i][0] < minX) minX = gpuRdp[i][0];
            if (gpuRdp[i][0] > maxX) maxX = gpuRdp[i][0];
            if (Math.abs(gpuRdp[j][1] - gpuRdp[i][1]) >= 1e-15) nonHorizEdges++;
        }
        assert.ok(starSql.includes(`"x" BETWEEN ${minX} AND ${maxX}`));
        const caseCount = (starSql.match(/CASE WHEN/g) ?? []).length;
        assert.equal(caseCount, nonHorizEdges);
    });

    it('integrates with real @uwdata/mosaic-core Coordinator: Path A (0 SQL), self-skip crossfilter, and Path B (SELECT row_id)', async () => {
        const state: {
            currentFilter: DeepgraphFilter | null;
            lastRowIds: (number | bigint)[] | null;
        } = {
            currentFilter: null,
            lastRowIds: null,
        };
        const brushListeners = new Set<(c: SelectionGeometryClause) => void>();
        const selectListeners = new Set<(e: PointSelectEvent) => void>();

        const schemaCols = new Set(['row_id', 'x_u16', 'y_u16', 'model', 'num_of_tokens']);
        const cats: Record<string, string[]> = {
            model: ['gpt-4', 'mixtral', 'llama-3'],
        };

        const fakePlot = {
            getMetadata: () => ({
                formatVersion: '0.1',
                totalRows: 100_000,
                maxZoom: 5,
                bounds: { minX: 0, maxX: 100, minY: 0, maxY: 100 },
                schemaFields: Array.from(schemaCols),
            }),
            getViewState: () => ({
                xField: null,
                yField: null,
                colorField: 'model',
                sizeField: null,
                renderMode: 'points',
                pointSize: 3,
                opacity: 0.85,
                densityBandwidth: 20,
                transitionDuration: 2000,
            }),
            getCategories: (field: string) => cats[field],
            canApplyFilterOnGpu: (f: DeepgraphFilter | null) =>
                planGpuFilterExecution(f, {
                    isXSpatial: true,
                    isYSpatial: true,
                    xField: 'x_u16',
                    yField: 'y_u16',
                    colorField: 'model',
                    sizeField: '',
                    auxField1: '',
                    auxField2: '',
                    schemaFields: Array.from(schemaCols),
                    categories: cats,
                }).supported,
            setFilter: (f: DeepgraphFilter | null) => {
                state.currentFilter = f;
            },
            getFilter: () => state.currentFilter,
            setFilteredRowIds: async (ids: Iterable<number | bigint> | null) => {
                state.lastRowIds = ids ? Array.from(ids) : null;
            },
            clearSelection: () => {
                state.currentFilter = null;
                state.lastRowIds = null;
            },
            on: (event: string, handler: any) => {
                if (event === 'brush') {
                    brushListeners.add(handler);
                    return () => brushListeners.delete(handler);
                }
                if (event === 'select') {
                    selectListeners.add(handler);
                    return () => selectListeners.delete(handler);
                }
                return () => {};
            },
        } as unknown as DeepgraphPlot;

        const executedSql: string[] = [];
        const rowIdIpcBytes = tableToIPC(
            tableFromColumns({
                row_id: columnFromArray(new BigUint64Array([40064n, 9_007_199_254_740_995n])),
            }),
            { format: 'stream' }
        )!;
        const countIpcBytes = tableToIPC(
            tableFromArrays({ cnt: new Int32Array([42]) }),
            { format: 'stream' }
        )!;
        const connector = {
            query: async (q: { type: string; sql: string }) => {
                executedSql.push(q.sql);
                return q.sql.includes('SELECT "row_id"') ? rowIdIpcBytes : countIpcBytes;
            },
        };
        const mc = new Coordinator(connector as any, { cache: false, preagg: { enabled: false } });
        const brush = Selection.crossfilter();

        // Peer client simulating a linked Mosaic histogram that queries DuckDB on brush changes
        let peerLastFilterSql = '';
        class PeerHistogramClient extends MosaicClient {
            constructor() {
                super(brush);
            }
            override query(filter: any = []) {
                peerLastFilterSql = String(filter ?? '');
                return Query.from('lmsys_umap')
                    .select({ cnt: sql`count(*)` })
                    .where(filter);
            }
        }
        const peer = new PeerHistogramClient();
        mc.connect(peer);

        const dgClient = new DeepgraphMosaicClient({
            plot: fakePlot,
            table: 'lmsys_umap',
            rowIdColumn: 'row_id',
            xColumn: 'umap_x',
            yColumn: 'umap_y',
            as: brush as any,
            filterBy: brush as any,
            coordinator: mc as any,
        });

        assert.ok(dgClient instanceof MosaicClient);
        await Promise.all([dgClient.pending, peer.pending]);
        const rowIdQueries = () => executedSql.filter((s) => s.includes('SELECT "row_id"'));
        assert.equal(rowIdQueries().length, 0);

        // 1. External histogram brushes `num_of_tokens` (both 1D `clauseInterval` and 1-field `clauseIntervals`) + `model` category -> Path A (0 row_id SQL!)
        const extSource = { id: 'hist_tokens' };
        brush.update(clauseInterval('num_of_tokens', [100, 400], { source: extSource }));
        await dgClient.filterSettled;
        assert.equal(dgClient.lastExecutionPath, 'gpu');
        assert.equal(dgClient.sqlQueryCount, 0);
        assert.equal(rowIdQueries().length, 0);
        assert.deepEqual(state.currentFilter?.ranges?.num_of_tokens, [100, 400]);

        brush.update(clauseIntervals(['num_of_tokens'], [[120, 380]], { source: extSource }));
        await dgClient.filterSettled;
        assert.equal(dgClient.lastExecutionPath, 'gpu');
        assert.equal(dgClient.sqlQueryCount, 0);
        assert.deepEqual(state.currentFilter?.ranges?.num_of_tokens, [120, 380]);

        const extCatSource = { id: 'legend_model' };
        brush.update(clausePoint('model', 'mixtral', { source: extCatSource }));
        await dgClient.filterSettled;
        assert.equal(dgClient.lastExecutionPath, 'gpu');
        assert.equal(dgClient.sqlQueryCount, 0);
        assert.deepEqual(state.currentFilter?.categories?.model, ['mixtral']);
        assert.deepEqual(state.currentFilter?.ranges?.num_of_tokens, [120, 380]);

        // 2. Outbound lasso from Deepgraph -> updates peer client with division-free SQL while self-skipping Deepgraph!
        for (const fn of brushListeners) {
            fn({
                kind: 'lasso',
                xField: null,
                yField: null,
                lasso: {
                    points: [
                        [10, 10],
                        [60, 10],
                        [35, 80],
                    ],
                },
            });
        }
        await dgClient.filterSettled;
        assert.equal(dgClient.sqlQueryCount, 0);
        assert.equal(rowIdQueries().length, 0);
        assert.ok(peerLastFilterSql.includes('"umap_x" BETWEEN 10 AND 60'));

        // 3. External text search (`ILIKE`) -> triggers Path B (`SELECT "row_id" FROM "lmsys_umap" ...`)!
        const searchSource = { id: 'search_box' };
        brush.update(clauseMatch('prompt_text', 'quantum', { source: searchSource }) as any);
        await dgClient.filterSettled;
        assert.equal(dgClient.lastExecutionPath, 'sql');
        assert.equal(dgClient.sqlQueryCount, 1);
        assert.equal(rowIdQueries().length, 1);
        assert.ok(rowIdQueries()[0].includes('SELECT "row_id"'));
        assert.ok(rowIdQueries()[0].includes('"lmsys_umap"'));
        assert.deepEqual(state.lastRowIds, [40064n, 9_007_199_254_740_995n]);

        // 4. Clearing external search clause restores Path A GPU filter & clears Path B row_id mask!
        brush.update({
            source: searchSource,
            value: null,
            predicate: null,
        } as any);
        await dgClient.filterSettled;
        assert.equal(dgClient.lastExecutionPath, 'gpu');
        assert.equal(state.lastRowIds, null);
        assert.deepEqual(state.currentFilter?.ranges?.num_of_tokens, [120, 380]);

        dgClient.destroy();
        mc.disconnect(peer);
    });
});
