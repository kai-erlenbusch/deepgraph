import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { ViewState } from '../../src/lib/ViewState.ts';
import type { DatasetSession } from '../../src/lib/DatasetSession.ts';

function makeMockSession(overrides?: Partial<{
    rootBounds: { minX: number; maxX: number; minY: number; maxY: number };
    dataBounds: { min_x: number; max_x: number; min_y: number; max_y: number } | undefined;
    stats: Record<string, { min: number; max: number; null_count: number }>;
    categories: Record<string, string[]>;
    colOrigins: Record<string, number>;
}>): DatasetSession {
    const rootBounds = overrides?.rootBounds ?? { minX: 0, maxX: 100, minY: 0, maxY: 100 };
    const dataBounds =
        overrides && 'dataBounds' in overrides
            ? overrides.dataBounds
            : { min_x: 10, max_x: 90, min_y: 20, max_y: 80 };
    const stats = overrides?.stats ?? {
        cad_x: { min: -500, max: 1500, null_count: 0 },
        cad_y: { min: 100, max: 900, null_count: 0 },
        unix_ts: { min: 1_710_000_000.0, max: 1_710_003_600.0, null_count: 0 },
        bp_rp: { min: -0.5, max: 4.5, null_count: 0 },
    };
    const categories = overrides?.categories ?? {
        model: ['gpt-4', 'claude-3', 'llama-3', 'mixtral'],
    };
    const colOrigins = overrides?.colOrigins ?? {
        unix_ts: 1_710_000_000.0,
    };

    return {
        colOrigins,
        embeddedView: {},
        setRequestedColumns: () => {},
        summary: {
            format: 'arrowtiles',
            version: '0.1',
            tiling: 'quadtree',
            archiveMaxZoom: 4,
            totalRows: 1000,
            isLegacyV00: false,
            magnitudeField: 'bp_rp',
            droppedColumns: [],
            schemaFields: ['cad_x', 'cad_y', 'unix_ts', 'bp_rp', 'model'],
            rootBounds,
            dataBounds,
            stats,
            categories,
        },
    } as unknown as DatasetSession;
}

describe('ViewState — literal transform, spatial flipY, optical preservation & 1D RTO', () => {
    it('keeps literal custom axes non-spatial (world = data) instead of hijacking x_u16/y_u16', () => {
        const session = makeMockSession();
        const vs = new ViewState();
        vs.applyEncoding(
            {
                x: { field: 'cad_x', transform: 'literal' },
                y: { field: 'cad_y', transform: 'literal' },
            },
            session.summary.stats,
            session.summary.categories
        );

        assert.equal(vs.isSpatialX, false);
        assert.equal(vs.isSpatialY, false);
        assert.equal(vs.hasCustomWorldScale, true);

        const m = vs.resolveAxisMapping(session);
        assert.deepEqual(m.xRange, [-500, 1500]);
        assert.deepEqual(m.yRange, [100, 900]);
        assert.deepEqual(m.dataXSpan, [-500, 1500]);
        assert.deepEqual(m.dataYSpan, [100, 900]);
    });

    it('avoids double-negating spatial dataYSpan when flipY: true is set on spatial coordinates', () => {
        const session = makeMockSession();
        const vsNormal = new ViewState();
        const mNormal = vsNormal.resolveAxisMapping(session);
        // Normal spatial: yRange=[100, 0], dataYSpan=[max_y=80, min_y=20] -> world 100 maps to max_y=80 (top)
        assert.deepEqual(mNormal.yRange, [100, 0]);
        assert.deepEqual(mNormal.dataYSpan, [80, 20]);

        const vsFlipped = new ViewState();
        vsFlipped.applyEncoding({ flipY: true }, session.summary.stats, session.summary.categories);
        const mFlipped = vsFlipped.resolveAxisMapping(session);
        // Flipped spatial: yRange=[0, 100], dataYSpan=[80, 20] -> world 0 (bottom) maps to max_y=80, world 100 (top) maps to min_y=20!
        assert.deepEqual(mFlipped.yRange, [0, 100]);
        assert.deepEqual(mFlipped.dataYSpan, [80, 20]);

        // Verify setViewport / getViewport round-trip math on flipped spatial Y
        const targetY = 65;
        const ty = (targetY - mFlipped.dataYSpan[0]) / (mFlipped.dataYSpan[1] - mFlipped.dataYSpan[0]);
        const worldY = mFlipped.yRange[0] + ty * (mFlipped.yRange[1] - mFlipped.yRange[0]);
        const roundTripTy = (worldY - mFlipped.yRange[0]) / (mFlipped.yRange[1] - mFlipped.yRange[0]);
        const roundTripDataY =
            mFlipped.dataYSpan[0] + roundTripTy * (mFlipped.dataYSpan[1] - mFlipped.dataYSpan[0]);
        assert.ok(Math.abs(roundTripDataY - targetY) < 1e-12);
        // Since 65 > midpoint 50 and Y is flipped, worldY must be in bottom half (< 50)
        assert.ok(worldY < 50);
    });

    it('preserves optical pipeline when updating opticalColorField domain without explicit scale', () => {
        const session = makeMockSession();
        const vs = new ViewState();
        vs.initFromSession(session, {
            name: 'gaia-dr3',
            renderPipeline: 'optical',
            defaultEncoding: {
                color: { field: 'bp_rp', domain: [-0.5, 4.0] },
            },
            experimental: {
                opticalColorField: 'bp_rp',
            },
        });
        assert.equal(vs.snapshot().pipeline, 'optical');

        // Updating color domain on bp_rp without scale keeps optical pipeline
        vs.applyEncoding({ color: { field: 'bp_rp', domain: [0.0, 3.0] } }, session.summary.stats);
        assert.equal(vs.snapshot().pipeline, 'optical');

        // Explicitly supplying a colormap scale switches to cartesian points
        vs.applyEncoding(
            { color: { field: 'bp_rp', scale: 'viridis', domain: [0.0, 3.0] } },
            session.summary.stats
        );
        assert.equal(vs.snapshot().pipeline, 'cartesian');
    });

    it('defaults categorical chart axes to [0, K-1] and keeps 1D RTO strictly internal to GPU config', () => {
        const session = makeMockSession();
        const vs = new ViewState();
        vs.applyEncoding(
            {
                x: 'model',
                y: 'unix_ts',
                color: { field: 'unix_ts', scale: 'viridis' },
            },
            session.summary.stats,
            session.summary.categories
        );

        // Public snapshot & resolved mapping stay 100% in unshifted user coordinates!
        const snap = vs.snapshot();
        const resolved = vs.resolveAxisMapping(session);
        assert.equal(snap.xDomain, undefined);
        assert.deepEqual(resolved.xDomain, [0, 3]);
        assert.deepEqual(resolved.yDomain, [1_710_000_000.0, 1_710_003_600.0]);
        assert.equal(snap.colorMin, 1_710_000_000.0);
        assert.equal(snap.colorMax, 1_710_003_600.0);

        // GPU TileStore config shifts unix_ts by -colOrigins.unix_ts -> [0, 3600]!
        let capturedNext: any = null;
        const mockStore = {
            computeMode: 1,
            categoricalDictionaries: session.summary.categories,
            colOrigins: {} as Record<string, number>,
            updateEncoding: (next: any) => {
                capturedNext = next;
                return ['xy_u16', 'model', 'unix_ts'];
            },
        };
        vs.syncToTileStore(mockStore as any, session, false);
        assert.deepEqual(capturedNext.x.domain, [0, 3]);
        assert.deepEqual(capturedNext.y.domain, [0, 3600]);
        assert.deepEqual(capturedNext.color.domain, [0, 3600]);
        assert.equal(mockStore.colOrigins.unix_ts, 1_710_000_000.0);
    });
});
