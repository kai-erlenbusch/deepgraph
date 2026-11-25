import { test, expect, type Page } from '@playwright/test';

const allowDgpu = process.env.DEEPGRAPH_ALLOW_DGPU === '1';
const expectedPowerPref: 'high-performance' | 'low-power' = allowDgpu
    ? 'high-performance'
    : 'low-power';

async function countLitCanvasPixels(page: Page) {
    const canvas = page.locator('canvas').first();
    const pngB64 = (await canvas.screenshot({ type: 'png' })).toString('base64');
    return await page.evaluate(async (b64) => {
        const res = await fetch(`data:image/png;base64,${b64}`);
        const bmp = await createImageBitmap(await res.blob());
        const c = new OffscreenCanvas(bmp.width, bmp.height);
        const ctx = c.getContext('2d')!;
        ctx.drawImage(bmp, 0, 0);
        const d = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
        let lit = 0;
        let bright = 0;
        for (let i = 0; i < d.length; i += 4) {
            const dr = Math.abs(d[i] - 17);
            const dg = Math.abs(d[i + 1] - 17);
            const db = Math.abs(d[i + 2] - 17);
            const m = Math.max(dr, dg, db);
            if (m > 3) lit++;
            if (m > 45) bright++;
        }
        return { lit, bright };
    }, pngB64);
}

async function measureStationaryShimmer(page: Page) {
    await page.waitForTimeout(300);
    const canvas = page.locator('canvas').first();
    const pngA = (await canvas.screenshot({ type: 'png' })).toString('base64');
    await page.waitForTimeout(80);
    const pngB = (await canvas.screenshot({ type: 'png' })).toString('base64');

    return await page.evaluate(
        async ({ a, b }) => {
            async function decodePng(b64: string) {
                const res = await fetch(`data:image/png;base64,${b64}`);
                const bmp = await createImageBitmap(await res.blob());
                const c = new OffscreenCanvas(bmp.width, bmp.height);
                const ctx = c.getContext('2d')!;
                ctx.drawImage(bmp, 0, 0);
                return ctx.getImageData(0, 0, bmp.width, bmp.height).data;
            }
            const dA = await decodePng(a);
            const dB = await decodePng(b);
            let litPixels = 0;
            let changedPixels = 0;
            for (let i = 0; i < dA.length; i += 4) {
                const isLit =
                    Math.abs(dA[i] - 17) > 3 ||
                    Math.abs(dA[i + 1] - 17) > 3 ||
                    Math.abs(dA[i + 2] - 17) > 3 ||
                    Math.abs(dB[i] - 17) > 3 ||
                    Math.abs(dB[i + 1] - 17) > 3 ||
                    Math.abs(dB[i + 2] - 17) > 3;
                if (!isLit) continue;
                litPixels++;
                if (
                    dA[i] !== dB[i] ||
                    dA[i + 1] !== dB[i + 1] ||
                    dA[i + 2] !== dB[i + 2]
                ) {
                    changedPixels++;
                }
            }
            return {
                litPixels,
                changedPixels,
                changedPct: litPixels > 0 ? (changedPixels / litPixels) * 100 : 0,
            };
        },
        { a: pngA, b: pngB }
    );
}

test.describe('Step 3A — Baseline-Safe GPU Cross-Filtering, 2D Brush/Spotlight/Lasso & UW Mosaic (Intel iGPU Sample-Only)', () => {
    test('Verifies Path A GPU uniform filtering (incl. unmapped spatial columns & 2-pass context dim), brush/spotlight/lasso UX, worker 32 KiB slot masks, real Mosaic Coordinator & zero-leak teardown', async ({
        page,
    }) => {
        test.setTimeout(120000);
        const pageErrors: string[] = [];
        const consoleErrors: string[] = [];
        page.on('pageerror', (e) => pageErrors.push(e.message || String(e)));
        page.on('console', (m) => {
            if (m.type() === 'error') consoleErrors.push(m.text());
        });

        await page.route('**/__step3a_brush_mosaic.html', async (route) => {
            await route.fulfill({
                status: 200,
                contentType: 'text/html; charset=utf-8',
                body: `<!doctype html><html><body style="margin:0;background:#111"><div id="stage" style="width:960px;height:600px"></div></body></html>`,
            });
        });
        await page.goto('/__step3a_brush_mosaic.html', { waitUntil: 'domcontentloaded' });

        // =========================================================================
        // 0. STRICT GPU COEXISTENCE PRE-CHECK (0 B ON DISCRETE NVIDIA RTX 5090!)
        // =========================================================================
        const igpuGuard = await page.evaluate(async (pref) => {
            const adapter = await navigator.gpu.requestAdapter({ powerPreference: pref });
            if (!adapter) {
                throw new Error(`ABORT: requestAdapter(${pref}) returned null`);
            }
            const info: any = (adapter as any).info || {};
            return {
                vendor: String(info.vendor || '').toLowerCase(),
                architecture: String(info.architecture || ''),
                isFallbackAdapter: Boolean((adapter as any).isFallbackAdapter),
            };
        }, expectedPowerPref);
        console.log('[Step 3A Brush/Mosaic GPU Pre-Check]', JSON.stringify(igpuGuard));
        if (!allowDgpu) {
            expect(igpuGuard.vendor).not.toBe('nvidia');
        }
        expect(igpuGuard.isFallbackAdapter).toBe(false);

        await page.evaluate(async () => {
            const { installGpuStatsInstrumentation } = await import('/src/demo/gpuStats.ts');
            installGpuStatsInstrumentation();
            (window as any).__wgpuUncaptured = [];
            const origReq = GPUAdapter.prototype.requestDevice;
            GPUAdapter.prototype.requestDevice = async function (...args: any[]) {
                const dev = await origReq.apply(this, args as any);
                dev.addEventListener('uncapturederror', (ev: any) => {
                    (window as any).__wgpuUncaptured.push(ev?.error?.message || String(ev));
                });
                return dev;
            };
        });

        // =========================================================================
        // 1. INITIALIZE LMSYS PLOT & VERIFY 33 KiB IDLE FILTER VRAM
        // =========================================================================
        const initInfo = await page.evaluate(async (pref) => {
            const { Deepgraph } = await import('/src/index.ts');
            const { INTERNAL_DIAGNOSTICS } = await import('/src/lib/diagnostics.ts');
            const stage = document.getElementById('stage')!;
            const plot = await Deepgraph.create(stage, {
                url: '/samples/lmsys_sample.arrowtiles',
                powerPreference: pref,
                enableRowIdIndex: true,
                encoding: {
                    color: { field: 'model' },
                    aggregation: 'points',
                },
            });
            plot.fitView();
            const diag = (plot as any)[INTERNAL_DIAGNOSTICS]();
            const events = {
                brushes: [] as any[],
                selects: [] as any[],
                clicks: [] as any[],
            };
            plot.on('brush', (c) => events.brushes.push(c));
            plot.on('select', (s) => events.selects.push(s));
            plot.on('click', (h) => events.clicks.push(h));
            (window as any).__s3 = { plot, diag, events };

            const deadline = performance.now() + 10000;
            while (performance.now() < deadline) {
                await new Promise((r) => setTimeout(r, 60));
                const tm: any = diag.getTileManager();
                if (tm && tm.activeTiles.size >= 5 && tm.loadingTiles.size === 0) break;
            }
            await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

            return {
                idleFilterGpuBytes: diag.filterResources.getLiveGpuBytes(),
                canFilterUnmappedSpatial: plot.canApplyFilterOnGpu({
                    ranges: { num_of_tokens: [250, 600] },
                    categories: { model: ['mixtral'] },
                }),
                canFilterColdCol: plot.canApplyFilterOnGpu({
                    ranges: { nonexistent_cold_col: [0, 10] },
                }),
            };
        }, expectedPowerPref);

        console.log('[1. Idle Filter & Capability Check]', JSON.stringify(initInfo));
        // 1,024 B GlobalFilterUniforms + 32,768 B dummy all-ones slice = 33,792 B (0 B lazy slotMaskBuffer!)
        expect(initInfo.idleFilterGpuBytes).toBe(33792);
        expect(initInfo.canFilterUnmappedSpatial).toBe(true);
        expect(initInfo.canFilterColdCol).toBe(false);

        const baselinePx = await countLitCanvasPixels(page);
        expect(baselinePx.lit).toBeGreaterThan(20000);

        // =========================================================================
        // 2. UNMAPPED SPATIAL COLUMN (`num_of_tokens`) + CATEGORY (`model`) + 2-PASS DIM
        // =========================================================================
        await page.evaluate(async () => {
            const { plot, diag } = (window as any).__s3;
            plot.setFilter({
                ranges: { num_of_tokens: [250, 600] },
                categories: { model: ['mixtral'] },
                unselectedOpacity: 0.0,
            });
            const deadline = performance.now() + 8000;
            while (performance.now() < deadline) {
                await new Promise((r) => setTimeout(r, 60));
                const tm: any = diag.getTileManager();
                if (tm && tm.loadingTiles.size === 0) break;
            }
            await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
        });

        const hardCullPx = await countLitCanvasPixels(page);
        const hardCullShimmer = await measureStationaryShimmer(page);
        console.log('[2a. Hard-Cull Unmapped Spatial Filter]', JSON.stringify({ hardCullPx, hardCullShimmer }));
        expect(hardCullPx.lit).toBeGreaterThan(30);
        expect(hardCullPx.lit).toBeLessThan(baselinePx.lit * 0.25);
        expect(hardCullShimmer.changedPixels).toBe(0);

        // Switch to 2-subpass context dimming (`unselectedOpacity: 0.22`) -> dimmed background reappears with 0.0000% shimmer!
        await page.evaluate(async () => {
            const { plot } = (window as any).__s3;
            plot.setFilter({
                ranges: { num_of_tokens: [250, 600] },
                categories: { model: ['mixtral'] },
                unselectedOpacity: 0.22,
            });
            await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
        });

        const dimMode1Px = await countLitCanvasPixels(page);
        const dimMode1Shimmer = await measureStationaryShimmer(page);
        console.log('[2b. Mode 1 Two-Pass Context Dim]', JSON.stringify({ dimMode1Px, dimMode1Shimmer }));
        expect(dimMode1Px.lit).toBeGreaterThan(hardCullPx.lit * 3);
        expect(dimMode1Shimmer.changedPixels).toBe(0);

        // Mode 2 (`aggregation: 'mean_field'`) MUST always hard-cull unselected points even if unselectedOpacity > 0!
        const densityParity = await page.evaluate(async () => {
            const { plot, diag } = (window as any).__s3;
            plot.setEncoding({
                color: { field: 'num_of_tokens', scale: 'viridis', domain: [1, 600] },
                aggregation: 'mean_field',
            });
            plot.setFilter({
                rect: { minX: 60, maxX: 95, minY: 50, maxY: 90 },
                unselectedOpacity: 0.0,
            });
            const deadline = performance.now() + 8000;
            while (performance.now() < deadline) {
                await new Promise((r) => setTimeout(r, 60));
                const tm: any = diag.getTileManager();
                const visSlots = diag.tileStore.slots.filter((s: any) => s && s.visible);
                const visibleReady =
                    visSlots.length > 0 &&
                    visSlots.every((s: any) => Boolean(s.buffers?.has('num_of_tokens')));
                if (tm && tm.loadingTiles.size === 0 && visibleReady) {
                    break;
                }
            }
            // Settle spawn fade-in so both snapshots compare identical converged frames
            for (const s of diag.tileStore.slots) {
                if (s) s.spawnTime = diag.tileStore.globalTimeSec - 5.0;
            }
            (plot as any).renderLoop.markDirty();
            await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
            return diag.tileStore.computeMode;
        });
        expect(densityParity).toBe(2);
        const densityHardPx = await countLitCanvasPixels(page);
        await page.evaluate(async () => {
            const { plot } = (window as any).__s3;
            plot.setFilter({
                rect: { minX: 60, maxX: 95, minY: 50, maxY: 90 },
                unselectedOpacity: 0.35,
            });
            await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
        });
        const densityWithDimReqPx = await countLitCanvasPixels(page);
        const densityShimmer = await measureStationaryShimmer(page);
        expect(densityWithDimReqPx.lit).toBe(densityHardPx.lit);
        expect(densityShimmer.changedPixels).toBe(0);

        // =========================================================================
        // 3. PICKINGLAYER & HIGHLIGHTOVERLAY SKIP DIMMED/CULLED POINTS
        // =========================================================================
        const pickFilterCheck = await page.evaluate(async () => {
            const { plot, diag } = (window as any).__s3;
            plot.setEncoding({
                color: { field: 'model' },
                aggregation: 'points',
            });
            plot.setFilter(null);
            // Center on ground-truth LMSys row_id=40064 (model='mixtral', num_of_tokens=308)
            plot.setViewport({ x: 88.76073587009975, y: 69.12738843695004, zoom: 128 });
            const deadline = performance.now() + 8000;
            while (performance.now() < deadline) {
                await new Promise((r) => setTimeout(r, 60));
                const tm: any = diag.getTileManager();
                if (tm && tm.loadingTiles.size === 0 && tm.currentMaxZ >= 4) break;
            }
            await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

            const hitUnfiltered = await plot.pickAt(480, 300);
            // Simulate pinning selection ring on row_id=40064, then filter it out with unselectedOpacity: 0.2
            diag.highlightOverlay.selectedMarkers = [
                {
                    kind: 'selected',
                    rowId: 40064,
                    rawRowIdx: 0,
                    tileZ: 0,
                    tileX: 0,
                    tileY: 0,
                    xU16: 32768,
                    yU16: 32768,
                    attributes: { model: 'mixtral', num_of_tokens: 308 },
                },
            ];
            plot.setFilter({
                categories: { model: ['gpt-4'] },
                unselectedOpacity: 0.2,
            });
            await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
            const hitWhenDimmed = await plot.pickAt(480, 300);
            const markerVisibleWhenCulled = (plot as any).isHighlightMarkerVisible(
                diag.highlightOverlay.selectedMarkers[0]
            );
            diag.highlightOverlay.selectedMarkers = [];
            plot.setFilter(null);
            plot.fitView();
            await new Promise((r) => setTimeout(r, 250));
            return {
                unfilteredRowId: hitUnfiltered?.rowId ?? -1,
                dimmedPicked40064: hitWhenDimmed?.rowId === 40064,
                markerVisibleWhenCulled,
            };
        });
        console.log('[3. Picking & Highlight Ring Filter Parity]', JSON.stringify(pickFilterCheck));
        expect(pickFilterCheck.unfilteredRowId).toBe(40064);
        expect(pickFilterCheck.dimmedPicked40064).toBe(false);
        expect(pickFilterCheck.markerVisibleWhenCulled).toBe(false);

        // =========================================================================
        // 4. POINTER 'brush' (DRAW + SPOTLIGHT-LENS TRANSLATE + <4PX CLEAR), 'lasso' & Shift+Drag
        // =========================================================================
        const stageBox = await page.locator('#stage').boundingBox();
        expect(stageBox).not.toBeNull();
        const sx = stageBox!.x;
        const sy = stageBox!.y;

        await page.evaluate(() => {
            const { plot, events } = (window as any).__s3;
            events.brushes.length = 0;
            events.selects.length = 0;
            events.clicks.length = 0;
            plot.setInteractionMode('brush');
        });

        // 4a. Draw 2D brush rectangle [300, 180] -> [620, 420]
        await page.mouse.move(Math.round(sx + 300), Math.round(sy + 180));
        await page.mouse.down();
        await page.mouse.move(Math.round(sx + 620), Math.round(sy + 420), { steps: 6 });
        await page.mouse.up();
        await page.waitForTimeout(200);

        const afterDrawRect = await page.evaluate(async () => {
            const { plot, events } = (window as any).__s3;
            const f = plot.getFilter();
            const selIds = await plot.getSelectedRowIds();
            return {
                mode: plot.getInteractionMode(),
                rect: f?.rect ?? null,
                brushCount: events.brushes.length,
                selectCount: events.selects.length,
                lastSelectKind: events.selects[events.selects.length - 1]?.clause?.kind ?? null,
                selectedRowIdCount: selIds.rowIds.length,
            };
        });
        console.log('[4a. Draw Brush Rect]', JSON.stringify(afterDrawRect));
        expect(afterDrawRect.mode).toBe('brush');
        expect(afterDrawRect.rect).not.toBeNull();
        expect(afterDrawRect.brushCount).toBeGreaterThan(0);
        expect(afterDrawRect.lastSelectKind).toBe('rect');
        expect(afterDrawRect.selectedRowIdCount).toBeGreaterThan(50);

        // 4b. Spotlight-lens drag INSIDE active brush rect [460, 300] -> [540, 300] translates rect right!
        await page.mouse.move(Math.round(sx + 460), Math.round(sy + 300));
        await page.mouse.down();
        await page.mouse.move(Math.round(sx + 540), Math.round(sy + 300), { steps: 5 });
        await page.mouse.up();
        await page.waitForTimeout(150);

        const afterTranslateRect = await page.evaluate(() => {
            const { plot } = (window as any).__s3;
            return plot.getFilter()?.rect ?? null;
        });
        expect(afterTranslateRect).not.toBeNull();
        const oldWidth = afterDrawRect.rect!.maxX - afterDrawRect.rect!.minX;
        const newWidth = afterTranslateRect!.maxX - afterTranslateRect!.minX;
        // Width stays preserved while minX shifts positively!
        expect(Math.abs(newWidth - oldWidth)).toBeLessThan(1e-4);
        expect(afterTranslateRect!.minX).toBeGreaterThan(afterDrawRect.rect!.minX + 1.0);

        // 4c. Sub-4px click clears active brush AND suppresses accidental single-point 'click'
        await page.mouse.click(Math.round(sx + 120), Math.round(sy + 120));
        await page.waitForTimeout(120);
        const afterSub4Clear = await page.evaluate(() => {
            const { plot, events } = (window as any).__s3;
            return {
                filter: plot.getFilter(),
                clickCount: events.clicks.length,
                lastClauseKind: events.brushes[events.brushes.length - 1]?.kind ?? null,
            };
        });
        expect(afterSub4Clear.filter).toBeNull();
        expect(afterSub4Clear.clickCount).toBe(0);
        expect(afterSub4Clear.lastClauseKind).toBe('none');

        // 4d. Concave freehand 'lasso' mode + SelectionOverlay stationary shimmer
        await page.evaluate(() => {
            const { plot } = (window as any).__s3;
            plot.setInteractionMode('lasso');
        });
        const lassoPath: [number, number][] = [
            [320, 180],
            [620, 190],
            [500, 300], // concave inward notch
            [640, 430],
            [310, 420],
        ];
        await page.mouse.move(Math.round(sx + lassoPath[0][0]), Math.round(sy + lassoPath[0][1]));
        await page.mouse.down();
        for (let i = 1; i < lassoPath.length; i++) {
            await page.mouse.move(
                Math.round(sx + lassoPath[i][0]),
                Math.round(sy + lassoPath[i][1]),
                { steps: 4 }
            );
        }
        await page.mouse.up();
        await page.waitForTimeout(180);

        const lassoCheck = await page.evaluate(async () => {
            const { plot } = (window as any).__s3;
            const f = plot.getFilter();
            const ids = await plot.getSelectedRowIds();
            return {
                vertCount: f?.lasso?.points.length ?? 0,
                selectedCount: ids.rowIds.length,
            };
        });
        const lassoOverlayShimmer = await measureStationaryShimmer(page);
        console.log('[4d. Concave Lasso & Overlay Shimmer]', JSON.stringify({ lassoCheck, lassoOverlayShimmer }));
        expect(lassoCheck.vertCount).toBeGreaterThanOrEqual(4);
        expect(lassoCheck.selectedCount).toBeGreaterThan(50);
        expect(lassoOverlayShimmer.changedPixels).toBe(0);

        // 4e. Shift+drag in 'pan' mode temporarily brushes without mutating interactionMode
        await page.evaluate(() => {
            const { plot } = (window as any).__s3;
            plot.clearSelection();
            plot.setInteractionMode('pan');
        });
        await page.keyboard.down('Shift');
        await page.mouse.move(Math.round(sx + 350), Math.round(sy + 220));
        await page.mouse.down();
        await page.mouse.move(Math.round(sx + 580), Math.round(sy + 390), { steps: 4 });
        await page.mouse.up();
        await page.keyboard.up('Shift');
        await page.waitForTimeout(120);

        const shiftBrushCheck = await page.evaluate(() => {
            const { plot } = (window as any).__s3;
            const hasRect = Boolean(plot.getFilter()?.rect);
            const modeStillPan = plot.getInteractionMode() === 'pan';
            plot.clearSelection();
            return { hasRect, modeStillPan };
        });
        expect(shiftBrushCheck.hasRect).toBe(true);
        expect(shiftBrushCheck.modeStillPan).toBe(true);

        // =========================================================================
        // 5. WORKER 32 KiB BITMASK (`setFilteredRowIds`) + REAL `@uwdata/mosaic-core` E2E
        // =========================================================================
        const mosaicAndMaskCheck = await page.evaluate(async () => {
            const { plot, diag } = (window as any).__s3;
            const {
                Coordinator,
                Selection,
                clauseInterval,
                clauseMatch,
                columnFromArray,
                tableFromColumns,
                tableToIPC,
                DeepgraphMosaicClient,
            } = await import('/tests/helpers/mosaicBrowserBridge.ts');

            // 5a. Direct setFilteredRowIds([40064]) allocates lazy slotMaskBuffer & isolates row 40064
            //     AND survives fresh tile streaming on column switch (`Reviewer 2 P1` regression guard)!
            await plot.setFilteredRowIds(new Uint32Array([40064]));
            const maskActiveBytes = diag.filterResources.getLiveGpuBytes();
            // Switch color column to force `setRequestedColumns` -> fresh worker decode + GPU upload while mask is active!
            plot.setEncoding({
                color: { field: 'num_of_tokens', scale: 'viridis', domain: [1, 600] },
                aggregation: 'points',
            });
            plot.setViewport({ x: 88.76073587009975, y: 69.12738843695004, zoom: 128 });
            const d1 = performance.now() + 6000;
            while (performance.now() < d1) {
                await new Promise((r) => setTimeout(r, 60));
                const tm: any = diag.getTileManager();
                const visSlots = diag.tileStore.slots.filter((s: any) => s && s.visible);
                if (
                    tm &&
                    tm.loadingTiles.size === 0 &&
                    tm.currentMaxZ >= 4 &&
                    visSlots.length > 0 &&
                    visSlots.every((s: any) => Boolean(s.buffers?.has('num_of_tokens')))
                ) {
                    break;
                }
            }
            await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
            const hitIncluded = await plot.pickAt(480, 300);

            // Verify `getSelectedRowIds()` intersects spatial brush with active `row_id` bitmask (`Reviewer 1 P2-1`)
            plot.setFilter({ rect: { minX: 0, maxX: 100, minY: 0, maxY: 100 } });
            const intersectedSel = await plot.getSelectedRowIds();
            plot.setFilter(null);

            // Switch to [38479] while still centered on 40064 -> 40064 is now masked out!
            await plot.setFilteredRowIds([38479]);
            await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
            const hitExcluded = await plot.pickAt(480, 300);

            // Clear row_id mask -> slotMaskBuffer is .destroy()ed back to 33,792 B!
            await plot.setFilteredRowIds(null);
            const maskClearedBytes = diag.filterResources.getLiveGpuBytes();
            plot.setEncoding({ color: { field: 'model' }, aggregation: 'points' });

            // 5b. Real Coordinator + DeepgraphMosaicClient Path A (0 SQL) & Path B (SELECT row_id)
            const ipc40064 = tableToIPC(
                tableFromColumns({ row_id: columnFromArray(new Uint32Array([40064])) }),
                { format: 'stream' }
            )!;
            const sqlQueries: string[] = [];
            const mc = new Coordinator(
                {
                    query: async (q: { sql: string }) => {
                        sqlQueries.push(q.sql);
                        return ipc40064;
                    },
                } as any,
                { cache: false, preagg: { enabled: false } }
            );
            const sel = Selection.crossfilter();
            const client = new DeepgraphMosaicClient({
                plot,
                table: 'lmsys_sample',
                as: sel as any,
                filterBy: sel as any,
                coordinator: mc as any,
            });
            await client.pending;

            const extHist = { id: 'ext_hist' };
            sel.update(clauseInterval('num_of_tokens', [250, 350], { source: extHist }));
            await client.filterSettled;
            const pathAPath = client.lastExecutionPath;
            const pathASqlCount = client.sqlQueryCount;

            const extSearch = { id: 'ext_search' };
            sel.update(clauseMatch('prompt', 'mixtral', { source: extSearch }) as any);
            await client.filterSettled;
            await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
            const pathBPath = client.lastExecutionPath;
            const pathBSqlCount = client.sqlQueryCount;
            const hitViaMosaicPathB = await plot.pickAt(480, 300);

            client.destroy();
            plot.destroy();

            return {
                maskActiveBytes,
                maskClearedBytes,
                hitIncludedRowId: hitIncluded?.rowId ?? -1,
                intersectedSelIds: intersectedSel.rowIds,
                hitExcludedIs40064: hitExcluded?.rowId === 40064,
                pathAPath,
                pathASqlCount,
                pathBPath,
                pathBSqlCount,
                sqlQueries,
                hitViaMosaicPathBRowId: hitViaMosaicPathB?.rowId ?? -1,
            };
        });

        console.log('[5. Slot Bitmask & Live Mosaic Client]', JSON.stringify(mosaicAndMaskCheck));
        expect(mosaicAndMaskCheck.maskActiveBytes).toBeGreaterThan(33792);
        expect(mosaicAndMaskCheck.maskClearedBytes).toBe(33792);
        expect(mosaicAndMaskCheck.hitIncludedRowId).toBe(40064);
        expect(mosaicAndMaskCheck.intersectedSelIds).toEqual([40064]);
        expect(mosaicAndMaskCheck.hitExcludedIs40064).toBe(false);
        expect(mosaicAndMaskCheck.pathAPath).toBe('gpu');
        expect(mosaicAndMaskCheck.pathASqlCount).toBe(0);
        expect(mosaicAndMaskCheck.pathBPath).toBe('sql');
        expect(mosaicAndMaskCheck.pathBSqlCount).toBe(1);
        expect(mosaicAndMaskCheck.sqlQueries[0]).toContain('SELECT "row_id"');
        expect(mosaicAndMaskCheck.hitViaMosaicPathBRowId).toBe(40064);

        // =========================================================================
        // 6. OPTICAL MODE 0 (`gaia_sample.arrowtiles`) 2-PASS CONTEXT DIM + ZERO-LEAK TEARDOWN
        // =========================================================================
        const gaiaAndTeardown = await page.evaluate(async (pref) => {
            const { Deepgraph } = await import('/src/index.ts');
            const { INTERNAL_DIAGNOSTICS } = await import('/src/lib/diagnostics.ts');
            const { getGpuStats } = await import('/src/demo/gpuStats.ts');
            const stage = document.getElementById('stage')!;
            const gaia = await Deepgraph.create(stage, {
                url: '/samples/gaia_sample.arrowtiles',
                powerPreference: pref,
                encoding: {
                    color: { field: 'bp_rp' },
                    size: { field: 'phot_g_mean_mag' },
                },
            });
            gaia.fitView();
            const diag = (gaia as any)[INTERNAL_DIAGNOSTICS]();
            const d = performance.now() + 8000;
            while (performance.now() < d) {
                await new Promise((r) => setTimeout(r, 60));
                const tm: any = diag.getTileManager();
                if (tm && tm.activeTiles.size >= 1 && tm.loadingTiles.size === 0) break;
            }
            gaia.setFilter({
                ranges: { bp_rp: [0.5, 2.0] },
                unselectedOpacity: 0.15,
            });
            await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
            (window as any).__gaiaPlot = gaia;
            return getGpuStats();
        }, expectedPowerPref);
        expect(gaiaAndTeardown.liveDevices).toBe(1);

        const gaiaShimmer = await measureStationaryShimmer(page);
        console.log('[6. Gaia Mode 0 Context-Dim Shimmer]', JSON.stringify(gaiaShimmer));
        expect(gaiaShimmer.litPixels).toBeGreaterThan(200);
        expect(gaiaShimmer.changedPixels).toBe(0);

        const finalTeardown = await page.evaluate(async () => {
            const { getGpuStats } = await import('/src/demo/gpuStats.ts');
            (window as any).__gaiaPlot.destroy();
            return {
                gs: getGpuStats(),
                uncaptured: (window as any).__wgpuUncaptured,
            };
        });
        console.log('[7. Final GPU Teardown]', JSON.stringify(finalTeardown));
        expect(finalTeardown.gs.liveDevices).toBe(0);
        expect(finalTeardown.gs.liveBuffers).toBe(0);
        expect(finalTeardown.gs.liveBytes).toBe(0);
        expect(finalTeardown.uncaptured).toEqual([]);
        expect(pageErrors).toEqual([]);
        expect(consoleErrors).toEqual([]);
    });
});
