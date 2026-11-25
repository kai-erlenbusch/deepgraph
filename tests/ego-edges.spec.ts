import { test, expect, type Page } from '@playwright/test';

const allowDgpu = process.env.DEEPGRAPH_ALLOW_DGPU === '1';
const expectedPowerPref: 'high-performance' | 'low-power' = allowDgpu
    ? 'high-performance'
    : 'low-power';

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

test.describe('Step 3A — Curved Quadratic Bézier Graph & Ego-Network Arcs (Intel iGPU Sample-Only)', () => {
    test('Verifies 25,000-edge instanced Bézier pass, bidirectional A->B / B->A separation, async resolveEdges (AbortSignal + click-pin + 2s axis transition) & zero-leak teardown', async ({
        page,
    }) => {
        test.setTimeout(120000);
        const pageErrors: string[] = [];
        const consoleErrors: string[] = [];
        page.on('pageerror', (e) => pageErrors.push(e.message || String(e)));
        page.on('console', (m) => {
            if (m.type() === 'error') consoleErrors.push(m.text());
        });

        await page.route('**/__step3a_ego_edges.html', async (route) => {
            await route.fulfill({
                status: 200,
                contentType: 'text/html; charset=utf-8',
                body: `<!doctype html><html><body style="margin:0;background:#111"><div id="stage" style="width:960px;height:600px"></div></body></html>`,
            });
        });
        await page.goto('/__step3a_ego_edges.html', { waitUntil: 'domcontentloaded' });

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
        console.log('[Step 3A Ego-Edges GPU Pre-Check]', JSON.stringify(igpuGuard));
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
        // 1. BIDIRECTIONAL A->B vs B->A QUADRATIC BÉZIER SEPARATION + 25,000 STATIC ARCS
        // =========================================================================
        const setupInfo = await page.evaluate(async (pref) => {
            const { Deepgraph } = await import('/src/index.ts');
            const { INTERNAL_DIAGNOSTICS } = await import('/src/lib/diagnostics.ts');
            const stage = document.getElementById('stage')!;

            const resolverLog: { rowId: number; trigger: string; aborted: boolean }[] = [];
            const plot = await Deepgraph.create(stage, {
                url: '/samples/lmsys_sample.arrowtiles',
                powerPreference: pref,
                encoding: {
                    color: { field: 'model' },
                    aggregation: 'points',
                },
                edgeStyle: {
                    width: 2.2,
                    curvature: 0.25,
                    opacity: 0.9,
                    colorHex: 0x38bdf8,
                    animatePulse: false,
                },
                resolveEdges: async (point, trigger, signal) => {
                    const entry = { rowId: point.rowId, trigger, aborted: false };
                    resolverLog.push(entry);
                    signal.addEventListener('abort', () => {
                        entry.aborted = true;
                    });
                    // Slow down the first hover invocation (1200ms) so superseding hover on 40064 aborts it deterministically
                    const hoverCallsSoFar = resolverLog.filter(
                        (e) => e.trigger === 'hover'
                    ).length;
                    if (trigger === 'hover' && hoverCallsSoFar === 1) {
                        await new Promise((r) => setTimeout(r, 1200));
                    } else {
                        await new Promise((r) => setTimeout(r, 15));
                    }
                    if (signal.aborted) return [];
                    return [
                        {
                            source: { x: point.dataX, y: point.dataY },
                            target: { x: 20.0, y: 80.0 },
                            weight: 0.9,
                            colorHex: 0xfbbf24,
                            sourceAttributes: { num_of_tokens: 308 },
                            targetAttributes: { num_of_tokens: 45 },
                        },
                        {
                            source: { x: point.dataX, y: point.dataY },
                            target: { x: 75.0, y: 15.0 },
                            weight: 0.6,
                            colorHex: 0xf472b6,
                            sourceAttributes: { num_of_tokens: 308 },
                            targetAttributes: { num_of_tokens: 420 },
                        },
                    ];
                },
            });
            plot.fitView();
            const diag = (plot as any)[INTERNAL_DIAGNOSTICS]();
            (window as any).__edgesCtx = { plot, diag, resolverLog };

            const deadline = performance.now() + 10000;
            while (performance.now() < deadline) {
                await new Promise((r) => setTimeout(r, 60));
                const tm: any = diag.getTileManager();
                if (tm && tm.activeTiles.size >= 5 && tm.loadingTiles.size === 0) break;
            }
            await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

            const idleEdgeBytes = diag.edgeOverlay.getLiveGpuBytes();

            // Isolate empty dark sky via empty filter rect so ONLY the bidirectional pair renders!
            plot.setFilter({ rect: { minX: -9999, maxX: -9998, minY: -9999, maxY: -9998 } });
            plot.setViewport({ x: 50, y: 50, zoom: 2.0 });
            plot.setEdges(
                [
                    {
                        source: { x: 20, y: 50 },
                        target: { x: 80, y: 50 },
                        weight: 1.0,
                        curvature: 0.3,
                        colorHex: 0x00ffff,
                    },
                    {
                        source: { x: 80, y: 50 },
                        target: { x: 20, y: 50 },
                        weight: 1.0,
                        curvature: 0.3,
                        colorHex: 0xff00ff,
                    },
                ],
                { width: 3.0, opacity: 1.0, animatePulse: false }
            );
            await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
            return { idleEdgeBytes };
        }, expectedPowerPref);

        expect(setupInfo.idleEdgeBytes).toBe(0);

        // Verify bidirectional separation: chord midpoint (y=300) is dark #111 while upper & lower arcs are lit!
        const bidirPng = (await page.locator('canvas').first().screenshot({ type: 'png' })).toString(
            'base64'
        );
        const bidirProbe = await page.evaluate(async (b64) => {
            const res = await fetch(`data:image/png;base64,${b64}`);
            const bmp = await createImageBitmap(await res.blob());
            const c = new OffscreenCanvas(bmp.width, bmp.height);
            const ctx = c.getContext('2d')!;
            ctx.drawImage(bmp, 0, 0);
            const midX = Math.floor(bmp.width * 0.5);
            const midY = Math.floor(bmp.height * 0.5);
            const col = ctx.getImageData(midX, 0, 1, bmp.height).data;
            const lumDiff = (y: number) => {
                const i = y * 4;
                return Math.max(
                    Math.abs(col[i] - 17),
                    Math.abs(col[i + 1] - 17),
                    Math.abs(col[i + 2] - 17)
                );
            };
            const chordCenterDiff = lumDiff(midY);
            let upperArcHits = 0;
            let lowerArcHits = 0;
            for (let y = 0; y < midY - 12; y++) {
                if (lumDiff(y) > 40) upperArcHits++;
            }
            for (let y = midY + 12; y < bmp.height; y++) {
                if (lumDiff(y) > 40) lowerArcHits++;
            }
            return { chordCenterDiff, upperArcHits, lowerArcHits };
        }, bidirPng);
        console.log('[1a. Bidirectional A->B / B->A Separation]', JSON.stringify(bidirProbe));
        expect(bidirProbe.chordCenterDiff).toBeLessThanOrEqual(4);
        expect(bidirProbe.upperArcHits).toBeGreaterThan(0);
        expect(bidirProbe.lowerArcHits).toBeGreaterThan(0);

        // Ground-truth LMSys node-to-node endpoint + curved Bézier apex pixel proof (`40064 -> 38479`)
        const gtNodeArcGeometry = await page.evaluate(async () => {
            const { plot } = (window as any).__edgesCtx;
            const { projectDataToWorld64 } = await import('/src/core/FilterGpuResources.ts');
            plot.setFilter({ rect: { minX: -9999, maxX: -9998, minY: -9999, maxY: -9998 } });
            plot.fitView();
            const src = { x: 88.76073587009975, y: 69.12738843695004 }; // row_id=40064
            const dst = { x: 87.7994427209227, y: 29.497411031549206 }; // row_id=38479
            const curv = 0.28;
            plot.setEdges(
                [{ sourceRowId: 40064, targetRowId: 38479, source: src, target: dst, curvature: curv, colorHex: 0x00ffaa }],
                { width: 3.6, opacity: 1.0, directed: false, animatePulse: false }
            );
            await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

            const { current } = (plot as any).buildProjectionContexts((plot as any).session);
            const cam = (plot as any).rendererWrapper.camera;
            const canvas = document.querySelector('canvas')!;
            const camLeft = cam.position.x + cam.left / cam.zoom;
            const camRight = cam.position.x + cam.right / cam.zoom;
            const camBottom = cam.position.y + cam.bottom / cam.zoom;
            const camTop = cam.position.y + cam.top / cam.zoom;

            const wS = projectDataToWorld64(src.x, src.y, current);
            const wT = projectDataToWorld64(dst.x, dst.y, current);
            const p0x = ((wS.worldX - camLeft) / (camRight - camLeft)) * canvas.width;
            const p0y = ((camTop - wS.worldY) / (camTop - camBottom)) * canvas.height;
            const p2x = ((wT.worldX - camLeft) / (camRight - camLeft)) * canvas.width;
            const p2y = ((camTop - wT.worldY) / (camTop - camBottom)) * canvas.height;

            const dx = p2x - p0x;
            const dy = p2y - p0y;
            const len = Math.hypot(dx, dy);
            const perpX = -dy / len;
            const perpY = dx / len;
            // At t = 0.5, quadratic Bézier B(0.5) = 0.25*p0 + 0.5*p1 + 0.25*p2 = 0.5*(p0+p2) + 0.5*curv*len*perp
            const chordMidX = 0.5 * (p0x + p2x);
            const chordMidY = 0.5 * (p0y + p2y);
            const apexX = chordMidX + 0.5 * curv * len * perpX;
            const apexY = chordMidY + 0.5 * curv * len * perpY;
            return { p0x, p0y, p2x, p2y, apexX, apexY, chordMidX, chordMidY };
        });

        const gtPng = (await page.locator('canvas').first().screenshot({ type: 'png' })).toString('base64');
        const gtNodeCheck = await page.evaluate(
            async ({ b64, g }) => {
                const res = await fetch(`data:image/png;base64,${b64}`);
                const bmp = await createImageBitmap(await res.blob());
                const c = new OffscreenCanvas(bmp.width, bmp.height);
                const ctx = c.getContext('2d')!;
                ctx.drawImage(bmp, 0, 0);
                const maxPatchDelta = (cx: number, cy: number, r = 1) => {
                    let best = 0;
                    for (let dy = -r; dy <= r; dy++) {
                        for (let dx = -r; dx <= r; dx++) {
                            const x = Math.max(0, Math.min(bmp.width - 1, Math.round(cx) + dx));
                            const y = Math.max(0, Math.min(bmp.height - 1, Math.round(cy) + dy));
                            const p = ctx.getImageData(x, y, 1, 1).data;
                            const d = Math.max(
                                Math.abs(p[0] - 17),
                                Math.abs(p[1] - 17),
                                Math.abs(p[2] - 17)
                            );
                            if (d > best) best = d;
                        }
                    }
                    return best;
                };
                return {
                    srcDelta: maxPatchDelta(g.p0x, g.p0y, 1),
                    apexDelta: maxPatchDelta(g.apexX, g.apexY, 1),
                    dstDelta: maxPatchDelta(g.p2x, g.p2y, 1),
                    chordMidDelta: maxPatchDelta(g.chordMidX, g.chordMidY, 0),
                };
            },
            { b64: gtPng, g: gtNodeArcGeometry }
        );
        console.log('[1a-GT. LMSys 40064->38479 Endpoint & Apex Proof]', JSON.stringify(gtNodeCheck));
        expect(gtNodeCheck.srcDelta).toBeGreaterThan(60);
        expect(gtNodeCheck.apexDelta).toBeGreaterThan(60);
        expect(gtNodeCheck.dstDelta).toBeGreaterThan(60);
        expect(gtNodeCheck.chordMidDelta).toBeLessThanOrEqual(4);

        // Now upload 25,000 curved static arcs on top of LMSys points and verify 0.0000% stationary shimmer!
        const edges25kBytes = await page.evaluate(async () => {
            const { plot, diag } = (window as any).__edgesCtx;
            plot.setFilter(null);
            plot.fitView();
            const edges = [];
            for (let i = 0; i < 25_000; i++) {
                const a = (i * 0.6180339887) % 1;
                const b = ((i * 0.3819660113) + 0.17) % 1;
                edges.push({
                    source: { x: 5 + a * 90, y: 5 + ((i * 0.271828) % 1) * 90 },
                    target: { x: 5 + b * 90, y: 5 + ((i * 0.718281) % 1) * 90 },
                    weight: (i % 10) / 10,
                });
            }
            plot.setEdges(edges, { width: 1.1, opacity: 0.45, animatePulse: false });
            await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
            return diag.edgeOverlay.getLiveGpuBytes();
        });
        // 512 B dynamic uniform buffer + 25,000 * 48 B (1,200,000 B) = 1,200,512 B
        expect(edges25kBytes).toBe(25_000 * 48 + 512);

        const shimmer25k = await measureStationaryShimmer(page);
        console.log('[1b. 25,000 Static Bézier Arcs Shimmer]', JSON.stringify(shimmer25k));
        expect(shimmer25k.litPixels).toBeGreaterThan(80_000);
        expect(shimmer25k.changedPixels).toBe(0);

        // Clear static edges -> staticLayer GPUBuffer immediately destroyed (leaves 512 B uniform buffer)
        const afterClearStaticBytes = await page.evaluate(async () => {
            const { plot, diag } = (window as any).__edgesCtx;
            plot.setEdges([]);
            await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
            return diag.edgeOverlay.getLiveGpuBytes();
        });
        expect(afterClearStaticBytes).toBe(512);

        // =========================================================================
        // 2. ASYNC `resolveEdges` ('hover' ABORT SUPERSESSION + 'click' PIN + 2s TRANSITION)
        // =========================================================================
        const stageBox = await page.locator('#stage').boundingBox();
        expect(stageBox).not.toBeNull();
        const centerPageX = Math.round(stageBox!.x + 480);
        const centerPageY = Math.round(stageBox!.y + 300);

        // 2a. Hover row_id=38479 (180ms slow resolver), then immediately jump to row_id=40064 (15ms fast resolver) -> 38479 aborted!
        await page.evaluate(async () => {
            const { plot, diag, resolverLog } = (window as any).__edgesCtx;
            resolverLog.length = 0;
            plot.setViewport({ x: 87.7994427209227, y: 29.497411031549206, zoom: 128 });
            const d = performance.now() + 6000;
            while (performance.now() < d) {
                await new Promise((r) => setTimeout(r, 50));
                const tm: any = diag.getTileManager();
                if (tm && tm.loadingTiles.size === 0 && tm.currentMaxZ >= 4) break;
            }
        });
        await page.mouse.move(centerPageX - 40, centerPageY - 40);
        await page.mouse.move(centerPageX, centerPageY);
        await expect
            .poll(
                async () => {
                    await page.mouse.move(centerPageX, centerPageY);
                    return await page.evaluate(
                        () => (window as any).__edgesCtx.resolverLog.length
                    );
                },
                { timeout: 5000, intervals: [40, 60] }
            )
            .toBeGreaterThanOrEqual(1);

        // Immediately jump viewport & pointer to row_id=40064 while 38479's 1200ms resolver is still in-flight!
        await page.evaluate(async () => {
            const { plot, diag } = (window as any).__edgesCtx;
            plot.setViewport({ x: 88.76073587009975, y: 69.12738843695004, zoom: 128 });
            const d = performance.now() + 6000;
            while (performance.now() < d) {
                await new Promise((r) => setTimeout(r, 40));
                const tm: any = diag.getTileManager();
                if (tm && tm.loadingTiles.size === 0 && tm.currentMaxZ >= 4) break;
            }
        });
        await page.mouse.move(centerPageX + 2, centerPageY + 2);
        await page.mouse.move(centerPageX, centerPageY);
        await page.waitForTimeout(260);

        const hoverSupersedeCheck = await page.evaluate(() => {
            const { resolverLog } = (window as any).__edgesCtx;
            const firstHover = resolverLog.find((e: any) => e.trigger === 'hover');
            const has40064 = resolverLog.some(
                (e: any) => e.rowId === 40064 && e.trigger === 'hover'
            );
            return {
                firstHoverRowId: firstHover?.rowId ?? -1,
                abortedFirstHover: Boolean(firstHover?.aborted),
                has40064,
            };
        });
        console.log('[2a. Hover AbortSignal Supersession]', JSON.stringify(hoverSupersedeCheck));
        expect(hoverSupersedeCheck.abortedFirstHover).toBe(true);
        expect(hoverSupersedeCheck.has40064).toBe(true);

        // 2b. Click row_id=40064 to pin ego-arcs, move pointer to empty corner, and freeze mid-2s axis transition!
        await page.mouse.click(centerPageX, centerPageY);
        await expect
            .poll(
                async () =>
                    await page.evaluate(() =>
                        (window as any).__edgesCtx.resolverLog.some(
                            (e: any) => e.rowId === 40064 && e.trigger === 'click'
                        )
                    ),
                { timeout: 5000 }
            )
            .toBe(true);
        await page.waitForTimeout(80);
        // Move mouse to empty corner -> click-pinned ego edges MUST stay active!
        await page.mouse.move(Math.round(stageBox!.x + 15), Math.round(stageBox!.y + 15));
        await page.waitForTimeout(150);

        const pinnedAndMidTransition = await page.evaluate(async () => {
            const { plot, diag } = (window as any).__edgesCtx;
            const pinnedBytesBeforeMove = diag.edgeOverlay.getLiveGpuBytes();

            // Isolate dark sky so mid-transition pixel check measures ONLY the interpolated ego-arcs!
            plot.setFilter({ rect: { minX: -9999, maxX: -9998, minY: -9999, maxY: -9998 } });

            // Also verify null chart-axis attribute culling (`Reviewer 1 P1-3`):
            // Setting a static edge with `targetAttributes: { num_of_tokens: null }` on non-spatial `num_of_tokens` X axis culls it!
            plot.setEdges([
                {
                    source: { x: 10, y: 10 },
                    target: { x: 90, y: 90 },
                    sourceAttributes: { num_of_tokens: 150 },
                    targetAttributes: { num_of_tokens: null },
                },
            ]);

            // Trigger 2-second axis transition to `num_of_tokens` and freeze at transitionT = 0.5
            plot.setEncoding({
                x: { field: 'num_of_tokens', transform: 'literal', domain: [0, 500] },
                y: null,
            });
            plot.fitView();
            const store = diag.tileStore;
            store.isTransitioning = true;
            store.transitionT = 0.5;
            (plot as any).renderLoop.stop();
            (plot as any).rendererWrapper.updateCameraUniforms();
            const { current, previous } = (plot as any).buildProjectionContexts((plot as any).session);
            (plot as any).computeRasterizer.render(diag.camera, store, (enc: any, view: any) => {
                diag.edgeOverlay.encodePass(enc, view, diag.camera, store, current, previous);
            });
            const bytesMidTransition = diag.edgeOverlay.getLiveGpuBytes();
            const staticNullCulledInstances = (diag.edgeOverlay as any).staticLayer.instanceCount;
            const egoValidInstances = (diag.edgeOverlay as any).egoLayer.instanceCount;
            return {
                pinnedBytesBeforeMove,
                bytesMidTransition,
                staticNullCulledInstances,
                egoValidInstances,
            };
        });
        const midTransPng = (await page.locator('canvas').first().screenshot({ type: 'png' })).toString('base64');
        const midTransLit = await page.evaluate(async (b64) => {
            const res = await fetch(`data:image/png;base64,${b64}`);
            const bmp = await createImageBitmap(await res.blob());
            const c = new OffscreenCanvas(bmp.width, bmp.height);
            const ctx = c.getContext('2d')!;
            ctx.drawImage(bmp, 0, 0);
            const d = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
            let lit = 0;
            for (let i = 0; i < d.length; i += 4) {
                if (
                    Math.abs(d[i] - 17) > 15 ||
                    Math.abs(d[i + 1] - 17) > 15 ||
                    Math.abs(d[i + 2] - 17) > 15
                ) {
                    lit++;
                }
            }
            const { plot, diag } = (window as any).__edgesCtx;
            plot.setEdges([]);
            plot.clearSelection();
            return { lit, bytesAfterClearSelection: diag.edgeOverlay.getLiveGpuBytes() };
        }, midTransPng);
        console.log(
            '[2b. Click-Pinned Ego Arcs, Null-Chart Cull & 2s Transition]',
            JSON.stringify({ ...pinnedAndMidTransition, ...midTransLit })
        );
        // 512 B uniform + 2 * 48 B = 608 B (static null-attr edge was culled -> 0 static buffer bytes!)
        expect(pinnedAndMidTransition.pinnedBytesBeforeMove).toBe(512 + 2 * 48);
        expect(pinnedAndMidTransition.staticNullCulledInstances).toBe(0);
        expect(pinnedAndMidTransition.egoValidInstances).toBe(2);
        expect(pinnedAndMidTransition.bytesMidTransition).toBe(512 + 2 * 48);
        expect(midTransLit.lit).toBeGreaterThan(50);
        expect(midTransLit.bytesAfterClearSelection).toBe(512);

        // =========================================================================
        // 3. DETERMINISTIC ZERO-LEAK GPU TEARDOWN
        // =========================================================================
        const finalTeardown = await page.evaluate(async () => {
            const { getGpuStats } = await import('/src/demo/gpuStats.ts');
            (window as any).__edgesCtx.plot.destroy();
            return {
                gs: getGpuStats(),
                uncaptured: (window as any).__wgpuUncaptured,
            };
        });
        console.log('[3. Final GPU Teardown]', JSON.stringify(finalTeardown));
        expect(finalTeardown.gs.liveDevices).toBe(0);
        expect(finalTeardown.gs.liveBuffers).toBe(0);
        expect(finalTeardown.gs.liveBytes).toBe(0);
        expect(finalTeardown.uncaptured).toEqual([]);
        expect(pageErrors).toEqual([]);
        expect(consoleErrors).toEqual([]);
    });
});
