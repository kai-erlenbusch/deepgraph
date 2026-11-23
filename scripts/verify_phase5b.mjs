import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const viewerRoot = path.resolve(__dirname, '..');
process.chdir(viewerRoot);

const require = createRequire(path.join(viewerRoot, 'package.json'));
const { createServer } = require('vite');
const { chromium } = require('@playwright/test');

function assert(cond, msg) {
    if (!cond) {
        throw new Error(`ASSERTION FAILED: ${msg}`);
    }
}

async function analyzeScreenshotPng(page) {
    const b64 = (await page.screenshot({ type: 'png' })).toString('base64');
    return await page.evaluate(
        async ({ b64Str }) => {
            const res = await fetch(`data:image/png;base64,${b64Str}`);
            const blob = await res.blob();
            const bmp = await createImageBitmap(blob);
            const c = new OffscreenCanvas(bmp.width, bmp.height);
            const ctx = c.getContext('2d');
            ctx.drawImage(bmp, 0, 0);
            const x0 = Math.floor(bmp.width * 0.15);
            const y0 = Math.floor(bmp.height * 0.15);
            const w = Math.floor(bmp.width * 0.7);
            const h = Math.floor(bmp.height * 0.7);
            const data = ctx.getImageData(x0, y0, w, h).data;
            let nonBgPixels = 0;
            let colorfulPixels = 0;
            let pureWhitePixels = 0;
            const total = w * h;
            for (let i = 0; i < data.length; i += 4) {
                const r = data[i];
                const g = data[i + 1];
                const b = data[i + 2];
                if (Math.abs(r - 17) > 6 || Math.abs(g - 17) > 6 || Math.abs(b - 17) > 6) {
                    nonBgPixels++;
                }
                if (Math.max(r, g, b) - Math.min(r, g, b) > 25) {
                    colorfulPixels++;
                }
                if (r >= 250 && g >= 250 && b >= 250) {
                    pureWhitePixels++;
                }
            }
            return {
                total,
                nonBgPixels,
                colorfulPixels,
                pureWhitePixels,
                whiteRatio: pureWhitePixels / Math.max(1, nonBgPixels),
            };
        },
        { b64Str: b64 }
    );
}

async function main() {
    const server = await createServer({
        root: viewerRoot,
        configFile: path.join(viewerRoot, 'vite.config.ts'),
        server: {
            host: '127.0.0.1',
            port: 0,
            strictPort: false,
        },
        logLevel: 'error',
    });
    await server.listen();
    const addr = server.httpServer.address();
    const baseUrl = `http://127.0.0.1:${addr.port}`;
    console.log(`[verify_phase5b] Ephemeral Vite server running at ${baseUrl}`);

    const browser = await chromium.launch({
        channel: 'chrome',
        headless: false,
        args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'],
    });

    const pageErrors = [];
    const consoleErrors = [];
    const results = {};

    try {
        const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
        await context.addInitScript(() => {
            window.__wgpuUncapturedErrors = [];
            window.__wgpuDevices = [];
            if (navigator.gpu && GPUAdapter && GPUAdapter.prototype.requestDevice) {
                const origReq = GPUAdapter.prototype.requestDevice;
                GPUAdapter.prototype.requestDevice = async function (...args) {
                    const dev = await origReq.apply(this, args);
                    dev.addEventListener('uncapturederror', (ev) => {
                        window.__wgpuUncapturedErrors.push(
                            ev?.error?.message || String(ev?.error || ev)
                        );
                    });
                    window.__wgpuDevices.push(dev);
                    return dev;
                };
            }
        });

        const page = await context.newPage();
        page.on('pageerror', (err) => pageErrors.push(err.message || String(err)));
        page.on('console', (msg) => {
            if (msg.type() === 'error') {
                consoleErrors.push(msg.text());
            }
        });

        await page.route('**/__embed_5b.html', async (route) => {
            await route.fulfill({
                status: 200,
                contentType: 'text/html; charset=utf-8',
                body: `<!doctype html><html><body style="margin:0;background:#111"><div id="embed" style="width:960px;height:600px"></div><div id="race" style="width:320px;height:240px"></div></body></html>`,
            });
        });

        // =========================================================================
        // CHECK 1 & 2: Sealed src/index.ts barrel, Deepgraph.isSupported(),
        //              standalone embed (0 lil-gui), mid-flight load() + destroy() race,
        //              and superseded load(A) -> load(B) AbortError semantics
        // =========================================================================
        await page.goto(`${baseUrl}/__embed_5b.html`, { waitUntil: 'domcontentloaded' });
        const check12 = await page.evaluate(async () => {
            const { installGpuStatsInstrumentation } = await import('/src/demo/gpuStats.ts');
            installGpuStatsInstrumentation();
            const pkg = await import('/src/index.ts');
            const exportedKeys = Object.keys(pkg);
            const supported = pkg.Deepgraph.isSupported();
            const lilGuiInDom = document.querySelectorAll('.lil-gui').length;

            // Check 2a: Mid-flight load() + destroy() race guard
            const raceBox = document.getElementById('race');
            const pRace = await pkg.Deepgraph.create(raceBox);
            const inflightLoad = pRace.load('/samples/lmsys_d7_proportional_25k.arrowtiles');
            pRace.destroy();
            let raceErrorMsg = '';
            try {
                await inflightLoad;
            } catch (err) {
                raceErrorMsg = err instanceof Error ? err.message : String(err);
            }
            await new Promise((r) => setTimeout(r, 250));
            const raceCanvasesAfterDestroy = raceBox.querySelectorAll('canvas').length;

            // Check 2b: Superseded load(A) -> load(B) rejects A with AbortError while B wins
            const embedBox = document.getElementById('embed');
            const pEmbed = await pkg.Deepgraph.create(embedBox);
            const loadA = pEmbed.load('/samples/lmsys_sample.arrowtiles');
            const loadB = pEmbed.load('/samples/gaia_sample.arrowtiles', {
                encoding: { color: 'bp_rp' },
            });
            let supersededErrName = '';
            try {
                await loadA;
            } catch (err) {
                supersededErrName = err?.name || String(err);
            }
            const summaryB = await loadB;
            await new Promise((r) => setTimeout(r, 500));

            const embedVp0 = pEmbed.getViewport();
            pEmbed.setViewport({ x: 1.5, y: -0.75, zoom: 3.2 });
            const embedVp1 = pEmbed.getViewport();
            pEmbed.fitView();
            const embedVpFit = pEmbed.getViewport();
            pEmbed.destroy();
            const embedCanvasesAfterDestroy = embedBox.querySelectorAll('canvas').length;

            return {
                exportedKeys,
                supported,
                lilGuiInDom,
                raceErrorMsg,
                raceCanvasesAfterDestroy,
                supersededErrName,
                activeTotalRows: pEmbed.getMetadata()?.totalRows ?? summaryB.totalRows,
                embedVp0,
                embedVp1,
                embedVpFit,
                embedCanvasesAfterDestroy,
                gpuStatsAfterEmbedDestroy: { ...window.__gpuStats },
                wgpuErrors: window.__wgpuUncapturedErrors.slice(),
            };
        });
        console.log('[Check 1 & 2]', JSON.stringify(check12));
        assert(
            check12.exportedKeys.length === 1 && check12.exportedKeys[0] === 'Deepgraph',
            `Expected sealed barrel ['Deepgraph'], got ${JSON.stringify(check12.exportedKeys)}`
        );
        assert(check12.supported === true, 'Deepgraph.isSupported() must be true');
        assert(check12.lilGuiInDom === 0, 'Standalone embed must inject 0 .lil-gui nodes');
        assert(
            check12.raceErrorMsg.includes('destroyed'),
            `Mid-flight load()+destroy() should reject cleanly, got: ${check12.raceErrorMsg}`
        );
        assert(
            check12.supersededErrName === 'AbortError',
            `Superseded load(A) must reject with AbortError, got: ${check12.supersededErrName}`
        );
        assert(
            check12.activeTotalRows === 149942,
            `Expected B (gaia_sample = 149,942 rows) to win, got ${check12.activeTotalRows}`
        );
        assert(check12.raceCanvasesAfterDestroy === 0, 'Destroyed race plot left canvas in DOM');
        assert(check12.embedCanvasesAfterDestroy === 0, 'Destroyed embed plot left canvas in DOM');
        assert(
            check12.gpuStatsAfterEmbedDestroy.liveDevices === 0 &&
                check12.gpuStatsAfterEmbedDestroy.liveBytes === 0,
            `Expected 0 liveDevices and 0 liveBytes after destroy(), got ${JSON.stringify(check12.gpuStatsAfterEmbedDestroy)}`
        );
        assert(Math.abs(check12.embedVp1.x - 1.5) < 1e-5, 'Embed setViewport/getViewport X mismatch');
        assert(Math.abs(check12.embedVp1.y - -0.75) < 1e-5, 'Embed setViewport/getViewport Y mismatch');
        assert(check12.wgpuErrors.length === 0, `WebGPU errors in Check 1&2: ${check12.wgpuErrors}`);
        results.check1_2_barrel_and_race = check12;

        // =========================================================================
        // CHECK 3: URL sanitization in src/main.ts (?file=//... and ?file=/\...)
        // =========================================================================
        for (const badQuery of ['?file=//evil.example/x.arrowtiles', '?file=/%5Cevil.arrowtiles']) {
            await page.goto(`${baseUrl}/${badQuery}`, { waitUntil: 'domcontentloaded' });
            await page.waitForFunction(() => Boolean(window.__viewer?.plot?.getMetadata()), null, {
                timeout: 15000,
            });
            const presetName = await page.evaluate(() => window.__viewer.plot.getMetadata()?.presetName);
            assert(
                presetName === 'gaia-dr3',
                `URL query ${badQuery} did not fall back safely to /gaia_full.arrowtiles (got ${presetName})`
            );
        }
        await page.evaluate(async () => {
            const { INTERNAL_DIAGNOSTICS } = await import('/src/lib/diagnostics.ts');
            Object.assign(window.__viewer, window.__viewer.plot[INTERNAL_DIAGNOSTICS]());
        });
        results.check3_url_sanitization = 'PASS (both // and /\\ rejected)';

        // =========================================================================
        // CHECK 4: Modes 0, 2, 3, 4 on gaia_full.arrowtiles + M_G <= 0 bright-star bin check
        // =========================================================================
        await page.waitForFunction(
            () => {
                const tm = window.__viewer?.getTileManager?.();
                return tm && tm.activeTiles && tm.activeTiles.size >= 4;
            },
            null,
            { timeout: 15000 }
        );
        await page.waitForTimeout(800);
        const mode0Stats = await analyzeScreenshotPng(page);
        const magBinCheck = await page.evaluate(async () => {
            const dev = window.__wgpuDevices[window.__wgpuDevices.length - 1];
            dev.pushErrorScope('validation');
            const v = window.__viewer;
            const store = v.tileStore;

            const fakeTile = {
                key: '99/0/0',
                z: 0,
                x: 0,
                y: 0,
                numRows: 3,
                bounds: { minX: -1, maxX: 1, minY: -1, maxY: 1 },
                uploadedToGpu: false,
                columns: {
                    xy_u16: new Uint32Array([0x80008000, 0x40004000, 0x20002000]).buffer,
                    M_G: new Float32Array([-1.46, 0.0, 25.0]).buffer,
                },
            };
            const prevMagField = store.magnitudeField;
            store.magnitudeField = 'M_G';
            store.updateVisibleTiles([fakeTile]);
            const slotIdx = store.tileKeyToSlot.get('99/0/0');
            const bin0Count = slotIdx !== undefined ? store.slotMagBins[slotIdx * 16 + 0] : -1;
            store.unloadTile('99/0/0');
            store.magnitudeField = prevMagField;

            const valErr = await dev.popErrorScope();
            return {
                mode0ComputeMode: store.computeMode,
                bin0Count,
                validationError: valErr ? valErr.message : null,
            };
        });
        console.log('[Check 4 Mode 0 + M_G <= 0]', JSON.stringify({ mode0Stats, magBinCheck }));
        assert(mode0Stats.nonBgPixels > 1000, `Mode 0 rendered blank canvas (${mode0Stats.nonBgPixels})`);
        assert(
            magBinCheck.bin0Count === 2,
            `Expected 2 bright stars (M_G = -1.46 and 0.0) in bin 0, got ${magBinCheck.bin0Count}`
        );
        assert(!magBinCheck.validationError, `Mode 0 validation error: ${magBinCheck.validationError}`);

        const modes234 = await page.evaluate(async () => {
            const dev = window.__wgpuDevices[window.__wgpuDevices.length - 1];
            const plot = window.__viewer.plot;
            const store = window.__viewer.tileStore;
            const out = {};

            dev.pushErrorScope('validation');
            plot.setViewOptions({ pipeline: 'cartesian', experimental: { metricType: 0 } });
            plot.setEncoding({
                aggregation: 'mean_field',
                color: { field: 'teff_gspphot', scale: 'coolwarm', domain: [3200, 9500] },
            });
            await new Promise((r) => setTimeout(r, 800));
            const err2 = await dev.popErrorScope();
            out.mode2 = { computeMode: store.computeMode, err: err2 ? err2.message : null };

            dev.pushErrorScope('validation');
            plot.setViewOptions({
                pipeline: 'optical',
                experimental: {
                    aggregation: 'trails',
                    auxFields: ['pmra', 'pmdec', 'parallax'],
                    minParallax: 5.0,
                    trailTimeKyr: 400,
                },
            });
            await new Promise((r) => setTimeout(r, 800));
            const err3 = await dev.popErrorScope();
            out.mode3 = { computeMode: store.computeMode, err: err3 ? err3.message : null };

            dev.pushErrorScope('validation');
            plot.setEncoding({
                color: { field: 'radial_velocity', scale: 'rdbu', domain: [-75, 75] },
            });
            plot.setViewOptions({
                pipeline: 'cartesian',
                experimental: {
                    aggregation: 'streamlines',
                    auxFields: ['pmra', 'pmdec', 'parallax'],
                    animateFlow: true,
                },
            });
            await new Promise((r) => setTimeout(r, 800));
            const err4 = await dev.popErrorScope();
            out.mode4 = { computeMode: store.computeMode, err: err4 ? err4.message : null };

            return out;
        });
        console.log('[Check 4 Modes 2/3/4]', JSON.stringify(modes234));
        assert(modes234.mode2.computeMode === 2 && !modes234.mode2.err, `Mode 2 failed: ${JSON.stringify(modes234.mode2)}`);
        assert(modes234.mode3.computeMode === 3 && !modes234.mode3.err, `Mode 3 failed: ${JSON.stringify(modes234.mode3)}`);
        assert(modes234.mode4.computeMode === 4 && !modes234.mode4.err, `Mode 4 failed: ${JSON.stringify(modes234.mode4)}`);
        results.check4_gaia_modes = { mode0Stats, magBinCheck, modes234 };

        // =========================================================================
        // CHECK 5, 6, 7: gaia_sample.arrowtiles (v0.1):
        //   - D9 2-second WGSL smooth axis transition
        //   - Custom { domain, range, transform } on encoding.x & encoding.y
        //   - setEncoding({ x: null, y: null }) spatial reset & hidden x_u16/y_u16
        //   - Data-coordinate getViewport()/setViewport() & fitView()
        //   - CPU Float64 RTC deep-zoom (zoom = 131072) & zoomT <= 1.0 clamp
        // =========================================================================
        const check567 = await page.evaluate(async () => {
            const dev = window.__wgpuDevices[window.__wgpuDevices.length - 1];
            dev.pushErrorScope('validation');
            const v = window.__viewer;
            const plot = v.plot;
            const store = v.tileStore;

            await v.loadDataset('/samples/gaia_sample.arrowtiles');
            await new Promise((r) => setTimeout(r, 800));

            plot.fitView();
            const vpFit = plot.getViewport();
            plot.setViewport({ x: 1.41421356, y: -0.70710678, zoom: 4.0 });
            const vpData = plot.getViewport();
            const worldCamPos = { x: v.camera.position.x, y: v.camera.position.y };
            plot.fitView();

            plot.setViewOptions({
                pipeline: 'cartesian',
                experimental: { aggregation: undefined, metricType: 0 },
            });
            plot.setEncoding({
                x: { field: 'bp_rp', domain: [-0.5, 4.5], range: [-1.8, 1.8], transform: 'linear' },
                y: { field: 'phot_g_mean_mag', domain: [3.0, 21.0], range: [-1.6, 1.6], transform: 'linear' },
                flipY: true,
                aggregation: 'points',
                color: { field: 'bp_rp', scale: 'inferno' },
            });

            const vsChart = plot.getViewState();
            const cfgChart = JSON.parse(JSON.stringify(store.currentConfig));
            await new Promise((r) => setTimeout(r, 350));
            const midTransition = {
                isTransitioning: store.isTransitioning,
                transitionT: store.transitionT,
            };
            await new Promise((r) => setTimeout(r, 1900));
            const endTransition = {
                isTransitioning: store.isTransitioning,
                transitionT: store.transitionT,
            };

            plot.setEncoding({ x: null, y: null });
            const vsReset = plot.getViewState();
            const encReset = plot.getEncoding();
            const internalXAfterReset = store.currentConfig.x.field;
            const internalYAfterReset = store.currentConfig.y.field;
            await new Promise((r) => setTimeout(r, 2150));

            plot.setViewport({ x: 0.123456789, y: -0.098765432, zoom: 131072 });
            await new Promise((r) => setTimeout(r, 400));
            const deepZoomCamera = v.camera.zoom;
            plot.fitView();

            const valErr = await dev.popErrorScope();
            return {
                vpFit,
                vpData,
                worldCamPos,
                vsChartXDomain: vsChart.xDomain,
                vsChartXRange: vsChart.xRange,
                cfgChartX: cfgChart.x,
                cfgChartY: cfgChart.y,
                midTransition,
                endTransition,
                vsResetXField: vsReset.xField,
                vsResetYField: vsReset.yField,
                encResetHasX: 'x' in encReset,
                encResetHasY: 'y' in encReset,
                internalXAfterReset,
                internalYAfterReset,
                deepZoomCamera,
                validationError: valErr ? valErr.message : null,
            };
        });
        console.log('[Check 5, 6, 7]', JSON.stringify(check567));
        assert(!check567.validationError, `Check 5/6/7 WebGPU error: ${check567.validationError}`);
        assert(Math.abs(check567.vpData.x - 1.41421356) < 1e-5, 'Data-coord viewport X round-trip failed');
        assert(Math.abs(check567.vpData.y - -0.70710678) < 1e-5, 'Data-coord viewport Y round-trip failed');
        assert(
            Math.abs(check567.worldCamPos.x - 1.0) < 0.05,
            `World camera X should be ~1.0 (half of [-2,2]) for data X=1.4142, got ${check567.worldCamPos.x}`
        );
        assert(
            check567.cfgChartX.domain[0] === -0.5 &&
                check567.cfgChartX.domain[1] === 4.5 &&
                check567.cfgChartX.range[0] === -1.8 &&
                check567.cfgChartX.range[1] === 1.8,
            `Custom x domain/range ignored: ${JSON.stringify(check567.cfgChartX)}`
        );
        assert(
            check567.cfgChartY.domain[0] === 3.0 &&
                check567.cfgChartY.domain[1] === 21.0 &&
                check567.cfgChartY.range[0] === 1.6 &&
                check567.cfgChartY.range[1] === -1.6,
            `Custom y domain/range + flipY ignored: ${JSON.stringify(check567.cfgChartY)}`
        );
        assert(
            check567.midTransition.isTransitioning === true &&
                check567.midTransition.transitionT > 0.01 &&
                check567.midTransition.transitionT < 0.99,
            `Expected active 2s WGSL transition at +350ms, got ${JSON.stringify(check567.midTransition)}`
        );
        assert(
            check567.endTransition.isTransitioning === false &&
                check567.endTransition.transitionT === 1.0,
            `Expected completed 2s WGSL transition after +2250ms, got ${JSON.stringify(check567.endTransition)}`
        );
        assert(
            check567.vsResetXField === null &&
                check567.vsResetYField === null &&
                !check567.encResetHasX &&
                !check567.encResetHasY &&
                check567.internalXAfterReset === 'x_u16' &&
                check567.internalYAfterReset === 'y_u16',
            'setEncoding({ x: null, y: null }) failed to reset cleanly while hiding x_u16/y_u16'
        );
        results.check5_6_7_transition_viewport_rtc = check567;

        // =========================================================================
        // CHECK 8: lmsys_d7_proportional_25k.arrowtiles (2.83M rows, 36 cols):
        //   - Mode 1 Categorical (clusters_14, 64x15) + Mode 1 Numeric (num_of_tokens, 16x63)
        //   - Lazy active column decode (<= 4 cols, NOT 36!)
        //   - Immediate CPU tile.columns release (0 retained CPU column arrays after GPU upload)
        // =========================================================================
        await page.evaluate(async () => {
            const dev = window.__wgpuDevices[window.__wgpuDevices.length - 1];
            dev.pushErrorScope('validation');
            const v = window.__viewer;
            await v.loadDataset('/samples/lmsys_d7_proportional_25k.arrowtiles');
            v.plot.setEncoding({ color: 'clusters_14', aggregation: 'points' });
            v.plot.fitView();
        });
        await page.waitForFunction(
            () => {
                const tm = window.__viewer?.getTileManager?.();
                return tm && tm.activeTiles && tm.activeTiles.size >= 5 && tm.loadingTiles.size === 0;
            },
            null,
            { timeout: 20000 }
        );
        await page.waitForTimeout(700);
        const lmsysCatPixels = await analyzeScreenshotPng(page);

        const check8NumAndMem = await page.evaluate(async () => {
            const v = window.__viewer;
            const tm = v.getTileManager();
            const store = v.tileStore;

            const requestedColsCat = Array.from(tm.requestedColumns || []);
            let uploadedTiles = 0;
            let retainedCpuColArrays = 0;
            for (const tile of tm.activeTiles.values()) {
                if (tile.uploadedToGpu) {
                    uploadedTiles++;
                    retainedCpuColArrays += Object.keys(tile.columns || {}).length;
                }
            }
            const telemetrySnapshot = {
                coldBootMs: tm.fetchTelemetry.coldBootMs,
                steadySamples: tm.fetchTelemetry.steadyTotals.length,
                steadyP95Ms: tm.fetchTelemetry.p95,
            };

            v.plot.setEncoding({
                color: { field: 'num_of_tokens', scale: 'viridis', domain: [1, 600] },
                aggregation: 'points',
            });
            return {
                requestedColsCat,
                uploadedTiles,
                retainedCpuColArrays,
                telemetrySnapshot,
                catComputeMode: store.computeMode,
            };
        });
        await page.waitForFunction(
            () => {
                const tm = window.__viewer?.getTileManager?.();
                return tm && tm.activeTiles && tm.activeTiles.size >= 5 && tm.loadingTiles.size === 0;
            },
            null,
            { timeout: 20000 }
        );
        await page.waitForTimeout(700);
        const lmsysNumPixels = await analyzeScreenshotPng(page);

        const check8Final = await page.evaluate(async () => {
            const dev = window.__wgpuDevices[window.__wgpuDevices.length - 1];
            const valErr = await dev.popErrorScope();
            const tm = window.__viewer.getTileManager();
            let retainedAfterSwap = 0;
            for (const tile of tm.activeTiles.values()) {
                if (tile.uploadedToGpu) {
                    retainedAfterSwap += Object.keys(tile.columns || {}).length;
                }
            }
            return {
                numComputeMode: window.__viewer.tileStore.computeMode,
                requestedColsNum: Array.from(tm.requestedColumns || []),
                retainedAfterSwap,
                validationError: valErr ? valErr.message : null,
                uncapturedErrors: window.__wgpuUncapturedErrors.slice(),
            };
        });

        console.log(
            '[Check 8 LMSys Mode 1 & Memory]',
            JSON.stringify({ lmsysCatPixels, lmsysNumPixels, check8NumAndMem, check8Final })
        );
        assert(!check8Final.validationError, `Check 8 WebGPU error: ${check8Final.validationError}`);
        assert(check8Final.uncapturedErrors.length === 0, `Uncaptured WebGPU errors: ${check8Final.uncapturedErrors}`);
        assert(
            lmsysCatPixels.nonBgPixels > 5000 && lmsysCatPixels.colorfulPixels > 2000 && lmsysCatPixels.whiteRatio < 0.02,
            `Categorical Mode 1 failed visual check: ${JSON.stringify(lmsysCatPixels)}`
        );
        assert(
            lmsysNumPixels.nonBgPixels > 5000 && lmsysNumPixels.colorfulPixels > 1000 && lmsysNumPixels.whiteRatio < 0.02,
            `Numeric Mode 1 failed visual/log-density check: ${JSON.stringify(lmsysNumPixels)}`
        );
        assert(
            check8NumAndMem.requestedColsCat.length <= 4,
            `Expected lazy active columns (<=4), got ${check8NumAndMem.requestedColsCat.length}`
        );
        assert(
            check8NumAndMem.uploadedTiles >= 5 &&
                check8NumAndMem.retainedCpuColArrays === 0 &&
                check8Final.retainedAfterSwap === 0,
            'Expected 0 retained CPU column arrays on uploaded tiles'
        );

        results.check8_lmsys_log_density_and_zero_cpu_retention = {
            lmsysCatPixels,
            lmsysNumPixels,
            check8NumAndMem,
            check8Final,
        };

        assert(pageErrors.length === 0, `Page errors detected: ${JSON.stringify(pageErrors)}`);
        assert(consoleErrors.length === 0, `Console errors detected: ${JSON.stringify(consoleErrors)}`);

        console.log('\n=== ALL PHASE 5B LIVE WEBGPU CHECKS PASSED ===');
        console.log(JSON.stringify(results, null, 2));
    } finally {
        await browser.close();
        await server.close();
    }
}

main().catch((err) => {
    console.error('[verify_phase5b FAILED]', err);
    process.exit(1);
});
