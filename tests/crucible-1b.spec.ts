import { test, expect, type Page } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const GAIA_FILE = process.env.DEEPGRAPH_GAIA_FILE || '/gaia_full.arrowtiles';
const ALLOW_DGPU = process.env.DEEPGRAPH_ALLOW_DGPU === '1';
const POWER_PREF: 'high-performance' | 'low-power' = ALLOW_DGPU
  ? 'high-performance'
  : 'low-power';
const GPU_TAG = ALLOW_DGPU ? 'dgpu_rtx5090' : 'igpu_xelpg';
const SCRATCH_DIR =
  process.env.CRUCIBLE_SCRATCH_DIR || path.join(os.tmpdir(), 'deepgraph-crucible');
const CRUCIBLE_BUILD_JSON = path.join(SCRATCH_DIR, 'crucible_v011_results.json');

const DEFAULT_BILLION_ROWID_STARS = [
  {
    row_id: 1202742788,
    tile_z: 0,
    tile_x: 0,
    tile_y: 0,
    raw_row_idx: 4,
    x_u16: 33481,
    y_u16: 36677,
    data_x: 0.061587937850199115,
    data_y: -0.16873023337446003,
    tile_phot_g_mean_mag: 2.116495132446289,
    tile_bp_rp: 1.8272674083709717,
    tile_parallax: 23.885852813720703,
    parquet_file: 'batch_032.parquet',
    parquet_file_row_number: 3179113,
    parquet_ra: 274.4060904518451,
    parquet_dec: -36.76242931758906,
    parquet_hammer_x: 0.06160321223935359,
    parquet_hammer_y: -0.1687139412525914,
    parquet_magnitude: 2.116495132446289,
    parquet_bv: 1.8272674083709717,
    parquet_parallax: 23.885852068095137,
    parquet_teff_gspphot: null,
  },
  {
    row_id: 1228887815,
    tile_z: 0,
    tile_x: 0,
    tile_y: 0,
    raw_row_idx: 11,
    x_u16: 32164,
    y_u16: 35659,
    data_x: -0.052092950935662685,
    data_y: -0.12479433937901274,
    tile_phot_g_mean_mag: 2.2969820499420166,
    tile_bp_rp: 1.426199197769165,
    tile_parallax: 7.846769332885742,
    parquet_file: 'batch_033.parquet',
    parquet_file_row_number: 8922713,
    parquet_ra: 275.24868220695066,
    parquet_dec: -29.82822386700995,
    parquet_hammer_x: -0.05206686010702422,
    parquet_hammer_y: -0.12480364634141525,
    parquet_magnitude: 2.2969820499420166,
    parquet_bv: 1.426199197769165,
    parquet_parallax: 7.846769491115783,
    parquet_teff_gspphot: null,
  },
  {
    row_id: 1060650830,
    tile_z: 0,
    tile_x: 0,
    tile_y: 0,
    raw_row_idx: 16,
    x_u16: 37669,
    y_u16: 37240,
    data_x: 0.4230879851054685,
    data_y: -0.19302876806938785,
    tile_phot_g_mean_mag: 2.391281843185425,
    tile_bp_rp: 1.4427056312561035,
    tile_parallax: 4.560053825378418,
    parquet_file: 'batch_027.parquet',
    parquet_file_row_number: 27680549,
    parquet_ra: 261.3248860596135,
    parquet_dec: -55.52999691705136,
    parquet_hammer_x: 0.4231260597969053,
    parquet_hammer_y: -0.19301885995133003,
    parquet_magnitude: 2.391281843185425,
    parquet_bv: 1.4427056312561035,
    parquet_parallax: 4.560053792219337,
    parquet_teff_gspphot: null,
  },
];

async function measureTwoFrameShimmer(page: Page) {
  await page.waitForTimeout(450);
  const canvas = page.locator('#stage canvas').first();
  const pngA = (await canvas.screenshot({ type: 'png' })).toString('base64');
  await page.evaluate(
    () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))
  );
  await page.waitForTimeout(60);
  const pngB = (await canvas.screenshot({ type: 'png' })).toString('base64');

  return await page.evaluate(
    async ({ a, b }) => {
      async function decode(b64: string) {
        const res = await fetch(`data:image/png;base64,${b64}`);
        const bmp = await createImageBitmap(await res.blob());
        const oc = new OffscreenCanvas(bmp.width, bmp.height);
        const ctx = oc.getContext('2d')!;
        ctx.drawImage(bmp, 0, 0);
        return ctx.getImageData(0, 0, bmp.width, bmp.height).data;
      }
      const dA = await decode(a);
      const dB = await decode(b);
      let litPixels = 0;
      let changedPixels = 0;
      let maxDelta = 0;
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
        const m = Math.max(
          Math.abs(dA[i] - dB[i]),
          Math.abs(dA[i + 1] - dB[i + 1]),
          Math.abs(dA[i + 2] - dB[i + 2])
        );
        if (m > 0) {
          changedPixels++;
          if (m > maxDelta) maxDelta = m;
        }
      }
      return { litPixels, changedPixels, maxDelta };
    },
    { a: pngA, b: pngB }
  );
}

test.describe('Step 2 — 1.35B Real Gaia DR3 v0.1.1 Hardware Crucible (TTFF, WAN, Multi-Scale FPS, Shimmer & >1B Pick Parity)', () => {
  test('benchmarks Cold Loopback + 50ms/50Mbps WAN TTFF, Z=2..10 steady-state FPS, 0.0000% Mode 0/1/2 shimmer, row_id > 1B Parquet parity & zero-leak teardown', async ({
    page,
    context,
  }) => {
    test.setTimeout(240000);
    await page.bringToFront();

    const pageErrors: string[] = [];
    const consoleErrors: string[] = [];
    page.on('pageerror', (e) => pageErrors.push(e.message || String(e)));
    page.on('console', (m) => {
      if (m.type() === 'error') consoleErrors.push(m.text());
    });

    await page.route('**/__crucible_1b.html', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/html; charset=utf-8',
        body: '<!doctype html><html><body style="margin:0;background:#111"><div id="stage" style="width:1280px;height:720px"></div></body></html>',
      });
    });
    await page.goto('/__crucible_1b.html', { waitUntil: 'domcontentloaded' });

    // 0. Verify hardware GPU adapter & coexistence rule
    const adapterInfo = await page.evaluate(async (pref) => {
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

      const adapter = await navigator.gpu.requestAdapter({ powerPreference: pref });
      if (!adapter) throw new Error(`No WebGPU adapter for ${pref}`);
      const info: any = (adapter as any).info || {};
      return {
        vendor: String(info.vendor || ''),
        architecture: String(info.architecture || ''),
        device: String(info.device || ''),
        description: String(info.description || ''),
        isFallbackAdapter: Boolean((adapter as any).isFallbackAdapter),
      };
    }, POWER_PREF);

    console.log(`[Crucible Adapter (${GPU_TAG})]`, JSON.stringify(adapterInfo));
    expect(adapterInfo.isFallbackAdapter).toBe(false);
    if (!ALLOW_DGPU) {
      expect(adapterInfo.vendor.toLowerCase()).not.toBe('nvidia');
    } else {
      expect(adapterInfo.vendor.toLowerCase()).toBe('nvidia');
    }

    // Track Range requests & bytes transferred for `.arrowtiles`
    let rangeReqs = 0;
    let rangeBytes = 0;
    page.on('response', (res) => {
      if (res.url().includes('.arrowtiles') && res.status() === 206) {
        rangeReqs++;
        const cl = Number(res.headers()['content-length'] || 0);
        if (Number.isFinite(cl)) rangeBytes += cl;
      }
    });

    // =========================================================================
    // 1. COLD LOOPBACK TTFF (Production `dist/standalone.js`: ms, Range requests, MB)
    // =========================================================================
    rangeReqs = 0;
    rangeBytes = 0;
    const coldLoopback = await page.evaluate(
      async ({ gaiaFile, pref }) => {
        const { Deepgraph } = await import('/dist/standalone.js');
        const { resolveDemoPreset } = await import('/src/demo/gaiaPreset.ts');
        const stage = document.getElementById('stage')!;

        const t0 = performance.now();
        const plot = await Deepgraph.create(stage, {
          url: `${gaiaFile}?cold=loopback_${Date.now()}`,
          preset: resolveDemoPreset,
          powerPreference: pref,
        });

        // Wait until first tile GPU storage buffer is uploaded and presented on rAF
        while (performance.now() - t0 < 30000) {
          if ((window as any).__gpuStats.tileLiveBytes > 0) {
            await new Promise((r) => requestAnimationFrame(() => r(null)));
            break;
          }
          await new Promise((r) => setTimeout(r, 4));
        }
        const ttffMs = performance.now() - t0;
        const meta = plot.getMetadata()!;

        // Allow initial Z=0..2 overview tiles to settle and capture public telemetry snapshot
        let latestSnap: any = null;
        const off = plot.on('telemetry', (snap: any) => {
          latestSnap = snap;
        });
        const settleDeadline = performance.now() + 5000;
        while (performance.now() < settleDeadline) {
          if (latestSnap && latestSnap.activeTiles >= 21 && latestSnap.activePoints > 1_000_000) {
            break;
          }
          await new Promise((r) => setTimeout(r, 40));
        }
        off();
        const settledPoints = latestSnap?.activePoints ?? 0;
        const settledTiles = latestSnap?.activeTiles ?? 0;
        plot.destroy();

        return {
          ttffMs: Number(ttffMs.toFixed(2)),
          version: meta.version,
          totalRows: meta.totalRows,
          archiveMaxZoom: meta.archiveMaxZoom,
          settledPoints,
          settledTiles,
        };
      },
      { gaiaFile: GAIA_FILE, pref: POWER_PREF }
    );
    const loopbackOverviewReqs = rangeReqs;
    const loopbackOverviewMB = Number((rangeBytes / (1024 * 1024)).toFixed(2));
    console.log(
      `[1. Cold Loopback TTFF (${GPU_TAG})]`,
      JSON.stringify({ ...coldLoopback, loopbackOverviewReqs, loopbackOverviewMB })
    );
    expect(coldLoopback.version).toBe('0.1');
    expect(coldLoopback.totalRows).toBeGreaterThan(1_300_000_000);
    expect(coldLoopback.settledPoints).toBeGreaterThan(50_000);

    // =========================================================================
    // 2. SIMULATED WAN (50 ms RTT / 50 Mbps) COLD TTFF (Production `dist/standalone.js`)
    //    Uses production Blob-inlined worker (`0` Vite dev-module waterfall) and cache-busts
    //    `.arrowtiles` (`?cold=wan_...`) so 100% of HTTP 206 Range requests (`header`,
    //    `metadata`, `leafDirectory`, `Z=0/0/0`) traverse the 50 ms RTT / 50 Mbps WAN link cold.
    // =========================================================================
    const cdp = await context.newCDPSession(page);
    await cdp.send('Network.enable');
    await cdp.send('Network.emulateNetworkConditions', {
      offline: false,
      latency: 50,
      downloadThroughput: Math.floor((50 * 1024 * 1024) / 8),
      uploadThroughput: Math.floor((50 * 1024 * 1024) / 8),
    });

    rangeReqs = 0;
    rangeBytes = 0;
    let wanCold: { ttffMs: number; firstFramePoints: number } = {
      ttffMs: 0,
      firstFramePoints: 0,
    };
    let wanFirstFrameReqs = 0;
    let wanFirstFrameMB = 0;
    try {
      wanCold = await page.evaluate(
        async ({ gaiaFile, pref }) => {
          const { Deepgraph } = await import('/dist/standalone.js');
          const { resolveDemoPreset } = await import('/src/demo/gaiaPreset.ts');
          const stage = document.getElementById('stage')!;

          let latestPoints = 0;
          const t0 = performance.now();
          const plot = await Deepgraph.create(stage, {
            url: `${gaiaFile}?cold=wan_${Date.now()}`,
            preset: resolveDemoPreset,
            powerPreference: pref,
            detailOffset: 0, // Cold overview root tile first
          });
          const off = plot.on('telemetry', (snap: any) => {
            if (snap.activePoints > 0) latestPoints = snap.activePoints;
          });
          while (performance.now() - t0 < 45000) {
            if ((window as any).__gpuStats.tileLiveBytes > 0) {
              await new Promise((r) => requestAnimationFrame(() => r(null)));
              break;
            }
            await new Promise((r) => setTimeout(r, 4));
          }
          const ttffMs = performance.now() - t0;

          // Wait briefly (post-TTFF) for telemetry snapshot to report exact activePoints
          const snapDeadline = performance.now() + 500;
          while (latestPoints === 0 && performance.now() < snapDeadline) {
            await new Promise((r) => setTimeout(r, 25));
          }
          off();
          plot.destroy();
          return {
            ttffMs: Number(ttffMs.toFixed(2)),
            firstFramePoints: latestPoints,
          };
        },
        { gaiaFile: GAIA_FILE, pref: POWER_PREF }
      );
      wanFirstFrameReqs = rangeReqs;
      wanFirstFrameMB = Number((rangeBytes / (1024 * 1024)).toFixed(2));
    } finally {
      await cdp.send('Network.emulateNetworkConditions', {
        offline: false,
        latency: 0,
        downloadThroughput: -1,
        uploadThroughput: -1,
      });
    }
    console.log(
      `[2. WAN 50ms/50Mbps Cold TTFF (${GPU_TAG})]`,
      JSON.stringify({ ...wanCold, wanFirstFrameReqs, wanFirstFrameMB })
    );
    expect(wanFirstFrameReqs).toBeGreaterThanOrEqual(2);
    expect(wanFirstFrameMB).toBeGreaterThan(1.0);
    expect(wanCold.firstFramePoints).toBeGreaterThan(10_000);
    expect(wanCold.ttffMs).toBeLessThan(4000);

    // =========================================================================
    // 3. MULTI-SCALE STEADY-STATE FPS (Full-Sky Z=2, Galactic Core Z=3, LMC Z=7/10, M31 Z=7)
    // =========================================================================
    await page.evaluate(
      async ({ gaiaFile, pref }) => {
        const { Deepgraph } = await import('/src/index.ts');
        const { INTERNAL_DIAGNOSTICS } = await import('/src/lib/diagnostics.ts');
        const { resolveDemoPreset } = await import('/src/demo/gaiaPreset.ts');
        const stage = document.getElementById('stage')!;
        const plot = await Deepgraph.create(stage, {
          url: gaiaFile,
          preset: resolveDemoPreset,
          powerPreference: pref,
          detailOffset: 2,
        });
        (window as any).__crucible = {
          plot,
          diag: (plot as any)[INTERNAL_DIAGNOSTICS](),
        };
      },
      { gaiaFile: GAIA_FILE, pref: POWER_PREF }
    );

    const viewTargets = [
      { name: 'full_sky_z2', x: 0.0, y: 0.0, zoom: 1.5 },
      { name: 'galactic_core_z3', x: 0.0, y: 0.0, zoom: 2.17 },
      { name: 'lmc_wide_z7', x: 1.1844, y: -0.5986, zoom: 36.0 },
      { name: 'lmc_core_z10', x: 1.1844, y: -0.5986, zoom: 288.0 },
      { name: 'andromeda_m31_z7', x: -1.8983, y: -0.4308, zoom: 48.0 },
    ];

    const steadyFpsResults: Record<string, any> = {};
    for (const vt of viewTargets) {
      steadyFpsResults[vt.name] = await page.evaluate(async (target) => {
        const { plot, diag } = (window as any).__crucible;
        plot.setViewport({ x: target.x, y: target.y, zoom: target.zoom });
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

        const deadline = performance.now() + 20000;
        while (performance.now() < deadline) {
          const tm = diag.getTileManager();
          const nowSec = diag.tileStore.globalTimeSec;
          const vis = diag.tileStore.slots.filter((s: any) => s && s.visible && s.activeRows > 0);
          const allFaded = vis.length > 0 && vis.every((s: any) => nowSec - s.spawnTime >= 0.35);
          if (
            tm &&
            allFaded &&
            tm.loadingTiles.size === 0 &&
            tm.queue.size === 0 &&
            tm.queue.pending === 0
          ) {
            break;
          }
          await new Promise((r) => setTimeout(r, 35));
        }

        // Measure 1.0s of pure stationary rAF frames + telemetry renderMs
        const renderSamples: number[] = [];
        let latestSnap: any = null;
        const off = plot.on('telemetry', (snap: any) => {
          latestSnap = snap;
          renderSamples.push(snap.renderMs);
        });
        let rafCount = 0;
        const tStart = performance.now();
        await new Promise<void>((resolve) => {
          const step = () => {
            rafCount++;
            if (performance.now() - tStart >= 1000) {
              resolve();
              return;
            }
            requestAnimationFrame(step);
          };
          requestAnimationFrame(step);
        });
        const elapsed = performance.now() - tStart;
        off();

        const avgRenderMs =
          renderSamples.length > 0
            ? renderSamples.reduce((a, b) => a + b, 0) / renderSamples.length
            : latestSnap?.renderMs ?? 0;

        return {
          fps: Number(((rafCount * 1000) / Math.max(1, elapsed)).toFixed(1)),
          renderMs: Number(avgRenderMs.toFixed(3)),
          currentZ: latestSnap?.currentZ ?? diag.getTileManager()?.currentMaxZ ?? 0,
          activeTiles: diag.tileStore.slots.filter((s: any) => s && s.visible).length,
          activePoints: diag.tileStore.getActivePointsCount(),
          tileVramMB: Number((diag.tileStore.getLiveTileGpuBytes() / (1024 * 1024)).toFixed(2)),
        };
      }, vt);
      console.log(`[3. Steady FPS ${vt.name} (${GPU_TAG})]`, JSON.stringify(steadyFpsResults[vt.name]));
      expect(steadyFpsResults[vt.name].activePoints).toBeGreaterThan(0);
      expect(steadyFpsResults[vt.name].fps).toBeGreaterThan(15);
    }

    // =========================================================================
    // 4. 0.0000% 2-FRAME STATIONARY SHIMMER ACROSS MODES 0, 1, AND 2
    // =========================================================================
    const waitSettledAndFaded = async () => {
      await page.evaluate(async () => {
        const { diag } = (window as any).__crucible;
        const deadline = performance.now() + 15000;
        while (performance.now() < deadline) {
          const tm = diag.getTileManager();
          const nowSec = diag.tileStore.globalTimeSec;
          const vis = diag.tileStore.slots.filter((s: any) => s && s.visible && s.activeRows > 0);
          const allFaded = vis.length > 0 && vis.every((s: any) => nowSec - s.spawnTime >= 0.35);
          if (
            tm &&
            allFaded &&
            tm.loadingTiles.size === 0 &&
            tm.queue.size === 0 &&
            tm.queue.pending === 0
          ) {
            break;
          }
          await new Promise((r) => setTimeout(r, 35));
        }
      });
    };

    await page.evaluate(async () => {
      const { plot } = (window as any).__crucible;
      plot.setViewOptions({ pipeline: 'optical' });
      plot.setViewport({ x: 0.0, y: 0.0, zoom: 2.17 });
    });
    await waitSettledAndFaded();
    const shimmerMode0 = await measureTwoFrameShimmer(page);

    await page.evaluate(async () => {
      const { plot } = (window as any).__crucible;
      plot.setViewOptions({ pipeline: 'cartesian', experimental: { metricType: 0 } });
      plot.setEncoding({
        color: { field: 'bp_rp', scale: 'viridis', domain: [-0.5, 3.5] },
        aggregation: 'points',
      });
    });
    await waitSettledAndFaded();
    const shimmerMode1 = await measureTwoFrameShimmer(page);

    await page.evaluate(async () => {
      const { plot } = (window as any).__crucible;
      plot.setEncoding({
        color: { field: 'bp_rp', scale: 'coolwarm', domain: [-0.2, 2.8] },
        aggregation: 'mean_field',
      });
    });
    await waitSettledAndFaded();
    const shimmerMode2 = await measureTwoFrameShimmer(page);

    console.log(
      `[4. Stationary Shimmer Modes 0/1/2 (${GPU_TAG})]`,
      JSON.stringify({ shimmerMode0, shimmerMode1, shimmerMode2 })
    );
    expect(shimmerMode0.litPixels).toBeGreaterThan(50_000);
    expect(shimmerMode0.changedPixels).toBe(0);
    expect(shimmerMode1.litPixels).toBeGreaterThan(50_000);
    expect(shimmerMode1.changedPixels).toBe(0);
    expect(shimmerMode2.litPixels).toBeGreaterThan(50_000);
    expect(shimmerMode2.changedPixels).toBe(0);

    // =========================================================================
    // 5. row_id > 1,000,000,000 GPU pickAt PARITY vs RAW s3_cache PARQUET
    // =========================================================================
    let refStars: any[] = DEFAULT_BILLION_ROWID_STARS;
    if (fs.existsSync(CRUCIBLE_BUILD_JSON)) {
      const parsed = JSON.parse(fs.readFileSync(CRUCIBLE_BUILD_JSON, 'utf8'));
      const fromJson = parsed?.rungs?.rung4_gaia_1_35b?.billion_rowid_reference_stars;
      if (Array.isArray(fromJson) && fromJson.length > 0) {
        refStars = fromJson;
      }
    }
    expect(refStars.length).toBeGreaterThan(0);

    const billionPickChecks: any[] = [];
    // Test up to 3 reference stars from batch_031+ (row_id > 1,000,000,000)
    for (const star of refStars.slice(0, 3)) {
      const hitRes = await page.evaluate(async (s) => {
        const { plot, diag } = (window as any).__crucible;
        plot.setViewOptions({ pipeline: 'optical' });
        plot.setViewport({ x: s.data_x, y: s.data_y, zoom: 256.0 });
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
        const deadline = performance.now() + 15000;
        while (performance.now() < deadline) {
          const tm = diag.getTileManager();
          if (tm && tm.loadingTiles.size === 0 && tm.queue.size === 0 && tm.queue.pending === 0) {
            break;
          }
          await new Promise((r) => setTimeout(r, 30));
        }
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
        // Canvas is 1280 x 720 -> center is (640, 360)
        const tPick0 = performance.now();
        const hit = await plot.pickAt(640, 360);
        const pickMs = performance.now() - tPick0;
        return { hit, pickMs: Number(pickMs.toFixed(2)) };
      }, star);

      console.log(
        `[5. >1B Pick Parity row_id=${star.row_id} (${GPU_TAG})]`,
        JSON.stringify(hitRes)
      );
      expect(hitRes.hit).not.toBeNull();
      expect(hitRes.hit.rowId).toBe(star.row_id);
      expect(hitRes.hit.rowId).toBeGreaterThan(1_000_000_000);
      expect(
        Math.abs(Number(hitRes.hit.attributes.phot_g_mean_mag) - star.parquet_magnitude)
      ).toBeLessThan(1e-4);
      if (star.parquet_bv !== null) {
        expect(Math.abs(Number(hitRes.hit.attributes.bp_rp) - star.parquet_bv)).toBeLessThan(1e-4);
      }
      expect(Math.abs(hitRes.hit.dataX - star.parquet_hammer_x)).toBeLessThan(5e-4);
      expect(Math.abs(hitRes.hit.dataY - star.parquet_hammer_y)).toBeLessThan(5e-4);

      billionPickChecks.push({
        expectedRowId: star.row_id,
        pickedRowId: hitRes.hit.rowId,
        pickMs: hitRes.pickMs,
        parquetFile: star.parquet_file,
        magDiff: Math.abs(
          Number(hitRes.hit.attributes.phot_g_mean_mag) - star.parquet_magnitude
        ),
        hammerXDiff: Math.abs(hitRes.hit.dataX - star.parquet_hammer_x),
        hammerYDiff: Math.abs(hitRes.hit.dataY - star.parquet_hammer_y),
      });
    }

    // =========================================================================
    // 6. DETERMINISTIC ZERO-LEAK GPU TEARDOWN
    // =========================================================================
    const teardown = await page.evaluate(async () => {
      const { getGpuStats } = await import('/src/demo/gpuStats.ts');
      const { plot } = (window as any).__crucible;
      plot.destroy();
      return {
        gpuStats: getGpuStats(),
        uncapturedErrors: (window as any).__wgpuUncaptured.slice(),
        canvasesLeft: document.querySelectorAll('#stage canvas').length,
      };
    });

    console.log(`[6. Crucible Teardown (${GPU_TAG})]`, JSON.stringify(teardown));
    expect(teardown.uncapturedErrors).toEqual([]);
    expect(teardown.canvasesLeft).toBe(0);
    expect(teardown.gpuStats.liveDevices).toBe(0);
    expect(teardown.gpuStats.liveBuffers).toBe(0);
    expect(teardown.gpuStats.liveBytes).toBe(0);
    expect(teardown.gpuStats.totalDestroyedBytes).toBe(teardown.gpuStats.totalCreatedBytes);
    expect(pageErrors).toEqual([]);
    expect(consoleErrors).toEqual([]);

    // Persist per-GPU benchmark receipt to scratch directory
    fs.mkdirSync(SCRATCH_DIR, { recursive: true });
    const gpuReportPath = path.join(SCRATCH_DIR, `crucible_gpu_${GPU_TAG}.json`);
    fs.writeFileSync(
      gpuReportPath,
      JSON.stringify(
        {
          timestampUtc: new Date().toISOString(),
          gpuTag: GPU_TAG,
          gaiaFile: GAIA_FILE,
          adapter: adapterInfo,
          coldLoopback: { ...coldLoopback, loopbackOverviewReqs, loopbackOverviewMB },
          wan50ms50Mbps: { ...wanCold, wanFirstFrameReqs, wanFirstFrameMB },
          steadyFps: steadyFpsResults,
          shimmer: { shimmerMode0, shimmerMode1, shimmerMode2 },
          billionRowIdParity: billionPickChecks,
          teardown: teardown.gpuStats,
        },
        null,
        2
      ),
      'utf8'
    );
  });
});
