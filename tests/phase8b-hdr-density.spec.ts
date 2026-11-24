import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import zlib from 'node:zlib';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const allowDgpu = process.env.DEEPGRAPH_ALLOW_DGPU === '1';
const expectedPowerPref: 'high-performance' | 'low-power' = allowDgpu
  ? 'high-performance'
  : 'low-power';

test.describe('Phase 8b — Optical RGB Restoration, Deep-Zoom Rescue & SPEC v0.1 Gates (Intel iGPU Sample-Only)', () => {
  test('enforces 0.0000% Mode 0 shimmer, non-fading deep zoom past zoom=64, multiscale API controls, and Mode 1 deep-zoom clarity on < 5 MiB samples', async ({
    page,
  }) => {
    test.setTimeout(60000);
    const pageErrors: string[] = [];
    const consoleErrors: string[] = [];

    page.on('pageerror', (e) => pageErrors.push(e.message || String(e)));
    page.on('console', (m) => {
      if (m.type() === 'error') consoleErrors.push(m.text());
    });

    await page.route('**/__phase8b_harness.html', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/html; charset=utf-8',
        body: '<!doctype html><html><body style="margin:0;background:#111"><div id="stage" style="width:960px;height:600px"></div></body></html>',
      });
    });
    await page.goto('/__phase8b_harness.html', { waitUntil: 'domcontentloaded' });

    // 0. Mandatory GPU Coexistence Pre-Check BEFORE any requestDevice()
    const igpu = await page.evaluate(async (pref) => {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: pref });
      if (!adapter) throw new Error(`ABORT: ${pref} WebGPU adapter unavailable`);
      const info: any = (adapter as any).info || {};
      const vendor = String(info.vendor || '').toLowerCase();
      return {
        vendor,
        architecture: String(info.architecture || ''),
        isFallbackAdapter: Boolean((adapter as any).isFallbackAdapter),
      };
    }, expectedPowerPref);
    if (!allowDgpu) {
      expect(igpu.vendor).not.toBe('nvidia');
    }
    expect(igpu.isFallbackAdapter).toBe(false);

    // Helper to wait until streaming tiles & 300ms spawn fade-in settle
    const waitSettled = async () => {
      await expect
        .poll(
          async () => {
            return await page.evaluate(() => {
              const d = (window as any).__p8b?.diag;
              if (!d) return false;
              const tm = d.getTileManager();
              const nowSec = d.tileStore.globalTimeSec;
              const visSlots = d.tileStore.slots.filter(
                (s: any) => s && s.visible && s.activeRows > 0
              );
              const allFadedIn = visSlots.every((s: any) => nowSec - s.spawnTime >= 0.35);
              return Boolean(
                tm &&
                  visSlots.length >= 1 &&
                  allFadedIn &&
                  tm.loadingTiles.size === 0 &&
                  tm.queue.size === 0 &&
                  tm.queue.pending === 0
              );
            });
          },
          { timeout: 15000, intervals: [50, 80, 100] }
        )
        .toBe(true);
      await page.evaluate(
        () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))
      );
    };

    const readCanvasStats = async () => {
      const b64 = (await page.locator('#stage canvas').first().screenshot({ type: 'png' })).toString(
        'base64'
      );
      return await page.evaluate(async (b64Str) => {
        const res = await fetch(`data:image/png;base64,${b64Str}`);
        const bmp = await createImageBitmap(await res.blob());
        const oc = new OffscreenCanvas(bmp.width, bmp.height);
        const ctx = oc.getContext('2d')!;
        ctx.drawImage(bmp, 0, 0);
        const data = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
        let lit = 0;
        let lumSum = 0;
        let chromaSum = 0;
        let darkPixels = 0;
        let pureWhite = 0;
        const totalPixels = data.length / 4;
        const hist = new Uint32Array(256);
        for (let i = 0; i < data.length; i += 4) {
          const r = data[i];
          const g = data[i + 1];
          const b = data[i + 2];
          const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
          if (lum < 28.0) darkPixels++;
          if (r > 20 || g > 20 || b > 20) {
            lit++;
            const lumInt = Math.round(lum);
            lumSum += lumInt;
            chromaSum += Math.max(r, g, b) - Math.min(r, g, b);
            hist[Math.min(255, Math.max(0, lumInt))]++;
            if (r >= 250 && g >= 250 && b >= 250) pureWhite++;
          }
        }
        let p99 = 0;
        if (lit > 0) {
          const target = Math.floor(lit * 0.99);
          let acc = 0;
          for (let k = 0; k < 256; k++) {
            acc += hist[k];
            if (acc >= target) {
              p99 = k;
              break;
            }
          }
        }
        return {
          litPixels: lit,
          meanLitBrightness: lit > 0 ? lumSum / lit : 0,
          meanLitChroma: lit > 0 ? chromaSum / lit : 0,
          darkSkyPct: totalPixels > 0 ? (darkPixels / totalPixels) * 100 : 0,
          p99Brightness: p99,
          pureWhitePctOfLit: lit > 0 ? (pureWhite / lit) * 100 : 0,
        };
      }, b64);
    };

    // 1. Load < 5 MiB `gaia_sample.arrowtiles` in Optical HDR Mode 0
    const initialState = await page.evaluate(async (pref) => {
      const { Deepgraph } = await import('/src/index.ts');
      const { INTERNAL_DIAGNOSTICS } = await import('/src/lib/diagnostics.ts');
      const { resolveDemoPreset } = await import('/src/demo/gaiaPreset.ts');
      const stage = document.getElementById('stage')!;
      const plot = await Deepgraph.create(stage, {
        url: '/samples/gaia_sample.arrowtiles',
        preset: resolveDemoPreset,
        powerPreference: pref,
      });
      plot.fitView();
      (window as any).__p8b = { plot, diag: (plot as any)[INTERNAL_DIAGNOSTICS]() };
      return plot.getViewState();
    }, expectedPowerPref);

    expect(initialState.pipeline).toBe('optical');
    expect(initialState.pointSize).toBe(1.0);
    expect(initialState.zoomBalance).toBeCloseTo(0.22, 5);
    expect(initialState.pointOpacity).toBe('auto');
    expect(initialState.exposure).toBe('auto');

    await waitSettled();

    // 2. Stationary Mode 0 Shimmer Proof: 0.0000% changed pixels across consecutive frames
    const pngFrameA = (
      await page.locator('#stage canvas').first().screenshot({ type: 'png' })
    ).toString('base64');
    await page.evaluate(
      () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))
    );
    const pngFrameB = (
      await page.locator('#stage canvas').first().screenshot({ type: 'png' })
    ).toString('base64');

    const shimmer = await page.evaluate(
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
        let lit = 0;
        let changed = 0;
        let maxDelta = 0;
        for (let i = 0; i < dA.length; i += 4) {
          const isLit =
            dA[i] > 20 ||
            dA[i + 1] > 20 ||
            dA[i + 2] > 20 ||
            dB[i] > 20 ||
            dB[i + 1] > 20 ||
            dB[i + 2] > 20;
          if (!isLit) continue;
          lit++;
          const m = Math.max(
            Math.abs(dA[i] - dB[i]),
            Math.abs(dA[i + 1] - dB[i + 1]),
            Math.abs(dA[i + 2] - dB[i + 2])
          );
          if (m > 0) {
            changed++;
            if (m > maxDelta) maxDelta = m;
          }
        }
        return { litPixels: lit, changedPixels: changed, maxDelta };
      },
      { a: pngFrameA, b: pngFrameB }
    );
    expect(shimmer.litPixels).toBeGreaterThan(20000);
    expect(shimmer.changedPixels).toBe(0);
    expect(shimmer.maxDelta).toBe(0);

    const zOverviewStats = await readCanvasStats();
    expect(zOverviewStats.pureWhitePctOfLit).toBeLessThan(1.0);
    expect(zOverviewStats.darkSkyPct).toBeGreaterThan(20.0);
    expect(zOverviewStats.meanLitChroma).toBeGreaterThan(6.0);

    // 3. Deep-Zoom Non-Fading Proof past zoom=64 (zoom = 36 -> 114 -> 358 around dense Galactic plane center)
    await page.evaluate(() => (window as any).__p8b.plot.setViewport({ x: 0.0, y: 0.0, zoom: 36 }));
    await waitSettled();
    const zoom36Stats = await readCanvasStats();

    await page.evaluate(() => (window as any).__p8b.plot.setViewport({ x: 0.0, y: 0.0, zoom: 114 }));
    await waitSettled();
    const zoom114Stats = await readCanvasStats();

    await page.evaluate(() => (window as any).__p8b.plot.setViewport({ x: 0.0, y: 0.0, zoom: 358 }));
    await waitSettled();
    const zoom358Stats = await readCanvasStats();

    expect(zoom36Stats.litPixels).toBeGreaterThan(100);
    expect(zoom114Stats.litPixels).toBeGreaterThan(30);
    expect(zoom358Stats.litPixels).toBeGreaterThan(10);
    // Past zoom=64, bright resolved stars stay crisp & luminous without fading (p99 > 105 vs ~49 pre-8b; mean > 55 vs ~24 pre-8b)
    expect(zoom114Stats.meanLitBrightness).toBeGreaterThan(55);
    expect(zoom114Stats.p99Brightness).toBeGreaterThan(105);
    expect(zoom358Stats.meanLitBrightness).toBeGreaterThan(55);
    expect(zoom358Stats.p99Brightness).toBeGreaterThan(105);
    expect(zoom358Stats.p99Brightness).toBeGreaterThanOrEqual(zoom114Stats.p99Brightness * 0.9);

    // 4. Public API Multiscale & HDR Controls (`pointSize`, `zoomBalance`, `pointOpacity`, `exposure`)
    const boostedState = await page.evaluate(async () => {
      const plot = (window as any).__p8b.plot;
      plot.setViewOptions({
        pointSize: 2.2,
        zoomBalance: 0.3,
        exposure: 2.6,
        pointOpacity: 0.25,
      });
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      return plot.getViewState();
    });
    expect(boostedState.pointSize).toBeCloseTo(2.2, 5);
    expect(boostedState.zoomBalance).toBeCloseTo(0.3, 5);
    expect(boostedState.exposure).toBeCloseTo(2.6, 5);
    expect(boostedState.pointOpacity).toBeCloseTo(0.25, 5);

    const boostedStats = await readCanvasStats();
    expect(boostedStats.litPixels).toBeGreaterThan(zoom358Stats.litPixels);
    expect(boostedStats.meanLitBrightness).toBeGreaterThan(zoom358Stats.meanLitBrightness);

    // Reset back to auto
    const resetState = await page.evaluate(() => {
      const plot = (window as any).__p8b.plot;
      plot.setEncoding({
        pointSize: 1.0,
        zoomBalance: 0.22,
        exposure: 'auto',
        pointOpacity: 'auto',
      });
      return plot.getViewState();
    });
    expect(resetState.exposure).toBe('auto');
    expect(resetState.pointOpacity).toBe('auto');

    // 5. Mode 1 Deep-Zoom `refCount` + Expanded Point-Size Picking Parity on < 5 MiB `lmsys_sample.arrowtiles`
    await page.evaluate(async () => {
      const plot = (window as any).__p8b.plot;
      await plot.load('/samples/lmsys_sample.arrowtiles', {
        encoding: { color: 'model', aggregation: 'points', pointSize: 2.4 },
      });
      plot.setViewport({ x: 88.76073587009975, y: 69.12738843695004, zoom: 128 });
    });
    await waitSettled();

    const lmsysDeepStats = await readCanvasStats();
    expect(lmsysDeepStats.p99Brightness).toBeGreaterThanOrEqual(165);

    // Pick slightly off-center (+5 CSS px, outside default 4.5 px half-window, inside expanded pointSize=2.4 window!)
    const hitOffCenter = await page.evaluate(async () => {
      const plot = (window as any).__p8b.plot;
      return await plot.pickAt(480 + 5, 300);
    });
    expect(hitOffCenter).not.toBeNull();
    expect(hitOffCenter.rowId).toBe(40064);

    // 5b. SPEC.md §1 0.x Minor-Version ("0.2" rejected) & `required_features` Gate
    const baseMeta = await page.evaluate(async () => {
      const { diag } = (window as any).__p8b;
      const { metadata } = await diag.getTileManager().getHeaderAndMetadata();
      return metadata as Record<string, unknown>;
    });

    const preferredSamplePath = path.resolve(
      rootDir,
      '..',
      'arrowtiles',
      'samples',
      'lmsys_sample.arrowtiles'
    );
    const origSampleBuf = fs.readFileSync(
      fs.existsSync(preferredSamplePath)
        ? preferredSamplePath
        : path.resolve(rootDir, '..', 'duckdb-arrowtiles', 'samples', 'lmsys_sample.arrowtiles')
    );
    const internalComp = origSampleBuf[97]; // 1 = None, 2 = Gzip, 4 = Zstd

    function buildPatchedArchiveBuffer(patch: Record<string, unknown>): Buffer {
      const rawJson = Buffer.from(JSON.stringify({ ...baseMeta, ...patch }), 'utf8');
      let encodedMeta: Buffer;
      if (internalComp === 4) {
        encodedMeta = zlib.zstdCompressSync(rawJson);
      } else if (internalComp === 2) {
        encodedMeta = zlib.gzipSync(rawJson);
      } else {
        encodedMeta = rawJson;
      }
      const out = Buffer.alloc(origSampleBuf.length + encodedMeta.length);
      origSampleBuf.copy(out, 0);
      encodedMeta.copy(out, origSampleBuf.length);
      out.writeBigUInt64LE(BigInt(origSampleBuf.length), 24);
      out.writeBigUInt64LE(BigInt(encodedMeta.length), 32);
      return out;
    }

    const okEmptyFeatB64 = buildPatchedArchiveBuffer({ required_features: [] }).toString('base64');
    const v02ArchiveB64 = buildPatchedArchiveBuffer({ version: '0.2' }).toString('base64');
    const unkFeatB64 = buildPatchedArchiveBuffer({
      required_features: ['column-groups-v1'],
    }).toString('base64');

    const specGateResults = await page.evaluate(
      async ({ okB64, v02B64, unkB64 }) => {
        const { plot } = (window as any).__p8b;
        function b64ToFile(b64: string, name: string): File {
          const bin = atob(b64);
          const u8 = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
          return new File([u8], name);
        }

        const okSummary = await plot.load(b64ToFile(okB64, 'ok_empty_feat.arrowtiles'));

        let v02Error = '';
        try {
          await plot.load(b64ToFile(v02B64, 'future_v02.arrowtiles'));
        } catch (e: any) {
          v02Error = String(e?.message ?? e);
        }

        let unkFeatError = '';
        try {
          await plot.load(b64ToFile(unkB64, 'unk_feat.arrowtiles'));
        } catch (e: any) {
          unkFeatError = String(e?.message ?? e);
        }

        return { okVersion: okSummary.version, v02Error, unkFeatError };
      },
      { okB64: okEmptyFeatB64, v02B64: v02ArchiveB64, unkB64: unkFeatB64 }
    );
    expect(specGateResults.okVersion).toBe('0.1');
    expect(specGateResults.v02Error).toMatch(/minor version.*0\.2/i);
    expect(specGateResults.unkFeatError).toContain('column-groups-v1');

    // Clear expected error notifications emitted during negative load() checks
    pageErrors.length = 0;

    // 6. Clean Teardown
    const canvasesLeft = await page.evaluate(() => {
      (window as any).__p8b.plot.destroy();
      return document.querySelectorAll('#stage canvas').length;
    });
    expect(canvasesLeft).toBe(0);
    expect(pageErrors).toEqual([]);
    expect(consoleErrors).toEqual([]);
  });
});
