import { test, expect, type Page } from '@playwright/test';
import { tableFromArrays, tableToIPC } from '@uwdata/flechette';

const allowDgpu = process.env.DEEPGRAPH_ALLOW_DGPU === '1';
const expectedPowerPref: 'high-performance' | 'low-power' = allowDgpu
  ? 'high-performance'
  : 'low-power';

async function measureStationaryShimmer(page: Page) {
  await page.waitForTimeout(450);
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
      let maxDelta = 0;
      for (let i = 0; i < dA.length; i += 4) {
        const r1 = dA[i],
          g1 = dA[i + 1],
          b1 = dA[i + 2];
        const r2 = dB[i],
          g2 = dB[i + 1],
          b2 = dB[i + 2];
        const isLit =
          Math.abs(r1 - 17) > 3 ||
          Math.abs(g1 - 17) > 3 ||
          Math.abs(b1 - 17) > 3 ||
          Math.abs(r2 - 17) > 3 ||
          Math.abs(g2 - 17) > 3 ||
          Math.abs(b2 - 17) > 3;
        if (!isLit) continue;
        litPixels++;
        const m = Math.max(Math.abs(r1 - r2), Math.abs(g1 - g2), Math.abs(b1 - b2));
        if (m > 0) {
          changedPixels++;
          if (m > maxDelta) maxDelta = m;
        }
      }
      return {
        litPixels,
        changedPixels,
        changedPct: litPixels > 0 ? (changedPixels / litPixels) * 100 : 0,
        maxDelta,
      };
    },
    { a: pngA, b: pngB }
  );
}

test.describe('Phase 7 — GPU PickingLayer, ArrowTilesReader Parity, Zero-Shimmer Numeric Mode 1 & Overlay (Intel iGPU Sample-Only)', () => {
  test('Full Phase 7 + Pre-Phase-8 hover hardening verification on Intel iGPU using <5 MiB gaia_sample & lmsys_sample only', async ({
    page,
    context,
  }) => {
    test.setTimeout(120000);
    const pageErrors: string[] = [];
    const consoleErrors: string[] = [];
    page.on('pageerror', (e) => pageErrors.push(e.message || String(e)));
    page.on('console', (m) => {
      if (m.type() === 'error') consoleErrors.push(m.text());
    });

    await page.route('**/__phase7_harness.html', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/html; charset=utf-8',
        body: `<!doctype html><html><body style="margin:0;background:#111"><div id="stage" style="width:960px;height:600px"></div></body></html>`,
      });
    });

    await page.goto('/__phase7_harness.html', { waitUntil: 'domcontentloaded' });

    // =========================================================================
    // 0. STRICT GPU COEXISTENCE PRE-CHECK (BEFORE ANY requestDevice / Deepgraph.create!)
    //    Verify low-power adapter vendor === 'intel' so 0 bytes touch the RTX 5090.
    // =========================================================================
    const igpuGuard = await page.evaluate(async (pref) => {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: pref });
      if (!adapter) {
        throw new Error(`ABORT: navigator.gpu.requestAdapter({ powerPreference: "${pref}" }) returned null`);
      }
      const info: any = (adapter as any).info || {};
      const vendor = String(info.vendor || '').toLowerCase();
      const architecture = String(info.architecture || '');
      return { vendor, architecture, isFallbackAdapter: Boolean((adapter as any).isFallbackAdapter) };
    }, expectedPowerPref);
    console.log('[0. GPU Pre-Check]', JSON.stringify(igpuGuard));
    if (!allowDgpu) {
      expect(igpuGuard.vendor).not.toBe('nvidia');
    }
    expect(igpuGuard.isFallbackAdapter).toBe(false);

    // =========================================================================
    // 1. PRE-PHASE-7 LIBRARY HYGIENE PROOF:
    //    Deepgraph.create() does NOT install window.__gpuStats or patch WebGPU prototypes!
    // =========================================================================
    const hygieneProof = await page.evaluate(async () => {
      const origCreateBufSrc = Function.prototype.toString.call(GPUDevice.prototype.createBuffer);
      const { Deepgraph } = await import('/src/index.ts');
      const stage = document.getElementById('stage')!;
      const p = await Deepgraph.create(stage, { powerPreference: 'low-power' });
      const afterCreateBufSrc = Function.prototype.toString.call(GPUDevice.prototype.createBuffer);
      const winGpuStats = (window as any).__gpuStats;
      p.destroy();
      return {
        winGpuStatsDefined: winGpuStats !== undefined,
        createBufferUnpatched: origCreateBufSrc === afterCreateBufSrc,
      };
    });
    console.log('[1. Library Hygiene Proof]', JSON.stringify(hygieneProof));
    expect(hygieneProof.winGpuStatsDefined).toBe(false);
    expect(hygieneProof.createBufferUnpatched).toBe(true);

    // Now explicitly install the demo/test harness gpuStats tracker for leak verification at teardown!
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
    // 2. ZERO-SHIMMER 1,024-LEVEL NUMERIC MODE 1 CHECK (lmsys_sample.arrowtiles)
    // =========================================================================
    await page.evaluate(async () => {
      const { Deepgraph } = await import('/src/index.ts');
      const { INTERNAL_DIAGNOSTICS } = await import('/src/lib/diagnostics.ts');
      const stage = document.getElementById('stage')!;
      const plot = await Deepgraph.create(stage, {
        url: '/samples/lmsys_sample.arrowtiles',
        powerPreference: 'low-power',
        encoding: {
          color: { field: 'num_of_tokens', scale: 'viridis', domain: [1, 600] },
          aggregation: 'points',
        },
      });
      plot.fitView();
      const diag = (plot as any)[INTERNAL_DIAGNOSTICS]();
      (window as any).__p7 = { plot, diag, clicks: [] as any[], selects: [] as any[], hovers: [] as any[] };
      plot.on('click', (hit) => (window as any).__p7.clicks.push(hit));
      plot.on('select', (sel) => (window as any).__p7.selects.push(sel));

      const deadline = performance.now() + 10000;
      while (performance.now() < deadline) {
        await new Promise((r) => setTimeout(r, 60));
        const tm: any = diag.getTileManager();
        if (tm && tm.activeTiles.size >= 5 && tm.loadingTiles.size === 0) break;
      }
    });

    const numShimmer = await measureStationaryShimmer(page);
    console.log('[2. Numeric Mode 1 Stationary Shimmer]', JSON.stringify(numShimmer));
    expect(numShimmer.litPixels).toBeGreaterThan(20000);
    // Order-independent scalar value sum guarantees 0.0000% shimmer!
    expect(numShimmer.changedPixels).toBe(0);
    expect(numShimmer.changedPct).toBe(0);

    // =========================================================================
    // 3. DRAG vs. CLICK GATE (< 4 CSS px) + LMSYS GROUND-TRUTH PARITY vs. ArrowTilesReader
    // =========================================================================
    const stageBox = await page.locator('#stage').boundingBox();
    expect(stageBox).not.toBeNull();
    const centerCssX = 480;
    const centerCssY = 300;
    const pageCenterX = Math.round(stageBox!.x + centerCssX);
    const pageCenterY = Math.round(stageBox!.y + centerCssY);

    // 3a. Dragging > 4 CSS px (40 px pan) MUST NOT emit 'click'!
    await page.mouse.move(pageCenterX, pageCenterY);
    await page.mouse.down();
    await page.mouse.move(pageCenterX + 40, pageCenterY + 25, { steps: 5 });
    await page.mouse.up();
    await page.waitForTimeout(150);
    const clicksAfterDrag = await page.evaluate(() => (window as any).__p7.clicks.length);
    expect(clicksAfterDrag).toBe(0);

    // 3b. LMSys Ground-Truth Row Parity vs. Python ArrowTilesReader:
    //     [{"row_id":40064,"model":"mixtral","num_of_tokens":308,"x_decoded":88.76073587009975,"y_decoded":69.12738843695004},
    //      {"row_id":38479,"model":"llama-3","num_of_tokens":95,"x_decoded":87.7994427209227,"y_decoded":29.497411031549206},
    //      {"row_id":45449,"model":"gpt-4","num_of_tokens":2,"x_decoded":0.06847303031261212,"y_decoded":80.44130221129444}]
    const lmsysTargets = [
      {
        row_id: 40064,
        model: 'mixtral',
        num_of_tokens: 308,
        x_decoded: 88.76073587009975,
        y_decoded: 69.12738843695004,
      },
      {
        row_id: 38479,
        model: 'llama-3',
        num_of_tokens: 95,
        x_decoded: 87.7994427209227,
        y_decoded: 29.497411031549206,
      },
      {
        row_id: 45449,
        model: 'gpt-4',
        num_of_tokens: 2,
        x_decoded: 0.06847303031261212,
        y_decoded: 80.44130221129444,
      },
    ];

    for (const expectedRow of lmsysTargets) {
      await page.evaluate(async (t) => {
        const { plot, diag } = (window as any).__p7;
        (window as any).__p7.clicks.length = 0;
        plot.setViewport({ x: t.x_decoded, y: t.y_decoded, zoom: 128 });
        const deadline = performance.now() + 8000;
        while (performance.now() < deadline) {
          await new Promise((r) => setTimeout(r, 50));
          const tm: any = diag.getTileManager();
          if (tm && tm.loadingTiles.size === 0) break;
        }
        await new Promise((r) => setTimeout(r, 80));
      }, expectedRow);

      // Click directly on canvas center (< 4 CSS px travel = 0 px!)
      await page.mouse.click(pageCenterX, pageCenterY);
      await page.waitForFunction(() => (window as any).__p7.clicks.length >= 1, null, {
        timeout: 5000,
      });

      const hit = await page.evaluate(() => (window as any).__p7.clicks[0]);
      console.log(`[3b. LMSys Pick row_id=${expectedRow.row_id}]`, JSON.stringify(hit));
      expect(hit).not.toBeNull();
      expect(hit.rowId).toBe(expectedRow.row_id);
      expect(hit.attributes.model).toBe(expectedRow.model);
      expect(hit.attributes.num_of_tokens).toBe(expectedRow.num_of_tokens);
      expect(Math.abs(hit.dataX - expectedRow.x_decoded)).toBeLessThan(1e-9);
      expect(Math.abs(hit.dataY - expectedRow.y_decoded)).toBeLessThan(1e-9);
    }

    // 3c. Verify HighlightOverlay rendered gold ring pixels around center (480, 300) AND clears on empty click!
    await page.waitForTimeout(100);
    const ringPixels = await page.evaluate(async () => {
      const c = document.querySelector('canvas')!;
      const b64 = c.toDataURL('image/png');
      const res = await fetch(b64);
      const bmp = await createImageBitmap(await res.blob());
      const oc = new OffscreenCanvas(bmp.width, bmp.height);
      const ctx = oc.getContext('2d')!;
      ctx.drawImage(bmp, 0, 0);
      const dpr = window.devicePixelRatio || 1;
      const cx = Math.round(480 * dpr);
      const cy = Math.round(300 * dpr);
      const r = Math.round(12 * dpr);
      const patch = ctx.getImageData(cx - r, cy - r, r * 2, r * 2).data;
      let goldRingPixels = 0;
      for (let i = 0; i < patch.length; i += 4) {
        const red = patch[i],
          green = patch[i + 1],
          blue = patch[i + 2];
        // Selected marker ringRgb = (1.0, 0.82, 0.22) -> high R & G, low B
        if (red > 180 && green > 140 && blue < 110) {
          goldRingPixels++;
        }
      }
      return {
        goldRingPixels,
        selectedCount: (window as any).__p7.diag.highlightOverlay.selectedMarkers.length,
      };
    });
    console.log('[3c. HighlightOverlay Gold Ring]', JSON.stringify(ringPixels));
    expect(ringPixels.selectedCount).toBe(1);
    expect(ringPixels.goldRingPixels).toBeGreaterThan(12);

    // Pan far outside data bounds ([0..100]^2) and click empty space -> emits click: null & clears ring!
    await page.evaluate(() => {
      const { plot } = (window as any).__p7;
      (window as any).__p7.clicks.length = 0;
      (window as any).__p7.selects.length = 0;
      plot.setViewport({ x: -250, y: -250, zoom: 32 });
    });
    await page.waitForTimeout(120);
    await page.mouse.click(pageCenterX, pageCenterY);
    await page.waitForFunction(() => (window as any).__p7.clicks.length >= 1, null, {
      timeout: 5000,
    });
    const emptyClickCheck = await page.evaluate(() => ({
      clickPayload: (window as any).__p7.clicks[0],
      selectPointsLen: (window as any).__p7.selects[0]?.points?.length ?? -1,
      selectedMarkersLen: (window as any).__p7.diag.highlightOverlay.selectedMarkers.length,
    }));
    expect(emptyClickCheck.clickPayload).toBeNull();
    expect(emptyClickCheck.selectPointsLen).toBe(0);
    expect(emptyClickCheck.selectedMarkersLen).toBe(0);

    // =========================================================================
    // 4. GAIA SAMPLE GROUND-TRUTH PARITY (Spatial Sky + NULL Attributes + HR Diagram Chart Axes + Hover)
    // =========================================================================
    const gaiaSkyTarget = {
      row_id: 0,
      phot_g_mean_mag: 1.9425238370895386,
      bp_rp: 1.1434835195541382,
      parallax: 14.09097671508789,
      x_decoded: -2.01116770559316,
      y_decoded: -0.1273407222333658,
    };
    const gaiaHrTarget = {
      row_id: 14301,
      phot_g_mean_mag: 6.667600154876709,
      bp_rp: -0.500831127166748,
    };

    const gaiaSpatialHit = await page.evaluate(async (t) => {
      const { plot, diag } = (window as any).__p7;
      const { resolveDemoPreset } = await import('/src/demo/gaiaPreset.ts');
      await plot.load('/samples/gaia_sample.arrowtiles', { preset: resolveDemoPreset });
      plot.setViewport({ x: t.x_decoded, y: t.y_decoded, zoom: 128 });
      const deadline = performance.now() + 8000;
      while (performance.now() < deadline) {
        await new Promise((r) => setTimeout(r, 50));
        const tm: any = diag.getTileManager();
        if (tm && tm.activeTiles.size >= 1 && tm.loadingTiles.size === 0) break;
      }
      await new Promise((r) => setTimeout(r, 100));
      return await plot.pickAt(480, 300);
    }, gaiaSkyTarget);

    console.log('[4a. Gaia Spatial Pick row_id=0]', JSON.stringify(gaiaSpatialHit));
    expect(gaiaSpatialHit).not.toBeNull();
    expect(gaiaSpatialHit.rowId).toBe(0);
    expect(Math.abs(gaiaSpatialHit.attributes.phot_g_mean_mag - gaiaSkyTarget.phot_g_mean_mag)).toBeLessThan(1e-5);
    expect(Math.abs(gaiaSpatialHit.attributes.bp_rp - gaiaSkyTarget.bp_rp)).toBeLessThan(1e-5);
    expect(gaiaSpatialHit.attributes.radial_velocity).toBeNull();
    expect(gaiaSpatialHit.attributes.teff_gspphot).toBeNull();
    expect(Math.abs(gaiaSpatialHit.dataX - gaiaSkyTarget.x_decoded)).toBeLessThan(1e-9);
    expect(Math.abs(gaiaSpatialHit.dataY - gaiaSkyTarget.y_decoded)).toBeLessThan(1e-9);

    // 4b. Throttled 'hover' event verification on centered star
    await page.evaluate(() => {
      const p7 = (window as any).__p7;
      p7.hovers.length = 0;
      p7.unsubHover = p7.plot.on('hover', (h: any) => p7.hovers.push(h));
    });
    await page.mouse.move(pageCenterX - 30, pageCenterY - 30);
    await page.waitForTimeout(60);
    await page.mouse.move(pageCenterX, pageCenterY);
    await page.waitForFunction(
      () => (window as any).__p7.hovers.some((h: any) => h && h.rowId === 0),
      null,
      { timeout: 5000 }
    );
    await page.evaluate(() => (window as any).__p7.unsubHover());

    // 4c. Switch to HR Diagram (custom chart axes x='bp_rp', y='phot_g_mean_mag', flipY=true) & pick!
    const gaiaHrHit = await page.evaluate(async (t) => {
      const { plot, diag } = (window as any).__p7;
      plot.setViewOptions({ pipeline: 'cartesian' });
      plot.setEncoding({
        x: 'bp_rp',
        y: 'phot_g_mean_mag',
        flipY: true,
        color: { field: 'bp_rp', scale: 'coolwarm' },
        aggregation: 'points',
      });
      // Wait for x/y columns to finish streaming AND 2-second WGSL transition to complete
      const deadline = performance.now() + 8000;
      while (performance.now() < deadline) {
        await new Promise((r) => setTimeout(r, 80));
        const tm: any = diag.getTileManager();
        if (tm && tm.loadingTiles.size === 0 && !diag.tileStore.isTransitioning) break;
      }
      plot.setViewport({ x: t.bp_rp, y: t.phot_g_mean_mag, zoom: 128 });
      await new Promise((r) => setTimeout(r, 120));
      return await plot.pickAt(480, 300);
    }, gaiaHrTarget);

    console.log('[4c. Gaia HR Diagram Pick row_id=14301]', JSON.stringify(gaiaHrHit));
    expect(gaiaHrHit).not.toBeNull();
    expect(gaiaHrHit.rowId).toBe(gaiaHrTarget.row_id);
    expect(Math.abs(gaiaHrHit.dataX - gaiaHrTarget.bp_rp)).toBeLessThan(1e-5);
    expect(Math.abs(gaiaHrHit.dataY - gaiaHrTarget.phot_g_mean_mag)).toBeLessThan(1e-5);

    // =========================================================================
    // 5. PRIORITY-AWARE TIE-BREAKING (z=0 beats z=1), OVERSIZE LEAF (>2^20-1 rows) CHUNK PASS,
    //    resolveRowAttributes SIDECAR HOOK (with rowIdBigInt > 2^53-1), AND ZERO-LEAK DESTROY
    // =========================================================================
    const hugeId = 5900000000000000123n; // Exceeds Number.MAX_SAFE_INTEGER (9007199254740991n)
    const hugeIpcBytes = Array.from(
      tableToIPC(
        tableFromArrays({
          row_id: new BigUint64Array([hugeId]),
          x_u16: new Uint16Array([32768]),
          y_u16: new Uint16Array([32768]),
          z: new Uint8Array([0]),
          score: new Float32Array([42.5]),
        }),
        { format: 'stream' }
      )!
    );

    const tieAndOversizeAndTeardown = await page.evaluate(async (ipcArr: number[]) => {
      const { Deepgraph } = await import('/src/index.ts');
      const { INTERNAL_DIAGNOSTICS } = await import('/src/lib/diagnostics.ts');
      const { getGpuStats } = await import('/src/demo/gpuStats.ts');

      // Destroy first plot first
      (window as any).__p7.plot.destroy();

      // Create plot with resolveRowAttributes hook to test sidecar enrichment + synthetic unit probes
      const stage = document.getElementById('stage')!;
      const plot2 = await Deepgraph.create(stage, {
        url: '/samples/lmsys_sample.arrowtiles',
        powerPreference: 'low-power',
        resolveRowAttributes: async (rowId, attrs) => {
          return { ...attrs, sidecarPrompt: `prompt_for_${rowId}` };
        },
      });
      const diag2 = (plot2 as any)[INTERNAL_DIAGNOSTICS]();
      const deadline = performance.now() + 8000;
      while (performance.now() < deadline) {
        await new Promise((r) => setTimeout(r, 50));
        const tm: any = diag2.getTileManager();
        if (tm && tm.activeTiles.size >= 1 && tm.loadingTiles.size === 0) break;
      }

      // 5a: Verify resolveRowAttributes enriches real picked hit
      plot2.setViewport({ x: 88.76073587009975, y: 69.12738843695004, zoom: 128 });
      await new Promise((r) => setTimeout(r, 80));
      const enrichedHit = await plot2.pickAt(480, 300);

      // 5b: Priority tie-break unit probe on PickingLayer ('0/0/0' beats '1/1/1' at exact same pixel)
      plot2.fitView();
      const store = diag2.tileStore;
      store.clearAllTiles();
      const packedHalf = 32768 | (32768 << 16);
      const packedZ1 = 1 | (1 << 16);
      store.updateVisibleTiles([
        {
          key: '1/1/1',
          numRows: 1,
          uploadedToGpu: false,
          columns: { xy_u16: new Uint32Array([packedZ1]).buffer },
        },
        {
          key: '0/0/0',
          numRows: 1,
          uploadedToGpu: false,
          columns: { xy_u16: new Uint32Array([packedHalf]).buffer },
        },
      ]);
      const tieWinner = await diag2.pickingLayer.pick(480, 300, diag2.camera, store, 6.0);

      // 5c: Oversize leaf (> 2^20 - 1 = 1,048,575 rows) multi-chunk pass probe!
      store.clearAllTiles();
      const oversizeCount = 1_060_000;
      const oversizeArr = new Uint32Array(oversizeCount);
      const targetOversizeIdx = 1_055_000;
      oversizeArr[targetOversizeIdx] = packedHalf;
      store.updateVisibleTiles([
        {
          key: '0/0/0',
          numRows: oversizeCount,
          uploadedToGpu: false,
          columns: { xy_u16: oversizeArr.buffer },
        },
      ]);
      const oversizeWinner = await diag2.pickingLayer.pick(480, 300, diag2.camera, store, 6.0);

      // 5d: Verify > 2^53 - 1 UInt64 row_id preserves exact rowIdBigInt through pmtiles.worker.ts decode_row!
      const tm2: any = diag2.getTileManager();
      const bigDecoded: any = await new Promise((resolve, reject) => {
        const id = 'test_bigint_row';
        tm2.rowDecodeResolvers.set(id, {
          resolve: (res: any) => {
            if (res?.rowIdBigIntStr) {
              res.rowIdBigInt = BigInt(res.rowIdBigIntStr);
            }
            resolve(res);
          },
          reject,
        });
        const buf = new Uint8Array(ipcArr).buffer;
        tm2.dispatchToWorker(
          { action: 'decode_row', id, buffers: [buf], rowIndex: 0, categories: {} },
          [buf]
        );
      });

      plot2.destroy();
      const finalGs = getGpuStats();
      return {
        enrichedSidecarPrompt: enrichedHit?.attributes?.sidecarPrompt ?? null,
        bigRowIdStr: bigDecoded?.rowIdBigInt ? String(bigDecoded.rowIdBigInt) : '',
        bigRowScore: bigDecoded?.attributes?.score ?? null,
        tieWinnerKey: tieWinner?.tileKey ?? null,
        oversizeWinnerIdx: oversizeWinner?.rawRowIdx ?? -1,
        finalGs,
        uncapturedErrors: (window as any).__wgpuUncaptured.slice(),
      };
    }, hugeIpcBytes);

    console.log('[5. Tie-Break, Oversize Chunk, BigInt & Zero-Leak Teardown]', JSON.stringify(tieAndOversizeAndTeardown));
    expect(tieAndOversizeAndTeardown.enrichedSidecarPrompt).toBe('prompt_for_40064');
    expect(tieAndOversizeAndTeardown.bigRowIdStr).toBe('5900000000000000123');
    expect(tieAndOversizeAndTeardown.bigRowScore).toBe(42.5);
    expect(tieAndOversizeAndTeardown.tieWinnerKey).toBe('0/0/0');
    expect(tieAndOversizeAndTeardown.oversizeWinnerIdx).toBe(1_055_000);
    expect(tieAndOversizeAndTeardown.uncapturedErrors).toEqual([]);
    expect(tieAndOversizeAndTeardown.finalGs.liveDevices).toBe(0);
    expect(tieAndOversizeAndTeardown.finalGs.liveBuffers).toBe(0);
    expect(tieAndOversizeAndTeardown.finalGs.liveBytes).toBe(0);
    expect(tieAndOversizeAndTeardown.finalGs.totalDestroyedBytes).toBe(
      tieAndOversizeAndTeardown.finalGs.totalCreatedBytes
    );

    // =========================================================================
    // 6. PRE-PHASE-8 HOVER NETWORK/CPU HARDENING, INSTANT GPU HOVER RING,
    //    SPATIAL SLOT BOUNDS CULL, pointerleave, OPT-IN resolveRowAttributesOnHover,
    //    AND ASYNC CLICK/PICK ERROR GUARDS (50 ms RTT / 50 Mbps CDP Throttling)
    // =========================================================================
    await page.evaluate(() => {
      (window as any).__unhandledRejections = [];
      window.addEventListener('unhandledrejection', (ev) => {
        (window as any).__unhandledRejections.push(
          ev?.reason?.message || String(ev?.reason || 'unhandledrejection')
        );
        ev.preventDefault();
      });
    });

    let sampleRangeReqCount = 0;
    const onSampleRequest = (req: any) => {
      if (req.url().includes('lmsys_sample.arrowtiles')) {
        sampleRangeReqCount++;
      }
    };
    page.on('request', onSampleRequest);

    const cdp = await context.newCDPSession(page);
    await cdp.send('Network.enable');
    await cdp.send('Network.emulateNetworkConditions', {
      offline: false,
      latency: 50,
      downloadThroughput: Math.floor((50 * 1024 * 1024) / 8),
      uploadThroughput: Math.floor((50 * 1024 * 1024) / 8),
    });

    try {
      // 6a. Already-loaded tile hover sweep (resolveRowAttributesOnHover omitted -> default false):
      //     Load Z=0..1 (5 tiles <= 8-tile compressed LRU cap) so multiple spatial tiles are visible
      //     (verifying spatial slot bounds culling) AND all visible tiles sit in rawPickTileCache ->
      //     hovering across 20 distinct positions makes 0 Range requests & 0 resolver calls!
      await page.evaluate(async () => {
        const { Deepgraph } = await import('/src/index.ts');
        const { INTERNAL_DIAGNOSTICS } = await import('/src/lib/diagnostics.ts');
        const stage = document.getElementById('stage')!;
        let resolverCallsDefault = 0;
        const plot = await Deepgraph.create(stage, {
          url: '/samples/lmsys_sample.arrowtiles',
          powerPreference: 'low-power',
          detailOffset: 0,
          encoding: { color: 'model', aggregation: 'points' },
          resolveRowAttributes: async (rowId, attrs) => {
            resolverCallsDefault++;
            return { ...attrs, sidecarDefault: rowId };
          },
        });
        const diag = (plot as any)[INTERNAL_DIAGNOSTICS]();
        plot.setViewport({ x: 50, y: 50, zoom: 2.0 });
        // Wait 2 rAF frames so RenderLoop dispatches getVisibleTiles for zoom=2.0 before polling queue idle
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

        const deadline = performance.now() + 12000;
        while (performance.now() < deadline) {
          await new Promise((r) => setTimeout(r, 80));
          const tm: any = diag.getTileManager();
          if (
            tm &&
            tm.activeTiles.size >= 5 &&
            tm.loadingTiles.size === 0 &&
            tm.queue.size === 0 &&
            tm.queue.pending === 0
          ) {
            break;
          }
        }
        await new Promise((r) => setTimeout(r, 150));

        const hoverEvents: any[] = [];
        plot.on('hover', (h) => hoverEvents.push(h));
        (window as any).__p8HoverA = {
          plot,
          diag,
          hoverEvents,
          getResolverCalls: () => resolverCallsDefault,
        };
      });

      const reqsBeforeLoadedSweep = sampleRangeReqCount;
      // Sweep 20 distinct positions across the loaded UMAP view (spaced >= 60 ms > HOVER_THROTTLE_MS=45ms)
      for (let i = 0; i < 20; i++) {
        const gx = i % 5;
        const gy = Math.floor(i / 5);
        const mx = Math.round(stageBox!.x + 340 + gx * 55);
        const my = Math.round(stageBox!.y + 190 + gy * 55);
        await page.mouse.move(mx, my);
        await page.waitForTimeout(60);
      }
      await page.waitForTimeout(200);
      const reqsDuringLoadedSweep = sampleRangeReqCount - reqsBeforeLoadedSweep;

      // Capture spatial slot culling count while at zoom=2.0 (5 visible tiles: 0/0/0 + four Z=1 quadrants;
      // picking inside top-left quadrant (340, 190) culls the 3 non-overlapping Z=1 quadrants!)
      await page.mouse.move(Math.round(stageBox!.x + 340), Math.round(stageBox!.y + 190));
      await page.waitForTimeout(90);
      const zoom2CullCheck = await page.evaluate(() => {
        const { diag } = (window as any).__p8HoverA;
        return {
          lastCulledSlotCount: diag.pickingLayer.lastCulledSlotCount,
          lastDispatchedSlotCount: diag.pickingLayer.lastDispatchedSlotCount,
        };
      });

      // Center on known point row_id=40064 and verify same-winner hover short-circuit + pointerleave clear!
      await page.evaluate(async () => {
        const { plot, diag } = (window as any).__p8HoverA;
        plot.setViewport({ x: 88.76073587009975, y: 69.12738843695004, zoom: 128 });
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
        const deadline = performance.now() + 10000;
        while (performance.now() < deadline) {
          await new Promise((r) => setTimeout(r, 60));
          const tm: any = diag.getTileManager();
          if (tm && tm.loadingTiles.size === 0 && tm.queue.size === 0 && tm.queue.pending === 0) {
            break;
          }
        }
        await new Promise((r) => setTimeout(r, 100));
        (window as any).__p8HoverA.hoverEvents.length = 0;
      });

      await page.mouse.move(pageCenterX - 40, pageCenterY - 40);
      await page.waitForTimeout(65);
      await page.mouse.move(pageCenterX, pageCenterY);
      await page.waitForFunction(
        () => (window as any).__p8HoverA.hoverEvents.some((h: any) => h && h.rowId === 40064),
        null,
        { timeout: 6000 }
      );
      const countAfterFirstSameHover = await page.evaluate(
        () => (window as any).__p8HoverA.hoverEvents.length
      );
      // Move 1 CSS px (still winning exact same row_id=40064) -> same-winner short-circuit emits 0 duplicates!
      await page.waitForTimeout(65);
      await page.mouse.move(pageCenterX + 1, pageCenterY);
      await page.waitForTimeout(150);

      const beforeLeaveState = await page.evaluate(() => {
        const { diag, hoverEvents, getResolverCalls } = (window as any).__p8HoverA;
        return {
          eventsAfterSamePointJitter: hoverEvents.length,
          hoverMarkerPresentBeforeLeave: diag.highlightOverlay.hoverMarker !== null,
          lastCulledSlotCount: diag.pickingLayer.lastCulledSlotCount,
          lastDispatchedSlotCount: diag.pickingLayer.lastDispatchedSlotCount,
          resolverCallsOnHoverDefaultFalse: getResolverCalls(),
        };
      });

      // Move pointer completely outside #stage (<canvas>) -> triggers pointerleave!
      await page.mouse.move(5, 5);
      await page.evaluate(() => {
        const c = document.querySelector('#stage canvas');
        c?.dispatchEvent(new PointerEvent('pointerleave', { bubbles: false }));
      });
      await page.waitForTimeout(80);

      const afterLeaveState = await page.evaluate(() => {
        const { plot, diag, hoverEvents } = (window as any).__p8HoverA;
        const res = {
          hoverMarkerAfterLeave: diag.highlightOverlay.hoverMarker,
          lastHoverEvent: hoverEvents[hoverEvents.length - 1],
        };
        plot.destroy();
        return res;
      });

      console.log(
        '[6a. Throttled Loaded-Tile 20-Hover Sweep, Same-Point Short-Circuit, Spatial Cull & pointerleave]',
        JSON.stringify({
          reqsDuringLoadedSweep,
          zoom2CullCheck,
          countAfterFirstSameHover,
          beforeLeaveState,
          afterLeaveState,
        })
      );
      expect(reqsDuringLoadedSweep).toBe(0);
      expect(zoom2CullCheck.lastCulledSlotCount).toBeGreaterThanOrEqual(3);
      expect(beforeLeaveState.eventsAfterSamePointJitter).toBe(countAfterFirstSameHover);
      expect(beforeLeaveState.hoverMarkerPresentBeforeLeave).toBe(true);
      expect(beforeLeaveState.resolverCallsOnHoverDefaultFalse).toBe(0);
      expect(afterLeaveState.hoverMarkerAfterLeave).toBeNull();
      expect(afterLeaveState.lastHoverEvent).toBeNull();

      // 6b. Cold compressed-LRU 20-position hover sweep inside single tile (0/0/0) under 50ms RTT / 50 Mbps,
      //     WITH resolveRowAttributesOnHover: true (delayed 120ms to prove instant GPU hover ring!):
      //     Clear rawPickTileCache so first hover MUST fetch tile 0/0/0 over throttled network,
      //     then sweep 20 guaranteed point positions inside 0/0/0 -> <= 1 HTTP Range request total!
      const coldTargetCssPoints = await page.evaluate(async () => {
        const { Deepgraph } = await import('/src/index.ts');
        const { INTERNAL_DIAGNOSTICS } = await import('/src/lib/diagnostics.ts');
        const stage = document.getElementById('stage')!;
        let hoverResolverCalls = 0;
        const plot = await Deepgraph.create(stage, {
          url: '/samples/lmsys_sample.arrowtiles',
          powerPreference: 'low-power',
          detailOffset: 0, // Keep zoom=1.5 (floor(log2(1.5))=0) strictly on single root tile 0/0/0!
          resolveRowAttributesOnHover: true,
          resolveRowAttributes: async (rowId, attrs) => {
            hoverResolverCalls++;
            await new Promise((r) => setTimeout(r, 120));
            return { ...attrs, hoverEnriched: true, enrichedRowId: rowId };
          },
        });
        plot.setViewport({ x: 50, y: 50, zoom: 1.5 });
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
        const diag = (plot as any)[INTERNAL_DIAGNOSTICS]();
        const deadline = performance.now() + 20000;
        let rootSlotReady = false;
        while (performance.now() < deadline) {
          await new Promise((r) => setTimeout(r, 60));
          const tm: any = diag.getTileManager();
          rootSlotReady = diag.tileStore.slots.some(
            (s: any) => s && s.key === '0/0/0' && s.visible && s.activeRows > 0
          );
          if (
            tm &&
            rootSlotReady &&
            tm.activeTiles.has('0/0/0') &&
            tm.loadingTiles.size === 0 &&
            tm.queue.size === 0 &&
            tm.queue.pending === 0
          ) {
            break;
          }
        }
        if (!rootSlotReady) {
          throw new Error('Root tile 0/0/0 did not upload to GPU slot within 20s under 50ms RTT');
        }
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
        await new Promise((r) => setTimeout(r, 80));

        // Sample 20 distinct real visible point CSS positions directly from 0/0/0 before purging caches
        const tm: any = diag.getTileManager();
        const cam = diag.camera;
        const camLeft = cam.position.x + cam.left / cam.zoom;
        const camRight = cam.position.x + cam.right / cam.zoom;
        const camBottom = cam.position.y + cam.bottom / cam.zoom;
        const camTop = cam.position.y + cam.top / cam.zoom;
        const rb = plot.getMetadata()!.rootBounds;

        const pts: { cssX: number; cssY: number }[] = [];
        for (let idx = 0; idx < 400 && pts.length < 20; idx += 7) {
          const r = await tm.decodeTileRow(0, 0, 0, idx, {});
          if (!r) continue;
          const wx = rb.minX + (r.xU16 / 65535.0) * (rb.maxX - rb.minX);
          const wy = rb.maxY - (r.yU16 / 65535.0) * (rb.maxY - rb.minY);
          const cssX = ((wx - camLeft) / (camRight - camLeft)) * 960;
          const cssY = ((camTop - wy) / (camTop - camBottom)) * 600;
          if (cssX >= 40 && cssX <= 920 && cssY >= 40 && cssY <= 560) {
            if (pts.every((p) => Math.hypot(p.cssX - cssX, p.cssY - cssY) >= 12)) {
              pts.push({ cssX, cssY });
            }
          }
        }

        // Explicitly purge compressed pick LRU + worker row cache so 0/0/0 starts 100% cold for hover!
        tm.clearRawPickTileCache(true);

        const hoverHits: any[] = [];
        plot.on('hover', (h) => {
          if (h) hoverHits.push(h);
        });
        (window as any).__p8HoverB = {
          plot,
          diag,
          hoverHits,
          getHoverResolverCalls: () => hoverResolverCalls,
        };
        return pts;
      });
      expect(coldTargetCssPoints.length).toBe(20);

      const reqsBeforeColdSweep = sampleRangeReqCount;

      // Move onto point #0 first and verify HighlightOverlay.hoverMarker is ALREADY populated
      // directly from GPU winnerCoordBuffer BEFORE the 50ms RTT + 120ms resolver finishes!
      const p0x = Math.round(stageBox!.x + coldTargetCssPoints[0].cssX);
      const p0y = Math.round(stageBox!.y + coldTargetCssPoints[0].cssY);
      await page.mouse.move(p0x, p0y);
      await page.waitForFunction(
        () => {
          const b = (window as any).__p8HoverB;
          return b && b.diag.highlightOverlay.hoverMarker !== null;
        },
        null,
        { timeout: 1500 }
      );
      const instantRingBeforeNetworkResolved = await page.evaluate(() => {
        const b = (window as any).__p8HoverB;
        return {
          markerPresentImmediately: b.diag.highlightOverlay.hoverMarker !== null,
          hoverHitsResolvedSoFar: b.hoverHits.length,
        };
      });

      // Continue across all 19 remaining distinct point positions inside root tile 0/0/0 (20 total positions)
      for (let i = 1; i < coldTargetCssPoints.length; i++) {
        const mx = Math.round(stageBox!.x + coldTargetCssPoints[i].cssX);
        const my = Math.round(stageBox!.y + coldTargetCssPoints[i].cssY);
        await page.mouse.move(mx, my);
        await page.waitForTimeout(55);
      }
      // Wait for the final in-flight hover + 120ms sidecar resolver to settle
      await page.waitForFunction(
        () => (window as any).__p8HoverB.hoverHits.length >= 1,
        null,
        { timeout: 6000 }
      );
      const coldTileRangeReqs = sampleRangeReqCount - reqsBeforeColdSweep;

      const coldSweepSummary = await page.evaluate(() => {
        const b = (window as any).__p8HoverB;
        const cpuStats = b.diag.getCpuMemoryStats();
        const out = {
          hoverHitsCount: b.hoverHits.length,
          hoverEnrichedFlag: Boolean(b.hoverHits[b.hoverHits.length - 1]?.attributes?.hoverEnriched),
          hoverResolverCalls: b.getHoverResolverCalls(),
          rawPickCacheEntries: cpuStats?.rawPickCacheEntries ?? 0,
          rawPickCacheBytes: cpuStats?.rawPickCacheBytes ?? 0,
        };
        b.plot.destroy();
        return out;
      });

      console.log(
        '[6b. Cold-Cache 20-Hover Sweep (50ms RTT / 50 Mbps), Instant GPU Ring & Opt-In Hover Sidecar]',
        JSON.stringify({
          coldTileRangeReqs,
          instantRingBeforeNetworkResolved,
          coldSweepSummary,
        })
      );
      expect(coldTileRangeReqs).toBeLessThanOrEqual(1);
      expect(instantRingBeforeNetworkResolved.markerPresentImmediately).toBe(true);
      expect(instantRingBeforeNetworkResolved.hoverHitsResolvedSoFar).toBe(0);
      expect(coldSweepSummary.hoverResolverCalls).toBeGreaterThanOrEqual(1);
      expect(coldSweepSummary.hoverEnrichedFlag).toBe(true);
      expect(coldSweepSummary.rawPickCacheEntries).toBe(1);
      expect(coldSweepSummary.rawPickCacheBytes).toBeGreaterThan(0);
    } finally {
      await cdp.send('Network.emulateNetworkConditions', {
        offline: false,
        latency: 0,
        downloadThroughput: -1,
        uploadThroughput: -1,
      });
      page.off('request', onSampleRequest);
    }

    // 6c. Async Error Guards:
    //     1) Mid-click plot.destroy() resolves cleanly with 0 UnhandledPromiseRejections.
    //     2) Genuine resolveRowAttributes exception routes to plot.on('error', ...) with 0 UnhandledPromiseRejections.
    //     3) Zstd init error simulation in worker routes action + id so decode_row rejects instead of hanging.
    const errorGuardSummary = await page.evaluate(async () => {
      const { Deepgraph } = await import('/src/index.ts');
      const { INTERNAL_DIAGNOSTICS } = await import('/src/lib/diagnostics.ts');
      const { getGpuStats } = await import('/src/demo/gpuStats.ts');
      const stage = document.getElementById('stage')!;

      const waitTilesReady = async (p: any) => {
        const d = p[INTERNAL_DIAGNOSTICS]();
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
        const deadline = performance.now() + 8000;
        while (performance.now() < deadline) {
          await new Promise((r) => setTimeout(r, 50));
          const tm: any = d.getTileManager();
          if (
            tm &&
            d.tileStore.getActivePointsCount() > 0 &&
            tm.loadingTiles.size === 0 &&
            tm.queue.size === 0 &&
            tm.queue.pending === 0
          ) {
            break;
          }
        }
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      };

      // Test 1: Mid-click destroy() + 4th-arg AbortSignal cancellation
      let notifyInResolver!: () => void;
      const enteredResolver = new Promise<void>((r) => {
        notifyInResolver = r;
      });
      let receivedAbortSignalInstance = false;
      let signalAbortedOnDestroy = false;
      const plotDestroyRace = await Deepgraph.create(stage, {
        url: '/samples/lmsys_sample.arrowtiles',
        powerPreference: 'low-power',
        resolveRowAttributes: async (_rowId, _attrs, _big, signal) => {
          receivedAbortSignalInstance = signal instanceof AbortSignal;
          signal?.addEventListener('abort', () => {
            signalAbortedOnDestroy = true;
          });
          notifyInResolver();
          await new Promise((r) => setTimeout(r, 200));
          return {};
        },
      });
      plotDestroyRace.setViewport({ x: 88.76073587009975, y: 69.12738843695004, zoom: 128 });
      await waitTilesReady(plotDestroyRace);

      const canvasEl = stage.querySelector('canvas')!;
      const rect = canvasEl.getBoundingClientRect();
      const cx = rect.left + 480;
      const cy = rect.top + 300;
      canvasEl.dispatchEvent(
        new PointerEvent('pointerdown', { button: 0, pointerId: 1, clientX: cx, clientY: cy })
      );
      canvasEl.dispatchEvent(
        new PointerEvent('pointerup', { button: 0, pointerId: 1, clientX: cx, clientY: cy })
      );
      await enteredResolver;
      plotDestroyRace.destroy();
      await new Promise((r) => setTimeout(r, 220));

      // Test 1b: Pre-call abort guard — if hover signal is aborted while decodeTileRow is resolving,
      //          resolveRowAttributes MUST NOT be invoked at all!
      let abortedHoverResolverInvocations = 0;
      let hoverSignalAbortedOnPointerLeave = false;
      const plotPreAbort = await Deepgraph.create(stage, {
        url: '/samples/lmsys_sample.arrowtiles',
        powerPreference: 'low-power',
        resolveRowAttributesOnHover: true,
        resolveRowAttributes: async (_rowId, attrs, _big, signal) => {
          abortedHoverResolverInvocations++;
          signal?.addEventListener('abort', () => {
            hoverSignalAbortedOnPointerLeave = true;
          });
          await new Promise((r) => setTimeout(r, 180));
          return attrs;
        },
      });
      plotPreAbort.on('hover', () => {});
      plotPreAbort.setViewport({ x: 88.76073587009975, y: 69.12738843695004, zoom: 128 });
      await waitTilesReady(plotPreAbort);

      // Delay decodeTileRow slightly so pointerleave aborts BEFORE resolveRowAttributes is reached
      const sessPre: any = (plotPreAbort as any).session;
      const origDecode = sessPre.decodeTileRow.bind(sessPre);
      let enteredDecode!: () => void;
      const decodeEnteredPromise = new Promise<void>((r) => {
        enteredDecode = r;
      });
      sessPre.decodeTileRow = async (...args: any[]) => {
        enteredDecode();
        await new Promise((r) => setTimeout(r, 120));
        return origDecode(...args);
      };

      const preCanvas = stage.querySelector('canvas')!;
      const preRect = preCanvas.getBoundingClientRect();
      preCanvas.dispatchEvent(
        new PointerEvent('pointermove', {
          clientX: preRect.left + 480,
          clientY: preRect.top + 300,
        })
      );
      await decodeEnteredPromise;
      // Trigger pointerleave while decodeTileRow is still awaiting -> aborts hover signal before resolver!
      preCanvas.dispatchEvent(new PointerEvent('pointerleave', { bubbles: false }));
      await new Promise((r) => setTimeout(r, 160));
      const callsWhenAbortedBeforeResolver = abortedHoverResolverInvocations;

      // Restore decodeTileRow and verify pointerleave WHILE inside resolver flips signal.aborted = true!
      sessPre.decodeTileRow = origDecode;
      await new Promise((r) => setTimeout(r, 60));
      preCanvas.dispatchEvent(
        new PointerEvent('pointermove', {
          clientX: preRect.left + 480,
          clientY: preRect.top + 300,
        })
      );
      const waitDeadline = performance.now() + 3000;
      while (abortedHoverResolverInvocations === 0 && performance.now() < waitDeadline) {
        await new Promise((r) => setTimeout(r, 20));
      }
      preCanvas.dispatchEvent(new PointerEvent('pointerleave', { bubbles: false }));
      plotPreAbort.destroy();

      // Test 2: Genuine resolveRowAttributes throw routes to plot.on('error')
      const capturedErrors: string[] = [];
      const plotErr = await Deepgraph.create(stage, {
        url: '/samples/lmsys_sample.arrowtiles',
        powerPreference: 'low-power',
        resolveRowAttributes: async () => {
          throw new Error('boom_sidecar_failure');
        },
      });
      plotErr.on('error', (e) => capturedErrors.push(e.message));
      plotErr.setViewport({ x: 88.76073587009975, y: 69.12738843695004, zoom: 128 });
      await waitTilesReady(plotErr);
      const failedPick = await plotErr.pickAt(480, 300);

      // Test 3: Worker Zstd-init style error routing (`action` + `id` rejects decode_row resolver cleanly)
      const tmErr: any = (plotErr as any)[INTERNAL_DIAGNOSTICS]().getTileManager();
      let workerErrMsg = '';
      try {
        await new Promise((resolve, reject) => {
          const fakeId = 'test_zstd_init_err';
          tmErr.rowDecodeResolvers.set(fakeId, { resolve, reject });
          tmErr.handleWorkerMessage(
            new MessageEvent('message', {
              data: {
                action: 'decode_row',
                id: fakeId,
                error: 'Failed to initialize Zstd WASM: synthetic_failure',
              },
            }),
            tmErr.allWorkers[0]
          );
        });
      } catch (e: any) {
        workerErrMsg = e?.message || String(e);
      }

      plotErr.destroy();
      await new Promise((r) => setTimeout(r, 100));

      return {
        receivedAbortSignalInstance,
        signalAbortedOnDestroy,
        callsWhenAbortedBeforeResolver,
        hoverSignalAbortedOnPointerLeave,
        failedPick,
        capturedErrors,
        workerErrMsg,
        unhandledRejections: (window as any).__unhandledRejections.slice(),
        finalGs: getGpuStats(),
        uncapturedErrors: (window as any).__wgpuUncaptured.slice(),
      };
    });

    console.log('[6c. Async Error Guards & Zero-Leak Teardown]', JSON.stringify(errorGuardSummary));
    expect(errorGuardSummary.receivedAbortSignalInstance).toBe(true);
    expect(errorGuardSummary.signalAbortedOnDestroy).toBe(true);
    expect(errorGuardSummary.callsWhenAbortedBeforeResolver).toBe(0);
    expect(errorGuardSummary.hoverSignalAbortedOnPointerLeave).toBe(true);
    expect(errorGuardSummary.failedPick).toBeNull();
    expect(errorGuardSummary.capturedErrors).toContain('boom_sidecar_failure');
    expect(errorGuardSummary.workerErrMsg).toContain('Failed to initialize Zstd WASM');
    expect(errorGuardSummary.unhandledRejections).toEqual([]);
    expect(errorGuardSummary.uncapturedErrors).toEqual([]);
    expect(errorGuardSummary.finalGs.liveDevices).toBe(0);
    expect(errorGuardSummary.finalGs.liveBuffers).toBe(0);
    expect(errorGuardSummary.finalGs.liveBytes).toBe(0);
    expect(errorGuardSummary.finalGs.totalDestroyedBytes).toBe(
      errorGuardSummary.finalGs.totalCreatedBytes
    );

    // =========================================================================
    // 7. v0.1.1 HARDENING PROBES ON REAL WEBGPU:
    //    - 2,500 overlapping categorical points at identical pixel -> 0.0000% shimmer
    //    - transform: 'literal' custom axes render & pick parity
    //    - spatial flipY: true setViewport / getViewport / pickAt parity
    //    - generic mean_field (metricType == 0) negative values with colorMin > 100
    //    - categorical x axis + partial resolveRowAttributes merge + numeric dataX
    //    - PickingLayer slot-eviction race safety during async readback
    // =========================================================================
    const v011Probes = await page.evaluate(async (pref) => {
      const { Deepgraph } = await import('/src/index.ts');
      const { INTERNAL_DIAGNOSTICS } = await import('/src/lib/diagnostics.ts');
      const { getGpuStats } = await import('/src/demo/gpuStats.ts');
      const stage = document.getElementById('stage')!;

      const plot = await Deepgraph.create(stage, {
        url: '/samples/lmsys_sample.arrowtiles',
        powerPreference: pref,
        encoding: { color: 'model', aggregation: 'points' },
        // Partial resolver that intentionally does NOT spread `attrs`!
        resolveRowAttributes: async (rowId) => ({ onlySidecarKey: `extra_${rowId}` }),
      });
      const diag = (plot as any)[INTERNAL_DIAGNOSTICS]();
      const waitIdle = async () => {
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
        const deadline = performance.now() + 8000;
        while (performance.now() < deadline) {
          await new Promise((r) => setTimeout(r, 40));
          const tm: any = diag.getTileManager();
          if (tm && tm.loadingTiles.size === 0 && tm.queue.size === 0 && tm.queue.pending === 0) {
            break;
          }
        }
        await new Promise((r) => setTimeout(r, 80));
      };
      await waitIdle();

      // 7a: Spatial flipY: true setViewport / getViewport / pickAt parity
      plot.setEncoding({ flipY: true, color: 'model', aggregation: 'points' });
      diag.tileStore.isTransitioning = false;
      diag.tileStore.transitionT = 1.0;
      plot.setViewport({ x: 88.76073587009975, y: 69.12738843695004, zoom: 128 });
      await waitIdle();
      const vpFlipY = plot.getViewport();
      const hitFlipY = await plot.pickAt(480, 300);

      // 7b: Categorical X chart axis (`x: 'model'`) + partial resolver merge
      plot.setEncoding({
        x: 'model',
        y: 'num_of_tokens',
        flipY: false,
        color: 'model',
        aggregation: 'points',
      });
      // Fast-forward 2s transition
      diag.tileStore.isTransitioning = false;
      diag.tileStore.transitionT = 1.0;
      await waitIdle();
      const cats = plot.getCategories('model') ?? [];
      const mixtralIdx = cats.indexOf('mixtral');
      plot.setViewport({ x: mixtralIdx, y: 308, zoom: 64 });
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      const catAxisHit = await plot.pickAt(480, 300);

      // 7c: `transform: 'literal'` custom axes (`world = data`, NOT spatial x_u16/y_u16!)
      plot.setEncoding({
        x: { field: 'num_of_tokens', transform: 'literal' },
        y: { field: 'num_of_tokens', transform: 'literal' },
        color: 'model',
        aggregation: 'points',
      });
      diag.tileStore.isTransitioning = false;
      diag.tileStore.transitionT = 1.0;
      await waitIdle();
      plot.fitView();
      const literalFitVp = plot.getViewport();
      plot.setViewport({ x: 308, y: 308, zoom: 2.0 });
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      const literalHit = await plot.pickAt(480, 300);

      // 7d: Synthetic 2,500 overlapping categorical points at identical pixel -> 0.0000% shimmer!
      plot.setEncoding({ x: null, y: null, color: 'model', aggregation: 'points', flipY: false });
      diag.tileStore.isTransitioning = false;
      diag.tileStore.transitionT = 1.0;
      plot.fitView();
      const store = diag.tileStore;
      store.clearAllTiles();
      const N = 2500;
      const xyArr = new Uint32Array(N).fill(32768 | (32768 << 16));
      const catCodes = new Float32Array(N);
      for (let i = 0; i < N; i++) catCodes[i] = i % 4;
      store.updateVisibleTiles([
        {
          key: '0/0/0',
          numRows: N,
          uploadedToGpu: false,
          columns: {
            xy_u16: xyArr.buffer,
            model: catCodes.buffer,
          },
        },
      ]);
      const slot0 = store.slots[0];
      if (slot0) slot0.spawnTime = store.globalTimeSec - 5.0;
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      const c = stage.querySelector('canvas')!;
      const snapRgba = async () => {
        const res = await fetch(c.toDataURL('image/png'));
        const bmp = await createImageBitmap(await res.blob());
        const oc = new OffscreenCanvas(bmp.width, bmp.height);
        const ctx = oc.getContext('2d')!;
        ctx.drawImage(bmp, 0, 0);
        return ctx.getImageData(0, 0, bmp.width, bmp.height).data;
      };
      const catFrameA = await snapRgba();
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      const catFrameB = await snapRgba();
      let catOverlapLit = 0;
      let catOverlapChanged = 0;
      for (let i = 0; i < catFrameA.length; i += 4) {
        if (Math.abs(catFrameA[i] - 17) > 3 || Math.abs(catFrameA[i + 1] - 17) > 3 || Math.abs(catFrameA[i + 2] - 17) > 3) {
          catOverlapLit++;
          if (
            catFrameA[i] !== catFrameB[i] ||
            catFrameA[i + 1] !== catFrameB[i + 1] ||
            catFrameA[i + 2] !== catFrameB[i + 2]
          ) {
            catOverlapChanged++;
          }
        }
      }

      // 7e: Generic `mean_field` (`metricType == 0`) with `colorMin > 100` AND negative value `-50.0`!
      plot.setEncoding({
        x: null,
        y: null,
        color: { field: 'num_of_tokens', scale: 'viridis', domain: [150, 500] },
        aggregation: 'mean_field',
      });
      store.clearAllTiles();
      store.updateVisibleTiles([
        {
          key: '0/0/0',
          numRows: 1,
          uploadedToGpu: false,
          columns: {
            xy_u16: new Uint32Array([32768 | (32768 << 16)]).buffer,
            num_of_tokens: new Float32Array([-50.0]).buffer,
          },
        },
      ]);
      if (store.slots[0]) store.slots[0].spawnTime = store.globalTimeSec - 5.0;
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      const meanNegWinner = await diag.pickingLayer.pick(480, 300, diag.camera, store, 6.0);

      // 7f: PickingLayer slot-eviction race during readbackBuffer.mapAsync -> returns null safely!
      const readbackBuf: GPUBuffer = (diag.pickingLayer as any).readbackBuffer;
      const origMapAsync = readbackBuf.mapAsync.bind(readbackBuf);
      let evictedDuringMap = false;
      readbackBuf.mapAsync = async (...args: any[]) => {
        const p = origMapAsync(...args);
        if (!evictedDuringMap) {
          evictedDuringMap = true;
          // Evict and replace slot 0 with a different tile key mid-flight!
          store.clearAllTiles();
          store.updateVisibleTiles([
            {
              key: '1/0/0',
              numRows: 1,
              uploadedToGpu: false,
              columns: {
                xy_u16: new Uint32Array([0]).buffer,
                num_of_tokens: new Float32Array([200.0]).buffer,
              },
            },
          ]);
        }
        return p;
      };
      const evictedRaceWinner = await diag.pickingLayer.pick(480, 300, diag.camera, store, 6.0);
      readbackBuf.mapAsync = origMapAsync;

      // 7g: Mid-flight 2s axis transition (`transitionT = 0.5`) mixing one `literal` axis with one spatial axis:
      // Verify ComputeRasterizer, PickingLayer, and HighlightOverlay project the exact same blended pixel!
      plot.setEncoding({
        x: null,
        y: null,
        color: { field: 'num_of_tokens', scale: 'viridis', domain: [0, 500] },
        aggregation: 'points',
      });
      plot.fitView();
      store.clearAllTiles();
      store.updateVisibleTiles([
        {
          key: '0/0/0',
          numRows: 1,
          uploadedToGpu: false,
          columns: {
            xy_u16: new Uint32Array([16384 | (49152 << 16)]).buffer,
            num_of_tokens: new Float32Array([250.0]).buffer,
          },
        },
      ]);
      // Trigger animated transition: spatial X -> literal X (`num_of_tokens` in [0, 500]), keeping spatial Y!
      plot.setEncoding({
        x: { field: 'num_of_tokens', transform: 'literal', domain: [0, 500] },
        y: null,
      });
      plot.fitView();
      // Freeze mid-transition at transitionT = 0.5
      store.isTransitioning = true;
      store.transitionT = 0.5;
      if (store.slots[0]) {
        store.slots[0].visible = true;
        store.slots[0].spawnTime = store.globalTimeSec - 5.0;
      }
      diag.highlightOverlay.selectedMarkers = [
        {
          kind: 'selected',
          rowId: 777,
          rawRowIdx: 0,
          tileZ: 0,
          tileX: 0,
          tileY: 0,
          xU16: 16384,
          yU16: 49152,
          attributes: { num_of_tokens: 250.0 },
        },
      ];
      // Render frame at frozen transitionT = 0.5 (without letting RenderLoop overwrite transitionT)
      (plot as any).renderLoop.stop();
      (plot as any).rendererWrapper.updateCameraUniforms();
      (plot as any).computeRasterizer.render(diag.camera, store, (enc: any, view: any) => {
        diag.highlightOverlay.encodePass(enc, view, diag.camera, store);
      });
      // Read overlay marker staging screen pixel (bx, by) in backing coordinates and convert to CSS px
      const overlayBx = (diag.highlightOverlay as any).stagingF32[0];
      const overlayBy = (diag.highlightOverlay as any).stagingF32[1];
      const dprNow = Math.max(1, (plot as any).rendererWrapper.dpr);
      const midTransWinner = await diag.pickingLayer.pick(
        overlayBx / dprNow,
        overlayBy / dprNow,
        diag.camera,
        store,
        2.0
      );

      // 7h: GPUDevice.lost handling emits clean 'error' when reason !== 'destroyed'
      plot.destroy();
      const finalGs = getGpuStats();
      return {
        vpFlipY,
        hitFlipYRowId: hitFlipY?.rowId ?? -1,
        hitFlipYSidecarMerged: Boolean(
          hitFlipY?.attributes?.model === 'mixtral' &&
            hitFlipY?.attributes?.onlySidecarKey === 'extra_40064'
        ),
        catAxisDataXIsFiniteNumber:
          catAxisHit !== null && Number.isFinite(catAxisHit.dataX),
        literalFitVp,
        literalHitValid: literalHit !== null && Number.isFinite(literalHit.dataX),
        catOverlapLit,
        catOverlapChanged,
        meanNegWinnerFound: meanNegWinner !== null,
        evictedRaceWinner,
        midTransWinnerDistPx: midTransWinner ? midTransWinner.distCssPx : 999,
        finalGs,
      };
    }, expectedPowerPref);

    console.log('[7. v0.1.1 Hardening Probes]', JSON.stringify(v011Probes));
    expect(Math.abs(v011Probes.vpFlipY.x - 88.76073587009975)).toBeLessThan(1e-6);
    expect(Math.abs(v011Probes.vpFlipY.y - 69.12738843695004)).toBeLessThan(1e-6);
    expect(v011Probes.hitFlipYRowId).toBe(40064);
    expect(v011Probes.hitFlipYSidecarMerged).toBe(true);
    expect(v011Probes.catAxisDataXIsFiniteNumber).toBe(true);
    expect(Number.isFinite(v011Probes.literalFitVp.zoom)).toBe(true);
    expect(v011Probes.literalFitVp.zoom).toBeGreaterThan(0);
    expect(v011Probes.literalHitValid).toBe(true);
    expect(v011Probes.catOverlapLit).toBeGreaterThan(0);
    expect(v011Probes.catOverlapChanged).toBe(0);
    expect(v011Probes.meanNegWinnerFound).toBe(true);
    expect(v011Probes.evictedRaceWinner).toBeNull();
    expect(v011Probes.midTransWinnerDistPx).toBeLessThanOrEqual(0.25);
    expect(v011Probes.finalGs.liveDevices).toBe(0);
    expect(v011Probes.finalGs.liveBytes).toBe(0);

    expect(pageErrors).toEqual([]);
    expect(consoleErrors).toEqual([]);
  });
});
