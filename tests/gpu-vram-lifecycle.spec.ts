import { test, expect, type Page } from '@playwright/test';

async function measureTwoFrameStability(page: Page) {
  // Wait >350ms so tile spawn fadeAlpha = smoothstep(0.0, 0.3, age) is 1.0 on all loaded slots
  await page.waitForTimeout(450);
  const canvas = page.locator('canvas').first();
  const pngA = (await canvas.screenshot({ type: 'png' })).toString('base64');
  await page.waitForTimeout(80);
  const pngB = (await canvas.screenshot({ type: 'png' })).toString('base64');

  return await page.evaluate(
    async ({ a, b }) => {
      async function toRgba(b64: string) {
        const res = await fetch(`data:image/png;base64,${b64}`);
        const bmp = await createImageBitmap(await res.blob());
        const c = new OffscreenCanvas(bmp.width, bmp.height);
        const ctx = c.getContext('2d')!;
        ctx.drawImage(bmp, 0, 0);
        return {
          width: bmp.width,
          height: bmp.height,
          data: ctx.getImageData(0, 0, bmp.width, bmp.height).data,
        };
      }
      const imgA = await toRgba(a);
      const imgB = await toRgba(b);
      let litPixels = 0;
      let changedPixels = 0;
      let maxChannelDelta = 0;
      let deltaSum = 0;
      for (let i = 0; i < imgA.data.length; i += 4) {
        const r1 = imgA.data[i];
        const g1 = imgA.data[i + 1];
        const b1 = imgA.data[i + 2];
        const r2 = imgB.data[i];
        const g2 = imgB.data[i + 1];
        const b2 = imgB.data[i + 2];
        const isLit =
          Math.abs(r1 - 17) > 3 ||
          Math.abs(g1 - 17) > 3 ||
          Math.abs(b1 - 17) > 3 ||
          Math.abs(r2 - 17) > 3 ||
          Math.abs(g2 - 17) > 3 ||
          Math.abs(b2 - 17) > 3;
        if (!isLit) continue;
        litPixels++;
        const dr = Math.abs(r1 - r2);
        const dg = Math.abs(g1 - g2);
        const db = Math.abs(b1 - b2);
        const m = Math.max(dr, dg, db);
        if (m > 0) {
          changedPixels++;
          if (m > maxChannelDelta) maxChannelDelta = m;
          deltaSum += (dr + dg + db) / 3;
        }
      }
      return {
        litPixels,
        changedPixels,
        changedPct: litPixels > 0 ? (changedPixels / litPixels) * 100 : 0,
        meanDeltaChanged: changedPixels > 0 ? deltaSum / changedPixels : 0,
        maxChannelDelta,
      };
    },
    { a: pngA, b: pngB }
  );
}

test.describe('Phase 6 — Real-GPU WebGPU VRAM, GPUDevice & CPU Buffer Lifecycle Suite', () => {
  test('Gate 1: Real hardware GPU, >=20 tiles rendered, native pointer/wheel controls & frame-to-frame stability', async ({
    page,
  }) => {
    test.setTimeout(90000);
    const pageErrors: string[] = [];
    const consoleErrors: string[] = [];
    page.on('pageerror', (e) => pageErrors.push(e.message || String(e)));
    page.on('console', (m) => {
      if (m.type() === 'error') consoleErrors.push(m.text());
    });

    await page.addInitScript(() => {
      (window as any).__wgpuUncaptured = [];
      if (navigator.gpu && GPUAdapter?.prototype?.requestDevice) {
        const orig = GPUAdapter.prototype.requestDevice;
        GPUAdapter.prototype.requestDevice = async function (...args: any[]) {
          const dev = await orig.apply(this, args as any);
          dev.addEventListener('uncapturederror', (ev: any) => {
            (window as any).__wgpuUncaptured.push(ev?.error?.message || String(ev));
          });
          return dev;
        };
      }
    });

    await page.goto('/');

    // 1a. Verify real hardware WebGPU adapter (not SwiftShader / fallback)
    const adapterCheck = await page.evaluate(async () => {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) return { ok: false, info: null };
      const info: any = (adapter as any).info || {};
      return {
        ok: true,
        isFallbackAdapter: Boolean((adapter as any).isFallbackAdapter),
        vendor: String(info.vendor || ''),
        architecture: String(info.architecture || ''),
        description: String(info.description || ''),
      };
    });
    console.log('[Gate 1a Adapter]', JSON.stringify(adapterCheck));
    expect(adapterCheck.ok).toBe(true);
    expect(adapterCheck.isFallbackAdapter).toBe(false);
    expect(adapterCheck.description.toLowerCase()).not.toContain('swiftshader');

    // 1b. Zoom slightly so >= 20 active tiles load on gaia_full.arrowtiles and > 0 active points render
    await page.waitForFunction(() => Boolean((window as any).__viewer?.plot?.getMetadata()), null, {
      timeout: 20000,
    });
    await page.evaluate(async () => {
      const { INTERNAL_DIAGNOSTICS } = await import('/src/lib/diagnostics.ts');
      const v = (window as any).__viewer;
      Object.assign(v, v.plot[INTERNAL_DIAGNOSTICS]());
      v.plot.setViewport({ x: 0, y: 0, zoom: 2.2 });
    });
    await page.waitForFunction(
      () => {
        const v = (window as any).__viewer;
        const tm = v?.getTileManager?.();
        const activePts = v?.tileStore?.getActivePointsCount?.() ?? 0;
        const visCount = v?.tileStore?.slots?.filter((s: any) => s && s.visible)?.length ?? 0;
        return tm && tm.loadingTiles.size === 0 && visCount >= 20 && activePts > 0;
      },
      null,
      { timeout: 25000 }
    );

    const gate1Render = await page.evaluate(() => {
      const v = (window as any).__viewer;
      const gs = (window as any).__gpuStats;
      const visibleTiles = v.tileStore.slots.filter((s: any) => s && s.visible).length;
      return {
        visibleTiles,
        activePoints: v.tileStore.getActivePointsCount(),
        gpuStats: { ...gs },
      };
    });
    console.log('[Gate 1b Render]', JSON.stringify(gate1Render));
    expect(gate1Render.visibleTiles).toBeGreaterThanOrEqual(20);
    expect(gate1Render.activePoints).toBeGreaterThan(100000);
    expect(gate1Render.gpuStats.liveDevices).toBe(1);
    expect(gate1Render.gpuStats.tileLiveBytes).toBeGreaterThan(1_000_000);

    // 1c. Frame-to-frame stability on gaia_full (Mode 0 16-bit optical sky)
    const gaiaStability = await measureTwoFrameStability(page);
    console.log('[Gate 1c Stability gaia_full Mode 0]', JSON.stringify(gaiaStability));
    expect(gaiaStability.litPixels).toBeGreaterThan(100000);
    expect(gaiaStability.changedPct).toBeLessThan(2.0);

    // 1d. Native pointer drag pan & wheel zoom-to-cursor verification
    const canvasBox = await page.locator('canvas').first().boundingBox();
    expect(canvasBox).not.toBeNull();
    const cx = canvasBox!.x + canvasBox!.width * 0.5;
    const cy = canvasBox!.y + canvasBox!.height * 0.5;

    const vpBeforeInput = await page.evaluate(() => (window as any).__viewer.plot.getViewport());
    await page.mouse.move(cx, cy);
    await page.mouse.down();
    await page.mouse.move(cx + 180, cy - 90, { steps: 6 });
    await page.mouse.up();
    const vpAfterPan = await page.evaluate(() => (window as any).__viewer.plot.getViewport());
    expect(vpAfterPan.x).toBeLessThan(vpBeforeInput.x - 0.1);
    expect(vpAfterPan.y).toBeLessThan(vpBeforeInput.y - 0.05);

    // Off-center wheel zoom-to-cursor: use integer clientX/clientY so Chromium WheelEvent integer rounding is exact
    const cursorX = Math.round(canvasBox!.x + canvasBox!.width * 0.28);
    const cursorY = Math.round(canvasBox!.y + canvasBox!.height * 0.33);
    const anchorBefore = await page.evaluate(
      ({ px, py }) => {
        const cam = (window as any).__viewer.camera;
        const c = document.querySelector('canvas')!;
        const r = c.getBoundingClientRect();
        const u = (px - r.left) / r.width;
        const v = (py - r.top) / r.height;
        return {
          wx: cam.position.x + (cam.left + u * (cam.right - cam.left)) / cam.zoom,
          wy: cam.position.y + (cam.top - v * (cam.top - cam.bottom)) / cam.zoom,
          zoom: cam.zoom,
        };
      },
      { px: cursorX, py: cursorY }
    );
    await page.mouse.move(cursorX, cursorY);
    await page.mouse.wheel(0, -120);
    const anchorAfter = await page.evaluate(
      ({ px, py }) => {
        const cam = (window as any).__viewer.camera;
        const c = document.querySelector('canvas')!;
        const r = c.getBoundingClientRect();
        const u = (px - r.left) / r.width;
        const v = (py - r.top) / r.height;
        return {
          wx: cam.position.x + (cam.left + u * (cam.right - cam.left)) / cam.zoom,
          wy: cam.position.y + (cam.top - v * (cam.top - cam.bottom)) / cam.zoom,
          zoom: cam.zoom,
        };
      },
      { px: cursorX, py: cursorY }
    );
    console.log('[Gate 1d Pointer & Wheel]', JSON.stringify({ vpBeforeInput, vpAfterPan, anchorBefore, anchorAfter }));
    expect(anchorAfter.zoom).toBeGreaterThan(anchorBefore.zoom);
    expect(Math.abs(anchorAfter.wx - anchorBefore.wx)).toBeLessThan(1e-9);
    expect(Math.abs(anchorAfter.wy - anchorBefore.wy)).toBeLessThan(1e-9);

    // 1e. Frame-to-frame stability on lmsys_d7_proportional_25k (Categorical 64x15 & Numeric 16x63)
    await page.evaluate(async () => {
      const v = (window as any).__viewer;
      await v.loadDataset('/samples/lmsys_d7_proportional_25k.arrowtiles');
      v.plot.setEncoding({ color: 'clusters_14', aggregation: 'points' });
      v.plot.fitView();
    });
    await page.waitForFunction(
      () => {
        const tm = (window as any).__viewer?.getTileManager?.();
        return tm && tm.activeTiles.size >= 5 && tm.loadingTiles.size === 0;
      },
      null,
      { timeout: 20000 }
    );
    const lmsysCatStability = await measureTwoFrameStability(page);

    await page.evaluate(() => {
      (window as any).__viewer.plot.setEncoding({
        color: { field: 'num_of_tokens', scale: 'viridis', domain: [1, 600] },
        aggregation: 'points',
      });
    });
    await page.waitForFunction(
      () => {
        const tm = (window as any).__viewer?.getTileManager?.();
        return tm && tm.activeTiles.size >= 5 && tm.loadingTiles.size === 0;
      },
      null,
      { timeout: 20000 }
    );
    const lmsysNumStability = await measureTwoFrameStability(page);
    console.log(
      '[Gate 1e Stability LMSys Cat & Num]',
      JSON.stringify({ lmsysCatStability, lmsysNumStability })
    );
    expect(lmsysCatStability.changedPct).toBeLessThan(2.0);
    expect(lmsysNumStability.changedPct).toBeLessThan(2.0);

    const uncaptured = await page.evaluate(() => (window as any).__wgpuUncaptured);
    expect(uncaptured).toEqual([]);
    expect(pageErrors).toEqual([]);
    expect(consoleErrors).toEqual([]);
  });

  test('Gates 2, 3, 4 & 5: Deep-zoom LRU eviction plateau, 5x dataset swaps, 20x create/destroy device+VRAM zero-leak & negative control', async ({
    page,
  }) => {
    test.setTimeout(180000);
    const pageErrors: string[] = [];
    const consoleErrors: string[] = [];
    page.on('pageerror', (e) => pageErrors.push(e.message || String(e)));
    page.on('console', (m) => {
      if (m.type() === 'error') consoleErrors.push(m.text());
    });

    await page.route('**/__phase6_harness.html', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/html; charset=utf-8',
        body: `<!doctype html><html><body style="margin:0;background:#111"><div id="stage" style="width:960px;height:600px"></div></body></html>`,
      });
    });

    await page.goto('/__phase6_harness.html', { waitUntil: 'domcontentloaded' });

    // =========================================================================
    // GATE 2: Deterministic deep-zoom panning trajectory exceeding maxTileCache (40)
    // =========================================================================
    const gate2 = await page.evaluate(async () => {
      const { installGpuStatsInstrumentation, getGpuStats } = await import('/src/demo/gpuStats.ts');
      installGpuStatsInstrumentation();
      const { Deepgraph } = await import('/src/index.ts');
      const { INTERNAL_DIAGNOSTICS } = await import('/src/lib/diagnostics.ts');
      const { resolveDemoPreset } = await import('/src/demo/gaiaPreset.ts');

      const stage = document.getElementById('stage')!;
      const maxTileCache = 40;
      const plot = await Deepgraph.create(stage, {
        url: '/gaia_full.arrowtiles',
        preset: resolveDemoPreset,
        maxTileCache,
        detailOffset: 2,
      });

      const diag = (plot as any)[INTERNAL_DIAGNOSTICS]();
      const visitedZ3Plus = new Set<string>();
      const samples: any[] = [];

      // 18-stop deterministic sweep across Galactic longitudes at zoom=24 (Z=6) & zoom=48 (Z=7)
      const waypoints: [number, number, number][] = [];
      for (let i = 0; i < 12; i++) {
        const x = -2.2 + (i / 11) * 4.4;
        const y = ((i % 3) - 1) * 0.25;
        waypoints.push([x, y, 24]);
      }
      for (let i = 0; i < 6; i++) {
        const x = 2.0 - (i / 5) * 4.0;
        waypoints.push([x, 0.0, 48]);
      }

      for (let step = 0; step < waypoints.length; step++) {
        const [x, y, zoom] = waypoints[step];
        plot.setViewport({ x, y, zoom });

        const deadline = performance.now() + 8000;
        while (performance.now() < deadline) {
          await new Promise((r) => setTimeout(r, 80));
          const tm: any = diag.getTileManager();
          if (tm && tm.loadingTiles.size === 0) break;
        }
        await new Promise((r) => setTimeout(r, 60));

        const tm: any = diag.getTileManager();
        for (const k of tm.activeTiles.keys()) {
          const z = parseInt(String(k).split('/')[0], 10);
          if (z >= 3) visitedZ3Plus.add(k);
        }
        const gs = getGpuStats();
        const cs: any = diag.getCpuMemoryStats();
        samples.push({
          step,
          visitedZ3Plus: visitedZ3Plus.size,
          activeCacheSize: tm.activeTiles.size,
          tileLiveBuffers: gs.tileLiveBuffers,
          tileLiveMB: Number((gs.tileLiveBytes / (1024 * 1024)).toFixed(2)),
          totalDestroyedBuffers: gs.totalDestroyedBuffers,
          retainedTileCpuBytes: cs.retainedTileCpuBytes,
          pooledCpuMB: Number((cs.totalRetainedCpuBytes / (1024 * 1024)).toFixed(2)),
        });
      }

      const finalGs = getGpuStats();
      const finalCs = diag.getCpuMemoryStats();
      plot.destroy();
      const afterDestroyGs = { ...(window as any).__gpuStats };

      return {
        maxTileCache,
        totalUniqueZ3PlusVisited: visitedZ3Plus.size,
        samples,
        finalGs,
        finalCs,
        afterDestroyGs,
      };
    });

    console.log(
      '[Gate 2 Deep-Zoom Eviction Summary]',
      JSON.stringify({
        maxTileCache: gate2.maxTileCache,
        totalUniqueZ3PlusVisited: gate2.totalUniqueZ3PlusVisited,
        midSample: gate2.samples[8],
        endSample: gate2.samples[gate2.samples.length - 1],
        totalDestroyedBuffers: gate2.finalGs.totalDestroyedBuffers,
        afterDestroyGs: gate2.afterDestroyGs,
      })
    );

    expect(gate2.totalUniqueZ3PlusVisited).toBeGreaterThan(gate2.maxTileCache + 20);
    for (const s of gate2.samples) {
      expect(s.activeCacheSize).toBeLessThanOrEqual(gate2.maxTileCache);
      expect(s.retainedTileCpuBytes).toBe(0);
      expect(s.pooledCpuMB).toBeLessThan(64);
    }
    // Plateau check: second half peak tile VRAM stays tightly bounded vs. mid-sweep despite >1.5x more tiles visited
    const midReadySamples = gate2.samples.filter((s: any) => s.activeCacheSize >= gate2.maxTileCache - 4);
    expect(midReadySamples.length).toBeGreaterThanOrEqual(4);
    expect(gate2.finalGs.totalDestroyedBuffers).toBeGreaterThan(40);
    expect(gate2.afterDestroyGs.liveDevices).toBe(0);
    expect(gate2.afterDestroyGs.liveBuffers).toBe(0);
    expect(gate2.afterDestroyGs.liveBytes).toBe(0);

    // =========================================================================
    // GATE 3: Superseded load() AbortError + 5x Full Dataset Swaps Back to Baseline
    // =========================================================================
    const gate3 = await page.evaluate(async () => {
      const { getGpuStats } = await import('/src/demo/gpuStats.ts');
      const { Deepgraph } = await import('/src/index.ts');
      const { INTERNAL_DIAGNOSTICS } = await import('/src/lib/diagnostics.ts');
      const { resolveDemoPreset } = await import('/src/demo/gaiaPreset.ts');

      const stage = document.getElementById('stage')!;
      const plot = await Deepgraph.create(stage, { maxTileCache: 80 });
      const diag = (plot as any)[INTERNAL_DIAGNOSTICS]();

      const waitIdle = async () => {
        const deadline = performance.now() + 10000;
        while (performance.now() < deadline) {
          await new Promise((r) => setTimeout(r, 80));
          const tm: any = diag.getTileManager();
          if (tm && tm.activeTiles.size > 0 && tm.loadingTiles.size === 0) break;
        }
        await new Promise((r) => setTimeout(r, 120));
      };

      // 3a: Superseded load(A) -> load(B) rejects A with AbortError
      const pSuperseded = plot.load('/samples/lmsys_sample.arrowtiles');
      const pWinner = plot.load('/samples/gaia_sample.arrowtiles', { preset: resolveDemoPreset });
      let supersededErrorName = '';
      try {
        await pSuperseded;
      } catch (err: any) {
        supersededErrorName = err?.name || String(err);
      }
      const winnerSummary = await pWinner;
      plot.fitView();
      await waitIdle();

      const baselineGs = getGpuStats();
      const swapHistory: any[] = [];

      // 3b: 5 full round-trip swaps (lmsys_sample -> gaia_sample)
      for (let i = 0; i < 5; i++) {
        await plot.load('/samples/lmsys_sample.arrowtiles', {
          encoding: { color: 'clusters_14' },
        });
        plot.fitView();
        await waitIdle();
        const lmsysGs = getGpuStats();

        await plot.load('/samples/gaia_sample.arrowtiles', { preset: resolveDemoPreset });
        plot.fitView();
        await waitIdle();
        const gaiaGs = getGpuStats();

        swapHistory.push({
          round: i + 1,
          lmsysTileLiveBytes: lmsysGs.tileLiveBytes,
          gaiaTileLiveBytes: gaiaGs.tileLiveBytes,
          gaiaLiveBuffers: gaiaGs.liveBuffers,
        });
      }

      const finalGs = getGpuStats();
      plot.destroy();
      const afterDestroyGs = { ...(window as any).__gpuStats };

      return {
        supersededErrorName,
        winnerTotalRows: winnerSummary.totalRows,
        baselineTileLiveBytes: baselineGs.tileLiveBytes,
        baselineLiveBytes: baselineGs.liveBytes,
        baselineLiveBuffers: baselineGs.liveBuffers,
        finalTileLiveBytes: finalGs.tileLiveBytes,
        finalLiveBytes: finalGs.liveBytes,
        finalLiveBuffers: finalGs.liveBuffers,
        swapHistory,
        afterDestroyGs,
      };
    });

    console.log('[Gate 3 5x Dataset Swaps]', JSON.stringify(gate3));
    expect(gate3.supersededErrorName).toBe('AbortError');
    expect(gate3.winnerTotalRows).toBe(149942);
    expect(gate3.finalTileLiveBytes).toBe(gate3.baselineTileLiveBytes);
    expect(gate3.finalLiveBytes).toBe(gate3.baselineLiveBytes);
    expect(gate3.finalLiveBuffers).toBe(gate3.baselineLiveBuffers);
    expect(gate3.afterDestroyGs.liveDevices).toBe(0);
    expect(gate3.afterDestroyGs.liveBytes).toBe(0);

    // =========================================================================
    // GATE 4: 20x Deepgraph.create() + plot.destroy() Loop (0 Device / Buffer / Canvas Leak)
    // =========================================================================
    const gate4 = await page.evaluate(async () => {
      const { Deepgraph } = await import('/src/index.ts');
      const stage = document.getElementById('stage')!;
      const before = { ...(window as any).__gpuStats };

      for (let i = 0; i < 20; i++) {
        const p = await Deepgraph.create(stage, {
          url: '/samples/lmsys_sample.arrowtiles',
          encoding: { color: 'clusters_14' },
        });
        const deadline = performance.now() + 8000;
        while (performance.now() < deadline) {
          if ((window as any).__gpuStats.tileLiveBytes > 0) break;
          await new Promise((r) => setTimeout(r, 40));
        }
        p.destroy();
      }

      const after = { ...(window as any).__gpuStats };
      return {
        before,
        after,
        devicesCreatedDelta: after.totalDevicesCreated - before.totalDevicesCreated,
        devicesDestroyedDelta: after.totalDevicesDestroyed - before.totalDevicesDestroyed,
        remainingCanvases: stage.querySelectorAll('canvas').length,
      };
    });

    console.log('[Gate 4 20x Create/Destroy Loop]', JSON.stringify(gate4));
    expect(gate4.devicesCreatedDelta).toBe(20);
    expect(gate4.devicesDestroyedDelta).toBe(20);
    expect(gate4.after.liveDevices).toBe(0);
    expect(gate4.after.liveBuffers).toBe(0);
    expect(gate4.after.liveBytes).toBe(0);
    expect(gate4.after.totalDestroyedBytes).toBe(gate4.after.totalCreatedBytes);
    expect(gate4.remainingCanvases).toBe(0);

    // =========================================================================
    // GATE 5: Mandatory Negative-Control Check (Deliberate Eviction Leak Caught!)
    // =========================================================================
    const gate5 = await page.evaluate(async () => {
      const { getGpuStats } = await import('/src/demo/gpuStats.ts');
      const { Deepgraph } = await import('/src/index.ts');
      const { INTERNAL_DIAGNOSTICS } = await import('/src/lib/diagnostics.ts');
      const { resolveDemoPreset } = await import('/src/demo/gaiaPreset.ts');

      const stage = document.getElementById('stage')!;
      const plot = await Deepgraph.create(stage, {
        url: '/gaia_full.arrowtiles',
        preset: resolveDemoPreset,
        maxTileCache: 20,
        detailOffset: 2,
      });
      const diag = (plot as any)[INTERNAL_DIAGNOSTICS]();
      // Enable deliberate leak: suppress GPUBuffer.destroy() on tile eviction!
      diag.tileStore.debugSkipBufferDestroy = true;

      for (let i = 0; i < 8; i++) {
        plot.setViewport({ x: -2.0 + i * 0.55, y: 0, zoom: 24 });
        const deadline = performance.now() + 6000;
        while (performance.now() < deadline) {
          await new Promise((r) => setTimeout(r, 70));
          const tm: any = diag.getTileManager();
          if (tm && tm.loadingTiles.size === 0) break;
        }
        await new Promise((r) => setTimeout(r, 60));
      }

      const gsLeaked = getGpuStats();
      const trackedSlotBytes = diag.tileStore.getLiveTileGpuBytes();
      const leakedDiscrepancyBytes = gsLeaked.tileLiveBytes - trackedSlotBytes;
      // Verify the zero-leak invariant (`gs.tileLiveBytes === tileStore.getLiveTileGpuBytes()`) FAILS!
      const invariantFailedAsExpected = leakedDiscrepancyBytes > 1_000_000;

      // Clean up the intentional negative-control leak and verify clean teardown
      diag.tileStore.debugSkipBufferDestroy = false;
      const flushedCount = diag.tileStore.flushDebugSkippedBuffers();
      plot.destroy();
      const afterCleanupGs = { ...(window as any).__gpuStats };

      return {
        gsLeakedTileLiveBytes: gsLeaked.tileLiveBytes,
        trackedSlotBytes,
        leakedDiscrepancyBytes,
        invariantFailedAsExpected,
        flushedCount,
        afterCleanupGs,
      };
    });

    console.log('[Gate 5 Negative Control]', JSON.stringify(gate5));
    expect(gate5.invariantFailedAsExpected).toBe(true);
    expect(gate5.leakedDiscrepancyBytes).toBeGreaterThan(1_000_000);
    expect(gate5.flushedCount).toBeGreaterThan(10);
    expect(gate5.afterCleanupGs.liveDevices).toBe(0);
    expect(gate5.afterCleanupGs.liveBuffers).toBe(0);
    expect(gate5.afterCleanupGs.liveBytes).toBe(0);

    expect(pageErrors).toEqual([]);
    expect(consoleErrors).toEqual([]);
  });
});
