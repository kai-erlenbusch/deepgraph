import { test, expect } from '@playwright/test';
import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');

test.describe('Phase 8 — Built `dist/index.js` Single-HTML Embed & Public-API Demo Proof (Intel iGPU Sample-Only)', () => {
  test.beforeAll(() => {
    if (!fs.existsSync(path.join(rootDir, 'dist', 'standalone.js'))) {
      execSync('npm run build:lib', { cwd: rootDir, stdio: 'inherit' });
    }
  });

  test('`examples/embed.html` loads built `../dist/standalone.js` + worker/WASM cleanly, picks points & tears down cleanly on Intel iGPU', async ({
    page,
  }) => {
    test.setTimeout(60000);
    const pageErrors: string[] = [];
    const consoleErrors: string[] = [];
    const failedUrls: string[] = [];

    page.on('pageerror', (e) => pageErrors.push(e.message || String(e)));
    page.on('console', (m) => {
      if (m.type() === 'error') consoleErrors.push(m.text());
    });
    page.on('response', (res) => {
      if (res.status() >= 400 && !res.url().endsWith('/favicon.ico')) {
        failedUrls.push(`${res.status()} ${res.url()}`);
      }
    });

    // 0. Strict Option B Intel iGPU Pre-Check BEFORE navigating to embed.html / calling requestDevice()
    await page.route('**/__phase8_igpu_guard.html', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/html; charset=utf-8',
        body: '<!doctype html><html><body></body></html>',
      });
    });
    await page.goto('/__phase8_igpu_guard.html', { waitUntil: 'domcontentloaded' });

    const igpuGuard = await page.evaluate(async () => {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'low-power' });
      if (!adapter) {
        throw new Error('ABORT: low-power WebGPU adapter unavailable');
      }
      const info: any = (adapter as any).info || {};
      const vendor = String(info.vendor || '').toLowerCase();
      if (vendor !== 'intel') {
        throw new Error(
          `ABORT: Expected Intel iGPU ('intel'), got vendor='${vendor}'. Refusing to touch discrete GPU.`
        );
      }
      return {
        vendor,
        architecture: String(info.architecture || ''),
        isFallbackAdapter: Boolean((adapter as any).isFallbackAdapter),
      };
    });
    console.log('[Phase 8 Intel iGPU Pre-Check]', JSON.stringify(igpuGuard));
    expect(igpuGuard.vendor).toBe('intel');
    expect(igpuGuard.isFallbackAdapter).toBe(false);

    // 1. Navigate to standalone `/examples/embed.html?power=low` importing `../dist/index.js`
    const loadedDistRequests: string[] = [];
    page.on('request', (req) => {
      if (req.url().includes('/dist/')) {
        loadedDistRequests.push(new URL(req.url()).pathname);
      }
    });

    await page.goto('/examples/embed.html?power=low', { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => Boolean((window as any).__embedPlot?.getMetadata()?.totalRows === 100000),
      null,
      { timeout: 15000 }
    );

    // Wait until rendered UMAP pixels appear on the canvas (proves built worker + base64 Zstd WASM decoded tiles!)
    await page.waitForFunction(
      async () => {
        const c = document.querySelector('#plot canvas') as HTMLCanvasElement | null;
        if (!c || c.width === 0) return false;
        const b64 = c.toDataURL('image/png');
        const res = await fetch(b64);
        const bmp = await createImageBitmap(await res.blob());
        const oc = new OffscreenCanvas(bmp.width, bmp.height);
        const ctx = oc.getContext('2d')!;
        ctx.drawImage(bmp, 0, 0);
        const d = ctx.getImageData(
          Math.floor(bmp.width * 0.2),
          Math.floor(bmp.height * 0.2),
          Math.floor(bmp.width * 0.6),
          Math.floor(bmp.height * 0.6)
        ).data;
        let lit = 0;
        for (let i = 0; i < d.length; i += 16) {
          if (d[i] > 28 || d[i + 1] > 28 || d[i + 2] > 28) lit++;
        }
        return lit > 500;
      },
      null,
      { timeout: 15000 }
    );

    // 2. Center on known LMSys row_id=40064 and click canvas center -> #hit-readout updates from public 'click' event!
    await page.evaluate(async () => {
      const plot = (window as any).__embedPlot;
      plot.setViewport({ x: 88.76073587009975, y: 69.12738843695004, zoom: 128 });
      await new Promise<void>((resolve) => {
        let seen = 0;
        const off = plot.on('telemetry', (snap: any) => {
          if (snap.currentZ >= 4 && snap.activePoints > 0 && ++seen >= 2) {
            off();
            resolve();
          }
        });
      });
    });

    const plotBox = await page.locator('#plot').boundingBox();
    expect(plotBox).not.toBeNull();
    await page.mouse.click(
      Math.round(plotBox!.x + plotBox!.width * 0.5),
      Math.round(plotBox!.y + plotBox!.height * 0.5)
    );

    await expect(page.locator('#hit-readout')).toContainText('Clicked row_id=40064', {
      timeout: 6000,
    });
    await expect(page.locator('#hit-readout')).toContainText('"model":"mixtral"');

    const embedCleanup = await page.evaluate(() => {
      const lilGuiCount = document.querySelectorAll('.lil-gui').length;
      const winGpuStatsDefined = (window as any).__gpuStats !== undefined;
      const readoutText = document.getElementById('hit-readout')?.textContent ?? '';
      (window as any).__embedPlot.destroy();
      const canvasesAfterDestroy = document.querySelectorAll('#plot canvas').length;
      return { lilGuiCount, winGpuStatsDefined, readoutText, canvasesAfterDestroy };
    });

    console.log(
      '[Phase 8 Standalone Embed Summary]',
      JSON.stringify({ loadedDistRequests, embedCleanup, failedUrls })
    );
    expect(loadedDistRequests).toContain('/dist/standalone.js');
    expect(loadedDistRequests.some((p) => p.startsWith('/dist/assets/pmtiles.worker-'))).toBe(true);
    expect(embedCleanup.lilGuiCount).toBe(0);
    expect(embedCleanup.winGpuStatsDefined).toBe(false);
    expect(embedCleanup.canvasesAfterDestroy).toBe(0);
    expect(failedUrls).toEqual([]);

    // 3. Verify pure-public-API Demo App (`/?file=/samples/lmsys_sample.arrowtiles&power=low`)
    //    works with 0 deep internal imports and shows hover/click inspect card!
    await page.goto('/?file=/samples/lmsys_sample.arrowtiles&power=low', {
      waitUntil: 'domcontentloaded',
    });
    await page.waitForFunction(
      () => Boolean((window as any).__viewer?.plot?.getMetadata()?.totalRows === 100000),
      null,
      { timeout: 15000 }
    );

    const demoPublicApiCheck = await page.evaluate(async () => {
      const v = (window as any).__viewer;
      const viewerKeys = Object.keys(v).sort();
      // Wait for initial overview tiles via public 'telemetry' event first
      await new Promise<void>((resolve) => {
        const off = v.plot.on('telemetry', (snap: any) => {
          if (snap.activePoints > 0 && snap.activeTiles >= 1) {
            off();
            resolve();
          }
        });
      });
      v.plot.setViewport({ x: 88.76073587009975, y: 69.12738843695004, zoom: 128 });
      // Wait 2 telemetry ticks (~500ms) + rAF after zooming to Z=4 leaf tile
      await new Promise<void>((resolve) => {
        let seen = 0;
        const off = v.plot.on('telemetry', (snap: any) => {
          if (snap.currentZ >= 4 && snap.activePoints > 0 && ++seen >= 2) {
            off();
            resolve();
          }
        });
      });
      return { viewerKeys };
    });
    expect(demoPublicApiCheck.viewerKeys).toEqual(['getGui', 'loadDataset', 'plot']);

    const demoCanvasBox = await page.locator('#app canvas').boundingBox();
    expect(demoCanvasBox).not.toBeNull();
    const demoCenterX = Math.round(demoCanvasBox!.x + demoCanvasBox!.width * 0.5);
    const demoCenterY = Math.round(demoCanvasBox!.y + demoCanvasBox!.height * 0.5);

    // Move first to verify live hover readout (`Hover row_id: 40064`), then click to pin (`Selected row_id: 40064`)!
    await page.mouse.move(demoCenterX - 40, demoCenterY - 40);
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
    await page.mouse.move(demoCenterX, demoCenterY);
    await expect
      .poll(
        async () => {
          await page.mouse.move(demoCenterX, demoCenterY);
          return (await page.locator('#point-inspect-card').textContent()) ?? '';
        },
        { timeout: 6000, intervals: [60, 80, 120] }
      )
      .toContain('Hover row_id: 40064');

    await page.mouse.click(demoCenterX, demoCenterY);
    await expect(page.locator('#point-inspect-card')).toContainText('Selected row_id: 40064', {
      timeout: 6000,
    });

    const demoDestroyGs = await page.evaluate(() => {
      (window as any).__viewer.plot.destroy();
      return { ...(window as any).__gpuStats };
    });
    console.log('[Phase 8 Public-API Demo Teardown]', JSON.stringify(demoDestroyGs));
    expect(demoDestroyGs.liveDevices).toBe(0);
    expect(demoDestroyGs.liveBuffers).toBe(0);
    expect(demoDestroyGs.liveBytes).toBe(0);

    expect(pageErrors).toEqual([]);
    expect(consoleErrors).toEqual([]);
  });
});
