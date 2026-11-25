import { test, expect, type Page } from '@playwright/test';
import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import { existsSync } from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');

const allowDgpu = process.env.DEEPGRAPH_ALLOW_DGPU === '1';
const expectedPowerPref: 'high-performance' | 'low-power' = allowDgpu
    ? 'high-performance'
    : 'low-power';

function resolvePythonBin(): string | null {
    const candidates =
        process.platform === 'win32'
            ? [path.join(REPO_ROOT, '.venv', 'Scripts', 'python.exe'), 'python']
            : [path.join(REPO_ROOT, '.venv', 'bin', 'python'), 'python3', 'python'];
    for (const bin of candidates) {
        if (bin.includes(path.sep) && !existsSync(bin)) continue;
        const probe = spawnSync(
            bin,
            ['-c', 'import marimo, pydeepgraph; print(pydeepgraph.__version__)'],
            { cwd: REPO_ROOT, encoding: 'utf-8', timeout: 10000 }
        );
        if (probe.status === 0) return bin;
    }
    return null;
}

function resolveSampleArchive(filename: string): string | null {
    const candidates = [
        path.resolve(REPO_ROOT, '..', 'arrowtiles', 'samples', filename),
        path.resolve(REPO_ROOT, '..', 'duckdb-arrowtiles', 'samples', filename),
        path.resolve(REPO_ROOT, 'public', filename),
    ];
    for (const c of candidates) {
        if (existsSync(c)) return c;
    }
    return null;
}

async function getFreePort(): Promise<number> {
    return await new Promise((resolve, reject) => {
        const srv = net.createServer();
        srv.listen(0, '127.0.0.1', () => {
            const addr = srv.address();
            const port = typeof addr === 'object' && addr ? addr.port : 0;
            srv.close((err) => (err ? reject(err) : resolve(port)));
        });
        srv.on('error', reject);
    });
}

async function waitForHttpReady(url: string, timeoutMs = 30000): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        try {
            const res = await fetch(url);
            if (res.ok) return;
        } catch {
            // Still booting
        }
        await new Promise((r) => setTimeout(r, 250));
    }
    throw new Error(`Timed out waiting for Marimo server at ${url}`);
}

function stopProcessTree(proc: ChildProcess | null): void {
    if (!proc || proc.pid === undefined) return;
    try {
        if (process.platform === 'win32') {
            spawnSync('taskkill', ['/PID', String(proc.pid), '/T', '/F'], {
                stdio: 'ignore',
            });
        } else {
            proc.kill('SIGTERM');
        }
    } catch {
        // Ignore already-exited process
    }
}

async function countNonBgCanvasPixels(page: Page, b64Png: string): Promise<number> {
    return await page.evaluate(async (b64) => {
        const res = await fetch(`data:image/png;base64,${b64}`);
        const bmp = await createImageBitmap(await res.blob());
        const c = new OffscreenCanvas(bmp.width, bmp.height);
        const ctx = c.getContext('2d')!;
        ctx.drawImage(bmp, 0, 0);
        const d = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
        let lit = 0;
        for (let i = 0; i < d.length; i += 4) {
            if (Math.abs(d[i] - 17) > 5 || Math.abs(d[i + 1] - 17) > 5 || Math.abs(d[i + 2] - 17) > 5) {
                lit++;
            }
        }
        return lit;
    }, b64Png);
}

test.describe('Step 3B — pydeepgraph AnyWidget + Marimo Reactive DuckDB kNN & Gaia E2E (Intel iGPU)', () => {
    test('Verifies LMSys exact DuckDB 10-NN + adaptive ego-zoom + Clear + SQL bitmasks (comm & http) and Gaia optical Sky / HR transitions', async ({
        page,
    }) => {
        test.setTimeout(180000);

        const pyBin = resolvePythonBin();
        const lmsysSamplePath = resolveSampleArchive('lmsys_sample.arrowtiles');
        const gaiaSamplePath = resolveSampleArchive('gaia_sample.arrowtiles');
        test.skip(
            !pyBin || !lmsysSamplePath || !gaiaSamplePath,
            'Python .venv with marimo + pydeepgraph or sample archives unavailable'
        );

        const pageErrors: string[] = [];
        const consoleErrors: string[] = [];
        page.on('pageerror', (e) => pageErrors.push(e.message || String(e)));
        page.on('console', (m) => {
            if (m.type() === 'error') {
                const txt = m.text();
                if (!txt.includes('favicon')) consoleErrors.push(txt);
            }
        });

        // =========================================================================
        // 0. MANDATORY INTEL iGPU PRE-CHECK BEFORE LAUNCHING MARIMO (0 B ON RTX 5090)
        // =========================================================================
        await page.route('**/__step3b_igpu_guard.html', async (route) => {
            await route.fulfill({
                status: 200,
                contentType: 'text/html; charset=utf-8',
                body: '<!doctype html><html><body>guard</body></html>',
            });
        });
        await page.goto('/__step3b_igpu_guard.html', { waitUntil: 'domcontentloaded' });
        const igpuGuard = await page.evaluate(async (pref) => {
            const adapter = await navigator.gpu.requestAdapter({ powerPreference: pref });
            if (!adapter) throw new Error(`requestAdapter(${pref}) returned null`);
            const info: any = (adapter as any).info || {};
            return {
                vendor: String(info.vendor || '').toLowerCase(),
                architecture: String(info.architecture || ''),
            };
        }, expectedPowerPref);
        if (!allowDgpu) {
            expect(igpuGuard.vendor).not.toContain('nvidia');
        }

        const lmsysNotebook = path.join(REPO_ROOT, 'python', 'examples', 'marimo_lmsys_mosaic.py');
        const gaiaNotebook = path.join(REPO_ROOT, 'python', 'examples', 'marimo_gaia_1_35b.py');

        // =========================================================================
        // PART A: LMSys with `PYDEEPGRAPH_TRANSPORT=comm` (<= 512 KiB Binary WS!)
        // =========================================================================
        const commPort = await getFreePort();
        const commUrl = `http://127.0.0.1:${commPort}`;
        let marimoCommProc: ChildProcess | null = null;

        try {
            marimoCommProc = spawn(
                pyBin!,
                [
                    '-m',
                    'marimo',
                    'run',
                    lmsysNotebook,
                    '--headless',
                    '--host',
                    '127.0.0.1',
                    '--port',
                    String(commPort),
                    '--no-token',
                ],
                {
                    cwd: REPO_ROOT,
                    env: {
                        ...process.env,
                        PYDEEPGRAPH_LMSYS_ARCHIVE: lmsysSamplePath!,
                        PYDEEPGRAPH_POWER_PREFERENCE: expectedPowerPref,
                        PYDEEPGRAPH_TRANSPORT: 'comm',
                    },
                    stdio: ['ignore', 'pipe', 'pipe'],
                }
            );

            await waitForHttpReady(commUrl, 40000);
            await page.goto(commUrl, { waitUntil: 'domcontentloaded' });

            const widgetRoot = page.locator('.dg-anywidget-root').first();
            await expect(widgetRoot).toBeVisible({ timeout: 45000 });
            await expect(widgetRoot).toHaveAttribute('data-dg-transport', 'comm');

            await expect
                .poll(
                    async () =>
                        await widgetRoot.evaluate((el: any) => el.__dgTelemetry?.activePoints ?? 0),
                    { timeout: 30000 }
                )
                .toBeGreaterThan(10000);

            const canvas = widgetRoot.locator('canvas').first();
            const pngBytes = await canvas.screenshot({ type: 'png' });
            expect(await countNonBgCanvasPixels(page, pngBytes.toString('base64'))).toBeGreaterThan(
                1000
            );

            // 1. Hover across the canvas MUST NOT trigger a reactive selection_sql update
            const box = await canvas.boundingBox();
            expect(box).not.toBeNull();
            await page.mouse.move(box!.x + box!.width * 0.45, box!.y + box!.height * 0.45);
            await page.mouse.move(box!.x + box!.width * 0.55, box!.y + box!.height * 0.52);
            await page.waitForTimeout(250);
            await expect(page.getByText('Committed Brush / Lasso / Click Selection')).toHaveCount(0);

            // 2. Click a rendered LMSys dot -> cold `record_summary` prioritized FIRST in badge (`I3`)
            //    + exact DuckDB 10-NN Neighbor Verification Table (`B2`)
            const targetHit = await widgetRoot.evaluate(async (el: any) => {
                const plot = el.__dgPlot;
                const c = el.querySelector('canvas')!;
                const w = c.clientWidth;
                const h = c.clientHeight;
                for (let gx = 0.35; gx <= 0.65; gx += 0.05) {
                    for (let gy = 0.35; gy <= 0.65; gy += 0.05) {
                        const hit = await plot.pickAt(w * gx, h * gy);
                        if (hit) {
                            return { cssX: w * gx, cssY: h * gy, rowId: hit.rowId };
                        }
                    }
                }
                return null;
            });
            expect(targetHit).not.toBeNull();

            await page.mouse.click(box!.x + targetHit!.cssX, box!.y + targetHit!.cssY);

            const inspectBadge = widgetRoot.locator('.dg-anywidget-inspect');
            await expect(inspectBadge).toBeVisible({ timeout: 10000 });
            await expect(inspectBadge).toContainText(`Selected #${targetHit!.rowId}`);
            await expect(inspectBadge).toContainText('record_summary:');
            const badgeText = (await inspectBadge.textContent()) ?? '';
            const badgeLines = badgeText.split('\n');
            expect(badgeLines[1]?.startsWith('record_summary:')).toBe(true);

            await expect(
                page.getByRole('heading', {
                    name: `Ground-Truth 1-Hop kNN Neighbors for row_id=${targetHit!.rowId}`,
                })
            ).toBeVisible({ timeout: 15000 });

            // 3. Click "Zoom to Clicked Ego-Neighborhood" (`B2` #1) -> adaptive camera zoom >= 8x!
            await page.getByRole('button', { name: 'Zoom to Clicked Ego-Neighborhood' }).click();
            await expect
                .poll(
                    async () =>
                        await widgetRoot.evaluate(
                            (el: any) => el.__dgPlot?.getViewport?.()?.zoom ?? 0
                        ),
                    { timeout: 10000 }
                )
                .toBeGreaterThanOrEqual(8.0);

            // Click "Reset Full View" -> returns near fit zoom (< 3x)
            await page.getByRole('button', { name: 'Reset Full View' }).click();
            await expect
                .poll(
                    async () =>
                        await widgetRoot.evaluate(
                            (el: any) => el.__dgPlot?.getViewport?.()?.zoom ?? 99
                        ),
                    { timeout: 10000 }
                )
                .toBeLessThan(3.0);

            // 4. Toolbar `Clear` hides `.dg-anywidget-inspect` cleanly (`I3`)
            await widgetRoot.locator('[data-dg-action="clear"]').click();
            await expect(inspectBadge).toBeHidden({ timeout: 5000 });

            // 5. Switch to 2D Brush mode and drag -> server-validated DuckDB SQL on pointerup (`R2 #3`)
            await widgetRoot.locator('[data-dg-mode="brush"]').click();
            const bx0 = box!.x + box!.width * 0.35;
            const by0 = box!.y + box!.height * 0.35;
            const bx1 = box!.x + box!.width * 0.65;
            const by1 = box!.y + box!.height * 0.65;

            await page.mouse.move(bx0, by0);
            await page.mouse.down();
            await page.mouse.move((bx0 + bx1) * 0.5, (by0 + by1) * 0.5, { steps: 4 });
            await page.mouse.move(bx1, by1, { steps: 4 });
            await page.mouse.up();

            await expect(page.getByText('Server-Validated DuckDB SQL:').first()).toBeVisible({
                timeout: 15000,
            });
            await expect(page.getByText(/"x" BETWEEN .* AND .*/).first()).toBeVisible({
                timeout: 15000,
            });

            await widgetRoot.locator('[data-dg-action="clear"]').click();
            await expect(page.getByText('Server-Validated DuckDB SQL:')).toHaveCount(0, {
                timeout: 15000,
            });

            // 6. Parameterized DuckDB search (`filter_by_sql` -> binary `u32` `row_id` bitmask -> WebGPU)
            const searchInput = page.locator('input[placeholder*="mixtral"]').first();
            await searchInput.fill('mixtral');
            await searchInput.press('Enter');
            await searchInput.blur();

            await expect(page.getByText(/Active DuckDB .* Bitmask Filter:/).first()).toBeVisible({
                timeout: 15000,
            });
            await expect(page.getByText(/matching nodes for/).first()).toBeVisible({
                timeout: 15000,
            });
            await expect
                .poll(
                    async () =>
                        await widgetRoot.evaluate((el: any) =>
                            Boolean(el.__dgPlot?.filterResources?.hasSlotBitmask)
                        ),
                    { timeout: 10000 }
                )
                .toBe(true);

            await page.goto('/__step3b_igpu_guard.html', { waitUntil: 'domcontentloaded' });
        } finally {
            stopProcessTree(marimoCommProc);
        }

        // =========================================================================
        // PART B: Gaia Cruiser (`marimo_gaia_1_35b.py`) on `http` Fast-Path (`B1` & `B6`)
        // =========================================================================
        const gaiaPort = await getFreePort();
        const gaiaUrl = `http://127.0.0.1:${gaiaPort}`;
        let marimoGaiaProc: ChildProcess | null = null;
        try {
            marimoGaiaProc = spawn(
                pyBin!,
                [
                    '-m',
                    'marimo',
                    'run',
                    gaiaNotebook,
                    '--headless',
                    '--host',
                    '127.0.0.1',
                    '--port',
                    String(gaiaPort),
                    '--no-token',
                ],
                {
                    cwd: REPO_ROOT,
                    env: {
                        ...process.env,
                        PYDEEPGRAPH_GAIA_ARCHIVE: gaiaSamplePath!,
                        PYDEEPGRAPH_POWER_PREFERENCE: expectedPowerPref,
                        PYDEEPGRAPH_TRANSPORT: 'auto',
                    },
                    stdio: ['ignore', 'pipe', 'pipe'],
                }
            );

            await waitForHttpReady(gaiaUrl, 40000);
            await page.goto(gaiaUrl, { waitUntil: 'domcontentloaded' });

            const gaiaRoot = page.locator('.dg-anywidget-root').first();
            await expect(gaiaRoot).toBeVisible({ timeout: 45000 });
            await expect(gaiaRoot).toHaveAttribute('data-dg-transport', 'http');

            await expect
                .poll(
                    async () =>
                        await gaiaRoot.evaluate((el: any) => el.__dgTelemetry?.activePoints ?? 0),
                    { timeout: 30000 }
                )
                .toBeGreaterThan(10000);

            // Verify `pipeline === 'optical'` and non-zero lit stars on Galactic Sky Map (`B1`)
            const skyState = await gaiaRoot.evaluate((el: any) => el.__dgPlot?.getViewState?.());
            expect(skyState?.pipeline).toBe('optical');

            const gaiaCanvas = gaiaRoot.locator('canvas').first();
            const skyPng = await gaiaCanvas.screenshot({ type: 'png' });
            expect(await countNonBgCanvasPixels(page, skyPng.toString('base64'))).toBeGreaterThan(
                500
            );

            // Brush Sky region with `max_selected_row_ids=0` -> compact `x_hammer` SQL (`B6`)
            const gbox = await gaiaCanvas.boundingBox();
            expect(gbox).not.toBeNull();
            await gaiaRoot.locator('[data-dg-mode="brush"]').click();
            await page.mouse.move(gbox!.x + gbox!.width * 0.3, gbox!.y + gbox!.height * 0.3);
            await page.mouse.down();
            await page.mouse.move(gbox!.x + gbox!.width * 0.7, gbox!.y + gbox!.height * 0.7, {
                steps: 4,
            });
            await page.mouse.up();

            await expect(page.getByText(/"x_hammer" BETWEEN .* AND .*/).first()).toBeVisible({
                timeout: 15000,
            });
            await gaiaRoot.locator('[data-dg-action="clear"]').click();

            // Toggle to Hertzsprung-Russell Diagram -> switches to cartesian + flipY + lit HR stars
            await page.getByText('Hertzsprung–Russell Diagram').first().click();
            await expect
                .poll(
                    async () =>
                        await gaiaRoot.evaluate((el: any) => {
                            const vs = el.__dgPlot?.getViewState?.();
                            return vs ? `${vs.pipeline}:${vs.flipY}` : '';
                        }),
                    { timeout: 15000 }
                )
                .toBe('cartesian:true');

            await page.waitForTimeout(2200); // Allow 2s WGSL transition to settle
            const hrPng = await gaiaCanvas.screenshot({ type: 'png' });
            expect(await countNonBgCanvasPixels(page, hrPng.toString('base64'))).toBeGreaterThan(
                200
            );

            // Toggle back HR -> Galactic Sky Map: preserves HR `previousConfig` (no vertical-flip clobber!)
            await page.getByText('Galactic Sky Map').first().click();
            await expect
                .poll(
                    async () =>
                        await gaiaRoot.evaluate((el: any) => {
                            const plot = el.__dgPlot;
                            const vs = plot?.getViewState?.();
                            const prevX = plot?.tileStore?.previousConfig?.x?.field ?? '';
                            const prevY = plot?.tileStore?.previousConfig?.y?.field ?? '';
                            return vs ? `${vs.pipeline}:${vs.flipY}:${prevX}:${prevY}` : '';
                        }),
                    { timeout: 15000 }
                )
                .toBe('optical:false:bp_rp:phot_g_mean_mag');

            await page.waitForTimeout(2200); // Allow 2s HR -> Sky WGSL transition to settle
            const skyBackPng = await gaiaCanvas.screenshot({ type: 'png' });
            expect(
                await countNonBgCanvasPixels(page, skyBackPng.toString('base64'))
            ).toBeGreaterThan(500);

            await page.goto('/__step3b_igpu_guard.html', { waitUntil: 'domcontentloaded' });
        } finally {
            stopProcessTree(marimoGaiaProc);
        }

        expect(pageErrors).toEqual([]);
        expect(consoleErrors).toEqual([]);
    });
});
