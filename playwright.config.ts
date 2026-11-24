import { defineConfig } from '@playwright/test';

const allowDgpu = process.env.DEEPGRAPH_ALLOW_DGPU === '1';

export default defineConfig({
  testDir: './tests',
  outputDir: process.env.PLAYWRIGHT_OUTPUT_DIR || './test-results',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: 'http://127.0.0.1:5179',
    viewport: { width: 1280, height: 800 },
    trace: 'off',
  },
  projects: [
    {
      name: 'chrome-webgpu',
      use: {
        channel: 'chrome',
        headless: false,
        launchOptions: {
          args: [
            '--enable-unsafe-webgpu',
            '--ignore-gpu-blocklist',
            allowDgpu ? '--force_high_performance_gpu' : '--force_low_power_gpu',
            '--disable-background-timer-throttling',
            '--disable-renderer-backgrounding',
            '--disable-backgrounding-occluded-windows',
            '--disable-features=CalculateNativeWinOcclusion',
          ],
        },
      },
    },
  ],
  webServer: {
    command: 'npx vite --host 127.0.0.1 --port 5179 --strictPort',
    url: 'http://127.0.0.1:5179',
    reuseExistingServer: false,
    timeout: 30000,
  },
});
