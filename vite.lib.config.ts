import { defineConfig } from 'vite';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * Vite ES-Module Library Build (`Phase 8 / 8c — Bundler Entry dist/index.js`).
 *
 * - Externalizes main-thread runtime dependencies (`pmtiles`, `p-queue`, `fflate`, `apache-arrow`,
 *   `@bokuweb/zstd-wasm`) so downstream Vite / Rollup / Webpack / esbuild consumers deduplicate
 *   shared packages cleanly instead of bundling duplicate copies of `apache-arrow`.
 * - Keeps the Web Worker chunk (`dist/assets/pmtiles.worker-*.js`) 100% self-contained
 *   (`worker.rollupOptions.external = []`) with inlined base64 `zstd.wasm` and relative
 *   `new URL("assets/pmtiles.worker-*.js", import.meta.url)` resolution so consumers need zero
 *   worker/WASM bundler plugins.
 * - For zero-build `<script type="module">` HTML pages, see `vite.standalone.config.ts` (`dist/standalone.js`).
 */
export default defineConfig({
    base: './',
    worker: {
        format: 'es',
        rollupOptions: {
            external: [],
        },
    },
    build: {
        outDir: 'dist',
        emptyOutDir: true,
        copyPublicDir: false,
        sourcemap: false,
        target: 'es2022',
        lib: {
            entry: path.resolve(__dirname, 'src/index.ts'),
            formats: ['es'],
            fileName: () => 'index.js',
        },
        rollupOptions: {
            external: [
                'pmtiles',
                'p-queue',
                'fflate',
                'apache-arrow',
                '@bokuweb/zstd-wasm',
                /^pmtiles\//,
                /^p-queue\//,
                /^fflate\//,
                /^apache-arrow\//,
                /^@bokuweb\/zstd-wasm\//,
            ],
        },
    },
});
