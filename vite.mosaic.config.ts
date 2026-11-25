import { defineConfig } from 'vite';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * Optional UW Mosaic Subpath Build (`v0.1.2 — dist/mosaic.js`).
 *
 * - Externalizes `@uwdata/mosaic-core`, `@uwdata/mosaic-sql`, and `deepgraph` so core
 *   `dist/index.js` and `dist/standalone.js` carry `0 B` of Mosaic / DuckDB-WASM weight.
 * - Runs immediately after `vite.standalone.config.ts` (`emptyOutDir: false`) into `dist/`.
 */
export default defineConfig({
    base: './',
    build: {
        outDir: 'dist',
        emptyOutDir: false,
        copyPublicDir: false,
        sourcemap: false,
        target: 'es2022',
        lib: {
            entry: path.resolve(__dirname, 'src/mosaic.ts'),
            formats: ['es'],
            fileName: () => 'mosaic.js',
        },
        rollupOptions: {
            external: [
                'deepgraph',
                '@uwdata/mosaic-core',
                '@uwdata/mosaic-sql',
                /^deepgraph\//,
                /^@uwdata\/mosaic-core\//,
                /^@uwdata\/mosaic-sql\//,
            ],
        },
    },
});
