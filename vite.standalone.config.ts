import { defineConfig } from 'vite';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * Self-Contained Standalone ES-Module Build (`Phase 8c — dist/standalone.js`).
 *
 * - Bundles all runtime streaming primitives (`pmtiles`, `p-queue`, `fflate`, `apache-arrow`,
 *   `@bokuweb/zstd-wasm`) so zero-build `<script type="module">` HTML pages (e.g. `examples/embed.html`)
 *   can `import { Deepgraph } from '../dist/standalone.js'` directly without an importmap or bundler.
 * - Runs immediately after `vite.lib.config.ts` (`emptyOutDir: false`) into `dist/`.
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
        emptyOutDir: false,
        copyPublicDir: false,
        sourcemap: false,
        target: 'es2022',
        lib: {
            entry: path.resolve(__dirname, 'src/index.ts'),
            formats: ['es'],
            fileName: () => 'standalone.js',
        },
    },
});
