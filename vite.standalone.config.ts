import { defineConfig, type Plugin } from 'vite';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * Inlines the emitted `pmtiles.worker-*.js` bundle into `dist/standalone.js` as a lazy,
 * cached `Blob` URL (`__getDgWorkerUrl()`) and deletes the duplicate worker entry from
 * the in-memory Rollup/Rolldown bundle ONLY (preserving `dist/index.js`'s worker chunk on disk).
 *
 * This makes `dist/standalone.js` a true single-file ES module that works from `blob:` URLs,
 * Jupyter `anywidget` ESM cells, Observable/Marimo notebooks, and cross-origin CDNs without
 * secondary `assets/pmtiles.worker-*.js` requests.
 */
function inlineStandaloneWorkerPlugin(): Plugin {
    return {
        name: 'deepgraph-inline-standalone-worker',
        enforce: 'post',
        generateBundle: {
            order: 'post',
            handler(_options, bundle) {
                let workerKey: string | null = null;
                let workerSrc: string | null = null;

                for (const [key, item] of Object.entries(bundle)) {
                    const base = path.posix.basename(key.replace(/\\/g, '/'));
                    if (base.startsWith('pmtiles.worker-') && base.endsWith('.js')) {
                        workerKey = key;
                        if (item.type === 'chunk') {
                            workerSrc = item.code;
                        } else {
                            workerSrc =
                                typeof item.source === 'string'
                                    ? item.source
                                    : new TextDecoder().decode(item.source);
                        }
                        break;
                    }
                }

                if (!workerKey || !workerSrc) {
                    throw new Error(
                        '[inlineStandaloneWorkerPlugin] Failed to locate pmtiles.worker-*.js in standalone bundle'
                    );
                }

                const standaloneItem = bundle['standalone.js'];
                if (!standaloneItem || standaloneItem.type !== 'chunk') {
                    throw new Error(
                        '[inlineStandaloneWorkerPlugin] Missing standalone.js chunk in bundle'
                    );
                }

                const workerCtorRe =
                    /new\s+Worker\(\s*new\s+URL\([\s\S]*?pmtiles\.worker-[^"']+?\.js[\s\S]*?\)\s*,\s*\{\s*type:\s*["']module["']\s*\}\s*\)/g;
                let matchCount = 0;
                const rewritten = standaloneItem.code.replace(workerCtorRe, () => {
                    matchCount++;
                    return 'new Worker(__getDgWorkerUrl(), { type: "module" })';
                });

                if (matchCount === 0 || rewritten.includes('pmtiles.worker-')) {
                    throw new Error(
                        `[inlineStandaloneWorkerPlugin] Failed to rewrite Worker URL constructor in standalone.js (matches=${matchCount})`
                    );
                }

                const helperPrefix =
                    `let __dgWorkerBlobUrl = null;\n` +
                    `function __getDgWorkerUrl() {\n` +
                    `  if (!__dgWorkerBlobUrl) {\n` +
                    `    __dgWorkerBlobUrl = URL.createObjectURL(new Blob([${JSON.stringify(workerSrc)}], { type: "text/javascript" }));\n` +
                    `  }\n` +
                    `  return __dgWorkerBlobUrl;\n` +
                    `}\n`;

                standaloneItem.code = helperPrefix + rewritten;
                delete bundle[workerKey];
            },
        },
    };
}

/**
 * Self-Contained Single-File Standalone ES-Module Build (`v0.1.1 — dist/standalone.js`).
 *
 * - Bundles all runtime streaming primitives (`pmtiles`, `p-queue`, `fflate`) and inlines the
 *   `@uwdata/flechette` + `@bokuweb/zstd-wasm` worker as a cached `Blob` URL so zero-build
 *   `<script type="module">` HTML pages, `anywidget` notebooks, and `blob:` module imports need
 *   only a single file.
 * - Runs immediately after `vite.lib.config.ts` (`emptyOutDir: false`) into `dist/`.
 */
export default defineConfig({
    base: './',
    plugins: [inlineStandaloneWorkerPlugin()],
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
