import { defineConfig, type Plugin } from 'vite';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function inlineWidgetWorkerPlugin(): Plugin {
    return {
        name: 'deepgraph-inline-widget-worker',
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
                        '[inlineWidgetWorkerPlugin] Failed to locate pmtiles.worker-*.js in widget bundle'
                    );
                }

                const widgetItem = bundle['widget.js'];
                if (!widgetItem || widgetItem.type !== 'chunk') {
                    throw new Error('[inlineWidgetWorkerPlugin] Missing widget.js chunk in bundle');
                }

                const workerCtorRe =
                    /new\s+Worker\(\s*new\s+URL\([\s\S]*?pmtiles\.worker-[^"']+?\.js[\s\S]*?\)\s*,\s*\{\s*type:\s*["']module["']\s*\}\s*\)/g;
                let matchCount = 0;
                const rewritten = widgetItem.code.replace(workerCtorRe, () => {
                    matchCount++;
                    return 'new Worker(__getDgWidgetWorkerUrl(), { type: "module" })';
                });

                if (matchCount === 0 || rewritten.includes('pmtiles.worker-')) {
                    throw new Error(
                        `[inlineWidgetWorkerPlugin] Failed to rewrite Worker URL constructor in widget.js (matches=${matchCount})`
                    );
                }

                const helperPrefix =
                    `let __dgWidgetWorkerBlobUrl = null;\n` +
                    `function __getDgWidgetWorkerUrl() {\n` +
                    `  if (!__dgWidgetWorkerBlobUrl) {\n` +
                    `    __dgWidgetWorkerBlobUrl = URL.createObjectURL(new Blob([${JSON.stringify(workerSrc)}], { type: "text/javascript" }));\n` +
                    `  }\n` +
                    `  return __dgWidgetWorkerBlobUrl;\n` +
                    `}\n`;

                widgetItem.code = helperPrefix + rewritten;
                delete bundle[workerKey];
            },
        },
    };
}

export default defineConfig({
    base: './',
    plugins: [inlineWidgetWorkerPlugin()],
    worker: {
        format: 'es',
        rollupOptions: {
            external: [],
        },
    },
    build: {
        outDir: path.resolve(__dirname, 'python/pydeepgraph/_static'),
        emptyOutDir: false,
        copyPublicDir: false,
        sourcemap: false,
        target: 'es2022',
        lib: {
            entry: path.resolve(__dirname, 'src/widget/widget_entry.ts'),
            formats: ['es'],
            fileName: () => 'widget.js',
        },
    },
});
