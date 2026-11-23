import { Deepgraph, type DeepgraphPlot } from './index.ts';
import { ControlsPanel, TelemetryHud } from './demo/ControlsPanel.ts';
import { resolveDemoPreset } from './demo/gaiaPreset.ts';
import { installGpuStatsInstrumentation } from './demo/gpuStats.ts';

function resolveSafeInitialUrl(search: string): string {
    const params = new URLSearchParams(search);
    const queryFile = params.get('file');
    if (
        queryFile &&
        queryFile.startsWith('/') &&
        !queryFile.startsWith('//') &&
        !queryFile.includes('\\') &&
        !queryFile.includes('..') &&
        queryFile.endsWith('.arrowtiles')
    ) {
        return queryFile;
    }
    return '/gaia_full.arrowtiles';
}

async function initDemoApp() {
    const container = document.getElementById('app')!;
    const uiText = document.querySelector('#ui p') as HTMLElement;
    const copyBtn = document.getElementById('copy-metrics-btn');
    const hud = new TelemetryHud(uiText, copyBtn);

    if (!Deepgraph.isSupported()) {
        hud.setStatus('WebGPU is not supported by your browser.');
        return;
    }

    installGpuStatsInstrumentation();
    const urlParams = new URLSearchParams(window.location.search);
    const powerPreference =
        urlParams.get('power') === 'low' ? 'low-power' : 'high-performance';

    let plot: DeepgraphPlot;
    try {
        plot = await Deepgraph.create(container, { powerPreference });
    } catch (err) {
        hud.setStatus(
            err instanceof Error ? err.message : 'WebGPU is not supported by your browser.'
        );
        return;
    }

    const controls = new ControlsPanel(plot);
    plot.on('telemetry', (snap) => hud.update(snap));
    plot.on('error', (err) => hud.setError(err.message));

    async function loadDataset(source: string | File) {
        hud.setStatus('Loading dataset schema...');
        try {
            const summary = await plot.load(source, { preset: resolveDemoPreset });
            controls.rebuild(summary, summary.presetName === 'gaia-dr3');
            hud.setStatus('WebGPU is supported!<br/>Streaming Quadtree Tiles...', true);
        } catch (err: any) {
            if (err?.name === 'AbortError') {
                return;
            }
            throw err;
        }
    }

    const dropOverlay = document.getElementById('drop-overlay');
    if (dropOverlay) {
        window.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropOverlay.style.display = 'flex';
        });
        window.addEventListener('dragleave', (e) => {
            e.preventDefault();
            if (e.clientX === 0 && e.clientY === 0) {
                dropOverlay.style.display = 'none';
            }
        });
        window.addEventListener('drop', (e) => {
            e.preventDefault();
            dropOverlay.style.display = 'none';
            const file = e.dataTransfer?.files?.[0];
            if (file && file.name.endsWith('.arrowtiles')) {
                loadDataset(file).catch(console.error);
            }
        });
    }

    (window as any).loadDataset = loadDataset;
    (window as any).__viewer = {
        plot,
        loadDataset,
        getGui: () => controls.getGui(),
    };

    try {
        await loadDataset(resolveSafeInitialUrl(window.location.search));
    } catch (err) {
        console.warn('Initial dataset failed to load, waiting for drag and drop.', err);
        hud.setStatus('Drag and drop an .arrowtiles file to begin.');
    }
}

initDemoApp().catch(console.error);
