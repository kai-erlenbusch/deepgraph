import type { ComputeRasterizer } from '../core/ComputeRasterizer.ts';
import type { EdgeOverlay } from '../core/EdgeOverlay.ts';
import type { AxisProjectionContext } from '../core/FilterGpuResources.ts';
import type { HighlightMarkerSpec, HighlightOverlay } from '../core/HighlightOverlay.ts';
import type { Renderer } from '../core/Renderer.ts';
import type { SelectionOverlay } from '../core/SelectionOverlay.ts';
import type { TileGpuStore } from '../core/TileGpuStore.ts';
import type { BoundingBox, TileData } from '../PMTilesClient.ts';
import type { DatasetSession } from './DatasetSession.ts';
import type { DeepgraphFilter, TelemetrySnapshot } from './types.ts';
import type { ViewState } from './ViewState.ts';

export class RenderLoop {
    private rendererWrapper: Renderer;
    private tileStore: TileGpuStore;
    private computeRasterizer: ComputeRasterizer;
    private edgeOverlay: EdgeOverlay;
    private selectionOverlay: SelectionOverlay;
    private highlightOverlay: HighlightOverlay;
    private getActiveFilter: () => DeepgraphFilter | null;
    private isMarkerVisible: (marker: HighlightMarkerSpec) => boolean;
    private getProjectionContexts: (session: DatasetSession) => {
        current: AxisProjectionContext;
        previous: AxisProjectionContext | null;
    };
    private getExtraGpuBytes: () => number;
    private viewState: ViewState;
    private getSession: () => DatasetSession | null;
    private onTelemetry: (snapshot: TelemetrySnapshot) => void;
    private onError: (err: Error) => void;

    private activeVisibleTiles: TileData[] = [];
    private hasPendingUpdates = false;
    private overlayDirty = false;
    private lastCamHash = '';
    private currentZDisplay = 0;
    private lastFrameTime = performance.now();
    private occlusionTimer: ReturnType<typeof setInterval> | null = null;
    private animatedEpochKyr = 400.0;
    private perfAccum = {
        frames: 0,
        getVisibleTiles: 0,
        updateTiles: 0,
        updateCam: 0,
        render: 0,
        totalFrame: 0,
        lastReport: performance.now(),
    };

    constructor(params: {
        rendererWrapper: Renderer;
        tileStore: TileGpuStore;
        computeRasterizer: ComputeRasterizer;
        edgeOverlay: EdgeOverlay;
        selectionOverlay: SelectionOverlay;
        highlightOverlay: HighlightOverlay;
        getActiveFilter: () => DeepgraphFilter | null;
        isMarkerVisible: (marker: HighlightMarkerSpec) => boolean;
        getProjectionContexts: (session: DatasetSession) => {
            current: AxisProjectionContext;
            previous: AxisProjectionContext | null;
        };
        getExtraGpuBytes: () => number;
        viewState: ViewState;
        getSession: () => DatasetSession | null;
        onTelemetry: (snapshot: TelemetrySnapshot) => void;
        onError: (err: Error) => void;
    }) {
        this.rendererWrapper = params.rendererWrapper;
        this.tileStore = params.tileStore;
        this.computeRasterizer = params.computeRasterizer;
        this.edgeOverlay = params.edgeOverlay;
        this.selectionOverlay = params.selectionOverlay;
        this.highlightOverlay = params.highlightOverlay;
        this.getActiveFilter = params.getActiveFilter;
        this.isMarkerVisible = params.isMarkerVisible;
        this.getProjectionContexts = params.getProjectionContexts;
        this.getExtraGpuBytes = params.getExtraGpuBytes;
        this.viewState = params.viewState;
        this.getSession = params.getSession;
        this.onTelemetry = params.onTelemetry;
        this.onError = params.onError;
    }

    public markDirty(): void {
        this.overlayDirty = true;
    }

    public syncEpoch(epochKyr: number): void {
        this.animatedEpochKyr = epochKyr;
    }

    public resetViewCache(): void {
        this.lastCamHash = '';
        this.activeVisibleTiles = [];
        this.hasPendingUpdates = false;
        this.overlayDirty = true;
        this.currentZDisplay = 0;
    }

    public start(): void {
        this.stop();
        this.lastFrameTime = performance.now();
        this.perfAccum.lastReport = this.lastFrameTime;

        const tickFrame = () => {
            try {
                const session = this.getSession();
                if (!session) return;

                const t0 = performance.now();
                const dtSec = Math.min(0.1, Math.max(0.0, (t0 - this.lastFrameTime) * 0.001));
                this.lastFrameTime = t0;
                this.overlayDirty = false;

                if (this.viewState.isFlowAnimated && this.tileStore.computeMode === 3) {
                    this.animatedEpochKyr = this.viewState.advanceFlowEpoch(
                        dtSec,
                        this.animatedEpochKyr
                    );
                    this.tileStore.trailTimeKyr =
                        this.viewState.snapshot().experimental.trailTimeKyr;
                }

                this.rendererWrapper.updateCameraUniforms();

                const cam = this.rendererWrapper.camera;
                const relZoom = cam.zoom / Math.max(1e-30, this.rendererWrapper.baseZoom);
                const currentZoom = Math.log2(Math.max(0.1, relZoom));
                let t1 = performance.now();

                const currentZ = Math.max(
                    0,
                    Math.min(
                        session.archiveMaxZoom,
                        Math.floor(currentZoom) + this.viewState.currentDetailOffset
                    )
                );
                const camHash = `${cam.position.x},${cam.position.y},${cam.zoom},${cam.left},${cam.right},${currentZ},${this.viewState.effectiveFlipY},${this.viewState.isSpatialX},${this.viewState.isSpatialY}`;

                if (
                    this.lastCamHash !== camHash ||
                    session.cacheChanged ||
                    this.hasPendingUpdates
                ) {
                    const frustumBox: BoundingBox = {
                        minX: cam.position.x + cam.left / cam.zoom,
                        maxX: cam.position.x + cam.right / cam.zoom,
                        minY: cam.position.y + cam.bottom / cam.zoom,
                        maxY: cam.position.y + cam.top / cam.zoom,
                    };

                    this.activeVisibleTiles = session.getVisibleTiles(
                        frustumBox,
                        currentZ,
                        this.viewState.effectiveFlipY,
                        this.viewState.isSpatialX,
                        this.viewState.isSpatialY
                    );

                    t1 = performance.now();
                    this.hasPendingUpdates = this.tileStore.updateVisibleTiles(
                        this.activeVisibleTiles
                    );
                    this.lastCamHash = camHash;
                    this.currentZDisplay = session.currentMaxZ;
                    session.cacheChanged = false;
                }

                const t2 = performance.now();
                this.tileStore.updateFrameState(
                    cam.zoom,
                    t2,
                    this.rendererWrapper.baseZoom
                );
                const projContexts = this.getProjectionContexts(session);
                const activeFilter = this.getActiveFilter();
                const t3 = performance.now();

                // Render order (`Step 3A`):
                // ComputeRasterizer -> EdgeOverlay -> SelectionOverlay -> HighlightOverlay (rings on top)
                this.computeRasterizer.render(
                    cam,
                    this.tileStore,
                    (encoder, targetView) => {
                        this.edgeOverlay.encodePass(
                            encoder,
                            targetView,
                            cam,
                            this.tileStore,
                            projContexts.current,
                            projContexts.previous
                        );
                        this.selectionOverlay.encodePass(
                            encoder,
                            targetView,
                            cam,
                            projContexts.current,
                            activeFilter?.rect,
                            activeFilter?.lasso,
                            this.tileStore.lightBackground
                        );
                        this.highlightOverlay.encodePass(
                            encoder,
                            targetView,
                            cam,
                            this.tileStore,
                            this.isMarkerVisible
                        );
                    },
                    projContexts.current
                );

                const t4 = performance.now();

                this.perfAccum.frames++;
                this.perfAccum.getVisibleTiles += t1 - t0;
                this.perfAccum.updateTiles += t2 - t1;
                this.perfAccum.updateCam += t3 - t2;
                this.perfAccum.render += t4 - t3;
                this.perfAccum.totalFrame += t4 - t0;

                if (t4 - this.perfAccum.lastReport >= 250) {
                    const p = this.perfAccum;
                    const frames = Math.max(1, p.frames);
                    const fps = p.frames * (1000 / Math.max(1, t4 - p.lastReport));
                    const cullMs = p.getVisibleTiles / frames;
                    const uploadMs = p.updateTiles / frames;
                    const renderMs = p.render / frames;
                    const totalFrameMs = p.totalFrame / frames;

                    this.perfAccum = {
                        frames: 0,
                        getVisibleTiles: 0,
                        updateTiles: 0,
                        updateCam: 0,
                        render: 0,
                        totalFrame: 0,
                        lastReport: t4,
                    };

                    let loadedPoints = 0;
                    for (const t of this.activeVisibleTiles) loadedPoints += t.numRows;
                    const activePoints = this.tileStore.getActivePointsCount();
                    const tileBytes = this.tileStore.getLiveTileGpuBytes();
                    const ownTotalBytes =
                        tileBytes +
                        this.computeRasterizer.getLiveGpuBytes() +
                        this.getExtraGpuBytes();
                    const cpuStats = session.getCpuMemoryStats();
                    const estimatedVramMB = tileBytes / (1024 * 1024);
                    const totalGpuMB = ownTotalBytes / (1024 * 1024);
                    const cpuPoolMB = cpuStats.totalRetainedCpuBytes / (1024 * 1024);

                    let netLatencyMs = 0;
                    let workerLatencyMs = 0;
                    let coldBootMs = 0;
                    let sloP95Ms = 0;
                    let sloBudgetMs = 50;
                    let isSloViolated = false;

                    const ft = session.getFetchTelemetry();
                    if (ft && ft.net.length > 0) {
                        const sumNet = ft.net.reduce((a, b) => a + b, 0);
                        const sumWorker = ft.worker.reduce((a, b) => a + b, 0);
                        netLatencyMs = sumNet / ft.net.length;
                        workerLatencyMs = sumWorker / Math.max(1, ft.worker.length);
                        coldBootMs = ft.coldBootMs;
                        sloP95Ms = ft.p95;
                        sloBudgetMs = ft.sloBudgetMs;
                        isSloViolated = ft.isSloViolated;
                    }

                    this.onTelemetry({
                        fps,
                        cameraZoom: cam.zoom,
                        currentZ: this.currentZDisplay ?? 0,
                        pyramidSummary: session.summary.pyramidSummary,
                        activeTiles: this.activeVisibleTiles.length,
                        estimatedVramMB,
                        totalGpuMB,
                        cpuPoolMB,
                        activePoints,
                        loadedPoints,
                        netLatencyMs,
                        workerLatencyMs,
                        coldBootMs,
                        sloP95Ms,
                        sloBudgetMs,
                        isSloViolated,
                        cullMs,
                        uploadMs,
                        renderMs,
                        totalFrameMs,
                        trailTimeKyr: this.tileStore.trailTimeKyr,
                    });
                }
            } catch (err) {
                const error = err instanceof Error ? err : new Error(String(err));
                console.error('Animation loop crash:', error);
                this.stop();
                this.onError(error);
            }
        };

        this.rendererWrapper.setAnimationLoop(tickFrame);

        this.occlusionTimer = setInterval(() => {
            const now = performance.now();
            if (now - this.lastFrameTime < 24) return;
            const session = this.getSession();
            if (!session) return;
            const cam = this.rendererWrapper.camera;
            const camPrefix = `${cam.position.x},${cam.position.y},${cam.zoom},${cam.left},${cam.right},`;
            const camMoved = !this.lastCamHash.startsWith(camPrefix);
            const nowSec = now * 0.001;
            let anyFading = false;
            for (let i = 0; i < this.tileStore.maxSlots; i++) {
                const s = this.tileStore.slots[i];
                if (s && s.visible && nowSec - s.spawnTime < 0.36) {
                    anyFading = true;
                    break;
                }
            }
            if (
                this.lastCamHash === '' ||
                camMoved ||
                session.cacheChanged ||
                this.hasPendingUpdates ||
                this.overlayDirty ||
                this.edgeOverlay.isPulseAnimated ||
                this.tileStore.isTransitioning ||
                anyFading
            ) {
                tickFrame();
                let guard = 0;
                while (this.hasPendingUpdates && guard++ < 8) {
                    this.hasPendingUpdates = this.tileStore.updateVisibleTiles(
                        this.activeVisibleTiles
                    );
                }
            }
        }, 16);
    }

    public stop(): void {
        if (this.occlusionTimer !== null) {
            clearInterval(this.occlusionTimer);
            this.occlusionTimer = null;
        }
        this.rendererWrapper.setAnimationLoop(null);
    }
}
