import { Renderer } from '../core/Renderer.ts';
import { ComputeRasterizer } from '../core/ComputeRasterizer.ts';
import { TileGpuStore } from '../core/TileGpuStore.ts';
import { PickingLayer, type GpuPickWinner } from '../core/PickingLayer.ts';
import { HighlightOverlay, type HighlightMarkerSpec } from '../core/HighlightOverlay.ts';
import { DatasetSession } from './DatasetSession.ts';
import { ViewState } from './ViewState.ts';
import { RenderLoop } from './RenderLoop.ts';
import {
    INTERNAL_DIAGNOSTICS,
    type InternalDiagnosticsHandle,
} from './diagnostics.ts';
import type {
    DatasetMetadataSummary,
    DeepgraphEncoding,
    DeepgraphEventMap,
    DeepgraphOptions,
    DeepgraphPlot,
    DeepgraphStatic,
    LoadDatasetOptions,
    PointHitEvent,
    RowAttributeResolver,
    TileSource,
    ViewOptionsPatch,
    ViewportState,
    ViewStateSnapshot,
} from './types.ts';

type ListenerSet<T> = Set<(payload: T) => void>;

const DEFAULT_FIT_ZOOM = 1.61;
const HOVER_THROTTLE_MS = 45;

export class ViewerLayer implements DeepgraphPlot {
    private rendererWrapper: Renderer;
    private tileStore: TileGpuStore;
    private computeRasterizer: ComputeRasterizer;
    private pickingLayer: PickingLayer;
    private highlightOverlay: HighlightOverlay;
    private viewState: ViewState;
    private renderLoop: RenderLoop;
    private session: DatasetSession | null = null;
    private maxTileCache: number;
    private defaultPreset?: DeepgraphOptions['preset'];
    private resolveRowAttributes?: RowAttributeResolver;
    private resolveRowAttributesOnHover: boolean;
    private loadGeneration = 0;
    private activeLoadController: AbortController | null = null;
    private hoverSeq = 0;
    private lastHoverDispatchMs = 0;
    private lastHoverWasNull = true;
    private activeHoverWinnerKey: string | null = null;
    private hoverAbortController: AbortController | null = null;
    private activePickAbortControllers = new Set<AbortController>();
    private destroyed = false;
    private device: GPUDevice | null = null;
    private boundUncapturedError: ((event: any) => void) | null = null;

    private listeners: {
        [K in keyof DeepgraphEventMap]: ListenerSet<DeepgraphEventMap[K]>;
    } = {
        click: new Set(),
        hover: new Set(),
        select: new Set(),
        telemetry: new Set(),
        load: new Set(),
        error: new Set(),
    };

    private constructor(
        rendererWrapper: Renderer,
        tileStore: TileGpuStore,
        computeRasterizer: ComputeRasterizer,
        pickingLayer: PickingLayer,
        highlightOverlay: HighlightOverlay,
        device: GPUDevice | null,
        options: DeepgraphOptions = {}
    ) {
        this.rendererWrapper = rendererWrapper;
        this.tileStore = tileStore;
        this.computeRasterizer = computeRasterizer;
        this.pickingLayer = pickingLayer;
        this.highlightOverlay = highlightOverlay;
        this.device = device;
        this.maxTileCache = options.maxTileCache ?? 250;
        this.defaultPreset = options.preset;
        this.resolveRowAttributes = options.resolveRowAttributes;
        this.resolveRowAttributesOnHover = Boolean(options.resolveRowAttributesOnHover);
        this.viewState = new ViewState(options.detailOffset ?? 2);

        this.tileStore.onSlotDisposed = (slot) => {
            this.computeRasterizer.invalidateSlot(slot);
            this.pickingLayer.invalidateSlot(slot);
        };

        this.rendererWrapper.onCanvasClick = (cssX, cssY) => {
            void this.handleCanvasClick(cssX, cssY);
        };

        this.rendererWrapper.onCanvasHover = (cssX, cssY) => {
            this.handleCanvasHover(cssX, cssY);
        };

        this.rendererWrapper.onCanvasPointerLeave = () => {
            this.clearHoverState(true);
        };

        this.renderLoop = new RenderLoop({
            rendererWrapper: this.rendererWrapper,
            tileStore: this.tileStore,
            computeRasterizer: this.computeRasterizer,
            highlightOverlay: this.highlightOverlay,
            getExtraGpuBytes: () =>
                this.pickingLayer.getLiveGpuBytes() + this.highlightOverlay.getLiveGpuBytes(),
            viewState: this.viewState,
            getSession: () => this.session,
            onTelemetry: (snap) => this.emit('telemetry', snap),
            onError: (err) => this.emit('error', err),
        });

        if (this.device) {
            this.boundUncapturedError = (event: any) => {
                const msg =
                    event?.error?.message ?? String(event?.error ?? 'Unknown WebGPU error');
                const err = new Error(`WebGPU VRAM Crash: ${msg}`);
                console.error('WebGPU uncapturederror:', event?.error);
                this.emit('error', err);
            };
            this.device.addEventListener('uncapturederror', this.boundUncapturedError);
            if (this.device.lost && typeof this.device.lost.then === 'function') {
                void this.device.lost.then((info: any) => {
                    if (!this.destroyed && info?.reason !== 'destroyed') {
                        this.renderLoop.stop();
                        this.emit(
                            'error',
                            new Error(
                                `WebGPU device lost (${info?.reason ?? 'unknown'}): ${
                                    info?.message ?? ''
                                }`
                            )
                        );
                    }
                });
            }
        }

        this.renderLoop.start();
    }

    public static isSupported(): boolean {
        return typeof navigator !== 'undefined' && 'gpu' in navigator && Boolean(navigator.gpu);
    }

    public static async create(
        container: HTMLElement,
        options: DeepgraphOptions = {}
    ): Promise<DeepgraphPlot> {
        if (!ViewerLayer.isSupported()) {
            throw new Error('WebGPU is not supported by your browser.');
        }
        const rendererWrapper = new Renderer(
            container,
            options.powerPreference ?? 'high-performance'
        );
        await rendererWrapper.init();

        const device = rendererWrapper.device;
        if (!device) {
            rendererWrapper.dispose();
            throw new Error('Failed to initialize WebGPU device.');
        }

        const maxTileCache = options.maxTileCache ?? 250;
        const maxSlots = Math.max(264, maxTileCache + 16);
        const initialBounds = { minX: -1.0, maxX: 1.0, minY: -1.0, maxY: 1.0 };

        let viewerRef: ViewerLayer | null = null;
        const tileStore = new TileGpuStore(device, initialBounds, maxSlots, (tile) => {
            viewerRef?.session?.releaseTileCpuBuffers(tile);
        });
        const computeRasterizer = new ComputeRasterizer(rendererWrapper, maxSlots);
        const pickingLayer = new PickingLayer(rendererWrapper, maxSlots);
        const highlightOverlay = new HighlightOverlay(rendererWrapper);

        const viewer = new ViewerLayer(
            rendererWrapper,
            tileStore,
            computeRasterizer,
            pickingLayer,
            highlightOverlay,
            device,
            options
        );
        viewerRef = viewer;

        if (options.url) {
            try {
                await viewer.load(options.url, {
                    preset: options.preset,
                    encoding: options.encoding,
                });
            } catch (err) {
                viewer.destroy();
                throw err;
            }
        }
        return viewer;
    }

    public async load(
        source: TileSource,
        options: LoadDatasetOptions = {}
    ): Promise<DatasetMetadataSummary> {
        if (this.destroyed) {
            throw new Error('Deepgraph instance has been destroyed.');
        }

        const gen = ++this.loadGeneration;
        if (this.activeLoadController) {
            this.activeLoadController.abort();
            this.activeLoadController = null;
        }
        const loadController = new AbortController();
        this.activeLoadController = loadController;

        this.clearHoverState(false);
        this.abortActivePickControllers();
        this.highlightOverlay.clear();

        if (this.session) {
            this.session.destroy();
            this.session = null;
            this.tileStore.clearAllTiles();
        }
        this.tileStore.clearDictionaries();
        this.renderLoop.resetViewCache();

        const activePreset = options.preset ?? this.defaultPreset;
        let session: DatasetSession;
        try {
            session = await DatasetSession.open(source, {
                preset: activePreset,
                maxTileCache: this.maxTileCache,
                signal: loadController.signal,
                onDictionariesLoaded: (dicts) => {
                    if (this.destroyed || this.session !== session) return;
                    this.tileStore.setDictionaries(dicts);
                    this.viewState.syncToTileStore(this.tileStore, session, false);
                    this.emit('load', session.summary);
                },
                onTileUnloaded: (tileId) => {
                    if (this.destroyed || this.session !== session) return;
                    this.tileStore.unloadTile(tileId);
                },
            });
        } catch (err) {
            if (this.destroyed) {
                throw new Error('Deepgraph instance was destroyed during dataset load.');
            }
            if (gen !== this.loadGeneration || loadController.signal.aborted) {
                throw new DOMException('Dataset load superseded', 'AbortError');
            }
            throw err;
        } finally {
            if (this.activeLoadController === loadController) {
                this.activeLoadController = null;
            }
        }

        // Guard against destroy() or a newer load() arriving while DatasetSession.open() was in flight!
        if (this.destroyed || gen !== this.loadGeneration) {
            session.destroy();
            if (this.destroyed) {
                throw new Error('Deepgraph instance was destroyed during dataset load.');
            }
            throw new DOMException('Dataset load superseded', 'AbortError');
        }

        this.session = session;
        this.tileStore.setRootBounds(session.summary.rootBounds);
        this.tileStore.colOrigins = { ...session.colOrigins };
        if (Object.keys(session.summary.categories).length > 0) {
            this.tileStore.setDictionaries(session.summary.categories);
        }

        this.viewState.initFromSession(session, session.preset, options.encoding);
        this.syncCameraBaseZoom();
        if (this.viewState.hasCustomWorldScale) {
            this.fitView();
        }
        this.viewState.syncToTileStore(this.tileStore, session, false);
        this.renderLoop.syncEpoch(this.viewState.snapshot().experimental.trailTimeKyr);

        this.emit('load', session.summary);
        return session.summary;
    }

    private clearHoverState(emitNull: boolean): void {
        this.hoverSeq++;
        if (this.hoverAbortController) {
            this.hoverAbortController.abort();
            this.hoverAbortController = null;
        }
        this.activeHoverWinnerKey = null;
        this.highlightOverlay.hoverMarker = null;
        if (emitNull) {
            if (!this.lastHoverWasNull) {
                this.lastHoverWasNull = true;
                this.emit('hover', null);
            }
        } else {
            this.lastHoverWasNull = true;
        }
    }

    private abortActivePickControllers(): void {
        for (const c of this.activePickAbortControllers) {
            c.abort();
        }
        this.activePickAbortControllers.clear();
    }

    private isExpectedPickAbort(
        err: unknown,
        expectedGen: number,
        signal?: AbortSignal
    ): boolean {
        if (this.destroyed || expectedGen !== this.loadGeneration || signal?.aborted) return true;
        if (!err || typeof err !== 'object') return false;
        const name = (err as any).name;
        const msg = String((err as any).message ?? '');
        return name === 'AbortError' || /PMTilesClient destroyed|instance.*destroyed/i.test(msg);
    }

    private async resolveWinnerToHitEvent(
        winner: GpuPickWinner,
        cssX: number,
        cssY: number,
        expectedGen: number,
        runResolver: boolean,
        signal?: AbortSignal
    ): Promise<{ hit: PointHitEvent; marker: Omit<HighlightMarkerSpec, 'kind'> } | null> {
        const session = this.session;
        if (!session || this.destroyed || expectedGen !== this.loadGeneration || signal?.aborted) {
            return null;
        }

        const decoded = await session.decodeTileRow(
            winner.z,
            winner.x,
            winner.y,
            winner.rawRowIdx,
            this.tileStore.categoricalDictionaries,
            signal
        );
        if (
            !decoded ||
            this.destroyed ||
            expectedGen !== this.loadGeneration ||
            signal?.aborted
        ) {
            return null;
        }

        let finalAttrs: Record<string, unknown> = { ...decoded.attributes };
        if (runResolver && this.resolveRowAttributes) {
            if (this.destroyed || expectedGen !== this.loadGeneration || signal?.aborted) {
                return null;
            }
            const enriched = await this.resolveRowAttributes(
                decoded.rowId,
                { ...decoded.attributes },
                decoded.rowIdBigInt,
                signal
            );
            if (this.destroyed || expectedGen !== this.loadGeneration || signal?.aborted) {
                return null;
            }
            if (enriched && typeof enriched === 'object') {
                finalAttrs = { ...decoded.attributes, ...enriched };
            }
        }

        const vs = this.viewState.snapshot();
        const scale = Math.pow(2.0, -winner.z);
        const xNorm = (winner.x + decoded.xU16 / 65535.0) * scale;
        const yNorm = (winner.y + decoded.yU16 / 65535.0) * scale;
        const db = session.summary.dataBounds;
        const rb = session.summary.rootBounds;
        const effFlip = vs.effectiveFlipY;

        const spatialX = db
            ? db.min_x + xNorm * (db.max_x - db.min_x)
            : rb.minX + xNorm * (rb.maxX - rb.minX);
        const spatialY = db
            ? db.max_y - yNorm * (db.max_y - db.min_y)
            : effFlip
              ? rb.minY + yNorm * (rb.maxY - rb.minY)
              : rb.maxY - yNorm * (rb.maxY - rb.minY);

        const resolveAxisHitCoord = (field: string | null, fallbackSpatial: number): number => {
            if (!field) return fallbackSpatial;
            const raw = finalAttrs[field];
            if (typeof raw === 'number' && Number.isFinite(raw)) return raw;
            if (typeof raw === 'string') {
                const cats = this.tileStore.categoricalDictionaries[field];
                const idx = cats ? cats.indexOf(raw) : -1;
                if (idx >= 0) return idx;
            }
            return fallbackSpatial;
        };

        const dataX = resolveAxisHitCoord(vs.xField, spatialX);
        const dataY = resolveAxisHitCoord(vs.yField, spatialY);

        const hit: PointHitEvent = {
            rowId: decoded.rowId,
            ...(decoded.rowIdBigInt !== undefined ? { rowIdBigInt: decoded.rowIdBigInt } : {}),
            screenX: cssX,
            screenY: cssY,
            dataX,
            dataY,
            attributes: finalAttrs,
        };

        return {
            hit,
            marker: {
                rowId: decoded.rowId,
                ...(decoded.rowIdBigInt !== undefined ? { rowIdBigInt: decoded.rowIdBigInt } : {}),
                rawRowIdx: winner.rawRowIdx,
                tileZ: winner.z,
                tileX: winner.x,
                tileY: winner.y,
                xU16: decoded.xU16,
                yU16: decoded.yU16,
                attributes: finalAttrs,
            },
        };
    }

    public async pickAt(cssX: number, cssY: number): Promise<PointHitEvent | null> {
        if (!this.session || this.destroyed) return null;
        const gen = this.loadGeneration;
        const controller = new AbortController();
        this.activePickAbortControllers.add(controller);
        try {
            const winner = await this.pickingLayer.pick(
                cssX,
                cssY,
                this.rendererWrapper.camera,
                this.tileStore
            );
            if (!winner || this.destroyed || gen !== this.loadGeneration || controller.signal.aborted) {
                return null;
            }
            const resolved = await this.resolveWinnerToHitEvent(
                winner,
                cssX,
                cssY,
                gen,
                true,
                controller.signal
            );
            return resolved ? resolved.hit : null;
        } catch (err) {
            if (this.isExpectedPickAbort(err, gen, controller.signal)) {
                return null;
            }
            const errorObj = err instanceof Error ? err : new Error(String(err));
            this.emit('error', errorObj);
            return null;
        } finally {
            this.activePickAbortControllers.delete(controller);
        }
    }

    private async handleCanvasClick(cssX: number, cssY: number): Promise<void> {
        if (!this.session || this.destroyed) return;
        const gen = this.loadGeneration;
        const controller = new AbortController();
        this.activePickAbortControllers.add(controller);
        try {
            const winner = await this.pickingLayer.pick(
                cssX,
                cssY,
                this.rendererWrapper.camera,
                this.tileStore
            );
            if (this.destroyed || gen !== this.loadGeneration || controller.signal.aborted) return;

            if (!winner) {
                this.highlightOverlay.selectedMarkers = [];
                this.emit('click', null);
                this.emit('select', { points: [] });
                return;
            }

            const resolved = await this.resolveWinnerToHitEvent(
                winner,
                cssX,
                cssY,
                gen,
                true,
                controller.signal
            );
            if (!resolved || this.destroyed || gen !== this.loadGeneration || controller.signal.aborted) {
                return;
            }

            this.highlightOverlay.selectedMarkers = [
                {
                    ...resolved.marker,
                    kind: 'selected',
                },
            ];
            this.emit('click', resolved.hit);
            this.emit('select', { points: [resolved.hit] });
        } catch (err) {
            if (this.isExpectedPickAbort(err, gen, controller.signal)) {
                return;
            }
            const errorObj = err instanceof Error ? err : new Error(String(err));
            this.emit('error', errorObj);
        } finally {
            this.activePickAbortControllers.delete(controller);
        }
    }

    private handleCanvasHover(cssX: number, cssY: number): void {
        if (!this.session || this.destroyed) return;
        if (this.listeners.hover.size === 0) return;

        const now = performance.now();
        if (now - this.lastHoverDispatchMs < HOVER_THROTTLE_MS || this.pickingLayer.isBusy) {
            return;
        }
        this.lastHoverDispatchMs = now;
        const seq = ++this.hoverSeq;
        const gen = this.loadGeneration;

        void (async () => {
            try {
                const winner = await this.pickingLayer.pick(
                    cssX,
                    cssY,
                    this.rendererWrapper.camera,
                    this.tileStore
                );
                if (this.destroyed || gen !== this.loadGeneration || seq !== this.hoverSeq) {
                    return;
                }
                if (!winner) {
                    this.clearHoverState(true);
                    return;
                }

                const winnerKey = `${winner.tileKey}:${winner.rawRowIdx}`;
                // Same-winner short-circuit: 0 network requests, 0 worker messages, 0 duplicate 'hover' events!
                if (winnerKey === this.activeHoverWinnerKey) {
                    return;
                }

                // Cancel any superseded in-flight hover decode immediately
                if (this.hoverAbortController) {
                    this.hoverAbortController.abort();
                    this.hoverAbortController = null;
                }

                this.activeHoverWinnerKey = winnerKey;

                // Instant GPU Hover Ring: render cyan ring immediately from GPU winner coordinates
                // before `decodeTileRow` finishes over a slow network!
                this.highlightOverlay.hoverMarker = {
                    kind: 'hover',
                    rowId: -1 - winner.rawRowIdx,
                    rawRowIdx: winner.rawRowIdx,
                    tileZ: winner.z,
                    tileX: winner.x,
                    tileY: winner.y,
                    xU16: winner.xU16,
                    yU16: winner.yU16,
                    attributes: winner.gpuAxisAttributes,
                };

                const controller = new AbortController();
                this.hoverAbortController = controller;

                try {
                    const resolved = await this.resolveWinnerToHitEvent(
                        winner,
                        cssX,
                        cssY,
                        gen,
                        this.resolveRowAttributesOnHover,
                        controller.signal
                    );
                    if (
                        !resolved ||
                        controller.signal.aborted ||
                        this.destroyed ||
                        gen !== this.loadGeneration ||
                        this.activeHoverWinnerKey !== winnerKey
                    ) {
                        return;
                    }
                    this.highlightOverlay.hoverMarker = {
                        ...resolved.marker,
                        kind: 'hover',
                    };
                    this.lastHoverWasNull = false;
                    this.emit('hover', resolved.hit);
                } catch (err) {
                    if (controller.signal.aborted || this.isExpectedPickAbort(err, gen)) {
                        return;
                    }
                    if (this.activeHoverWinnerKey === winnerKey) {
                        this.activeHoverWinnerKey = null;
                    }
                    const errorObj = err instanceof Error ? err : new Error(String(err));
                    this.emit('error', errorObj);
                } finally {
                    if (this.hoverAbortController === controller) {
                        this.hoverAbortController = null;
                    }
                }
            } catch {
                // Ignore transient GPU pick errors on teardown
            }
        })();
    }

    public setEncoding(encoding: DeepgraphEncoding): void {
        if (!this.session || this.destroyed) return;
        const prevSpatialX = this.viewState.isSpatialX;
        const prevSpatialY = this.viewState.isSpatialY;
        const prevFlip = this.viewState.effectiveFlipY;
        this.viewState.applyEncoding(
            encoding,
            this.session.summary.stats,
            this.tileStore.categoricalDictionaries
        );
        this.syncCameraBaseZoom();
        if (
            this.viewState.isSpatialX !== prevSpatialX ||
            this.viewState.isSpatialY !== prevSpatialY ||
            this.viewState.effectiveFlipY !== prevFlip
        ) {
            this.session.cacheChanged = true;
        }
        this.viewState.syncToTileStore(this.tileStore, this.session, true);
    }

    public getEncoding(): DeepgraphEncoding {
        return this.viewState.toEncoding();
    }

    public setViewOptions(patch: ViewOptionsPatch): void {
        if (this.destroyed) return;
        const prevSpatialX = this.viewState.isSpatialX;
        const prevSpatialY = this.viewState.isSpatialY;
        const prevFlip = this.viewState.effectiveFlipY;
        this.viewState.patch(patch);
        if (
            patch.experimental?.trailTimeKyr !== undefined ||
            patch.experimental?.animateFlow === true
        ) {
            this.renderLoop.syncEpoch(this.viewState.snapshot().experimental.trailTimeKyr);
        }
        if (!this.session) return;
        if (
            patch.detailOffset !== undefined ||
            this.viewState.isSpatialX !== prevSpatialX ||
            this.viewState.isSpatialY !== prevSpatialY ||
            this.viewState.effectiveFlipY !== prevFlip
        ) {
            this.session.cacheChanged = true;
        }
        if (patch.detailOffset !== undefined && Object.keys(patch).length === 1) {
            return;
        }
        this.syncCameraBaseZoom();
        this.viewState.syncToTileStore(this.tileStore, this.session, true);
    }

    public getViewState(): ViewStateSnapshot {
        return this.viewState.snapshot();
    }

    public getMetadata(): DatasetMetadataSummary | null {
        return this.session?.summary ?? null;
    }

    public getCategories(field: string): string[] | undefined {
        return this.tileStore.categoricalDictionaries?.[field];
    }

    public getCategoryColorHex(index: number): number {
        return this.tileStore.getCategoryColorHex(index);
    }

    /** @internal */
    public getRowAttributeResolver(): RowAttributeResolver | undefined {
        return this.resolveRowAttributes;
    }

    /** Converts world camera `(x, y)` to active **data coordinates**. */
    public getViewport(): ViewportState {
        const cam = this.rendererWrapper.camera;
        if (!this.session) {
            return { x: cam.position.x, y: cam.position.y, zoom: cam.zoom };
        }
        const m = this.viewState.resolveAxisMapping(this.session);
        const xSpanWorld = m.xRange[1] - m.xRange[0];
        const ySpanWorld = m.yRange[1] - m.yRange[0];
        const tx = Math.abs(xSpanWorld) < 1e-12 ? 0.5 : (cam.position.x - m.xRange[0]) / xSpanWorld;
        const ty = Math.abs(ySpanWorld) < 1e-12 ? 0.5 : (cam.position.y - m.yRange[0]) / ySpanWorld;
        const dataX = m.dataXSpan[0] + tx * (m.dataXSpan[1] - m.dataXSpan[0]);
        const dataY = m.dataYSpan[0] + ty * (m.dataYSpan[1] - m.dataYSpan[0]);
        return { x: dataX, y: dataY, zoom: cam.zoom };
    }

    private computeFitZoom(): number {
        if (!this.session || !this.viewState.hasCustomWorldScale) {
            return DEFAULT_FIT_ZOOM;
        }
        const cam = this.rendererWrapper.camera;
        const frustumW = Math.max(1e-6, cam.right - cam.left);
        const frustumH = Math.max(1e-6, cam.top - cam.bottom);
        const m = this.viewState.resolveAxisMapping(this.session);
        const spanX = Math.abs(m.xRange[1] - m.xRange[0]);
        const spanY = Math.abs(m.yRange[1] - m.yRange[0]);
        const effSpanX = spanX > 1e-9 ? spanX : spanY > 1e-9 ? spanY : frustumH;
        const effSpanY = spanY > 1e-9 ? spanY : spanX > 1e-9 ? spanX : frustumH;
        const fitZoom = Math.min(frustumW / effSpanX, frustumH / effSpanY) * 0.85;
        return Number.isFinite(fitZoom) && fitZoom > 0 ? fitZoom : DEFAULT_FIT_ZOOM;
    }

    private syncCameraBaseZoom(): number {
        const fitZoom = this.computeFitZoom();
        this.rendererWrapper.setBaseZoom(fitZoom / DEFAULT_FIT_ZOOM);
        return fitZoom;
    }

    /** Pans/zooms the viewport from **data coordinates** `(x, y)` and magnification `zoom`. */
    public setViewport(viewport: Partial<ViewportState>): void {
        const cam = this.rendererWrapper.camera;
        this.syncCameraBaseZoom();

        if (this.session) {
            const m = this.viewState.resolveAxisMapping(this.session);
            if (viewport.x !== undefined && Number.isFinite(viewport.x)) {
                const dSpanX = m.dataXSpan[1] - m.dataXSpan[0];
                const tx = Math.abs(dSpanX) < 1e-12 ? 0.5 : (viewport.x - m.dataXSpan[0]) / dSpanX;
                const wx = m.xRange[0] + tx * (m.xRange[1] - m.xRange[0]);
                cam.position.x = wx;
            }
            if (viewport.y !== undefined && Number.isFinite(viewport.y)) {
                const dSpanY = m.dataYSpan[1] - m.dataYSpan[0];
                const ty = Math.abs(dSpanY) < 1e-12 ? 0.5 : (viewport.y - m.dataYSpan[0]) / dSpanY;
                const wy = m.yRange[0] + ty * (m.yRange[1] - m.yRange[0]);
                cam.position.y = wy;
            }
        } else {
            if (viewport.x !== undefined && Number.isFinite(viewport.x)) {
                cam.position.x = viewport.x;
            }
            if (viewport.y !== undefined && Number.isFinite(viewport.y)) {
                cam.position.y = viewport.y;
            }
        }

        if (viewport.zoom !== undefined && Number.isFinite(viewport.zoom) && viewport.zoom > 0) {
            cam.zoom = this.rendererWrapper.clampZoom(viewport.zoom);
        }
        cam.updateProjectionMatrix();
        cam.updateMatrixWorld();
        if (this.session) {
            this.session.cacheChanged = true;
        }
    }

    /** Resets the camera to frame the center of the active dataset bounds. */
    public fitView(): void {
        const cam = this.rendererWrapper.camera;
        const fitZoom = this.syncCameraBaseZoom();
        let centerX = 0.0;
        let centerY = 0.0;
        if (this.session) {
            const m = this.viewState.resolveAxisMapping(this.session);
            centerX = 0.5 * (m.xRange[0] + m.xRange[1]);
            centerY = 0.5 * (m.yRange[0] + m.yRange[1]);
        }
        cam.position.set(centerX, centerY, cam.position.z);
        cam.zoom = fitZoom;
        cam.updateProjectionMatrix();
        cam.updateMatrixWorld();
        if (this.session) {
            this.session.cacheChanged = true;
        }
    }

    public on<K extends keyof DeepgraphEventMap>(
        event: K,
        handler: (payload: DeepgraphEventMap[K]) => void
    ): () => void {
        this.listeners[event].add(handler);
        return () => this.off(event, handler);
    }

    public off<K extends keyof DeepgraphEventMap>(
        event: K,
        handler: (payload: DeepgraphEventMap[K]) => void
    ): void {
        this.listeners[event].delete(handler);
        if (event === 'hover' && this.listeners.hover.size === 0) {
            this.clearHoverState(false);
        }
    }

    private emit<K extends keyof DeepgraphEventMap>(
        event: K,
        payload: DeepgraphEventMap[K]
    ): void {
        for (const fn of this.listeners[event]) {
            try {
                fn(payload);
            } catch (err) {
                console.error(`[Deepgraph] Error in '${event}' handler:`, err);
            }
        }
    }

    /** @internal */
    public [INTERNAL_DIAGNOSTICS](): InternalDiagnosticsHandle {
        return {
            camera: this.rendererWrapper.camera,
            tileStore: this.tileStore,
            pickingLayer: this.pickingLayer,
            highlightOverlay: this.highlightOverlay,
            getTileManager: () => this.session?.tileClient ?? null,
            getLibraryGpuBytes: () =>
                this.tileStore.getLiveTileGpuBytes() +
                this.computeRasterizer.getLiveGpuBytes() +
                this.pickingLayer.getLiveGpuBytes() +
                this.highlightOverlay.getLiveGpuBytes(),
            getCpuMemoryStats: () => this.session?.getCpuMemoryStats() ?? null,
        };
    }

    public destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        this.loadGeneration++;
        if (this.activeLoadController) {
            this.activeLoadController.abort();
            this.activeLoadController = null;
        }
        this.clearHoverState(false);
        this.abortActivePickControllers();
        this.renderLoop.stop();
        if (this.device && this.boundUncapturedError) {
            this.device.removeEventListener('uncapturederror', this.boundUncapturedError);
            this.boundUncapturedError = null;
        }
        if (this.session) {
            this.session.destroy();
            this.session = null;
        }
        this.tileStore.dispose();
        this.computeRasterizer.dispose();
        this.pickingLayer.dispose();
        this.highlightOverlay.dispose();
        for (const set of Object.values(this.listeners)) {
            set.clear();
        }
        this.rendererWrapper.dispose();
        this.device = null;
    }
}

/** Public namespace facade (`Deepgraph.create`, `Deepgraph.isSupported`). */
export const Deepgraph: DeepgraphStatic = {
    create: ViewerLayer.create,
    isSupported: ViewerLayer.isSupported,
};
