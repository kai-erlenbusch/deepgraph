import { ComputeRasterizer } from '../core/ComputeRasterizer.ts';
import { EdgeOverlay } from '../core/EdgeOverlay.ts';
import {
    evaluatePointMatchesFilterCpu,
    FilterGpuResources,
    MAX_LASSO_GPU_VERTICES,
    planGpuFilterExecution,
    simplifyRdpPolygon,
    unprojectCssToData64,
    type AxisProjectionContext,
} from '../core/FilterGpuResources.ts';
import { HighlightOverlay, type HighlightMarkerSpec } from '../core/HighlightOverlay.ts';
import { PickingLayer, type GpuPickWinner } from '../core/PickingLayer.ts';
import { Renderer } from '../core/Renderer.ts';
import { SelectionOverlay } from '../core/SelectionOverlay.ts';
import { TileGpuStore, type InternalAxisConfig } from '../core/TileGpuStore.ts';
import { DatasetSession } from './DatasetSession.ts';
import {
    INTERNAL_DIAGNOSTICS,
    type InternalDiagnosticsHandle,
} from './diagnostics.ts';
import { RenderLoop } from './RenderLoop.ts';
import type {
    DatasetMetadataSummary,
    DeepgraphEncoding,
    DeepgraphEventMap,
    DeepgraphFilter,
    DeepgraphOptions,
    DeepgraphPlot,
    DeepgraphStatic,
    EdgeSpec,
    EdgeStyleOptions,
    EgoEdgeResolver,
    InteractionMode,
    LassoPolygonClause,
    LoadDatasetOptions,
    PointHitEvent,
    RectBrushClause,
    RowAttributeResolver,
    SelectionGeometryClause,
    TileSource,
    ViewOptionsPatch,
    ViewportState,
    ViewStateSnapshot,
} from './types.ts';
import { ViewState } from './ViewState.ts';

type ListenerSet<T> = Set<(payload: T) => void>;

const DEFAULT_FIT_ZOOM = 1.61;
const HOVER_THROTTLE_MS = 45;
const BRUSH_EMIT_THROTTLE_MS = 16;

function cloneFilter(filter: DeepgraphFilter | null | undefined): DeepgraphFilter | null {
    if (!filter) return null;
    const out: DeepgraphFilter = {};
    if (filter.rect !== undefined) {
        out.rect = filter.rect ? { ...filter.rect } : null;
    }
    if (filter.lasso !== undefined) {
        out.lasso = filter.lasso
            ? { points: filter.lasso.points.map((p) => [p[0], p[1]] as const) }
            : null;
    }
    if (filter.ranges) {
        const ranges: Record<string, readonly [number, number]> = {};
        for (const [k, v] of Object.entries(filter.ranges)) {
            if (v && v.length >= 2) {
                ranges[k] = [v[0], v[1]];
            }
        }
        out.ranges = ranges;
    }
    if (filter.categories) {
        const cats: Record<string, ReadonlyArray<string | number>> = {};
        for (const [k, v] of Object.entries(filter.categories)) {
            if (Array.isArray(v)) {
                cats[k] = [...v];
            }
        }
        out.categories = cats;
    }
    if (filter.unselectedOpacity !== undefined) {
        out.unselectedOpacity = filter.unselectedOpacity;
    }
    return out;
}

function hasAnyFilterClause(filter: DeepgraphFilter | null | undefined): boolean {
    if (!filter) return false;
    if (filter.rect) return true;
    if (filter.lasso && filter.lasso.points.length >= 3) return true;
    if (filter.ranges && Object.keys(filter.ranges).length > 0) return true;
    if (filter.categories && Object.keys(filter.categories).length > 0) return true;
    return false;
}

export class ViewerLayer implements DeepgraphPlot {
    private rendererWrapper: Renderer;
    private tileStore: TileGpuStore;
    private filterResources: FilterGpuResources;
    private computeRasterizer: ComputeRasterizer;
    private pickingLayer: PickingLayer;
    private edgeOverlay: EdgeOverlay;
    private selectionOverlay: SelectionOverlay;
    private highlightOverlay: HighlightOverlay;
    private viewState: ViewState;
    private renderLoop: RenderLoop;
    private session: DatasetSession | null = null;
    private maxTileCache: number;
    private defaultPreset?: DeepgraphOptions['preset'];
    private resolveRowAttributes?: RowAttributeResolver;
    private resolveRowAttributesOnHover: boolean;
    private resolveEdges?: EgoEdgeResolver;
    private resolveEdgesOnHover: boolean;
    private enableRowIdIndex: boolean;
    private maxSelectedRowIds?: number;
    private evictedSelectedMarkerAllowed = new Map<bigint, boolean>();
    private rowIdFilterSeq = 0;

    private loadGeneration = 0;
    private activeLoadController: AbortController | null = null;
    private hoverSeq = 0;
    private lastHoverDispatchMs = 0;
    private lastHoverWasNull = true;
    private activeHoverWinnerKey: string | null = null;
    private hoverAbortController: AbortController | null = null;
    private egoHoverAbortController: AbortController | null = null;
    private egoClickAbortController: AbortController | null = null;
    private pinnedClickEgoEdges: EdgeSpec[] | null = null;
    private activePickAbortControllers = new Set<AbortController>();
    private destroyed = false;
    private device: GPUDevice | null = null;
    private boundUncapturedError: ((event: any) => void) | null = null;

    // Interactive 2D brush / spotlight-lens / lasso state
    private dragSubMode: 'draw-rect' | 'translate-rect' | 'draw-lasso' | null = null;
    private dragPreFilterSnapshot: DeepgraphFilter | null = null;
    private dragTranslateBaseRect: RectBrushClause | null = null;
    private dragTranslateAnchorData: { dataX: number; dataY: number } | null = null;
    private dragLassoCssPath: [number, number][] = [];
    private lastBrushEmitMs = 0;

    private listeners: {
        [K in keyof DeepgraphEventMap]: ListenerSet<DeepgraphEventMap[K]>;
    } = {
        click: new Set(),
        hover: new Set(),
        select: new Set(),
        brush: new Set(),
        telemetry: new Set(),
        load: new Set(),
        error: new Set(),
    };

    private constructor(
        rendererWrapper: Renderer,
        tileStore: TileGpuStore,
        filterResources: FilterGpuResources,
        computeRasterizer: ComputeRasterizer,
        pickingLayer: PickingLayer,
        edgeOverlay: EdgeOverlay,
        selectionOverlay: SelectionOverlay,
        highlightOverlay: HighlightOverlay,
        device: GPUDevice | null,
        options: DeepgraphOptions = {}
    ) {
        this.rendererWrapper = rendererWrapper;
        this.tileStore = tileStore;
        this.filterResources = filterResources;
        this.computeRasterizer = computeRasterizer;
        this.pickingLayer = pickingLayer;
        this.edgeOverlay = edgeOverlay;
        this.selectionOverlay = selectionOverlay;
        this.highlightOverlay = highlightOverlay;
        this.device = device;
        this.maxTileCache = options.maxTileCache ?? 250;
        this.defaultPreset = options.preset;
        this.resolveRowAttributes = options.resolveRowAttributes;
        this.resolveRowAttributesOnHover = Boolean(options.resolveRowAttributesOnHover);
        this.resolveEdges = options.resolveEdges;
        this.resolveEdgesOnHover = options.resolveEdgesOnHover ?? true;
        this.enableRowIdIndex = Boolean(options.enableRowIdIndex);
        this.maxSelectedRowIds = options.maxSelectedRowIds;
        this.viewState = new ViewState(options.detailOffset ?? 2);

        this.rendererWrapper.interactionMode = options.interactionMode ?? 'pan';
        this.rendererWrapper.shiftDragMode = options.shiftDragMode ?? 'brush';
        this.syncCanvasCursor();

        if (options.filter) {
            this.filterResources.currentFilter = cloneFilter(options.filter);
        }
        if (options.edges && options.edges.length > 0) {
            this.edgeOverlay.setStaticEdges(options.edges, options.edgeStyle);
        } else if (options.edgeStyle) {
            this.edgeOverlay.setStyle(options.edgeStyle);
        }

        this.tileStore.onSlotBindingsInvalidated = (slot) => {
            this.computeRasterizer.invalidateSlot(slot);
            this.pickingLayer.invalidateSlot(slot);
        };

        this.tileStore.onSlotDisposed = (slot) => {
            this.computeRasterizer.invalidateSlot(slot);
            this.pickingLayer.invalidateSlot(slot);
            this.filterResources.clearSingleSlotMask(slot);
        };

        this.tileStore.onSlotMaskUploaded = (slot, slices) => {
            this.filterResources.writeSlotMaskSlices(slot, slices);
        };

        this.rendererWrapper.onCanvasClick = (cssX, cssY) => {
            void this.handleCanvasClick(cssX, cssY);
        };

        this.rendererWrapper.onCanvasHover = (cssX, cssY) => {
            this.updateHoverCursorForBrush(cssX, cssY);
            this.handleCanvasHover(cssX, cssY);
        };

        this.rendererWrapper.onCanvasPointerLeave = () => {
            this.clearHoverState(true);
            this.syncCanvasCursor();
        };

        this.rendererWrapper.onSelectionDragStart = (mode, cssX, cssY) =>
            this.handleSelectionDragStart(mode, cssX, cssY);
        this.rendererWrapper.onSelectionDragMove = (mode, sX, sY, cssX, cssY) =>
            this.handleSelectionDragMove(mode, sX, sY, cssX, cssY);
        this.rendererWrapper.onSelectionDragEnd = (
            mode,
            sX,
            sY,
            cssX,
            cssY,
            travel,
            committed
        ) => this.handleSelectionDragEnd(mode, sX, sY, cssX, cssY, travel, committed);

        this.renderLoop = new RenderLoop({
            rendererWrapper: this.rendererWrapper,
            tileStore: this.tileStore,
            computeRasterizer: this.computeRasterizer,
            edgeOverlay: this.edgeOverlay,
            selectionOverlay: this.selectionOverlay,
            highlightOverlay: this.highlightOverlay,
            getActiveFilter: () => this.filterResources.currentFilter,
            isMarkerVisible: (marker) => this.isHighlightMarkerVisible(marker),
            getProjectionContexts: (sess) => this.buildProjectionContexts(sess),
            getExtraGpuBytes: () =>
                this.filterResources.getLiveGpuBytes() +
                this.pickingLayer.getLiveGpuBytes() +
                this.edgeOverlay.getLiveGpuBytes() +
                this.selectionOverlay.getLiveGpuBytes() +
                this.highlightOverlay.getLiveGpuBytes(),
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
        try {
            await rendererWrapper.init();
        } catch (err) {
            rendererWrapper.dispose();
            throw err;
        }

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
        const filterResources = new FilterGpuResources(device, maxSlots);
        filterResources.init();
        const computeRasterizer = new ComputeRasterizer(
            rendererWrapper,
            maxSlots,
            filterResources
        );
        const pickingLayer = new PickingLayer(rendererWrapper, maxSlots, filterResources);
        const edgeOverlay = new EdgeOverlay(rendererWrapper, options.edgeStyle);
        const selectionOverlay = new SelectionOverlay(rendererWrapper);
        const highlightOverlay = new HighlightOverlay(rendererWrapper);

        const viewer = new ViewerLayer(
            rendererWrapper,
            tileStore,
            filterResources,
            computeRasterizer,
            pickingLayer,
            edgeOverlay,
            selectionOverlay,
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

    private buildProjectionContexts(session: DatasetSession): {
        current: AxisProjectionContext;
        previous: AxisProjectionContext | null;
    } {
        const computeMode = this.tileStore.computeMode;
        const isOptical = computeMode === 0 || computeMode === 3;
        const cfg = this.tileStore.currentConfig;
        const prevCfg = this.tileStore.previousConfig;

        const xAxis: InternalAxisConfig = isOptical
            ? { field: 'x_u16', transform: 'linear', domain: cfg.x.domain, range: cfg.x.range }
            : cfg.x;
        const yAxis: InternalAxisConfig = isOptical
            ? { field: 'y_u16', transform: 'linear', domain: cfg.y.domain, range: cfg.y.range }
            : cfg.y;

        const current: AxisProjectionContext = {
            xAxis,
            yAxis,
            rootBounds: session.summary.rootBounds,
            dataBounds: session.summary.dataBounds,
            effectiveFlipY: this.viewState.effectiveFlipY,
            colOrigins: this.tileStore.colOrigins,
        };

        const isTransitioning =
            this.tileStore.isTransitioning && !isOptical && computeMode !== 3 && computeMode !== 4;
        const previous: AxisProjectionContext | null = isTransitioning
            ? {
                  xAxis: prevCfg.x,
                  yAxis: prevCfg.y,
                  rootBounds: session.summary.rootBounds,
                  dataBounds: session.summary.dataBounds,
                  effectiveFlipY: this.viewState.effectiveFlipY,
                  colOrigins: this.tileStore.colOrigins,
              }
            : null;

        return { current, previous };
    }

    private isHighlightMarkerVisible(marker: HighlightMarkerSpec): boolean {
        const filter = this.filterResources.currentFilter;
        const hasBitmask = this.filterResources.hasSlotBitmask;
        if (!hasBitmask && !hasAnyFilterClause(filter)) {
            return true;
        }
        // Transient GPU instant-hover ring (`rowId < 0` before `decodeTileRow` completes)
        // was already filter-tested directly inside `PickingLayer` compute shader!
        if (marker.rowId < 0 && marker.rowIdBigInt === undefined) {
            return true;
        }

        const session = this.session;
        if (!session) return true;

        const rowId64 =
            marker.rowIdBigInt !== undefined
                ? BigInt.asUintN(64, marker.rowIdBigInt)
                : BigInt.asUintN(64, BigInt(Math.trunc(marker.rowId)));
        if (hasBitmask) {
            const tileKey = `${marker.tileZ}/${marker.tileX}/${marker.tileY}`;
            const slotIdx = this.tileStore.tileKeyToSlot.get(tileKey);
            if (slotIdx !== undefined) {
                if (!this.filterResources.isSlotRowSelected(slotIdx, marker.rawRowIdx)) {
                    return false;
                }
            } else {
                const allowed = this.evictedSelectedMarkerAllowed.get(rowId64);
                if (!allowed) {
                    return false;
                }
            }
        }
        if (!filter) return true;

        // If `unselectedOpacity > 0` in Mode 0 / Mode 1, context points remain faintly visible,
        // wait: Reviewer Pushback #5 specified hiding rings on culled/filtered-out rows so
        // stale hover/selected rings never sit on unselected background points!
        const vs = this.viewState.snapshot();
        const scale = Math.pow(2.0, -marker.tileZ);
        const xNorm = (marker.tileX + marker.xU16 / 65535.0) * scale;
        const yNorm = (marker.tileY + marker.yU16 / 65535.0) * scale;
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

        const resolveCoord = (field: string | null, fallbackSpatial: number): number => {
            if (!field) return fallbackSpatial;
            const raw = marker.attributes[field];
            if (typeof raw === 'number' && Number.isFinite(raw)) return raw;
            if (typeof raw === 'string') {
                const dict = this.tileStore.categoricalDictionaries[field];
                const idx = dict ? dict.indexOf(raw) : -1;
                if (idx >= 0) return idx;
            }
            return fallbackSpatial;
        };

        const dataX = resolveCoord(vs.xField, spatialX);
        const dataY = resolveCoord(vs.yField, spatialY);
        return evaluatePointMatchesFilterCpu(
            dataX,
            dataY,
            marker.attributes,
            filter,
            this.tileStore.categoricalDictionaries,
            null,
            rowId64
        );
    }

    private getGpuFilterCapabilityContext() {
        const computeMode = this.tileStore.computeMode;
        const isOptical = computeMode === 0 || computeMode === 3;
        const cfg = this.tileStore.currentConfig;
        const xField = isOptical ? 'x_u16' : cfg.x.field;
        const yField = isOptical ? 'y_u16' : cfg.y.field;
        const isXSpatial = xField === 'x_u16' || xField === 'y_u16';
        const isYSpatial = yField === 'y_u16' || yField === 'x_u16';
        const colorField = isOptical
            ? this.tileStore.opticalColorField || 'bp_rp'
            : cfg.color.field;
        const sizeField = isOptical
            ? this.tileStore.magnitudeField || 'x_u16'
            : cfg.size.field;

        return {
            isXSpatial,
            isYSpatial,
            xField,
            yField,
            colorField,
            sizeField,
            auxField1: this.tileStore.auxField1,
            auxField2: this.tileStore.auxField2,
            schemaFields: this.session?.summary.schemaFields ?? [],
            categories: this.tileStore.categoricalDictionaries,
        };
    }

    private syncFilterExtraColumns(): void {
        if (!this.session) return;
        const plan = planGpuFilterExecution(
            this.filterResources.currentFilter,
            this.getGpuFilterCapabilityContext()
        );
        const next1 = plan.supported ? plan.extraFilterField1 : '';
        const next2 = plan.supported ? plan.extraFilterField2 : '';
        const changed =
            this.tileStore.filterExtraField1 !== next1 ||
            this.tileStore.filterExtraField2 !== next2;

        this.tileStore.filterExtraField1 = next1;
        this.tileStore.filterExtraField2 = next2;

        if (changed) {
            const activeCols = this.tileStore.getActiveColumns(this.tileStore.currentConfig);
            this.session.setRequestedColumns(activeCols);
        }
    }

    private syncCanvasCursor(): void {
        const mode = this.rendererWrapper.interactionMode;
        this.rendererWrapper.canvas.style.cursor =
            mode === 'brush' || mode === 'lasso' ? 'crosshair' : 'grab';
    }

    private updateHoverCursorForBrush(cssX: number, cssY: number): void {
        if (!this.session) return;
        const mode = this.rendererWrapper.interactionMode;
        if (mode !== 'brush') return;
        const rect = this.filterResources.currentFilter?.rect;
        if (!rect) {
            this.rendererWrapper.canvas.style.cursor = 'crosshair';
            return;
        }
        const { current } = this.buildProjectionContexts(this.session);
        const inside = this.selectionOverlay.isCssInsideRect(
            cssX,
            cssY,
            rect,
            this.rendererWrapper.camera,
            current
        );
        this.rendererWrapper.canvas.style.cursor = inside ? 'move' : 'crosshair';
    }

    private buildCurrentGeometryClause(): SelectionGeometryClause {
        const vs = this.viewState.snapshot();
        const f = this.filterResources.currentFilter;
        if (f?.lasso && f.lasso.points.length >= 3) {
            return {
                kind: 'lasso',
                xField: vs.xField,
                yField: vs.yField,
                lasso: { points: f.lasso.points.map((p) => [p[0], p[1]] as const) },
            };
        }
        if (f?.rect) {
            return {
                kind: 'rect',
                xField: vs.xField,
                yField: vs.yField,
                rect: { ...f.rect },
            };
        }
        return {
            kind: 'none',
            xField: vs.xField,
            yField: vs.yField,
        };
    }

    private handleSelectionDragStart(
        mode: 'brush' | 'lasso',
        cssX: number,
        cssY: number
    ): boolean {
        if (!this.session || this.destroyed) return false;
        this.clearHoverState(true);
        this.dragPreFilterSnapshot = cloneFilter(this.filterResources.currentFilter);
        this.dragTranslateBaseRect = null;
        this.dragTranslateAnchorData = null;
        this.dragLassoCssPath = [];
        this.selectionOverlay.liveLassoCssPoints = null;

        const { current: projCtx } = this.buildProjectionContexts(this.session);
        const { width: cssW, height: cssH } = this.rendererWrapper.getContainerSize();

        if (mode === 'brush') {
            const existingRect = this.filterResources.currentFilter?.rect;
            if (
                existingRect &&
                this.selectionOverlay.isCssInsideRect(
                    cssX,
                    cssY,
                    existingRect,
                    this.rendererWrapper.camera,
                    projCtx
                )
            ) {
                this.dragSubMode = 'translate-rect';
                this.dragTranslateBaseRect = { ...existingRect };
                this.dragTranslateAnchorData = unprojectCssToData64(
                    cssX,
                    cssY,
                    cssW,
                    cssH,
                    this.rendererWrapper.camera,
                    projCtx
                );
                this.rendererWrapper.canvas.style.cursor = 'move';
                return true;
            }
            this.dragSubMode = 'draw-rect';
            return true;
        }

        this.dragSubMode = 'draw-lasso';
        this.dragLassoCssPath = [[cssX, cssY]];
        return true;
    }

    private handleSelectionDragMove(
        _mode: 'brush' | 'lasso',
        startCssX: number,
        startCssY: number,
        cssX: number,
        cssY: number
    ): void {
        if (!this.session || this.destroyed || !this.dragSubMode) return;
        const travel = Math.hypot(cssX - startCssX, cssY - startCssY);
        const { current: projCtx } = this.buildProjectionContexts(this.session);
        const { width: cssW, height: cssH } = this.rendererWrapper.getContainerSize();
        const cam = this.rendererWrapper.camera;

        if (this.dragSubMode === 'translate-rect') {
            if (!this.dragTranslateBaseRect || !this.dragTranslateAnchorData) return;
            const cur = unprojectCssToData64(cssX, cssY, cssW, cssH, cam, projCtx);
            const dx = cur.dataX - this.dragTranslateAnchorData.dataX;
            const dy = cur.dataY - this.dragTranslateAnchorData.dataY;
            const base = this.dragTranslateBaseRect;
            const nextRect: RectBrushClause = {
                minX: base.minX + dx,
                maxX: base.maxX + dx,
                minY: base.minY + dy,
                maxY: base.maxY + dy,
            };
            const nextFilter: DeepgraphFilter = {
                ...(this.filterResources.currentFilter ?? {}),
                rect: nextRect,
                lasso: null,
            };
            this.filterResources.currentFilter = nextFilter;
            this.renderLoop.markDirty();
        } else if (this.dragSubMode === 'draw-rect') {
            if (travel < 3.0) return;
            const p0 = unprojectCssToData64(startCssX, startCssY, cssW, cssH, cam, projCtx);
            const p1 = unprojectCssToData64(cssX, cssY, cssW, cssH, cam, projCtx);
            const nextRect: RectBrushClause = {
                minX: Math.min(p0.dataX, p1.dataX),
                maxX: Math.max(p0.dataX, p1.dataX),
                minY: Math.min(p0.dataY, p1.dataY),
                maxY: Math.max(p0.dataY, p1.dataY),
            };
            const nextFilter: DeepgraphFilter = {
                ...(this.dragPreFilterSnapshot ?? {}),
                rect: nextRect,
                lasso: null,
            };
            this.filterResources.currentFilter = nextFilter;
            this.renderLoop.markDirty();
        } else if (this.dragSubMode === 'draw-lasso') {
            const lastPt = this.dragLassoCssPath[this.dragLassoCssPath.length - 1];
            if (!lastPt || Math.hypot(cssX - lastPt[0], cssY - lastPt[1]) >= 2.5) {
                this.dragLassoCssPath.push([cssX, cssY]);
            }
            if (this.dragLassoCssPath.length < 3) {
                this.selectionOverlay.liveLassoCssPoints = [...this.dragLassoCssPath];
                this.renderLoop.markDirty();
                return;
            }
            this.selectionOverlay.liveLassoCssPoints = null;
            const simpCss = simplifyRdpPolygon(this.dragLassoCssPath, MAX_LASSO_GPU_VERTICES);
            if (simpCss.length >= 3) {
                const dataPts = simpCss.map(([px, py]) => {
                    const d = unprojectCssToData64(px, py, cssW, cssH, cam, projCtx);
                    return [d.dataX, d.dataY] as const;
                });
                const lassoClause: LassoPolygonClause = { points: dataPts };
                const nextFilter: DeepgraphFilter = {
                    ...(this.dragPreFilterSnapshot ?? {}),
                    lasso: lassoClause,
                    rect: null,
                };
                this.filterResources.currentFilter = nextFilter;
                this.renderLoop.markDirty();
            }
        }

        const now = performance.now();
        if (now - this.lastBrushEmitMs >= BRUSH_EMIT_THROTTLE_MS) {
            this.lastBrushEmitMs = now;
            this.emit('brush', this.buildCurrentGeometryClause());
        }
    }

    private handleSelectionDragEnd(
        _mode: 'brush' | 'lasso',
        _startCssX: number,
        _startCssY: number,
        cssX: number,
        cssY: number,
        totalTravelCssPx: number,
        committed: boolean
    ): boolean {
        const subMode = this.dragSubMode;
        const preSnap = this.dragPreFilterSnapshot;
        this.dragSubMode = null;
        this.dragPreFilterSnapshot = null;
        this.dragTranslateBaseRect = null;
        this.dragTranslateAnchorData = null;
        this.dragLassoCssPath = [];
        this.selectionOverlay.liveLassoCssPoints = null;
        this.updateHoverCursorForBrush(cssX, cssY);

        if (!this.session || this.destroyed || !subMode) return false;

        if (!committed) {
            this.filterResources.currentFilter = preSnap;
            this.renderLoop.markDirty();
            return true;
        }

        // Sub-4px click while in brush/lasso mode:
        // If an active rect/lasso existed, clear it and suppress `onCanvasClick`!
        if (totalTravelCssPx < 4.0) {
            const hadBrushOrLasso = Boolean(preSnap?.rect || preSnap?.lasso);
            if (hadBrushOrLasso) {
                const next: DeepgraphFilter = { ...(preSnap ?? {}) };
                delete next.rect;
                delete next.lasso;
                this.filterResources.currentFilter = hasAnyFilterClause(next) ? next : null;
                this.renderLoop.markDirty();

                const vs = this.viewState.snapshot();
                const noneClause: SelectionGeometryClause = {
                    kind: 'none',
                    xField: vs.xField,
                    yField: vs.yField,
                };
                this.emit('brush', noneClause);
                this.emit('select', {
                    points: [],
                    clause: noneClause,
                    ...(this.enableRowIdIndex ? { rowIds: [], rowIdsBigInt: [] } : {}),
                });
                return true;
            }
            return false;
        }

        const clause = this.buildCurrentGeometryClause();
        if (clause.kind === 'none') {
            this.filterResources.currentFilter = preSnap;
            this.renderLoop.markDirty();
            return true;
        }

        this.highlightOverlay.selectedMarkers = [];
        this.pinnedClickEgoEdges = null;
        this.edgeOverlay.clearEgoEdges();
        this.emit('brush', clause);

        if (
            this.enableRowIdIndex &&
            this.maxSelectedRowIds !== 0 &&
            this.viewState.isSpatialX &&
            this.viewState.isSpatialY
        ) {
            const gen = this.loadGeneration;
            const sess = this.session;
            void sess
                .queryLoadedSpatialSelection(
                    {
                        rect: clause.rect ?? null,
                        lasso: clause.lasso?.points ?? null,
                        effectiveFlipY: this.viewState.effectiveFlipY,
                    },
                    this.maxSelectedRowIds
                )
                .then((res) => {
                    if (this.destroyed || gen !== this.loadGeneration) return;
                    this.emit('select', {
                        points: [],
                        clause,
                        rowIds: res.rowIds,
                        rowIdsBigInt: res.rowIdsBigInt,
                        totalMatched: res.totalMatched,
                        rowIdsTruncated: res.truncated,
                    });
                })
                .catch(() => {
                    if (!this.destroyed && gen === this.loadGeneration) {
                        this.emit('select', { points: [], clause });
                    }
                });
        } else {
            this.emit('select', {
                points: [],
                clause,
                ...(this.maxSelectedRowIds === 0 ? { rowIdsTruncated: true } : {}),
            });
        }

        return true;
    }

    private triggerEgoEdgeResolution(hit: PointHitEvent, trigger: 'hover' | 'click'): void {
        if (!this.resolveEdges || this.destroyed) return;

        if (trigger === 'click') {
            if (this.egoClickAbortController) {
                this.egoClickAbortController.abort();
            }
            const controller = new AbortController();
            this.egoClickAbortController = controller;
            const gen = this.loadGeneration;
            void Promise.resolve()
                .then(() => this.resolveEdges!(hit, 'click', controller.signal))
                .then((edges) => {
                    if (
                        this.destroyed ||
                        gen !== this.loadGeneration ||
                        controller.signal.aborted
                    ) {
                        return;
                    }
                    const list = Array.isArray(edges) ? edges : [];
                    this.pinnedClickEgoEdges = list;
                    this.edgeOverlay.setEgoEdges(list);
                    this.renderLoop.markDirty();
                })
                .catch((err) => {
                    if (controller.signal.aborted || this.isExpectedPickAbort(err, gen)) return;
                    this.emit('error', err instanceof Error ? err : new Error(String(err)));
                })
                .finally(() => {
                    if (this.egoClickAbortController === controller) {
                        this.egoClickAbortController = null;
                    }
                });
            return;
        }

        if (!this.resolveEdgesOnHover) return;
        if (this.egoHoverAbortController) {
            this.egoHoverAbortController.abort();
        }
        const controller = new AbortController();
        this.egoHoverAbortController = controller;
        const gen = this.loadGeneration;
        void Promise.resolve()
            .then(() => this.resolveEdges!(hit, 'hover', controller.signal))
            .then((edges) => {
                if (this.destroyed || gen !== this.loadGeneration || controller.signal.aborted) {
                    return;
                }
                this.edgeOverlay.setEgoEdges(Array.isArray(edges) ? edges : []);
                this.renderLoop.markDirty();
            })
            .catch((err) => {
                if (controller.signal.aborted || this.isExpectedPickAbort(err, gen)) return;
                this.emit('error', err instanceof Error ? err : new Error(String(err)));
            })
            .finally(() => {
                if (this.egoHoverAbortController === controller) {
                    this.egoHoverAbortController = null;
                }
            });
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
        if (this.egoClickAbortController) {
            this.egoClickAbortController.abort();
            this.egoClickAbortController = null;
        }
        this.pinnedClickEgoEdges = null;
        this.edgeOverlay.clearEgoEdges();
        this.highlightOverlay.clear();
        this.filterResources.clearSlotBitmasks();
        this.evictedSelectedMarkerAllowed.clear();

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
                    this.syncFilterExtraColumns();
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

        if (this.destroyed || gen !== this.loadGeneration) {
            session.destroy();
            if (this.destroyed) {
                throw new Error('Deepgraph instance was destroyed during dataset load.');
            }
            throw new DOMException('Dataset load superseded', 'AbortError');
        }

        this.session = session;
        if (this.enableRowIdIndex) {
            session.setRetainRowIds(true);
        }
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
        this.syncFilterExtraColumns();
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
        if (this.egoHoverAbortController) {
            this.egoHoverAbortController.abort();
            this.egoHoverAbortController = null;
        }
        if (this.resolveEdges && this.resolveEdgesOnHover) {
            if (this.pinnedClickEgoEdges) {
                this.edgeOverlay.setEgoEdges(this.pinnedClickEgoEdges);
            } else {
                this.edgeOverlay.clearEgoEdges();
            }
            this.renderLoop.markDirty();
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
                // Preserve `enriched` key order first (so cold sidecar summaries/prompts lead inspect UIs)
                // while retaining any `decoded.attributes` columns omitted by partial resolvers.
                const nextAttrs: Record<string, unknown> = { ...enriched };
                for (const [k, v] of Object.entries(decoded.attributes)) {
                    if (!(k in nextAttrs)) {
                        nextAttrs[k] = v;
                    }
                }
                finalAttrs = nextAttrs;
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
            const { current: projCtx } = this.buildProjectionContexts(this.session);
            const winner = await this.pickingLayer.pick(
                cssX,
                cssY,
                this.rendererWrapper.camera,
                this.tileStore,
                4.5,
                projCtx
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
            const { current: projCtx } = this.buildProjectionContexts(this.session);
            const winner = await this.pickingLayer.pick(
                cssX,
                cssY,
                this.rendererWrapper.camera,
                this.tileStore,
                4.5,
                projCtx
            );
            if (this.destroyed || gen !== this.loadGeneration || controller.signal.aborted) return;

            const vs = this.viewState.snapshot();
            if (!winner) {
                this.highlightOverlay.selectedMarkers = [];
                if (this.egoClickAbortController) {
                    this.egoClickAbortController.abort();
                    this.egoClickAbortController = null;
                }
                this.pinnedClickEgoEdges = null;
                this.edgeOverlay.clearEgoEdges();
                this.renderLoop.markDirty();
                this.emit('click', null);
                this.emit('select', {
                    points: [],
                    clause: { kind: 'none', xField: vs.xField, yField: vs.yField },
                });
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
            if (this.filterResources.hasSlotBitmask) {
                const clickedId64 =
                    resolved.hit.rowIdBigInt !== undefined
                        ? BigInt.asUintN(64, resolved.hit.rowIdBigInt)
                        : BigInt.asUintN(64, BigInt(Math.trunc(resolved.hit.rowId)));
                this.evictedSelectedMarkerAllowed.set(clickedId64, true);
            }
            this.renderLoop.markDirty();
            this.triggerEgoEdgeResolution(resolved.hit, 'click');
            this.emit('click', resolved.hit);
            this.emit('select', {
                points: [resolved.hit],
                clause: { kind: 'point', xField: vs.xField, yField: vs.yField },
                rowIds: [resolved.hit.rowId],
                ...(resolved.hit.rowIdBigInt !== undefined
                    ? { rowIdsBigInt: [resolved.hit.rowIdBigInt] }
                    : {}),
            });
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
        const shouldResolveHoverEdges = Boolean(this.resolveEdges && this.resolveEdgesOnHover);
        if (this.listeners.hover.size === 0 && !shouldResolveHoverEdges) return;

        const now = performance.now();
        if (now - this.lastHoverDispatchMs < HOVER_THROTTLE_MS || this.pickingLayer.isBusy) {
            return;
        }
        this.lastHoverDispatchMs = now;
        const seq = ++this.hoverSeq;
        const gen = this.loadGeneration;

        void (async () => {
            try {
                if (!this.session || this.destroyed) return;
                const { current: projCtx } = this.buildProjectionContexts(this.session);
                const winner = await this.pickingLayer.pick(
                    cssX,
                    cssY,
                    this.rendererWrapper.camera,
                    this.tileStore,
                    4.5,
                    projCtx
                );
                if (this.destroyed || gen !== this.loadGeneration || seq !== this.hoverSeq) {
                    return;
                }
                if (!winner) {
                    this.clearHoverState(true);
                    return;
                }

                const winnerKey = `${winner.tileKey}:${winner.rawRowIdx}`;
                if (winnerKey === this.activeHoverWinnerKey) {
                    return;
                }

                if (this.hoverAbortController) {
                    this.hoverAbortController.abort();
                    this.hoverAbortController = null;
                }
                if (this.egoHoverAbortController) {
                    this.egoHoverAbortController.abort();
                    this.egoHoverAbortController = null;
                }

                this.activeHoverWinnerKey = winnerKey;

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
                this.renderLoop.markDirty();

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
                    this.renderLoop.markDirty();
                    if (shouldResolveHoverEdges) {
                        this.triggerEgoEdgeResolution(resolved.hit, 'hover');
                    }
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

    public setInteractionMode(mode: InteractionMode): void {
        if (this.destroyed) return;
        this.rendererWrapper.interactionMode = mode;
        this.syncCanvasCursor();
    }

    public getInteractionMode(): InteractionMode {
        return this.rendererWrapper.interactionMode;
    }

    public canApplyFilterOnGpu(filter: DeepgraphFilter | null): boolean {
        if (!filter) return true;
        return planGpuFilterExecution(filter, this.getGpuFilterCapabilityContext()).supported;
    }

    public setFilter(filter: DeepgraphFilter | null): void {
        if (this.destroyed) return;
        const cloned = cloneFilter(filter);
        this.filterResources.currentFilter = hasAnyFilterClause(cloned) ? cloned : null;
        this.syncFilterExtraColumns();
        this.renderLoop.markDirty();
    }

    public getFilter(): DeepgraphFilter | null {
        return cloneFilter(this.filterResources.currentFilter);
    }

    public async setFilteredRowIds(
        ids: Iterable<number | bigint | string> | Uint32Array | BigUint64Array | null
    ): Promise<void> {
        if (this.destroyed) return;
        const seq = ++this.rowIdFilterSeq;

        if (ids === null) {
            this.evictedSelectedMarkerAllowed.clear();
            this.filterResources.clearSlotBitmasks();
            if (this.session) {
                await this.session.setRowIdFilter(null);
            }
            this.renderLoop.markDirty();
            return;
        }

        if (!this.session) {
            this.evictedSelectedMarkerAllowed.clear();
            this.filterResources.enableSlotBitmasks();
            this.renderLoop.markDirty();
            return;
        }

        const selMarkers = this.highlightOverlay.selectedMarkers;
        let probeIds: BigUint64Array | null = null;
        if (selMarkers.length > 0) {
            probeIds = new BigUint64Array(selMarkers.length);
            for (let i = 0; i < selMarkers.length; i++) {
                const m = selMarkers[i];
                probeIds[i] =
                    m.rowIdBigInt !== undefined
                        ? BigInt.asUintN(64, m.rowIdBigInt)
                        : BigInt.asUintN(64, BigInt(Math.trunc(m.rowId)));
            }
        }

        this.enableRowIdIndex = true;
        const { masks: maskMap, probeMatches } = await this.session.setRowIdFilter(ids, probeIds);
        if (this.destroyed || seq !== this.rowIdFilterSeq) return;

        this.evictedSelectedMarkerAllowed.clear();
        if (probeIds && probeMatches) {
            for (let i = 0; i < probeIds.length; i++) {
                this.evictedSelectedMarkerAllowed.set(probeIds[i], probeMatches[i] !== 0);
            }
        }

        this.filterResources.enableSlotBitmasks();
        for (let slotIdx = 0; slotIdx < this.tileStore.maxSlots; slotIdx++) {
            const slotObj = this.tileStore.slots[slotIdx];
            if (!slotObj) continue;
            const slices = maskMap.get(slotObj.key);
            if (slices) {
                this.filterResources.writeSlotMaskSlices(slotIdx, slices);
            } else {
                this.filterResources.clearSingleSlotMask(slotIdx);
            }
        }
        this.renderLoop.markDirty();
    }

    public clearSelection(): void {
        if (this.destroyed) return;
        this.highlightOverlay.selectedMarkers = [];
        if (this.egoClickAbortController) {
            this.egoClickAbortController.abort();
            this.egoClickAbortController = null;
        }
        this.pinnedClickEgoEdges = null;
        this.edgeOverlay.clearEgoEdges();

        const prev = this.filterResources.currentFilter;
        if (prev && (prev.rect || prev.lasso)) {
            const next: DeepgraphFilter = { ...prev };
            delete next.rect;
            delete next.lasso;
            this.filterResources.currentFilter = hasAnyFilterClause(next) ? next : null;
        }
        this.renderLoop.markDirty();
    }

    public async getSelectedRowIds(maxCount?: number): Promise<{
        rowIds: number[];
        rowIdsBigInt: bigint[];
        totalMatched: number;
        truncated: boolean;
    }> {
        if (!this.session || this.destroyed) {
            return { rowIds: [], rowIdsBigInt: [], totalMatched: 0, truncated: false };
        }
        const effCap = maxCount !== undefined ? maxCount : this.maxSelectedRowIds;
        const f = this.filterResources.currentFilter;
        if (f && (f.rect || f.lasso) && this.viewState.isSpatialX && this.viewState.isSpatialY) {
            this.enableRowIdIndex = true;
            return this.session.queryLoadedSpatialSelection(
                {
                    rect: f.rect ?? null,
                    lasso: f.lasso?.points ?? null,
                    effectiveFlipY: this.viewState.effectiveFlipY,
                },
                effCap
            );
        }
        const pts = this.highlightOverlay.selectedMarkers;
        const totalMatched = pts.length;
        const sliceLen =
            effCap !== undefined && Number.isFinite(effCap) && effCap >= 0
                ? Math.min(pts.length, Math.trunc(effCap))
                : pts.length;
        const sliced = pts.slice(0, sliceLen);
        const rowIds = sliced.map((m) => m.rowId);
        const rowIdsBigInt = sliced.map((m) =>
            m.rowIdBigInt !== undefined ? m.rowIdBigInt : BigInt(Math.trunc(m.rowId))
        );
        return {
            rowIds,
            rowIdsBigInt,
            totalMatched,
            truncated: totalMatched > sliceLen,
        };
    }

    public setEdges(edges: ReadonlyArray<EdgeSpec>, style?: EdgeStyleOptions): void {
        if (this.destroyed) return;
        this.edgeOverlay.setStaticEdges(edges, style);
        this.renderLoop.markDirty();
    }

    public setEdgeStyle(style: EdgeStyleOptions): void {
        if (this.destroyed) return;
        this.edgeOverlay.setStyle(style);
        this.renderLoop.markDirty();
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
        this.syncFilterExtraColumns();
        this.renderLoop.markDirty();
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
        this.syncFilterExtraColumns();
        this.renderLoop.markDirty();
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
        const shouldResolveHoverEdges = Boolean(this.resolveEdges && this.resolveEdgesOnHover);
        if (event === 'hover' && this.listeners.hover.size === 0 && !shouldResolveHoverEdges) {
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
            selectionOverlay: this.selectionOverlay,
            edgeOverlay: this.edgeOverlay,
            filterResources: this.filterResources,
            getTileManager: () => this.session?.tileClient ?? null,
            getLibraryGpuBytes: () =>
                this.tileStore.getLiveTileGpuBytes() +
                this.computeRasterizer.getLiveGpuBytes() +
                this.filterResources.getLiveGpuBytes() +
                this.pickingLayer.getLiveGpuBytes() +
                this.edgeOverlay.getLiveGpuBytes() +
                this.selectionOverlay.getLiveGpuBytes() +
                this.highlightOverlay.getLiveGpuBytes(),
            getCpuMemoryStats: () => this.session?.getCpuMemoryStats() ?? null,
        };
    }

    public destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        this.loadGeneration++;
        this.rowIdFilterSeq++;
        if (this.activeLoadController) {
            this.activeLoadController.abort();
            this.activeLoadController = null;
        }
        if (this.egoClickAbortController) {
            this.egoClickAbortController.abort();
            this.egoClickAbortController = null;
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
        this.filterResources.dispose();
        this.edgeOverlay.dispose();
        this.selectionOverlay.dispose();
        this.highlightOverlay.dispose();
        this.evictedSelectedMarkerAllowed.clear();
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
