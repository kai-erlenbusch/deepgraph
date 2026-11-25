/** Axis-aligned 2D bounding box in world or data space. */
export interface BoundingBox {
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
}

/**
 * Custom random-access byte-range source (compatible with PMTiles `Source`, e.g. Jupyter/Colab
 * `anywidget` `CommRangeSource` or custom authenticated storage adapters).
 */
export interface CustomRangeSource {
    /** Stable cache key identifying this archive instance (or provide `getKey()`). */
    key?: string;
    /** Optional PMTiles `Source`-compatible key getter. */
    getKey?(): string;
    /**
     * Optional cap on automatic PMTiles leaf-directory prefetch size in bytes
     * (`1 MiB` default on `CustomRangeSource`, or `0` to disable prefetching).
     */
    maxLeafPrefetchBytes?: number;
    /** Fetch `[offset, offset + length)` bytes from the archive. */
    getBytes(
        offset: number,
        length: number,
        signal?: AbortSignal,
        etag?: string
    ): Promise<{ data: ArrayBuffer; etag?: string; expires?: string; cacheControl?: string }>;
}

/** Input source accepted by `Deepgraph.create({ url })` and `plot.load(source)`. */
export type TileSource = string | File | CustomRangeSource;

/** Standard built-in color scales. */
export type ColorScaleName =
    | 'viridis'
    | 'plasma'
    | 'magma'
    | 'inferno'
    | 'rdbu'
    | 'coolwarm'
    | 'jet';

/** Experimental / domain-specific astronomy colormaps. */
export type ExperimentalColorScaleName = 'esa_age' | 'esa_dust' | 'esa_extinction';

/** Spatial/pixel aggregation mode in the Cartesian pipeline. */
export type AggregationStrategy = 'points' | 'mean_field';

/** Rendering shader pipeline: standard Cartesian plot or calibrated optical flux. */
export type RenderPipelineMode = 'cartesian' | 'optical';

/** Active pointer drag behavior on the canvas (`'pan'` default; `Shift+drag` triggers `shiftDragMode`). */
export type InteractionMode = 'pan' | 'brush' | 'lasso';

/** 2D axis-aligned rectangle brush in active **data coordinates** (`xField`/`yField` or spatial `dataBounds`). */
export interface RectBrushClause {
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
}

/** Closed 2D polygon lasso (`>= 3` vertices) in active **data coordinates**. */
export interface LassoPolygonClause {
    /** Data-space `[x, y]` vertices; simplified via Ramer-Douglas-Peucker to `<= 64` GPU vertices. */
    points: ReadonlyArray<readonly [number, number]>;
}

/** Zero-SQL GPU Fast-Path filter evaluated directly in `ComputeRasterizer` & `PickingLayer`. */
export interface DeepgraphFilter {
    /** 2D rectangle brush in current active X/Y data coordinates. */
    rect?: RectBrushClause | null;
    /** 2D freehand polygon lasso in current active X/Y data coordinates. */
    lasso?: LassoPolygonClause | null;
    /** Numeric `[min, max]` intervals keyed by GPU-resident column (`color`, `size`, `x`, `y`, plus up to 2 spatial extra columns). */
    ranges?: Record<string, readonly [number, number]>;
    /** Allowed category labels or integer codes keyed by categorical column (`<= 512` codes on GPU fast path). */
    categories?: Record<string, ReadonlyArray<string | number>>;
    /**
     * Opacity multiplier for unselected context points in Optical (`Mode 0`) and Point (`Mode 1`) pipelines
     * (`0.0` default = hard cull; e.g. `0.08` renders dimmed background context via a 2-pass resolve while `PickingLayer` skips unselected points).
     * Mean-field (`Mode 2`) and flow modes (`Modes 3/4`) always hard-cull unselected rows so averages remain unbiased.
     */
    unselectedOpacity?: number;
}

/** Geometric selection predicate emitted on `'brush'` (live drag) and `'select'` (committed `pointerup` / `click` / `clearSelection`). */
export interface SelectionGeometryClause {
    kind: 'point' | 'rect' | 'lasso' | 'none';
    xField: string | null;
    yField: string | null;
    rect?: RectBrushClause;
    lasso?: LassoPolygonClause;
}

/** Curved quadratic Bézier graph/kNN arc specification (`EdgeOverlay`). */
export interface EdgeSpec {
    sourceRowId?: number | bigint;
    targetRowId?: number | bigint;
    /** Data-space source/target coordinates (renders edges even when endpoints reside in unloaded deep tiles). */
    source?: { x: number; y: number };
    target?: { x: number; y: number };
    /** Optional per-column values so edges interpolate smoothly across 2-second chart axis transitions. */
    sourceAttributes?: Record<string, unknown>;
    targetAttributes?: Record<string, unknown>;
    weight?: number;
    /** Optional per-edge curvature override (`EdgeStyleOptions.curvature` default). */
    curvature?: number;
    colorHex?: number;
    directed?: boolean;
}

/** Visual styling options for curved quadratic Bézier graph arcs (`EdgeOverlay`). */
export interface EdgeStyleOptions {
    /** Base stroke width in CSS pixels (`1.35` default) or `[minPx, maxPx]` mapped across `weightDomain`. */
    width?: number | [number, number];
    weightDomain?: [number, number];
    /** Signed normal bow ratio relative to chord length (`0.18` default; `0` = straight segments). */
    curvature?: number;
    /** Base edge opacity (`0.68` default). */
    opacity?: number;
    /** Default `0xRRGGBB` edge tint when `EdgeSpec.colorHex` is omitted (`0x7dd3fc` cyan-sky default). */
    colorHex?: number;
    /** Taper alpha & brightness from source -> target (`true` default). */
    directed?: boolean;
    /** Optional subtle animated pulse along directed arcs (`false` default — stationary frames remain `0.0000%` shimmer!). */
    animatePulse?: boolean;
}

/** Async or synchronous 1-hop ego-network / kNN edge resolver invoked on point hover or click. */
export type EgoEdgeResolver = (
    hit: PointHitEvent,
    trigger: 'hover' | 'click',
    signal: AbortSignal
) => Promise<EdgeSpec[]> | EdgeSpec[];

/** Experimental / domain-specific astronomy & flow modes configured via `preset.experimental` or `setViewOptions({ experimental })`. */
export interface ExperimentalOptions {
    colorScale?: ExperimentalColorScaleName;
    aggregation?: 'trails' | 'streamlines';
    metricType?: number;
    auxFields?: [string, string, string];
    magnitudeFields?: string[];
    opticalColorField?: string;
    legacyFlipY?: boolean;
    trailMode?: number;
    trailTimeKyr?: number;
    minParallax?: number;
    animateFlow?: boolean;
}

export interface AxisChannelEncoding {
    /** Column name for chart mode, or `null` / omitted to use the archive's spatial quadtree positions. */
    field?: string | null;
    transform?: 'literal' | 'linear';
    domain?: [number, number];
    range?: [number, number];
}

export interface ColorChannelEncoding {
    field: string;
    scale?: ColorScaleName | ExperimentalColorScaleName;
    domain?: [number, number];
    invert?: boolean;
}

export interface SizeChannelEncoding {
    field: string;
    domain?: [number, number];
    range?: [number, number];
}

/**
 * Public declarative encoding specification (`Deepgraph.create` & `plot.setEncoding`).
 * - Omitting `x`/`y` on `create`/`load` uses the archive's spatial positions.
 * - Passing `x: null` or `y: null` in `setEncoding` resets an axis back to archive spatial positions.
 */
export interface DeepgraphEncoding {
    x?: string | null | AxisChannelEncoding;
    y?: string | null | AxisChannelEncoding;
    color?: string | ColorChannelEncoding;
    size?: string | SizeChannelEncoding;
    flipY?: boolean;
    aggregation?: AggregationStrategy;
    lightBackground?: boolean;
    smoothingKernel?: number;
    /** User multiplier on rendered point diameter (`1.0` default). */
    pointSize?: number;
    /** Continuous log-zoom point growth exponent in `pow(zoom, zoomBalance)` (`0.22` default). */
    zoomBalance?: number;
    /** Per-point opacity override (`0..1`), or `'auto'` (`'auto'` default) for zoom-adaptive opacity. */
    pointOpacity?: 'auto' | number;
    /** Tone-mapping exposure multiplier, or `'auto'` (`'auto'` default). */
    exposure?: 'auto' | number;
}

export interface NumericColumnStat {
    type?: 'numeric';
    min: number;
    max: number;
    null_count?: number;
}

export interface CategoricalColumnStat {
    type: 'categorical';
    categories: string[];
    null_count?: number;
}

export type ColumnStat = NumericColumnStat | CategoricalColumnStat;

/** Domain-agnostic dataset preset passed directly or resolved from archive metadata on `load()`. */
export interface DeepgraphPreset {
    name?: string;
    /** Default root world bounds when archive metadata omits `bounds` (e.g. legacy v0.0 files). */
    fallbackRootBounds?: BoundingBox;
    /** Fallback per-column numeric bounds merged beneath archive `metadata.stats`. */
    fallbackStats?: Record<string, { min: number; max: number }>;
    /** Initial shader pipeline mode (`'cartesian'` default, or `'optical'` for flux-calibrated sky maps). */
    renderPipeline?: RenderPipelineMode;
    /** Default declarative encoding applied when archive `metadata.view` omits channels. */
    defaultEncoding?: DeepgraphEncoding;
    /** Regex matching discrete small-integer (`0..255`) columns to auto-register as categorical. */
    discreteCategoryPattern?: RegExp;
    /** Optional domain/experimental extensions (custom astronomy ramps, proper-motion trails/streamlines). */
    experimental?: ExperimentalOptions;
}

export type DeepgraphPresetResolver = (
    rawMetadata: Record<string, unknown>,
    source: TileSource
) => DeepgraphPreset | undefined;

export type DeepgraphPresetInput = DeepgraphPreset | DeepgraphPresetResolver;

/** Low-level shader & viewport parameters controlled via `setViewOptions`. */
export interface ViewOptionsPatch {
    pipeline?: RenderPipelineMode;
    detailOffset?: number;
    aggregation?: AggregationStrategy;
    /** Column name or `null` for spatial X. */
    xField?: string | null;
    /** Column name or `null` for spatial Y. */
    yField?: string | null;
    colorField?: string;
    colorMin?: number;
    colorMax?: number;
    colorScale?: ColorScaleName | ExperimentalColorScaleName;
    invertScale?: boolean;
    lightBackground?: boolean;
    smoothingKernel?: number;
    sizeField?: string;
    sizeMin?: number;
    sizeMax?: number;
    sizeRange?: [number, number];
    flipY?: boolean;
    pointSize?: number;
    zoomBalance?: number;
    pointOpacity?: 'auto' | number;
    exposure?: 'auto' | number;
    experimental?: ExperimentalOptions;
}

export interface ViewStateSnapshot {
    pipeline: RenderPipelineMode;
    detailOffset: number;
    aggregation: AggregationStrategy | 'trails' | 'streamlines';
    /** `null` when using the archive's spatial X coordinate. */
    xField: string | null;
    /** `null` when using the archive's spatial Y coordinate. */
    yField: string | null;
    xTransform: 'literal' | 'linear';
    yTransform: 'literal' | 'linear';
    xDomain?: [number, number];
    yDomain?: [number, number];
    xRange?: [number, number];
    yRange?: [number, number];
    colorField: string;
    colorMin: number;
    colorMax: number;
    colorScale: ColorScaleName | ExperimentalColorScaleName;
    invertScale: boolean;
    lightBackground: boolean;
    smoothingKernel: number;
    sizeField: string;
    sizeMin: number;
    sizeMax: number;
    sizeRange: [number, number];
    flipY: boolean;
    effectiveFlipY: boolean;
    pointSize: number;
    zoomBalance: number;
    pointOpacity: 'auto' | number;
    exposure: 'auto' | number;
    experimental: Required<
        Pick<
            ExperimentalOptions,
            'metricType' | 'auxFields' | 'trailMode' | 'trailTimeKyr' | 'minParallax' | 'animateFlow'
        >
    >;
}

export interface DatasetMetadataSummary {
    format: string;
    version: string;
    tiling: string;
    isLegacyV00: boolean;
    archiveMaxZoom: number;
    totalRows?: number;
    droppedRows?: number;
    droppedColumns?: string[];
    rootBounds: BoundingBox;
    dataBounds?: { min_x: number; max_x: number; min_y: number; max_y: number };
    /** User payload columns available for visual encoding (excludes internal `x_u16`, `y_u16`, `z`, `row_id`). */
    schemaFields: string[];
    magnitudeField: string;
    presetName?: string;
    stats: Record<string, ColumnStat>;
    categories: Record<string, string[]>;
    pyramidSummary: string;
    rawMetadata: Record<string, unknown>;
}

export interface TelemetrySnapshot {
    fps: number;
    cameraZoom: number;
    currentZ: number;
    pyramidSummary: string;
    activeTiles: number;
    /** Live tile-slot `GPUBuffer` VRAM in MiB. */
    estimatedVramMB: number;
    /** Total live library-owned `GPUBuffer` bytes (tile slots + rasterizer + picking + overlay) in MiB. */
    totalGpuMB: number;
    /** Total retained/pooled CPU `ArrayBuffer` bytes in MiB. */
    cpuPoolMB: number;
    activePoints: number;
    loadedPoints: number;
    netLatencyMs: number;
    workerLatencyMs: number;
    coldBootMs: number;
    sloP95Ms: number;
    sloBudgetMs: number;
    isSloViolated: boolean;
    cullMs: number;
    uploadMs: number;
    renderMs: number;
    totalFrameMs: number;
    trailTimeKyr: number;
}

/** Point hit payload emitted by `'click'`, `'hover'`, and `plot.pickAt(...)` via the GPU `PickingLayer`. */
export interface PointHitEvent {
    /** Row identifier (`0 .. N-1` exact IEEE-754 integer for all standard archives up to `2^53 - 1`). */
    rowId: number;
    /** Exact 64-bit `BigInt` row identifier populated when `--id-col` exceeds `Number.MAX_SAFE_INTEGER` (`2^53 - 1`). */
    rowIdBigInt?: bigint;
    screenX: number;
    screenY: number;
    dataX: number;
    dataY: number;
    attributes: Record<string, unknown>;
}

export interface PointSelectEvent {
    points: PointHitEvent[];
    /** Geometric predicate over active X/Y data coordinates for querying all `N` rows in DuckDB/Mosaic/Marimo. */
    clause?: SelectionGeometryClause;
    /** Loaded-tile `row_id`s matching a committed brush/lasso when `enableRowIdIndex` is active. */
    rowIds?: number[];
    rowIdsBigInt?: bigint[];
}

/** Optional sidecar hook (`D8(2)`) to enrich picked `rowId`s with cold/dropped columns (e.g. from Parquet). */
export type RowAttributeResolver = (
    rowId: number,
    retainedAttributes: Record<string, unknown>,
    rowIdBigInt?: bigint,
    signal?: AbortSignal
) => Promise<Record<string, unknown>> | Record<string, unknown>;

export interface DeepgraphOptions {
    /** Initial `.arrowtiles` URL, local `File`, or `CustomRangeSource` to load immediately on creation. */
    url?: TileSource;
    /** Initial visual channel encoding. */
    encoding?: DeepgraphEncoding;
    /** Dataset preset object or `(rawMetadata, source) => DeepgraphPreset | undefined` resolver. */
    preset?: DeepgraphPresetInput;
    /** Initial Z-level overfetch offset relative to `floor(log2(zoom))` (default `2`). */
    detailOffset?: number;
    /** Maximum decoded tiles retained in the client LRU cache (default `250`, minimum `32`). */
    maxTileCache?: number;
    /** Optional WebGPU adapter power preference (`'high-performance'` default, or `'low-power'` to prefer integrated GPUs). */
    powerPreference?: 'high-performance' | 'low-power';
    /** Initial pointer drag mode (`'pan'` default). */
    interactionMode?: InteractionMode;
    /** Drag mode triggered when holding `Shift` while in `'pan'` mode (`'brush'` default). */
    shiftDragMode?: 'brush' | 'lasso' | 'none';
    /** Retain compact per-tile `row_id` arrays in workers for `setFilteredRowIds` / `getSelectedRowIds` (`false` default; auto-enabled on demand). */
    enableRowIdIndex?: boolean;
    /** Initial declarative GPU cross-filter. */
    filter?: DeepgraphFilter;
    /** Initial static curved graph/kNN edges (`EdgeOverlay`). */
    edges?: EdgeSpec[];
    /** Initial visual styling for `EdgeOverlay` arcs. */
    edgeStyle?: EdgeStyleOptions;
    /** Optional async/sync resolver for 1-hop ego-network / kNN arcs on hover or click. */
    resolveEdges?: EgoEdgeResolver;
    /** Whether `resolveEdges` runs on `'hover'` as well as `'click'` (`true` default when `resolveEdges` is set). */
    resolveEdgesOnHover?: boolean;
    /** Optional async resolver for dropped/sidecar columns on point inspect (`D8(2)`). */
    resolveRowAttributes?: RowAttributeResolver;
    /**
     * Whether `resolveRowAttributes` should also run on `'hover'` picks (`false` default).
     * By default, `resolveRowAttributes` runs only on `'click'` and `plot.pickAt(...)` to avoid
     * firing sidecar/backend requests during rapid pointer movement.
     */
    resolveRowAttributesOnHover?: boolean;
}

export interface LoadDatasetOptions {
    preset?: DeepgraphPresetInput;
    encoding?: DeepgraphEncoding;
}

/** Viewport state expressed in **data coordinates** (`dataBounds` when spatial, active axis `domain` on chart axes). */
export interface ViewportState {
    x: number;
    y: number;
    zoom: number;
}

export interface DeepgraphEventMap {
    click: PointHitEvent | null;
    hover: PointHitEvent | null;
    select: PointSelectEvent;
    brush: SelectionGeometryClause;
    telemetry: TelemetrySnapshot;
    load: DatasetMetadataSummary;
    error: Error;
}

export interface DeepgraphPlot {
    load(source: TileSource, options?: LoadDatasetOptions): Promise<DatasetMetadataSummary>;
    setEncoding(encoding: DeepgraphEncoding): void;
    getEncoding(): DeepgraphEncoding;
    setViewOptions(patch: ViewOptionsPatch): void;
    getViewState(): ViewStateSnapshot;
    getMetadata(): DatasetMetadataSummary | null;
    getCategories(field: string): string[] | undefined;
    getCategoryColorHex(index: number): number;
    /** Returns the current viewport center `(x, y)` in **data coordinates** and camera `zoom`. */
    getViewport(): ViewportState;
    /** Pans/zooms the viewport using **data coordinates** `(x, y)` and magnification `zoom`. */
    setViewport(viewport: Partial<ViewportState>): void;
    /** Resets the viewport to frame the full active dataset bounds. */
    fitView(): void;
    /** Programmatically picks the highest-priority rendered point within a `9x9 CSS px` window around `(cssX, cssY)`. */
    pickAt(cssX: number, cssY: number): Promise<PointHitEvent | null>;
    /** Sets the active left-drag pointer mode (`'pan'`, `'brush'`, or `'lasso'`). */
    setInteractionMode(mode: InteractionMode): void;
    /** Returns the active left-drag pointer mode. */
    getInteractionMode(): InteractionMode;
    /** Applies or clears a zero-SQL GPU cross-filter (`rect`, `lasso`, `ranges`, `categories`, `unselectedOpacity`). */
    setFilter(filter: DeepgraphFilter | null): void;
    /** Returns a snapshot of the active declarative `DeepgraphFilter`, or `null` when unfiltered. */
    getFilter(): DeepgraphFilter | null;
    /** Returns `true` when `filter` can execute 100% on the GPU uniform fast path (`<= 6` GPU-resident range slots, `<= 2` categorical columns with `< 512` codes). */
    canApplyFilterOnGpu(filter: DeepgraphFilter | null): boolean;
    /** Filters visible points by exact `row_id` membership compiled off-main-thread into `@binding(11)` `32 KiB` slot bitmasks (`null` clears). */
    setFilteredRowIds(
        ids: Iterable<number | bigint | string> | Uint32Array | BigUint64Array | null
    ): Promise<void>;
    /** Clears any active interactive brush/lasso, selected point rings, and ego-graph arcs. */
    clearSelection(): void;
    /** Queries loaded tiles in workers for `row_id`s matching the active selection/filter. */
    getSelectedRowIds(): Promise<{ rowIds: number[]; rowIdsBigInt: bigint[] }>;
    /** Replaces the static curved graph/kNN edge layer (`<= 100,000` instanced quadratic Bézier arcs). */
    setEdges(edges: ReadonlyArray<EdgeSpec>, style?: EdgeStyleOptions): void;
    /** Updates visual styling for static and ego-resolved graph arcs. */
    setEdgeStyle(style: EdgeStyleOptions): void;
    on<K extends keyof DeepgraphEventMap>(
        event: K,
        handler: (payload: DeepgraphEventMap[K]) => void
    ): () => void;
    off<K extends keyof DeepgraphEventMap>(
        event: K,
        handler: (payload: DeepgraphEventMap[K]) => void
    ): void;
    destroy(): void;
}

/** Static factory & WebGPU capability probe exported as `Deepgraph` from `src/index.ts`. */
export interface DeepgraphStatic {
    create(container: HTMLElement, options?: DeepgraphOptions): Promise<DeepgraphPlot>;
    isSupported(): boolean;
}
