/** Axis-aligned 2D bounding box in world or data space. */
export interface BoundingBox {
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
}

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
    source: string | File
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

/** Point hit payload emitted by `'click'` and `'hover'` once GPU PickingLayer lands in Phase 7. */
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
}

/** Optional sidecar hook (`D8(2)`) to enrich picked `rowId`s with cold/dropped columns (e.g. from Parquet). */
export type RowAttributeResolver = (
    rowId: number,
    retainedAttributes: Record<string, unknown>,
    rowIdBigInt?: bigint,
    signal?: AbortSignal
) => Promise<Record<string, unknown>> | Record<string, unknown>;

export interface DeepgraphOptions {
    /** Initial `.arrowtiles` URL or local `File` to load immediately on creation. */
    url?: string | File;
    /** Initial visual channel encoding. */
    encoding?: DeepgraphEncoding;
    /** Dataset preset object or `(rawMetadata, source) => DeepgraphPreset | undefined` resolver. */
    preset?: DeepgraphPresetInput;
    /** Initial Z-level overfetch offset relative to `floor(log2(zoom))` (default `2`). */
    detailOffset?: number;
    /** Maximum decoded tiles retained in the client LRU cache (default `250`). */
    maxTileCache?: number;
    /** Optional WebGPU adapter power preference (`'high-performance'` default, or `'low-power'` to prefer integrated GPUs). */
    powerPreference?: 'high-performance' | 'low-power';
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
    telemetry: TelemetrySnapshot;
    load: DatasetMetadataSummary;
    error: Error;
}

export interface DeepgraphPlot {
    load(source: string | File, options?: LoadDatasetOptions): Promise<DatasetMetadataSummary>;
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
