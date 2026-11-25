import { PMTilesClient } from '../PMTilesClient.ts';
import type { BoundingBox, DecodedTileRow, TileData } from '../PMTilesClient.ts';
import type { AnyColorScaleName } from '../core/TileGpuStore.ts';
import type {
    ColumnStat,
    DatasetMetadataSummary,
    DeepgraphPreset,
    DeepgraphPresetInput,
    TileSource,
} from './types.ts';

export interface EmbeddedViewConfig {
    mode?: string;
    colorField?: string;
    colorMin?: number;
    colorMax?: number;
    sizeField?: string;
    sizeMin?: number;
    sizeMax?: number;
    colorScale?: AnyColorScaleName;
    preserveAspect: boolean;
    xField?: string;
    yField?: string;
}

const DEFAULT_DISCRETE_CLUSTER_PATTERN = /^(clusters?(_|$)|cat_|label|class)/i;

function formatPyramidSummary(
    metadata: Record<string, any> | null | undefined,
    isLegacyV00: boolean
): string {
    let priorityText = isLegacyV00 ? 'Priority: legacy v0.0' : 'Priority: unknown';
    const prio = metadata?.priority;
    if (prio && typeof prio === 'object') {
        if (prio.kind === 'random') {
            priorityText = `Priority: random (seed=${prio.seed ?? 42})`;
        } else if (prio.kind === 'column') {
            priorityText = `Priority: ${String(prio.column ?? '?')} ${String(prio.order ?? 'asc').toUpperCase()}`;
        } else if (prio.kind) {
            priorityText = `Priority: ${String(prio.kind)}`;
        }
    }

    let samplingText = isLegacyV00 ? 'Sampling: legacy v0.0' : 'Sampling: unknown';
    if (typeof metadata?.sampling === 'string' && metadata.sampling.length > 0) {
        if (
            metadata.sampling === 'even_coverage' &&
            Number.isFinite(Number(metadata.grid_size))
        ) {
            samplingText = `Sampling: even_coverage (${Number(metadata.grid_size)}x${Number(metadata.grid_size)})`;
        } else {
            samplingText = `Sampling: ${metadata.sampling}`;
        }
    }

    return `${priorityText} | ${samplingText}`;
}

export class DatasetSession {
    public readonly tileClient: PMTilesClient;
    public readonly summary: DatasetMetadataSummary;
    public readonly embeddedView: EmbeddedViewConfig;
    public readonly preset?: DeepgraphPreset;
    public readonly colOrigins: Record<string, number>;

    private constructor(
        tileClient: PMTilesClient,
        summary: DatasetMetadataSummary,
        embeddedView: EmbeddedViewConfig,
        colOrigins: Record<string, number>,
        preset?: DeepgraphPreset
    ) {
        this.tileClient = tileClient;
        this.summary = summary;
        this.embeddedView = embeddedView;
        this.colOrigins = colOrigins;
        this.preset = preset;
    }

    public static async open(
        source: TileSource,
        options: {
            preset?: DeepgraphPresetInput;
            maxTileCache?: number;
            signal?: AbortSignal;
            onDictionariesLoaded?: (dicts: Record<string, string[]>) => void;
            onTileUnloaded?: (tileId: string) => void;
        } = {}
    ): Promise<DatasetSession> {
        const {
            preset: presetInput,
            maxTileCache = 250,
            signal,
            onDictionariesLoaded,
            onTileUnloaded,
        } = options;

        if (signal?.aborted) {
            throw new DOMException('Dataset load aborted', 'AbortError');
        }

        const initialBounds: BoundingBox = { minX: -1.0, maxX: 1.0, minY: -1.0, maxY: 1.0 };
        const tileClient = new PMTilesClient(
            source,
            [],
            initialBounds,
            maxTileCache,
            null,
            14
        );

        let onAbort: (() => void) | null = null;
        const abortPromise = signal
            ? new Promise<never>((_, reject) => {
                  onAbort = () => {
                      tileClient.destroy();
                      reject(new DOMException('Dataset load aborted', 'AbortError'));
                  };
                  if (signal.aborted) {
                      onAbort();
                  } else {
                      signal.addEventListener('abort', onAbort, { once: true });
                  }
              })
            : null;

        const openWork = async (): Promise<DatasetSession> => {
            const { header: archiveHeader, metadata } = await tileClient.getHeaderAndMetadata();
            const rawMeta: any = metadata || {};

            let isLegacyV00 = false;
            if (rawMeta && (rawMeta.format !== undefined || rawMeta.version !== undefined)) {
                if (rawMeta.format !== 'arrowtiles') {
                    throw new Error(
                        `Unsupported archive format '${rawMeta.format}' (expected 'arrowtiles')`
                    );
                }
                const verStr = typeof rawMeta.version === 'string' ? rawMeta.version : '';
                const verMatch = /^(\d+)\.(\d+)$/.exec(verStr);
                if (!verMatch) {
                    throw new Error(
                        `Invalid ArrowTiles version '${rawMeta.version}' (expected '<major>.<minor>')`
                    );
                }
                const major = Number(verMatch[1]);
                const minor = Number(verMatch[2]);
                if (major !== 0) {
                    throw new Error(
                        `Unsupported ArrowTiles major version '${verStr}' (viewer supports 0.x up to '0.1')`
                    );
                }
                if (minor > 1) {
                    throw new Error(
                        `Unsupported ArrowTiles minor version '${verStr}' (viewer supports up to '0.1')`
                    );
                }
                if (rawMeta.tiling !== undefined && rawMeta.tiling !== 'quadtree-2d') {
                    throw new Error(
                        `Unsupported ArrowTiles tiling '${rawMeta.tiling}' (viewer supports 'quadtree-2d')`
                    );
                }
                if (rawMeta.required_features !== undefined) {
                    if (
                        !Array.isArray(rawMeta.required_features) ||
                        !rawMeta.required_features.every((f: unknown) => typeof f === 'string')
                    ) {
                        throw new Error(
                            `Invalid ArrowTiles required_features (expected array of strings)`
                        );
                    }
                    if (rawMeta.required_features.length > 0) {
                        throw new Error(
                            `Unsupported ArrowTiles required_features [${rawMeta.required_features.join(', ')}] (v0.1 viewer supports none)`
                        );
                    }
                }
            } else {
                isLegacyV00 = true;
                console.warn(
                    '[ArrowTiles] Legacy v0.0 archive detected (missing format/version metadata); opening in compatibility mode.'
                );
            }

            const preset: DeepgraphPreset | undefined =
                typeof presetInput === 'function' ? presetInput(rawMeta, source) : presetInput;

            const archiveMaxZoom = isLegacyV00
                ? archiveHeader.maxZoom !== undefined && archiveHeader.maxZoom <= 20
                    ? archiveHeader.maxZoom
                    : 14
                : Math.max(0, Math.min(20, archiveHeader.maxZoom ?? 14));

            let schemaBuffer: Uint8Array | null = null;
            if (typeof rawMeta.arrow_schema === 'string' && rawMeta.arrow_schema.length > 0) {
                const binaryString = atob(rawMeta.arrow_schema);
                schemaBuffer = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    schemaBuffer[i] = binaryString.charCodeAt(i);
                }
            } else if (!isLegacyV00) {
                throw new Error(
                    'Missing or empty metadata.arrow_schema in ArrowTiles v0.1 archive'
                );
            }

            const viewObj =
                rawMeta && typeof rawMeta.view === 'object' && rawMeta.view !== null
                    ? rawMeta.view
                    : rawMeta;

            const parsedStatsRaw = rawMeta.stats
                ? typeof rawMeta.stats === 'string'
                    ? JSON.parse(rawMeta.stats)
                    : rawMeta.stats
                : undefined;

            const mergedStats: Record<string, ColumnStat> = {};
            if (preset?.fallbackStats) {
                for (const [k, v] of Object.entries(preset.fallbackStats)) {
                    mergedStats[k] = { type: 'numeric', min: v.min, max: v.max };
                }
            }
            const categories: Record<string, string[]> = {};
            if (parsedStatsRaw && typeof parsedStatsRaw === 'object') {
                for (const [colName, colStat] of Object.entries<any>(parsedStatsRaw)) {
                    if (!colStat || typeof colStat !== 'object') continue;
                    if (colStat.type === 'categorical' && Array.isArray(colStat.categories)) {
                        const labels = colStat.categories.map((c: any) => String(c));
                        categories[colName] = labels;
                        mergedStats[colName] = {
                            type: 'categorical',
                            categories: labels,
                            null_count:
                                typeof colStat.null_count === 'number'
                                    ? colStat.null_count
                                    : undefined,
                        };
                    } else if (Number.isFinite(colStat.min) && Number.isFinite(colStat.max)) {
                        mergedStats[colName] = {
                            type: 'numeric',
                            min: Number(colStat.min),
                            max: Number(colStat.max),
                            null_count:
                                typeof colStat.null_count === 'number'
                                    ? colStat.null_count
                                    : undefined,
                        };
                    }
                }
            }

            const parseOptionalNum = (val: unknown): number | undefined => {
                if (val === undefined || val === null || val === '') return undefined;
                const n = Number(val);
                return Number.isFinite(n) ? n : undefined;
            };

            const embeddedView: EmbeddedViewConfig = {
                mode: viewObj?.mode ?? rawMeta.mode,
                colorField: viewObj?.colorField ?? rawMeta.colorField,
                colorMin: parseOptionalNum(viewObj?.colorMin ?? rawMeta.colorMin),
                colorMax: parseOptionalNum(viewObj?.colorMax ?? rawMeta.colorMax),
                sizeField: viewObj?.sizeField ?? rawMeta.sizeField,
                sizeMin: parseOptionalNum(viewObj?.sizeMin ?? rawMeta.sizeMin),
                sizeMax: parseOptionalNum(viewObj?.sizeMax ?? rawMeta.sizeMax),
                colorScale: viewObj?.colorScale ?? rawMeta.colorScale,
                preserveAspect: viewObj?.preserveAspect ?? true,
                xField: viewObj?.xField ?? rawMeta.xField,
                yField: viewObj?.yField ?? rawMeta.yField,
            };

            let rootBounds: BoundingBox;
            let dataBounds: DatasetMetadataSummary['dataBounds'];
            const rawBounds = rawMeta.bounds;
            if (
                rawBounds &&
                typeof rawBounds === 'object' &&
                Number.isFinite(rawBounds.min_x) &&
                Number.isFinite(rawBounds.max_x) &&
                Number.isFinite(rawBounds.min_y) &&
                Number.isFinite(rawBounds.max_y)
            ) {
                dataBounds = {
                    min_x: Number(rawBounds.min_x),
                    max_x: Number(rawBounds.max_x),
                    min_y: Number(rawBounds.min_y),
                    max_y: Number(rawBounds.max_y),
                };
                if (embeddedView.preserveAspect !== false) {
                    const spanX = Math.max(1e-9, dataBounds.max_x - dataBounds.min_x);
                    const spanY = Math.max(1e-9, dataBounds.max_y - dataBounds.min_y);
                    const dataAspect = spanX / spanY;
                    if (dataAspect > 2.0) {
                        const halfW = 2.0;
                        const halfH = Math.max(0.1, 2.0 / dataAspect);
                        rootBounds = { minX: -halfW, maxX: halfW, minY: -halfH, maxY: halfH };
                    } else {
                        const halfH = 1.0;
                        const halfW = Math.max(0.1, dataAspect);
                        rootBounds = { minX: -halfW, maxX: halfW, minY: -halfH, maxY: halfH };
                    }
                } else {
                    rootBounds = preset?.fallbackRootBounds ?? initialBounds;
                }
            } else {
                rootBounds = preset?.fallbackRootBounds ?? initialBounds;
            }

            tileClient.configureArchive(schemaBuffer, archiveMaxZoom, rootBounds);

            if (onDictionariesLoaded) {
                tileClient.onDictionariesLoaded = (dicts) => {
                    Object.assign(categories, dicts);
                    onDictionariesLoaded(dicts);
                };
            }
            if (onTileUnloaded) {
                tileClient.onTileUnloaded = onTileUnloaded;
            }

            let schemaFields: string[] = [];
            let parsedSchema: { name: string; type: string }[] = [];
            const intFieldNames = new Set<string>();
            try {
                parsedSchema = await tileClient.getSchema();
                for (const f of parsedSchema) {
                    if (/^(u?int(8|16|32|64))$/i.test(f.type)) {
                        intFieldNames.add(f.name);
                    }
                }
                schemaFields = parsedSchema
                    .filter(
                        (f) =>
                            f.name !== 'x_u16' &&
                            f.name !== 'y_u16' &&
                            f.name !== 'z' &&
                            f.name !== 'row_id' &&
                            !/^(utf8|largeutf8|utf8view|binary|largebinary|binaryview|fixedsizebinary)$/i.test(
                                f.type
                            )
                    )
                    .map((f) => f.name);
            } catch (err) {
                if (!isLegacyV00) {
                    throw new Error(
                        `Failed to parse metadata.arrow_schema in ArrowTiles v0.1 archive: ${
                            err instanceof Error ? err.message : String(err)
                        }`
                    );
                }
                console.warn('Failed to parse legacy v0.0 schema, falling back to empty fields list', err);
            }

            const discretePattern =
                preset?.discreteCategoryPattern ?? DEFAULT_DISCRETE_CLUSTER_PATTERN;
            for (const colName of schemaFields) {
                if (categories[colName]) continue;
                if (!intFieldNames.has(colName)) continue;
                if (!discretePattern.test(colName)) continue;
                const st = mergedStats[colName];
                if (
                    st &&
                    st.type !== 'categorical' &&
                    Number.isFinite(st.min) &&
                    Number.isFinite(st.max) &&
                    st.min >= 0 &&
                    st.max >= st.min &&
                    st.max < 256 &&
                    Number.isInteger(st.min) &&
                    Number.isInteger(st.max)
                ) {
                    const labels: string[] = [];
                    for (let k = 0; k <= st.max; k++) labels.push(String(k));
                    categories[colName] = labels;
                }
            }

            // Compute internal 1D Relative-to-Origin (`RTO`) offsets for temporal columns and
            // narrow-span high-offset numeric columns BEFORE any tile worker decodes payloads!
            const excludedRtoCols = new Set<string>([
                'x_u16',
                'y_u16',
                'z',
                'row_id',
                'phot_g_mean_mag',
                'abs_m',
                'bp_rp',
                'parallax',
                'teff_gspphot',
                'pmra',
                'pmdec',
                'radial_velocity',
            ]);
            if (preset?.experimental?.opticalColorField) {
                excludedRtoCols.add(preset.experimental.opticalColorField);
            }
            for (const mf of preset?.experimental?.magnitudeFields ?? []) {
                if (mf) excludedRtoCols.add(mf);
            }
            for (const af of preset?.experimental?.auxFields ?? []) {
                if (af) excludedRtoCols.add(af);
            }

            const colOrigins: Record<string, number> = {};
            for (const f of parsedSchema) {
                if (excludedRtoCols.has(f.name) || categories[f.name]) continue;
                const st = mergedStats[f.name];
                if (
                    !st ||
                    st.type === 'categorical' ||
                    !Number.isFinite(st.min) ||
                    !Number.isFinite(st.max)
                ) {
                    continue;
                }
                const isTemporal = /^(date(32|64)?|timestamp)$/i.test(f.type);
                const maxAbs = Math.max(Math.abs(st.min), Math.abs(st.max));
                const isNarrowHighOffset =
                    maxAbs > 1e5 &&
                    st.max >= st.min &&
                    st.max - st.min < maxAbs * 1e-3;
                if ((isTemporal || isNarrowHighOffset) && st.min !== 0) {
                    colOrigins[f.name] = st.min;
                }
            }
            tileClient.setColOrigins(colOrigins);

            const candidateMagFields = preset?.experimental?.magnitudeFields ?? [];
            let magnitudeField = schemaFields[0] ?? 'x_u16';
            for (const cand of candidateMagFields) {
                if (schemaFields.includes(cand)) {
                    magnitudeField = cand;
                    break;
                }
            }
            if (candidateMagFields.length > 0 && !schemaFields.includes(magnitudeField)) {
                magnitudeField = candidateMagFields[candidateMagFields.length - 1];
            }

            const summary: DatasetMetadataSummary = {
                format: String(rawMeta.format ?? 'arrowtiles'),
                version: isLegacyV00 ? '0.0' : String(rawMeta.version ?? '0.1'),
                tiling: String(rawMeta.tiling ?? 'quadtree-2d'),
                isLegacyV00,
                archiveMaxZoom,
                totalRows: typeof rawMeta.total_rows === 'number' ? rawMeta.total_rows : undefined,
                droppedRows:
                    typeof rawMeta.dropped_rows === 'number' ? rawMeta.dropped_rows : undefined,
                droppedColumns: Array.isArray(rawMeta.dropped_columns)
                    ? rawMeta.dropped_columns
                    : undefined,
                rootBounds,
                dataBounds,
                schemaFields,
                magnitudeField,
                presetName: preset?.name,
                stats: mergedStats,
                categories,
                pyramidSummary: formatPyramidSummary(rawMeta, isLegacyV00),
                rawMetadata: rawMeta,
            };

            return new DatasetSession(tileClient, summary, embeddedView, colOrigins, preset);
        };

        try {
            if (abortPromise) {
                const workPromise = openWork();
                void workPromise.catch(() => {
                    // Suppress late rejection after abort already destroyed tileClient
                });
                return await Promise.race([workPromise, abortPromise]);
            }
            return await openWork();
        } catch (err) {
            tileClient.destroy();
            throw err;
        } finally {
            if (signal && onAbort) {
                signal.removeEventListener('abort', onAbort);
            }
        }
    }

    public get archiveMaxZoom(): number {
        return this.tileClient.archiveMaxZoom;
    }

    public get currentMaxZ(): number {
        return this.tileClient.currentMaxZ;
    }

    public get cacheChanged(): boolean {
        return this.tileClient.cacheChanged;
    }

    public set cacheChanged(val: boolean) {
        this.tileClient.cacheChanged = val;
    }

    public setRequestedColumns(cols: string[]): void {
        this.tileClient.setRequestedColumns(cols);
    }

    public setRetainRowIds(enabled: boolean): void {
        this.tileClient.setRetainRowIds(enabled);
    }

    public setRowIdFilter(
        ids: Iterable<number | bigint | string> | Uint32Array | BigUint64Array | null,
        probeRowIds?: BigUint64Array | null
    ): Promise<{ masks: Map<string, ArrayBuffer[]>; probeMatches: Uint8Array | null }> {
        return this.tileClient.setRowIdFilter(ids, probeRowIds);
    }

    public queryLoadedSpatialSelection(
        spec: {
            rect?: { minX: number; maxX: number; minY: number; maxY: number } | null;
            lasso?: ReadonlyArray<readonly [number, number]> | null;
            effectiveFlipY: boolean;
        },
        maxCount?: number
    ): Promise<{
        rowIds: number[];
        rowIdsBigInt: bigint[];
        totalMatched: number;
        truncated: boolean;
    }> {
        return this.tileClient.queryLoadedSpatialSelection(
            {
                rect: spec.rect,
                lasso: spec.lasso,
                dataBounds: this.summary.dataBounds,
                rootBounds: this.summary.rootBounds,
                effectiveFlipY: spec.effectiveFlipY,
            },
            maxCount
        );
    }

    public releaseTileCpuBuffers(tile: TileData): void {
        this.tileClient.releaseTileCpuBuffers(tile);
    }

    public getCpuMemoryStats() {
        return this.tileClient.getCpuMemoryStats();
    }

    public async decodeTileRow(
        z: number,
        x: number,
        y: number,
        rowIndex: number,
        categories: Record<string, string[]>,
        signal?: AbortSignal
    ): Promise<DecodedTileRow | null> {
        return this.tileClient.decodeTileRow(z, x, y, rowIndex, categories, signal);
    }

    public getVisibleTiles(
        frustumBox: BoundingBox,
        targetZ: number,
        flipY: boolean,
        isSpatialX: boolean,
        isSpatialY: boolean
    ): TileData[] {
        return this.tileClient.getVisibleTiles(
            frustumBox,
            targetZ,
            flipY,
            isSpatialX,
            isSpatialY
        );
    }

    public getFetchTelemetry(): {
        net: number[];
        worker: number[];
        coldBootMs: number;
        p95: number;
        sloBudgetMs: number;
        isSloViolated: boolean;
    } | null {
        return this.tileClient.fetchTelemetry ?? null;
    }

    public destroy(): void {
        this.tileClient.destroy();
    }
}
