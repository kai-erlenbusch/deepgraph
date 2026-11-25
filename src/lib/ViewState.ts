import type { AnyColorScaleName, TileGpuStore } from '../core/TileGpuStore.ts';
import type { DatasetSession } from './DatasetSession.ts';
import type {
    AggregationStrategy,
    DeepgraphEncoding,
    DeepgraphPreset,
    RenderPipelineMode,
    ViewOptionsPatch,
    ViewStateSnapshot,
} from './types.ts';

function normalizeSpatialField(
    raw: string | null | undefined,
    defaultSpatial: 'x_u16' | 'y_u16'
): string | null {
    if (
        raw === null ||
        raw === undefined ||
        raw === '' ||
        raw === defaultSpatial ||
        raw === '$spatial_x' ||
        raw === '$spatial_y'
    ) {
        return null;
    }
    return raw;
}

export class ViewState {
    private pipeline: RenderPipelineMode = 'cartesian';
    private detailOffset = 2;
    private aggregation: AggregationStrategy | 'trails' | 'streamlines' = 'mean_field';
    private metricType = 0;
    private auxFields: [string, string, string] = ['', '', ''];

    // `null` = archive spatial position (`x_u16` / `y_u16` internally)
    private xField: string | null = null;
    private yField: string | null = null;
    private xTransform: 'literal' | 'linear' = 'linear';
    private yTransform: 'literal' | 'linear' = 'linear';
    private customXDomain?: [number, number];
    private customYDomain?: [number, number];
    private customXRange?: [number, number];
    private customYRange?: [number, number];

    private colorField = '';
    private colorMin = 0.0;
    private colorMax = 100.0;
    private colorScale: AnyColorScaleName = 'viridis';
    private invertScale = false;
    private lightBackground = false;
    private smoothingKernel = 1;
    private sizeField = '';
    private sizeMin = 0.0;
    private sizeMax = 100.0;
    private sizeRange: [number, number] = [0.5, 3.0];
    private flipY = false;
    private pointSize = 1.0;
    private zoomBalance = 0.22;
    private pointOpacity: 'auto' | number = 'auto';
    private exposure: 'auto' | number = 'auto';

    private opticalColorField = 'bp_rp';
    private trailMode = 0;
    private trailTimeKyr = 400;
    private minParallax = 8.0;
    private animateFlow = false;

    constructor(initialDetailOffset = 2) {
        this.detailOffset = initialDetailOffset;
    }

    /** Single canonical flipY truth (`05_PLAN` §7.1). */
    public get effectiveFlipY(): boolean {
        return this.pipeline === 'optical' ? false : this.flipY;
    }

    public get isSpatialX(): boolean {
        return (
            this.pipeline === 'optical' ||
            this.xField === null ||
            this.xField === 'x_u16' ||
            this.xField === 'y_u16'
        );
    }

    public get isSpatialY(): boolean {
        return (
            this.pipeline === 'optical' ||
            this.yField === null ||
            this.yField === 'y_u16' ||
            this.yField === 'x_u16'
        );
    }

    public get hasCustomWorldScale(): boolean {
        return (
            (!this.isSpatialX &&
                (this.xTransform === 'literal' || this.customXRange !== undefined)) ||
            (!this.isSpatialY &&
                (this.yTransform === 'literal' || this.customYRange !== undefined))
        );
    }

    public get currentDetailOffset(): number {
        return this.detailOffset;
    }

    public get isFlowAnimated(): boolean {
        return (
            this.animateFlow &&
            this.pipeline === 'cartesian' &&
            this.aggregation === 'trails'
        );
    }

    public advanceFlowEpoch(dtSec: number, currentEpochKyr: number): number {
        let next = currentEpochKyr + dtSec * 120.0;
        if (next > 1600.0) {
            next = 40.0;
        }
        this.trailTimeKyr = Math.round(next);
        return next;
    }

    public initFromSession(
        session: DatasetSession,
        preset?: DeepgraphPreset,
        encodingOverride?: DeepgraphEncoding
    ): void {
        const { summary, embeddedView } = session;
        const schemaFields = summary.schemaFields;
        const defaultField = schemaFields.includes('sort_metric')
            ? 'sort_metric'
            : schemaFields.length > 0
              ? schemaFields[0]
              : 'x_u16';

        const initialPipeline: RenderPipelineMode = preset?.renderPipeline ?? 'cartesian';
        this.opticalColorField = preset?.experimental?.opticalColorField ?? 'bp_rp';

        this.pipeline = initialPipeline;
        this.metricType = 0;
        this.auxFields = ['', '', ''];
        this.xField = normalizeSpatialField(embeddedView.xField, 'x_u16');
        this.yField = normalizeSpatialField(embeddedView.yField, 'y_u16');
        this.xTransform = 'linear';
        this.yTransform = 'linear';
        this.customXDomain = undefined;
        this.customYDomain = undefined;
        this.customXRange = undefined;
        this.customYRange = undefined;

        const initialColorField = embeddedView.colorField || defaultField;
        const colorStat = summary.stats[initialColorField];
        const statColorMin =
            colorStat && colorStat.type !== 'categorical' && Number.isFinite(colorStat.min)
                ? colorStat.min
                : undefined;
        const statColorMax =
            colorStat && colorStat.type !== 'categorical' && Number.isFinite(colorStat.max)
                ? colorStat.max
                : undefined;

        const initialSizeField =
            embeddedView.sizeField ||
            (preset?.experimental?.magnitudeFields?.length ? summary.magnitudeField : defaultField);
        const sizeStat = summary.stats[initialSizeField];
        const statSizeMin =
            sizeStat && sizeStat.type !== 'categorical' && Number.isFinite(sizeStat.min)
                ? sizeStat.min
                : undefined;
        const statSizeMax =
            sizeStat && sizeStat.type !== 'categorical' && Number.isFinite(sizeStat.max)
                ? sizeStat.max
                : undefined;

        this.colorField = initialColorField;
        this.colorMin = embeddedView.colorMin ?? statColorMin ?? 0.0;
        this.colorMax = embeddedView.colorMax ?? statColorMax ?? 100.0;
        this.colorScale = embeddedView.colorScale || 'viridis';
        this.invertScale = false;
        this.lightBackground = false;
        this.smoothingKernel = 1;
        this.sizeField = initialSizeField;
        this.sizeMin = embeddedView.sizeMin ?? statSizeMin ?? 0.0;
        this.sizeMax = embeddedView.sizeMax ?? statSizeMax ?? 100.0;
        this.sizeRange = [0.5, 3.0];
        this.pointSize = 1.0;
        this.zoomBalance = 0.22;
        this.pointOpacity = 'auto';
        this.exposure = 'auto';

        const legacyFlipDefault = preset?.experimental?.legacyFlipY ?? true;
        this.flipY = summary.isLegacyV00 && legacyFlipDefault ? true : false;
        this.trailMode = preset?.experimental?.trailMode ?? 0;
        this.trailTimeKyr = preset?.experimental?.trailTimeKyr ?? 400;
        this.minParallax = preset?.experimental?.minParallax ?? 8.0;
        this.animateFlow = preset?.experimental?.animateFlow ?? false;

        // Apply preset's `defaultEncoding` (`DeepgraphEncoding`) when archive `metadata.view` omits channels
        if (preset?.defaultEncoding) {
            this.applyEncoding(preset.defaultEncoding, summary.stats, summary.categories, {
                preserveOpticalIfUnchanged: true,
                skipIfEmbeddedViewSet: embeddedView,
            });
        }

        const isInitialCategorical = Boolean(summary.categories[this.colorField]);
        this.aggregation =
            this.pipeline === 'cartesian' && isInitialCategorical ? 'points' : 'mean_field';

        if (encodingOverride) {
            this.applyEncoding(encodingOverride, summary.stats, summary.categories);
        }
    }

    public applyEncoding(
        encoding: DeepgraphEncoding,
        stats?: Record<string, { type?: string; min?: number; max?: number }>,
        categories?: Record<string, string[]>,
        internalOpts?: {
            preserveOpticalIfUnchanged?: boolean;
            skipIfEmbeddedViewSet?: {
                colorField?: string;
                colorMin?: number;
                colorMax?: number;
                colorScale?: string;
                sizeField?: string;
                sizeMin?: number;
                sizeMax?: number;
                xField?: string;
                yField?: string;
            };
        }
    ): void {
        const emb = internalOpts?.skipIfEmbeddedViewSet;

        if (encoding.x !== undefined && !emb?.xField) {
            if (encoding.x === null || typeof encoding.x === 'string') {
                const nextX = normalizeSpatialField(encoding.x, 'x_u16');
                if (nextX !== this.xField || encoding.x === null) {
                    this.customXDomain = undefined;
                    this.customXRange = undefined;
                }
                this.xField = nextX;
                this.xTransform = 'linear';
            } else {
                const nextX =
                    encoding.x.field === undefined
                        ? this.xField
                        : normalizeSpatialField(encoding.x.field, 'x_u16');
                const fieldChanged = nextX !== this.xField;
                this.xField = nextX;
                if (encoding.x.transform !== undefined) {
                    this.xTransform = encoding.x.transform;
                }
                if (encoding.x.domain !== undefined) {
                    this.customXDomain = [encoding.x.domain[0], encoding.x.domain[1]];
                } else if (fieldChanged) {
                    this.customXDomain = undefined;
                }
                if (encoding.x.range !== undefined) {
                    this.customXRange = [encoding.x.range[0], encoding.x.range[1]];
                } else if (fieldChanged) {
                    this.customXRange = undefined;
                }
            }
        }

        if (encoding.y !== undefined && !emb?.yField) {
            if (encoding.y === null || typeof encoding.y === 'string') {
                const nextY = normalizeSpatialField(encoding.y, 'y_u16');
                if (nextY !== this.yField || encoding.y === null) {
                    this.customYDomain = undefined;
                    this.customYRange = undefined;
                }
                this.yField = nextY;
                this.yTransform = 'linear';
            } else {
                const nextY =
                    encoding.y.field === undefined
                        ? this.yField
                        : normalizeSpatialField(encoding.y.field, 'y_u16');
                const fieldChanged = nextY !== this.yField;
                this.yField = nextY;
                if (encoding.y.transform !== undefined) {
                    this.yTransform = encoding.y.transform;
                }
                if (encoding.y.domain !== undefined) {
                    this.customYDomain = [encoding.y.domain[0], encoding.y.domain[1]];
                } else if (fieldChanged) {
                    this.customYDomain = undefined;
                }
                if (encoding.y.range !== undefined) {
                    this.customYRange = [encoding.y.range[0], encoding.y.range[1]];
                } else if (fieldChanged) {
                    this.customYRange = undefined;
                }
            }
        }

        if (encoding.color !== undefined) {
            if (encoding.color === null) {
                if (!emb?.colorField) {
                    this.colorField = '';
                }
            } else if (typeof encoding.color === 'string') {
                if (!emb?.colorField) {
                    this.colorField = encoding.color;
                    const st = stats?.[this.colorField];
                    if (
                        st &&
                        st.type !== 'categorical' &&
                        Number.isFinite(st.min) &&
                        Number.isFinite(st.max)
                    ) {
                        this.colorMin = st.min!;
                        this.colorMax = st.max!;
                    }
                }
            } else {
                const fieldChanged = !emb?.colorField && encoding.color.field !== this.colorField;
                if (!emb?.colorField) {
                    this.colorField = encoding.color.field;
                }
                if (encoding.color.scale && !emb?.colorScale) {
                    this.colorScale = encoding.color.scale;
                }
                if (encoding.color.invert !== undefined) {
                    this.invertScale = encoding.color.invert;
                }
                if (encoding.color.domain && emb?.colorMin === undefined) {
                    this.colorMin = encoding.color.domain[0];
                    this.colorMax = encoding.color.domain[1];
                } else if (fieldChanged) {
                    const st = stats?.[this.colorField];
                    if (
                        st &&
                        st.type !== 'categorical' &&
                        Number.isFinite(st.min) &&
                        Number.isFinite(st.max)
                    ) {
                        this.colorMin = st.min!;
                        this.colorMax = st.max!;
                    }
                }
            }

            const requestedScale =
                encoding.color !== null && typeof encoding.color === 'object'
                    ? encoding.color.scale
                    : undefined;
            const keepsOptical =
                internalOpts?.preserveOpticalIfUnchanged ||
                (this.colorField === this.opticalColorField && requestedScale === undefined);
            if (this.pipeline === 'optical' && !keepsOptical) {
                this.pipeline = 'cartesian';
            }
            if (
                encoding.aggregation === undefined &&
                categories?.[this.colorField] &&
                this.aggregation === 'mean_field'
            ) {
                this.aggregation = 'points';
            }
        }

        if (encoding.size !== undefined) {
            if (encoding.size === null) {
                if (!emb?.sizeField) {
                    this.sizeField = '';
                }
            } else if (typeof encoding.size === 'string') {
                if (!emb?.sizeField) {
                    this.sizeField = encoding.size;
                    const st = stats?.[this.sizeField];
                    if (
                        st &&
                        st.type !== 'categorical' &&
                        Number.isFinite(st.min) &&
                        Number.isFinite(st.max)
                    ) {
                        this.sizeMin = st.min!;
                        this.sizeMax = st.max!;
                    }
                }
            } else {
                const fieldChanged = !emb?.sizeField && encoding.size.field !== this.sizeField;
                if (!emb?.sizeField) {
                    this.sizeField = encoding.size.field;
                }
                if (encoding.size.domain && emb?.sizeMin === undefined) {
                    this.sizeMin = encoding.size.domain[0];
                    this.sizeMax = encoding.size.domain[1];
                } else if (fieldChanged) {
                    const st = stats?.[this.sizeField];
                    if (
                        st &&
                        st.type !== 'categorical' &&
                        Number.isFinite(st.min) &&
                        Number.isFinite(st.max)
                    ) {
                        this.sizeMin = st.min!;
                        this.sizeMax = st.max!;
                    }
                }
                if (encoding.size.range) {
                    this.sizeRange = [encoding.size.range[0], encoding.size.range[1]];
                }
            }
        }

        if (encoding.flipY !== undefined) this.flipY = encoding.flipY;
        if (encoding.aggregation !== undefined) this.aggregation = encoding.aggregation;
        if (encoding.lightBackground !== undefined) this.lightBackground = encoding.lightBackground;
        if (encoding.smoothingKernel !== undefined) this.smoothingKernel = encoding.smoothingKernel;
        if (encoding.pointSize !== undefined && Number.isFinite(encoding.pointSize)) {
            this.pointSize = Math.max(0.1, encoding.pointSize);
        }
        if (encoding.zoomBalance !== undefined && Number.isFinite(encoding.zoomBalance)) {
            this.zoomBalance = Math.max(0.0, Math.min(1.0, encoding.zoomBalance));
        }
        if (encoding.pointOpacity !== undefined) {
            this.pointOpacity =
                encoding.pointOpacity === 'auto'
                    ? 'auto'
                    : Math.max(0.004, Math.min(1.0, encoding.pointOpacity));
        }
        if (encoding.exposure !== undefined) {
            this.exposure =
                encoding.exposure === 'auto'
                    ? 'auto'
                    : Math.max(0.05, Math.min(16.0, encoding.exposure));
        }
    }

    public patch(patch: ViewOptionsPatch): void {
        if (patch.pipeline !== undefined) this.pipeline = patch.pipeline;
        if (patch.detailOffset !== undefined) this.detailOffset = patch.detailOffset;
        if (patch.aggregation !== undefined) this.aggregation = patch.aggregation;
        if (patch.xField !== undefined) {
            const nextX = normalizeSpatialField(patch.xField, 'x_u16');
            if (nextX !== this.xField) {
                this.customXDomain = undefined;
                this.customXRange = undefined;
            }
            this.xField = nextX;
        }
        if (patch.yField !== undefined) {
            const nextY = normalizeSpatialField(patch.yField, 'y_u16');
            if (nextY !== this.yField) {
                this.customYDomain = undefined;
                this.customYRange = undefined;
            }
            this.yField = nextY;
        }
        if (patch.colorField !== undefined) this.colorField = patch.colorField;
        if (patch.colorMin !== undefined) this.colorMin = patch.colorMin;
        if (patch.colorMax !== undefined) this.colorMax = patch.colorMax;
        if (patch.colorScale !== undefined) this.colorScale = patch.colorScale;
        if (patch.invertScale !== undefined) this.invertScale = patch.invertScale;
        if (patch.lightBackground !== undefined) this.lightBackground = patch.lightBackground;
        if (patch.smoothingKernel !== undefined) this.smoothingKernel = patch.smoothingKernel;
        if (patch.sizeField !== undefined) this.sizeField = patch.sizeField;
        if (patch.sizeMin !== undefined) this.sizeMin = patch.sizeMin;
        if (patch.sizeMax !== undefined) this.sizeMax = patch.sizeMax;
        if (patch.sizeRange !== undefined) {
            this.sizeRange = [patch.sizeRange[0], patch.sizeRange[1]];
        }
        if (patch.flipY !== undefined) this.flipY = patch.flipY;
        if (patch.pointSize !== undefined && Number.isFinite(patch.pointSize)) {
            this.pointSize = Math.max(0.1, patch.pointSize);
        }
        if (patch.zoomBalance !== undefined && Number.isFinite(patch.zoomBalance)) {
            this.zoomBalance = Math.max(0.0, Math.min(1.0, patch.zoomBalance));
        }
        if (patch.pointOpacity !== undefined) {
            this.pointOpacity =
                patch.pointOpacity === 'auto'
                    ? 'auto'
                    : Math.max(0.004, Math.min(1.0, patch.pointOpacity));
        }
        if (patch.exposure !== undefined) {
            this.exposure =
                patch.exposure === 'auto'
                    ? 'auto'
                    : Math.max(0.05, Math.min(16.0, patch.exposure));
        }

        if (patch.experimental) {
            const exp = patch.experimental;
            if (exp.colorScale !== undefined) this.colorScale = exp.colorScale;
            if (exp.aggregation !== undefined) {
                this.aggregation = exp.aggregation;
            } else if (
                ('aggregation' in exp || patch.pipeline === 'optical') &&
                patch.aggregation === undefined &&
                (this.aggregation === 'trails' || this.aggregation === 'streamlines')
            ) {
                this.aggregation = 'points';
            }
            if (exp.metricType !== undefined) this.metricType = exp.metricType;
            if (exp.auxFields !== undefined) {
                this.auxFields = [exp.auxFields[0], exp.auxFields[1], exp.auxFields[2]];
            }
            if (exp.opticalColorField !== undefined) this.opticalColorField = exp.opticalColorField;
            if (exp.trailMode !== undefined) this.trailMode = exp.trailMode;
            if (exp.trailTimeKyr !== undefined) this.trailTimeKyr = exp.trailTimeKyr;
            if (exp.minParallax !== undefined) this.minParallax = exp.minParallax;
            if (exp.animateFlow !== undefined) this.animateFlow = exp.animateFlow;
        } else if (
            patch.pipeline === 'optical' &&
            patch.aggregation === undefined &&
            (this.aggregation === 'trails' || this.aggregation === 'streamlines')
        ) {
            this.aggregation = 'points';
        }
    }

    public resolveAxisMapping(session: DatasetSession): {
        internalXField: string;
        internalYField: string;
        xDomain: [number, number];
        yDomain: [number, number];
        xRange: [number, number];
        yRange: [number, number];
        dataXSpan: [number, number];
        dataYSpan: [number, number];
    } {
        const rb = session.summary.rootBounds;
        const db = session.summary.dataBounds;
        const effFlip = this.effectiveFlipY;

        const internalXField = this.xField ?? 'x_u16';
        const internalYField = this.yField ?? 'y_u16';

        const xStat = internalXField !== 'x_u16' && internalXField !== 'y_u16'
            ? session.summary.stats[internalXField]
            : undefined;
        const yStat = internalYField !== 'y_u16' && internalYField !== 'x_u16'
            ? session.summary.stats[internalYField]
            : undefined;

        const xCats =
            internalXField !== 'x_u16' && internalXField !== 'y_u16'
                ? session.summary.categories[internalXField]
                : undefined;
        const yCats =
            internalYField !== 'y_u16' && internalYField !== 'x_u16'
                ? session.summary.categories[internalYField]
                : undefined;

        const defaultXDomain: [number, number] =
            xCats && xCats.length > 0
                ? [0, Math.max(1, xCats.length - 1)]
                : xStat &&
                    xStat.type !== 'categorical' &&
                    Number.isFinite(xStat.min) &&
                    Number.isFinite(xStat.max)
                  ? [xStat.min, xStat.max]
                  : [0, 1];
        const defaultYDomain: [number, number] =
            yCats && yCats.length > 0
                ? [0, Math.max(1, yCats.length - 1)]
                : yStat &&
                    yStat.type !== 'categorical' &&
                    Number.isFinite(yStat.min) &&
                    Number.isFinite(yStat.max)
                  ? [yStat.min, yStat.max]
                  : [0, 1];

        const xDomain = this.customXDomain ?? defaultXDomain;
        const yDomain = this.customYDomain ?? defaultYDomain;

        const isSpatialX = internalXField === 'x_u16' || internalXField === 'y_u16';
        const isSpatialY = internalYField === 'y_u16' || internalYField === 'x_u16';

        const defaultXRange: [number, number] =
            !isSpatialX && this.xTransform === 'literal'
                ? [xDomain[0], xDomain[1]]
                : [rb.minX, rb.maxX];
        // Top-down v0.1 spatial `y_u16` has 0 = max_y (top) and 1 = min_y (bottom);
        // Cartesian data columns have domMin at bottom (minY) unless `flipY: true` (e.g. HR Diagram).
        const defaultYRange: [number, number] = isSpatialY
            ? effFlip
                ? [rb.minY, rb.maxY]
                : [rb.maxY, rb.minY]
            : this.yTransform === 'literal'
              ? this.flipY
                  ? [yDomain[1], yDomain[0]]
                  : [yDomain[0], yDomain[1]]
              : this.flipY
                ? [rb.maxY, rb.minY]
                : [rb.minY, rb.maxY];

        const xRange: [number, number] = this.customXRange
            ? [this.customXRange[0], this.customXRange[1]]
            : defaultXRange;
        const yRange: [number, number] = this.customYRange
            ? (isSpatialY ? effFlip : this.flipY)
                ? [this.customYRange[1], this.customYRange[0]]
                : [this.customYRange[0], this.customYRange[1]]
            : defaultYRange;

        const dataXSpan: [number, number] = isSpatialX
            ? db
                ? [db.min_x, db.max_x]
                : [rb.minX, rb.maxX]
            : xDomain;
        const dataYSpan: [number, number] = isSpatialY
            ? db
                ? [db.max_y, db.min_y]
                : session.summary.isLegacyV00 && effFlip
                  ? [rb.minY, rb.maxY]
                  : [rb.maxY, rb.minY]
            : yDomain;

        return {
            internalXField,
            internalYField,
            xDomain,
            yDomain,
            xRange,
            yRange,
            dataXSpan,
            dataYSpan,
        };
    }

    public toEncoding(): DeepgraphEncoding {
        const out: DeepgraphEncoding = {
            color: {
                field: this.colorField,
                scale: this.colorScale,
                domain: [this.colorMin, this.colorMax],
                invert: this.invertScale,
            },
            size: {
                field: this.sizeField,
                domain: [this.sizeMin, this.sizeMax],
                range: [this.sizeRange[0], this.sizeRange[1]],
            },
            flipY: this.flipY,
            aggregation:
                this.aggregation === 'points' ? 'points' : 'mean_field',
            lightBackground: this.lightBackground,
            smoothingKernel: this.smoothingKernel,
            pointSize: this.pointSize,
            zoomBalance: this.zoomBalance,
            pointOpacity: this.pointOpacity,
            exposure: this.exposure,
        };
        if (
            this.xField !== null ||
            this.xTransform !== 'linear' ||
            this.customXDomain !== undefined ||
            this.customXRange !== undefined
        ) {
            out.x = {
                field: this.xField,
                transform: this.xTransform,
                ...(this.customXDomain ? { domain: [...this.customXDomain] as [number, number] } : {}),
                ...(this.customXRange ? { range: [...this.customXRange] as [number, number] } : {}),
            };
        }
        if (
            this.yField !== null ||
            this.yTransform !== 'linear' ||
            this.customYDomain !== undefined ||
            this.customYRange !== undefined
        ) {
            out.y = {
                field: this.yField,
                transform: this.yTransform,
                ...(this.customYDomain ? { domain: [...this.customYDomain] as [number, number] } : {}),
                ...(this.customYRange ? { range: [...this.customYRange] as [number, number] } : {}),
            };
        }
        return out;
    }

    public snapshot(): ViewStateSnapshot {
        return {
            pipeline: this.pipeline,
            detailOffset: this.detailOffset,
            aggregation: this.aggregation,
            xField: this.xField,
            yField: this.yField,
            xTransform: this.xTransform,
            yTransform: this.yTransform,
            xDomain: this.customXDomain ? [...this.customXDomain] : undefined,
            yDomain: this.customYDomain ? [...this.customYDomain] : undefined,
            xRange: this.customXRange ? [...this.customXRange] : undefined,
            yRange: this.customYRange ? [...this.customYRange] : undefined,
            colorField: this.colorField,
            colorMin: this.colorMin,
            colorMax: this.colorMax,
            colorScale: this.colorScale,
            invertScale: this.invertScale,
            lightBackground: this.lightBackground,
            smoothingKernel: this.smoothingKernel,
            sizeField: this.sizeField,
            sizeMin: this.sizeMin,
            sizeMax: this.sizeMax,
            sizeRange: [this.sizeRange[0], this.sizeRange[1]],
            flipY: this.flipY,
            effectiveFlipY: this.effectiveFlipY,
            pointSize: this.pointSize,
            zoomBalance: this.zoomBalance,
            pointOpacity: this.pointOpacity,
            exposure: this.exposure,
            experimental: {
                metricType: this.metricType,
                auxFields: [this.auxFields[0], this.auxFields[1], this.auxFields[2]],
                trailMode: this.trailMode,
                trailTimeKyr: this.trailTimeKyr,
                minParallax: this.minParallax,
                animateFlow: this.animateFlow,
            },
        };
    }

    public syncToTileStore(
        tileStore: TileGpuStore,
        session: DatasetSession,
        animateAxes = true
    ): void {
        const isOptical = this.pipeline === 'optical';
        tileStore.isOpticalPipeline = isOptical;
        tileStore.invertColorScale = this.invertScale;
        tileStore.lightBackground = this.lightBackground;
        tileStore.kernelRadius = this.smoothingKernel;
        tileStore.pointSize = this.pointSize;
        tileStore.zoomBalance = this.zoomBalance;
        tileStore.pointOpacity = this.pointOpacity;
        tileStore.exposure = this.exposure;
        tileStore.trailMode = this.trailMode;
        tileStore.trailTimeKyr = this.trailTimeKyr;
        tileStore.minParallax = this.minParallax;
        tileStore.animateFlow = this.animateFlow;
        tileStore.magnitudeField = session.summary.magnitudeField;
        tileStore.opticalColorField = this.opticalColorField;

        if (this.aggregation === 'trails') {
            tileStore.computeMode = 3;
            tileStore.metricType = this.metricType;
            tileStore.auxField1 = this.auxFields[0];
            tileStore.auxField2 = this.auxFields[1];
            tileStore.auxField3 = this.auxFields[2];
        } else if (this.aggregation === 'streamlines') {
            tileStore.computeMode = 4;
            tileStore.metricType = this.metricType;
            tileStore.auxField1 = this.auxFields[0];
            tileStore.auxField2 = this.auxFields[1];
            tileStore.auxField3 = this.auxFields[2];
        } else if (isOptical) {
            tileStore.computeMode = 0;
            tileStore.metricType = 0;
            tileStore.auxField1 = '';
            tileStore.auxField2 = '';
            tileStore.auxField3 = '';
        } else {
            const isColorCategorical = Boolean(
                tileStore.categoricalDictionaries?.[this.colorField]
            );
            tileStore.computeMode =
                !isColorCategorical && this.aggregation === 'mean_field' ? 2 : 1;
            tileStore.metricType = this.metricType;
            tileStore.auxField1 = this.auxFields[0];
            tileStore.auxField2 = this.auxFields[1];
            tileStore.auxField3 = this.auxFields[2];
        }

        const mapped = this.resolveAxisMapping(session);
        const origins = session.colOrigins ?? {};
        tileStore.colOrigins = { ...origins };
        const xOrig =
            mapped.internalXField !== 'x_u16' && mapped.internalXField !== 'y_u16'
                ? (origins[mapped.internalXField] ?? 0)
                : 0;
        const yOrig =
            mapped.internalYField !== 'y_u16' && mapped.internalYField !== 'x_u16'
                ? (origins[mapped.internalYField] ?? 0)
                : 0;
        const cOrig =
            this.colorField !== 'x_u16' && this.colorField !== 'y_u16'
                ? (origins[this.colorField] ?? 0)
                : 0;
        const sOrig =
            this.sizeField !== 'x_u16' && this.sizeField !== 'y_u16'
                ? (origins[this.sizeField] ?? 0)
                : 0;

        const activeCols = tileStore.updateEncoding(
            {
                x: {
                    field: mapped.internalXField,
                    transform: this.xTransform,
                    domain: [mapped.xDomain[0] - xOrig, mapped.xDomain[1] - xOrig],
                    range: mapped.xRange,
                },
                y: {
                    field: mapped.internalYField,
                    transform: this.yTransform,
                    domain: [mapped.yDomain[0] - yOrig, mapped.yDomain[1] - yOrig],
                    range: mapped.yRange,
                },
                color: {
                    field: this.colorField,
                    range: this.colorScale,
                    domain: [this.colorMin - cOrig, this.colorMax - cOrig],
                },
                size: {
                    field: this.sizeField,
                    domain: [this.sizeMin - sOrig, this.sizeMax - sOrig],
                    range: [this.sizeRange[0], this.sizeRange[1]],
                },
            },
            animateAxes
        );
        session.setRequestedColumns(activeCols);
    }
}
