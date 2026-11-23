import { Deepgraph as DeepgraphImpl } from './lib/ViewerLayer.ts';
import type { DeepgraphStatic } from './lib/types.ts';

/** Public namespace facade (`Deepgraph.create`, `Deepgraph.isSupported`). */
export const Deepgraph: DeepgraphStatic = DeepgraphImpl;

export type {
    AggregationStrategy,
    AxisChannelEncoding,
    BoundingBox,
    CategoricalColumnStat,
    ColorChannelEncoding,
    ColorScaleName,
    ColumnStat,
    DatasetMetadataSummary,
    DeepgraphEncoding,
    DeepgraphEventMap,
    DeepgraphOptions,
    DeepgraphPlot,
    DeepgraphPreset,
    DeepgraphPresetInput,
    DeepgraphPresetResolver,
    DeepgraphStatic,
    ExperimentalColorScaleName,
    ExperimentalOptions,
    LoadDatasetOptions,
    NumericColumnStat,
    PointHitEvent,
    PointSelectEvent,
    RenderPipelineMode,
    RowAttributeResolver,
    SizeChannelEncoding,
    TelemetrySnapshot,
    ViewOptionsPatch,
    ViewportState,
    ViewStateSnapshot,
} from './lib/types.ts';
