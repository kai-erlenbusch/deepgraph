import type { EdgeOverlay } from '../core/EdgeOverlay.ts';
import type { FilterGpuResources } from '../core/FilterGpuResources.ts';
import type { HighlightOverlay } from '../core/HighlightOverlay.ts';
import type { PickingLayer } from '../core/PickingLayer.ts';
import type { SelectionOverlay } from '../core/SelectionOverlay.ts';
import type { TileGpuStore } from '../core/TileGpuStore.ts';

/**
 * @internal
 * Non-public symbol used strictly by workspace Playwright / benchmark scripts (`window.__viewer`).
 * Never exported from `src/index.ts`.
 */
export const INTERNAL_DIAGNOSTICS = Symbol('deepgraph.internalDiagnostics');

export interface InternalDiagnosticsHandle {
    camera: unknown;
    tileStore: TileGpuStore;
    pickingLayer: PickingLayer;
    highlightOverlay: HighlightOverlay;
    selectionOverlay: SelectionOverlay;
    edgeOverlay: EdgeOverlay;
    filterResources: FilterGpuResources;
    getTileManager: () => unknown;
    getLibraryGpuBytes: () => number;
    getCpuMemoryStats: () => unknown;
}
