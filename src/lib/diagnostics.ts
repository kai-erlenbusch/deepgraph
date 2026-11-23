import type { TileGpuStore } from '../core/TileGpuStore.ts';
import type { PickingLayer } from '../core/PickingLayer.ts';
import type { HighlightOverlay } from '../core/HighlightOverlay.ts';

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
    getTileManager: () => unknown;
    getLibraryGpuBytes: () => number;
    getCpuMemoryStats: () => unknown;
}
