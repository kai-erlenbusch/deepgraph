import { init as initZstd, decompress as zstdDecompress } from '@bokuweb/zstd-wasm';
import {
    compileSlotBitmaskSlices,
    decodeRowFromTables,
    extractTileColumnsFromTables,
    extractTileRowIndexFromTables,
    parseArrowSchemaFields,
    parseTileTablesFromPayloads,
    queryTileSpatialSelection,
    safeZstdDecompress,
    testProbeRowIdsInFilter,
    type WorkerTileRowIndex,
} from './workerCodec.ts';

const isWorkerScope =
    typeof self !== 'undefined' && typeof (self as any).postMessage === 'function';

let zstdReady = false;
let zstdInitPromise: Promise<void> | null = isWorkerScope
    ? initZstd()
          .then(() => {
              zstdReady = true;
          })
          .catch(() => {
              zstdInitPromise = null;
          })
    : null;

let globalSchemaBuffer: Uint8Array | null = null;
let cachedDecodeRow: { key: string; tables: any[] } | null = null;

let retainRowIdsEnabled = false;
const cachedTileRowIndex = new Map<string, WorkerTileRowIndex>();
let hasActiveRowIdFilter = false;
let activeFilterU32: Uint32Array | null = null;
let activeFilterU64: BigUint64Array | null = null;

function computeRetainedIndexBytes(): number {
    let total = 0;
    for (const entry of cachedTileRowIndex.values()) {
        total += entry.byteLength;
    }
    return total;
}

async function ensureZstdReady(): Promise<boolean> {
    const wasWarm = zstdReady;
    if (!zstdInitPromise) {
        zstdInitPromise = initZstd()
            .then(() => {
                zstdReady = true;
            })
            .catch((err: any) => {
                zstdInitPromise = null;
                throw err;
            });
    }
    await zstdInitPromise;
    return wasWarm;
}

const decompressWithZstd = (raw: Uint8Array): Uint8Array =>
    safeZstdDecompress(raw, (input) =>
        zstdDecompress(input, { defaultHeapSize: 1024 * 1024 * 64 })
    );

if (isWorkerScope) {
    self.onmessage = async (e) => {
        const data = e.data;
        if (!data) return;

        // Schema & cache messages never need Zstd WASM — handle immediately!
        if (data.action === 'init_schema') {
            globalSchemaBuffer = new Uint8Array(data.schemaBuffer);
            cachedDecodeRow = null;
            cachedTileRowIndex.clear();
            return;
        }

        if (data.action === 'clear_row_cache') {
            cachedDecodeRow = null;
            if (data.clearRowIds) {
                cachedTileRowIndex.clear();
                hasActiveRowIdFilter = false;
                activeFilterU32 = null;
                activeFilterU64 = null;
            }
            return;
        }

        if (data.action === 'set_retain_row_ids') {
            retainRowIdsEnabled = Boolean(data.enabled);
            if (!retainRowIdsEnabled && !hasActiveRowIdFilter) {
                cachedTileRowIndex.clear();
            }
            return;
        }

        if (data.action === 'drop_tile') {
            if (typeof data.key === 'string') {
                cachedTileRowIndex.delete(data.key);
            }
            return;
        }

        if (data.action === 'set_rowid_filter') {
            try {
                hasActiveRowIdFilter = Boolean(data.hasFilter);
                if (hasActiveRowIdFilter && data.u32Buffer instanceof ArrayBuffer) {
                    const arr32 = new Uint32Array(data.u32Buffer);
                    if (!data.isSorted) arr32.sort();
                    activeFilterU32 = arr32;
                } else {
                    activeFilterU32 = null;
                }
                if (hasActiveRowIdFilter && data.u64Buffer instanceof ArrayBuffer) {
                    const arr64 = new BigUint64Array(data.u64Buffer);
                    if (!data.isSorted) arr64.sort();
                    activeFilterU64 = arr64;
                } else {
                    activeFilterU64 = null;
                }

                const masks: { key: string; slices: ArrayBuffer[] }[] = [];
                const transfers: ArrayBuffer[] = [];
                let probeMaskBuffer: ArrayBuffer | null = null;

                if (hasActiveRowIdFilter) {
                    if (data.probeRowIdsBuffer instanceof ArrayBuffer) {
                        const probeIds = new BigUint64Array(data.probeRowIdsBuffer);
                        const probeRes = testProbeRowIdsInFilter(
                            probeIds,
                            activeFilterU32,
                            activeFilterU64
                        );
                        probeMaskBuffer = probeRes.buffer as ArrayBuffer;
                        transfers.push(probeMaskBuffer);
                    }
                    for (const [tileKey, idxEntry] of cachedTileRowIndex.entries()) {
                        const slices = compileSlotBitmaskSlices(
                            idxEntry,
                            activeFilterU32,
                            activeFilterU64
                        );
                        masks.push({ key: tileKey, slices });
                        for (const s of slices) transfers.push(s);
                    }
                } else if (!retainRowIdsEnabled) {
                    cachedTileRowIndex.clear();
                }

                self.postMessage(
                    {
                        id: data.id,
                        action: 'set_rowid_filter',
                        masks,
                        probeMaskBuffer,
                        retainedIndexBytes: computeRetainedIndexBytes(),
                    },
                    { transfer: transfers } as any
                );
            } catch (error: any) {
                self.postMessage({
                    id: data.id,
                    action: 'set_rowid_filter',
                    error: error.stack || error.message || 'Unknown error',
                });
            }
            return;
        }

        if (data.action === 'query_loaded_selection') {
            try {
                const maxIds =
                    typeof data.maxIds === 'number' && Number.isFinite(data.maxIds) && data.maxIds >= 0
                        ? Math.trunc(data.maxIds)
                        : Infinity;
                const chunks: BigUint64Array[] = [];
                let totalMatched = 0;
                let bufferedLen = 0;
                for (const [tileKey, idxEntry] of cachedTileRowIndex.entries()) {
                    if (Array.isArray(data.visibleKeys) && !data.visibleKeys.includes(tileKey)) {
                        continue;
                    }
                    const matched = queryTileSpatialSelection(
                        tileKey,
                        idxEntry,
                        data.spec,
                        hasActiveRowIdFilter,
                        activeFilterU32,
                        activeFilterU64
                    );
                    if (matched.length > 0) {
                        totalMatched += matched.length;
                        if (bufferedLen < maxIds) {
                            const take = Math.min(matched.length, maxIds - bufferedLen);
                            chunks.push(take === matched.length ? matched : matched.subarray(0, take));
                            bufferedLen += take;
                        }
                    }
                }
                const combined = new BigUint64Array(bufferedLen);
                let dst = 0;
                for (const c of chunks) {
                    combined.set(c, dst);
                    dst += c.length;
                }
                const outBuf = combined.buffer as ArrayBuffer;
                self.postMessage(
                    {
                        id: data.id,
                        action: 'query_loaded_selection',
                        idsBuffer: outBuf,
                        totalMatched,
                    },
                    { transfer: [outBuf] } as any
                );
            } catch (error: any) {
                self.postMessage({
                    id: data.id,
                    action: 'query_loaded_selection',
                    error: error.stack || error.message || 'Unknown error',
                });
            }
            return;
        }

        if (data.action === 'get_schema') {
            try {
                if (!data.schemaBuffer) {
                    throw new Error('No schema buffer provided');
                }
                const fields = parseArrowSchemaFields(new Uint8Array(data.schemaBuffer));
                self.postMessage({ id: data.id, action: 'get_schema', fields });
            } catch (error: any) {
                self.postMessage({
                    id: data.id,
                    action: 'get_schema',
                    error: error.stack || error.message || 'Unknown error',
                });
            }
            return;
        }

        let wasWarm = zstdReady;
        try {
            wasWarm = await ensureZstdReady();
        } catch (err: any) {
            self.postMessage({
                action: data?.action,
                id: data?.id,
                error: `Failed to initialize Zstd WASM: ${err?.message ?? String(err)}`,
                reqId: data?.reqId,
                key: data?.key || data?.id,
            });
            return;
        }

        if (data.action === 'index_tile_rowids') {
            try {
                const { id, key, buffers } = data;
                let idxEntry = cachedTileRowIndex.get(key) ?? null;
                if (!idxEntry && Array.isArray(buffers)) {
                    const tables = parseTileTablesFromPayloads(
                        buffers,
                        globalSchemaBuffer,
                        decompressWithZstd
                    );
                    idxEntry = extractTileRowIndexFromTables(tables);
                    if (idxEntry && (retainRowIdsEnabled || hasActiveRowIdFilter)) {
                        cachedTileRowIndex.set(key, idxEntry);
                    }
                }
                const slices =
                    idxEntry && hasActiveRowIdFilter
                        ? compileSlotBitmaskSlices(idxEntry, activeFilterU32, activeFilterU64)
                        : [];
                self.postMessage(
                    {
                        id,
                        action: 'index_tile_rowids',
                        key,
                        slices,
                        rowIndexBytes: idxEntry?.byteLength ?? 0,
                    },
                    { transfer: slices } as any
                );
            } catch (error: any) {
                self.postMessage({
                    id: data.id,
                    action: 'index_tile_rowids',
                    error: error.stack || error.message || 'Unknown error',
                });
            }
            return;
        }

        if (data.action === 'decompress') {
            try {
                const decompressed = decompressWithZstd(new Uint8Array(data.buffer));
                const outBuf =
                    decompressed.byteOffset === 0 &&
                    decompressed.byteLength === decompressed.buffer.byteLength
                        ? (decompressed.buffer as ArrayBuffer)
                        : (decompressed.slice().buffer as ArrayBuffer);
                self.postMessage(
                    {
                        id: data.id,
                        action: 'decompress',
                        buffer: outBuf,
                    },
                    { transfer: [outBuf] } as any
                );
            } catch (error: any) {
                self.postMessage({
                    id: data.id,
                    action: 'decompress',
                    error: error.stack || error.message || 'Unknown error',
                });
            }
            return;
        }

        if (data.action === 'decompress_gzip') {
            try {
                const ds = new DecompressionStream('gzip');
                const response = new Response(data.buffer);
                const decompressedStream = response.body!.pipeThrough(ds);
                const decompressedBuffer = await new Response(decompressedStream).arrayBuffer();
                self.postMessage(
                    {
                        id: data.id,
                        action: 'decompress_gzip',
                        buffer: decompressedBuffer,
                    },
                    { transfer: [decompressedBuffer] } as any
                );
            } catch (error: any) {
                self.postMessage({
                    id: data.id,
                    action: 'decompress_gzip',
                    error: error.stack || error.message || 'Unknown error',
                });
            }
            return;
        }

        if (data.action === 'decode_row') {
            try {
                const { id, key, buffers, rowIndex, categories } = data;
                if (typeof rowIndex !== 'number' || rowIndex < 0) {
                    throw new Error('Invalid decode_row request');
                }

                const cacheKey = typeof key === 'string' && key.length > 0 ? key : null;
                let tables: any[];
                if (cacheKey && cachedDecodeRow && cachedDecodeRow.key === cacheKey) {
                    tables = cachedDecodeRow.tables;
                } else {
                    if (!Array.isArray(buffers)) {
                        self.postMessage({ id, action: 'decode_row', cacheMiss: true });
                        return;
                    }
                    tables = parseTileTablesFromPayloads(
                        buffers,
                        globalSchemaBuffer,
                        decompressWithZstd
                    );
                    if (cacheKey && tables.length > 0) {
                        cachedDecodeRow = { key: cacheKey, tables };
                    }
                }

                const result = decodeRowFromTables(tables, rowIndex, categories);
                self.postMessage({
                    id,
                    action: 'decode_row',
                    cachedKey: cacheKey ?? undefined,
                    result,
                });
            } catch (error: any) {
                self.postMessage({
                    id: data.id,
                    action: 'decode_row',
                    error: error.stack || error.message || 'Unknown error',
                });
            }
            return;
        }

        // Main tile column extraction
        const { reqId, key, buffers, requestedColumns, pooledColumns, colOrigins } = data;
        if (!key || !buffers) return;

        const tDecode0 = performance.now();
        try {
            const tables = parseTileTablesFromPayloads(
                buffers,
                globalSchemaBuffer,
                decompressWithZstd
            );
            const extracted = extractTileColumnsFromTables(
                tables,
                requestedColumns,
                pooledColumns,
                colOrigins
            );

            let rowMaskSlices: ArrayBuffer[] | undefined;
            let rowIndexBytes = 0;
            if (retainRowIdsEnabled || hasActiveRowIdFilter || data.retainRowIds) {
                const idxEntry = extractTileRowIndexFromTables(tables);
                if (idxEntry) {
                    cachedTileRowIndex.set(key, idxEntry);
                    rowIndexBytes = idxEntry.byteLength;
                    if (hasActiveRowIdFilter) {
                        rowMaskSlices = compileSlotBitmaskSlices(
                            idxEntry,
                            activeFilterU32,
                            activeFilterU64
                        );
                        for (const s of rowMaskSlices) {
                            extracted.transferables.push(s);
                        }
                    }
                }
            }

            const decodeMs = performance.now() - tDecode0;
            self.postMessage(
                {
                    reqId,
                    key,
                    columns: extracted.columns,
                    numRows: extracted.numRows,
                    decodeMs,
                    wasWarm,
                    dictionaries: extracted.dictionaries,
                    rowMaskSlices,
                    rowIndexBytes,
                },
                { transfer: extracted.transferables }
            );
        } catch (error: any) {
            self.postMessage({
                reqId,
                key,
                error: error.stack || error.message || 'Unknown error',
            });
        }
    };
}
