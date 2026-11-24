import { init as initZstd, decompress as zstdDecompress } from '@bokuweb/zstd-wasm';
import {
    decodeRowFromTables,
    extractTileColumnsFromTables,
    parseArrowSchemaFields,
    parseTileTablesFromPayloads,
    safeZstdDecompress,
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
            return;
        }

        if (data.action === 'clear_row_cache') {
            cachedDecodeRow = null;
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
