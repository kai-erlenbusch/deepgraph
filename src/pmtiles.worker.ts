import { tableFromIPC, Type } from 'apache-arrow';
import { init as initZstd, decompress as zstdDecompress } from '@bokuweb/zstd-wasm';

// Pre-warm WASM Zstd immediately on worker spawn so compilation overlaps header fetch
let zstdReady = false;
let zstdInitPromise: Promise<void> | null = initZstd()
    .then(() => {
        zstdReady = true;
    })
    .catch(() => {
        zstdInitPromise = null;
    });
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

self.onmessage = async (e) => {
    const data = e.data;
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

    if (data.action === 'init_schema') {
        globalSchemaBuffer = new Uint8Array(data.schemaBuffer);
        cachedDecodeRow = null;
        return;
    }

    if (data.action === 'clear_row_cache') {
        cachedDecodeRow = null;
        return;
    }

    if (data.action === 'decompress') {
        try {
            const decompressed = zstdDecompress(new Uint8Array(data.buffer), {
                defaultHeapSize: 1024 * 1024 * 64,
            });
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

    if (data.action === 'get_schema') {
        try {
            if (!data.schemaBuffer) {
                throw new Error('No schema buffer provided');
            }
            const bytes = new Uint8Array(data.schemaBuffer);
            const table = tableFromIPC([bytes]);
            const fields = table.schema.fields.map((f) => ({
                name: f.name,
                type: String(f.type),
            }));
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
                tables = buffers
                    .map((buffer: ArrayBuffer | null) => {
                        if (!buffer || buffer.byteLength === 0) return null;
                        let rawData: Uint8Array<ArrayBufferLike> = new Uint8Array(buffer);
                        if (
                            rawData.length >= 4 &&
                            rawData[0] === 0x28 &&
                            rawData[1] === 0xb5 &&
                            rawData[2] === 0x2f &&
                            rawData[3] === 0xfd
                        ) {
                            rawData = zstdDecompress(rawData, {
                                defaultHeapSize: 1024 * 1024 * 64,
                            });
                        }
                        let table: any = null;
                        if (globalSchemaBuffer) {
                            try {
                                const t2 = tableFromIPC([globalSchemaBuffer, rawData]);
                                if (t2 && t2.numRows > 0) table = t2;
                            } catch {
                                // Fallback below
                            }
                        }
                        if (!table || table.numRows === 0) {
                            table = tableFromIPC(rawData);
                        }
                        return table;
                    })
                    .filter((t: any) => t !== null && t.numRows > 0);

                if (cacheKey && tables.length > 0) {
                    cachedDecodeRow = { key: cacheKey, tables };
                }
            }

            if (tables.length === 0 || rowIndex >= tables[0].numRows) {
                self.postMessage({
                    id,
                    action: 'decode_row',
                    cachedKey: cacheKey ?? undefined,
                    result: null,
                });
                return;
            }

            let rowId = rowIndex;
            let rowIdBigIntStr: string | undefined = undefined;
            let xU16 = 0;
            let yU16 = 0;
            const attributes: Record<string, unknown> = {};
            const MAX_SAFE = 9007199254740991n;

            for (const table of tables) {
                for (const field of table.schema.fields) {
                    const name: string = field.name;
                    const child = table.getChild(name);
                    if (!child) continue;

                    if (name === 'row_id') {
                        if (child.isValid(rowIndex)) {
                            const rawId = child.get(rowIndex);
                            if (typeof rawId === 'bigint') {
                                rowId = Number(rawId);
                                if (rawId > MAX_SAFE || rawId < -MAX_SAFE) {
                                    rowIdBigIntStr = rawId.toString();
                                }
                            } else if (typeof rawId === 'number') {
                                rowId = rawId;
                            }
                        }
                        continue;
                    }
                    if (name === 'x_u16') {
                        xU16 = Number(child.get(rowIndex) ?? 0);
                        continue;
                    }
                    if (name === 'y_u16') {
                        yU16 = Number(child.get(rowIndex) ?? 0);
                        continue;
                    }
                    if (name === 'z') {
                        continue;
                    }

                    if (!child.isValid(rowIndex)) {
                        attributes[name] = null;
                        continue;
                    }

                    const val = child.get(rowIndex);
                    const catList: string[] | undefined = categories?.[name];
                    if (
                        catList &&
                        (typeof val === 'number' || typeof val === 'bigint')
                    ) {
                        const code = Number(val);
                        attributes[name] =
                            code >= 0 && code < catList.length
                                ? String(catList[code])
                                : code;
                    } else if (typeof val === 'bigint') {
                        attributes[name] =
                            val <= MAX_SAFE && val >= -MAX_SAFE
                                ? Number(val)
                                : val.toString();
                    } else {
                        attributes[name] = val;
                    }
                }
            }

            self.postMessage({
                id,
                action: 'decode_row',
                cachedKey: cacheKey ?? undefined,
                result: {
                    rowId,
                    rowIdBigIntStr,
                    xU16,
                    yU16,
                    attributes,
                },
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

    // Main data parsing
    const { reqId, key, buffers, requestedColumns, pooledColumns } = data;
    if (!key || !buffers) return;

    const tDecode0 = performance.now();
    try {
        const tables = buffers
            .map((buffer: ArrayBuffer | null) => {
                if (!buffer) return null;
                let rawData: Uint8Array<ArrayBufferLike> = new Uint8Array(buffer);

                if (
                    rawData.length >= 4 &&
                    rawData[0] === 0x28 &&
                    rawData[1] === 0xb5 &&
                    rawData[2] === 0x2f &&
                    rawData[3] === 0xfd
                ) {
                    rawData = zstdDecompress(rawData, { defaultHeapSize: 1024 * 1024 * 64 });
                }

                let table: any = null;
                if (globalSchemaBuffer) {
                    try {
                        const table2 = tableFromIPC([globalSchemaBuffer, rawData]);
                        if (table2 && table2.numRows > 0) {
                            table = table2;
                        }
                    } catch {
                        // Fallback below if tile embeds its own schema
                    }
                }

                if (!table || table.numRows === 0) {
                    try {
                        table = tableFromIPC(rawData);
                    } catch (e2) {
                        console.error('Worker: Fallback parsing failed', e2);
                    }
                }
                return table;
            })
            .filter((t: any) => t !== null && t.numRows > 0);

        if (tables.length === 0) {
            throw new Error('No tables found');
        }

        const getRawCol = (name: string) => {
            for (const table of tables) {
                const child = table.getChild(name);
                if (child) return child;
            }
            return null;
        };

        const dictionaries: Record<string, string[]> = {};

        const getCol = (name: string) => {
            const child = getRawCol(name);
            if (!child) return null;

            if (child.type.typeId === Type.Dictionary) {
                if (!dictionaries[name] && child.data.length > 0 && child.data[0].dictionary) {
                    dictionaries[name] = child.data[0].dictionary.toArray();
                }
                if (child.data.length === 1) {
                    return child.data[0].values;
                }
                let totalLen = 0;
                for (const chunk of child.data) totalLen += chunk.length;
                const combined = new Float32Array(totalLen);
                let offset = 0;
                for (const chunk of child.data) {
                    const vals = chunk.values;
                    for (let i = 0; i < chunk.length; i++) {
                        combined[offset++] = vals[i];
                    }
                }
                return combined;
            }

            if (child.data.length === 1) {
                return child.data[0].values;
            }
            return child.toArray();
        };

        const xU16 = getCol('x_u16') as Uint16Array;
        const yU16 = getCol('y_u16') as Uint16Array;

        if (xU16 && yU16) {
            const numRows = xU16.length;
            const columns: Record<string, ArrayBuffer> = {};
            const transferables: ArrayBuffer[] = [];

            const getU16Col = (raw: Uint16Array, pooled: ArrayBuffer | undefined): ArrayBuffer => {
                let buf: Uint16Array;
                if (pooled && pooled.byteLength === numRows * 2) {
                    buf = new Uint16Array(pooled, 0, numRows);
                } else {
                    buf = new Uint16Array(numRows);
                }
                buf.set(raw.subarray(0, numRows));
                return buf.buffer as ArrayBuffer;
            };

            const getXYU16Col = (
                rawX: Uint16Array,
                rawY: Uint16Array,
                pooled: ArrayBuffer | undefined
            ): ArrayBuffer => {
                const outBuf =
                    pooled && pooled.byteLength >= numRows * 4 && pooled.byteLength <= numRows * 8
                        ? pooled
                        : new ArrayBuffer(numRows * 4);
                const out32 = new Uint32Array(outBuf, 0, numRows);
                for (let i = 0; i < numRows; i++) {
                    out32[i] = rawX[i] | (rawY[i] << 16);
                }
                return outBuf;
            };

            const getFloat32Col = (
                name: string,
                pooled: ArrayBuffer | undefined
            ): ArrayBuffer => {
                const usePooled =
                    pooled && pooled.byteLength >= numRows * 4 && pooled.byteLength <= numRows * 8;
                const rawBuf = usePooled ? pooled! : new ArrayBuffer(numRows * 4);
                const buf = new Float32Array(rawBuf, 0, numRows);

                const child = getRawCol(name);
                if (!child || !child.data || child.data.length === 0) {
                    buf.fill(NaN);
                    return rawBuf;
                }

                if (child.type.typeId === Type.Dictionary) {
                    if (!dictionaries[name] && child.data[0].dictionary) {
                        dictionaries[name] = child.data[0].dictionary.toArray();
                    }
                }

                let dstOffset = 0;
                for (const chunk of child.data) {
                    const chunkLen = Math.min(chunk.length, numRows - dstOffset);
                    if (chunkLen <= 0) break;

                    const vals = chunk.values;
                    const hasNulls = chunk.nullCount > 0 && chunk.nullBitmap != null;
                    const bitmap: Uint8Array | null = hasNulls ? chunk.nullBitmap : null;
                    const bitOffset: number = chunk.offset || 0;
                    const isBool = child.type.typeId === Type.Bool;
                    const isBigInt =
                        vals instanceof BigInt64Array || vals instanceof BigUint64Array;

                    if (isBool) {
                        for (let i = 0; i < chunkLen; i++) {
                            const bitIdx = bitOffset + i;
                            if (hasNulls && (bitmap![bitIdx >> 3] & (1 << (bitIdx & 7))) === 0) {
                                buf[dstOffset + i] = NaN;
                            } else {
                                buf[dstOffset + i] =
                                    (vals[bitIdx >> 3] & (1 << (bitIdx & 7))) !== 0 ? 1.0 : 0.0;
                            }
                        }
                    } else if (!hasNulls && !isBigInt && vals instanceof Float32Array) {
                        buf.set(vals.subarray(0, chunkLen), dstOffset);
                    } else if (!hasNulls && !isBigInt) {
                        for (let i = 0; i < chunkLen; i++) {
                            buf[dstOffset + i] = vals[i];
                        }
                    } else if (!hasNulls && isBigInt) {
                        for (let i = 0; i < chunkLen; i++) {
                            buf[dstOffset + i] = Number(vals[i]);
                        }
                    } else {
                        for (let i = 0; i < chunkLen; i++) {
                            const bitIdx = bitOffset + i;
                            const isValid = (bitmap![bitIdx >> 3] & (1 << (bitIdx & 7))) !== 0;
                            if (!isValid) {
                                buf[dstOffset + i] = NaN;
                            } else {
                                const v = vals[i];
                                buf[dstOffset + i] = typeof v === 'bigint' ? Number(v) : v;
                            }
                        }
                    }
                    dstOffset += chunkLen;
                }

                if (dstOffset < numRows) {
                    buf.fill(NaN, dstOffset);
                }
                return rawBuf;
            };

            const reqCols = requestedColumns || ['xy_u16'];

            for (const col of reqCols) {
                let buf: ArrayBuffer;
                const pooled =
                    pooledColumns && pooledColumns[col] ? pooledColumns[col] : undefined;

                if (col === 'xy_u16') {
                    buf = getXYU16Col(xU16, yU16, pooled);
                } else if (col === 'x_u16') {
                    buf = getU16Col(xU16, pooled);
                } else if (col === 'y_u16') {
                    buf = getU16Col(yU16, pooled);
                } else {
                    buf = getFloat32Col(col, pooled);
                }

                columns[col] = buf;
                transferables.push(buf);
            }

            const decodeMs = performance.now() - tDecode0;
            self.postMessage(
                {
                    reqId,
                    key,
                    columns,
                    numRows,
                    decodeMs,
                    wasWarm,
                    dictionaries:
                        Object.keys(dictionaries).length > 0 ? dictionaries : undefined,
                },
                { transfer: transferables }
            );
        } else {
            self.postMessage({ reqId, key, error: 'Missing x_u16 or y_u16' });
        }
    } catch (error: any) {
        self.postMessage({
            reqId,
            key,
            error: error.stack || error.message || 'Unknown error',
        });
    }
};
