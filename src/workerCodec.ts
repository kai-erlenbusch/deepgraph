import {
    decimal as flDecimal,
    field as flField,
    Table,
    tableFromIPC,
    tableToIPC,
    Type,
} from '@uwdata/flechette';

const tableNegDecimalMultipliers = new WeakMap<object, Record<string, number>>();

interface NormalizedSchemaCacheEntry {
    schemaBytes: Uint8Array;
    negDecimalMultipliers: Record<string, number> | null;
}

const normalizedSchemaCache = new WeakMap<Uint8Array, NormalizedSchemaCacheEntry>();

function normalizeNegativeDecimalSchema(schemaBytes: Uint8Array): NormalizedSchemaCacheEntry {
    const cached = normalizedSchemaCache.get(schemaBytes);
    if (cached) return cached;

    const schemaTbl = tableFromIPC([schemaBytes], FLECHETTE_IPC_OPTIONS);
    let negDecimalMultipliers: Record<string, number> | null = null;
    const patchedFields = schemaTbl.schema.fields.map((f: any) => {
        if (
            f.type?.typeId === Type.Decimal &&
            typeof f.type.scale === 'number' &&
            f.type.scale < 0
        ) {
            if (!negDecimalMultipliers) negDecimalMultipliers = {};
            negDecimalMultipliers[String(f.name)] = Math.pow(10, -f.type.scale);
            return flField(
                f.name,
                flDecimal(f.type.precision ?? 38, 0, f.type.bitWidth ?? 128),
                f.nullable,
                f.metadata
            );
        }
        return f;
    });

    if (!negDecimalMultipliers) {
        const entry: NormalizedSchemaCacheEntry = { schemaBytes, negDecimalMultipliers: null };
        normalizedSchemaCache.set(schemaBytes, entry);
        return entry;
    }

    const fullPatched = tableToIPC(
        new Table({ ...schemaTbl.schema, fields: patchedFields }, []),
        { format: 'stream' }
    )!;
    const entry: NormalizedSchemaCacheEntry = {
        schemaBytes: fullPatched.subarray(0, fullPatched.byteLength - 8),
        negDecimalMultipliers,
    };
    normalizedSchemaCache.set(schemaBytes, entry);
    return entry;
}

function splitLeadingIpcMessage(
    raw: Uint8Array
): { firstMsg: Uint8Array; rest: Uint8Array } | null {
    if (raw.byteLength < 8) return null;
    const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    let offset = 0;
    if (dv.getUint32(0, true) === 0xffffffff) {
        offset = 4;
    }
    if (offset + 4 > raw.byteLength) return null;
    const metaLen = dv.getInt32(offset, true);
    if (metaLen <= 0) return null;
    const end = offset + 4 + metaLen;
    if (end > raw.byteLength) return null;
    return {
        firstMsg: raw.subarray(0, end),
        rest: raw.subarray(end),
    };
}

export const MAX_ZSTD_UNCOMPRESSED_BYTES = 256 * 1024 * 1024; // 256 MiB hard WASM allocation ceiling
const MAX_SAFE_BIGINT = 9007199254740991n;

export const FLECHETTE_IPC_OPTIONS = {
    useBigInt: true,
    useDate: false,
} as const;

export function hasZstdMagic(bytes: Uint8Array): boolean {
    return (
        bytes.byteLength >= 4 &&
        bytes[0] === 0x28 &&
        bytes[1] === 0xb5 &&
        bytes[2] === 0x2f &&
        bytes[3] === 0xfd
    );
}

/**
 * Parses the RFC 8878 §3.1.1.1 Zstandard Frame Header before WASM `_malloc` runs,
 * rejecting frames declaring `> 256 MiB` uncompressed content size (`DG-P2-3`).
 */
export function validateZstdFrameHeader(raw: Uint8Array): void {
    if (!hasZstdMagic(raw)) return;
    if (raw.byteLength < 5) {
        throw new Error('Truncated Zstd frame header (< 5 bytes)');
    }
    const fhd = raw[4];
    const fcsFlag = (fhd >>> 6) & 0x3;
    const singleSegment = ((fhd >>> 5) & 0x1) !== 0;
    const dictIdFlag = fhd & 0x3;
    const dictBytes =
        dictIdFlag === 0 ? 0 : dictIdFlag === 1 ? 1 : dictIdFlag === 2 ? 2 : 4;
    const fcsOffset = 5 + (singleSegment ? 0 : 1) + dictBytes;
    const fcsBytes = fcsFlag === 0 ? (singleSegment ? 1 : 0) : 1 << fcsFlag;

    if (fcsBytes === 0) return;
    if (raw.byteLength < fcsOffset + fcsBytes) {
        throw new Error('Truncated Zstd frame content size header');
    }

    const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    let contentSize: bigint;
    if (fcsBytes === 1) {
        contentSize = BigInt(raw[fcsOffset]);
    } else if (fcsBytes === 2) {
        contentSize = BigInt(dv.getUint16(fcsOffset, true) + 256);
    } else if (fcsBytes === 4) {
        contentSize = BigInt(dv.getUint32(fcsOffset, true));
    } else {
        contentSize = dv.getBigUint64(fcsOffset, true);
    }

    if (contentSize > BigInt(MAX_ZSTD_UNCOMPRESSED_BYTES)) {
        throw new Error(
            `Zstd frame content size (${contentSize.toString()} bytes) exceeds 256 MiB safety limit`
        );
    }
}

export function safeZstdDecompress(
    raw: Uint8Array,
    decompressFn: (input: Uint8Array) => Uint8Array
): Uint8Array {
    validateZstdFrameHeader(raw);
    const out = decompressFn(raw);
    if (out.byteLength > MAX_ZSTD_UNCOMPRESSED_BYTES) {
        throw new Error(
            `Decompressed Zstd payload (${out.byteLength} bytes) exceeds 256 MiB safety limit`
        );
    }
    return out;
}

export function flechetteTypeToString(type: any): string {
    if (!type || typeof type !== 'object') return String(type ?? 'unknown');
    switch (type.typeId) {
        case Type.Null:
            return 'null';
        case Type.Bool:
            return 'bool';
        case Type.Int: {
            const prefix = type.signed === false ? 'uint' : 'int';
            return `${prefix}${type.bitWidth ?? 32}`;
        }
        case Type.Float: {
            if (type.precision === 0) return 'float16';
            if (type.precision === 1) return 'float32';
            return 'float64';
        }
        case Type.Utf8:
            return 'utf8';
        case Type.LargeUtf8:
            return 'largeutf8';
        case Type.Utf8View:
            return 'utf8view';
        case Type.Binary:
            return 'binary';
        case Type.LargeBinary:
            return 'largebinary';
        case Type.BinaryView:
            return 'binaryview';
        case Type.FixedSizeBinary:
            return 'fixedsizebinary';
        case Type.Decimal:
            return `decimal${type.bitWidth ?? 128}`;
        case Type.Date:
            return type.unit === 0 ? 'date32' : 'date64';
        case Type.Timestamp:
            return 'timestamp';
        case Type.Time:
            return 'time';
        case Type.Duration:
            return 'duration';
        case Type.Interval:
            return 'interval';
        case Type.Dictionary:
            return 'dictionary';
        case Type.List:
        case Type.LargeList:
        case Type.FixedSizeList:
        case Type.ListView:
        case Type.LargeListView:
            return 'list';
        case Type.Struct:
            return 'struct';
        case Type.Map:
            return 'map';
        default:
            return `type_${String(type.typeId)}`;
    }
}

export function parseArrowSchemaFields(
    schemaBytes: Uint8Array
): { name: string; type: string }[] {
    const table = tableFromIPC([schemaBytes], FLECHETTE_IPC_OPTIONS);
    return table.schema.fields.map((f: any) => ({
        name: String(f.name),
        type: flechetteTypeToString(f.type),
    }));
}

export function parseTileTablesFromPayloads(
    buffers: (ArrayBuffer | Uint8Array | null)[],
    globalSchemaBuffer: Uint8Array | null,
    decompressFn: (input: Uint8Array) => Uint8Array
): any[] {
    return buffers
        .map((buffer) => {
            if (!buffer || buffer.byteLength === 0) return null;
            let rawData: Uint8Array =
                buffer instanceof Uint8Array
                    ? buffer
                    : new Uint8Array(buffer);
            if (hasZstdMagic(rawData)) {
                rawData = safeZstdDecompress(rawData, decompressFn);
            }

            let table: any = null;
            try {
                const direct = tableFromIPC(rawData, FLECHETTE_IPC_OPTIONS);
                if (direct && direct.numRows > 0 && direct.schema.fields.length > 0) {
                    table = direct;
                }
            } catch (err) {
                // Handle self-contained streams with negative-scale Decimal (`10n ** BigInt(scale)`)
                if (err instanceof RangeError) {
                    const split = splitLeadingIpcMessage(rawData);
                    if (split && split.rest.byteLength > 0) {
                        try {
                            const norm = normalizeNegativeDecimalSchema(split.firstMsg);
                            if (norm.negDecimalMultipliers) {
                                const patched = tableFromIPC(
                                    [norm.schemaBytes, split.rest],
                                    FLECHETTE_IPC_OPTIONS
                                );
                                if (patched && patched.numRows > 0) {
                                    tableNegDecimalMultipliers.set(
                                        patched,
                                        norm.negDecimalMultipliers
                                    );
                                    table = patched;
                                }
                            }
                        } catch {
                            // Fall through to globalSchemaBuffer fallback
                        }
                    }
                }
            }
            if ((!table || table.numRows === 0) && globalSchemaBuffer) {
                const norm = normalizeNegativeDecimalSchema(globalSchemaBuffer);
                table = tableFromIPC(
                    [norm.schemaBytes, rawData],
                    FLECHETTE_IPC_OPTIONS
                );
                if (table && norm.negDecimalMultipliers) {
                    tableNegDecimalMultipliers.set(table, norm.negDecimalMultipliers);
                }
            }
            return table;
        })
        .filter((t: any) => t !== null && t.numRows > 0);
}

export interface DecodedRowPayload {
    rowId: number;
    rowIdBigIntStr?: string;
    xU16: number;
    yU16: number;
    attributes: Record<string, unknown>;
}

export function decodeRowFromTables(
    tables: any[],
    rowIndex: number,
    categories?: Record<string, string[]>
): DecodedRowPayload | null {
    if (tables.length === 0 || rowIndex < 0 || rowIndex >= tables[0].numRows) {
        return null;
    }

    let rowId = rowIndex;
    let rowIdBigIntStr: string | undefined = undefined;
    let xU16 = 0;
    let yU16 = 0;
    const attributes: Record<string, unknown> = {};

    for (const table of tables) {
        for (const field of table.schema.fields) {
            const name: string = field.name;
            const child = table.getChild(name);
            if (!child) continue;

            if (name === 'row_id') {
                const rawId = child.at(rowIndex);
                if (typeof rawId === 'bigint') {
                    rowId = Number(rawId);
                    if (rawId > MAX_SAFE_BIGINT || rawId < -MAX_SAFE_BIGINT) {
                        rowIdBigIntStr = rawId.toString();
                    }
                } else if (typeof rawId === 'number') {
                    rowId = rawId;
                }
                continue;
            }
            if (name === 'x_u16') {
                xU16 = Number(child.at(rowIndex) ?? 0);
                continue;
            }
            if (name === 'y_u16') {
                yU16 = Number(child.at(rowIndex) ?? 0);
                continue;
            }
            if (name === 'z') {
                continue;
            }

            const rawVal = child.at(rowIndex);
            if (rawVal === null || rawVal === undefined) {
                attributes[name] = null;
                continue;
            }
            const decMult: number | undefined = tableNegDecimalMultipliers.get(table)?.[name];
            const val =
                decMult !== undefined && typeof rawVal === 'number' ? rawVal * decMult : rawVal;

            const catList: string[] | undefined = categories?.[name];
            if (catList && (typeof val === 'number' || typeof val === 'bigint')) {
                const code = Number(val);
                attributes[name] =
                    code >= 0 && code < catList.length ? String(catList[code]) : code;
            } else if (typeof val === 'bigint') {
                attributes[name] =
                    val <= MAX_SAFE_BIGINT && val >= -MAX_SAFE_BIGINT
                        ? Number(val)
                        : val.toString();
            } else {
                attributes[name] = val;
            }
        }
    }

    return {
        rowId,
        rowIdBigIntStr,
        xU16,
        yU16,
        attributes,
    };
}

export function extractTileColumnsFromTables(
    tables: any[],
    requestedColumns: string[] | undefined,
    pooledColumns?: Record<string, ArrayBuffer>,
    colOrigins?: Record<string, number>
): {
    columns: Record<string, ArrayBuffer>;
    transferables: ArrayBuffer[];
    numRows: number;
    dictionaries?: Record<string, string[]>;
} {
    if (tables.length === 0) {
        throw new Error('No tables found');
    }

    const numRows: number = tables[0].numRows;
    const getRawColEntry = (name: string): { child: any; decMult: number } | null => {
        for (const table of tables) {
            const child = table.getChild(name);
            if (child) {
                const decMult = tableNegDecimalMultipliers.get(table)?.[name] ?? 1;
                return { child, decMult };
            }
        }
        return null;
    };
    const getRawCol = (name: string) => getRawColEntry(name)?.child ?? null;

    const extractU16Coord = (name: string): Uint16Array | null => {
        const child = getRawCol(name);
        if (!child || !child.data || child.data.length === 0) return null;
        if (child.data.length === 1 && child.data[0].values instanceof Uint16Array) {
            return child.data[0].values.subarray(0, Math.min(numRows, child.data[0].length));
        }
        const out = new Uint16Array(numRows);
        let dst = 0;
        for (const chunk of child.data) {
            const len = Math.min(chunk.length, numRows - dst);
            if (len <= 0) break;
            if (chunk.values instanceof Uint16Array) {
                out.set(chunk.values.subarray(0, len), dst);
            } else {
                for (let i = 0; i < len; i++) {
                    out[dst + i] = Number(chunk.at(i) ?? 0);
                }
            }
            dst += len;
        }
        return out;
    };

    const xU16 = extractU16Coord('x_u16');
    const yU16 = extractU16Coord('y_u16');
    if (!xU16 || !yU16 || xU16.length < numRows || yU16.length < numRows) {
        throw new Error('Missing x_u16 or y_u16');
    }

    const dictionaries: Record<string, string[]> = {};
    const columns: Record<string, ArrayBuffer> = {};
    const transferables: ArrayBuffer[] = [];

    const getU16Col = (raw: Uint16Array, pooled: ArrayBuffer | undefined): ArrayBuffer => {
        const buf =
            pooled && pooled.byteLength === numRows * 2
                ? new Uint16Array(pooled, 0, numRows)
                : new Uint16Array(numRows);
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

        const colEntry = getRawColEntry(name);
        const child = colEntry?.child ?? null;
        const decMult = colEntry?.decMult ?? 1;
        if (!child || !child.data || child.data.length === 0) {
            buf.fill(NaN);
            return rawBuf;
        }

        const type = child.type;
        const typeId: number = type?.typeId;
        if (typeId === Type.Dictionary) {
            if (!dictionaries[name] && child.data[0]?.dictionary) {
                dictionaries[name] = child.data[0].dictionary.toArray().map((v: any) => String(v));
            }
        }

        const colOrigin =
            colOrigins &&
            typeof colOrigins[name] === 'number' &&
            Number.isFinite(colOrigins[name])
                ? colOrigins[name]
                : 0;
        const hasBigIntSafeOrigin =
            colOrigin !== 0 && Number.isSafeInteger(Math.trunc(colOrigin));
        const originBig = hasBigIntSafeOrigin ? BigInt(Math.trunc(colOrigin)) : 0n;
        const originFrac = hasBigIntSafeOrigin ? colOrigin - Math.trunc(colOrigin) : 0;

        const canUseValuesArray =
            (typeId === Type.Float && (type?.precision ?? 1) > 0) ||
            typeId === Type.Int ||
            typeId === Type.Dictionary;

        let dstOffset = 0;
        for (const chunk of child.data) {
            const chunkLen = Math.min(chunk.length, numRows - dstOffset);
            if (chunkLen <= 0) break;

            const hasNulls = chunk.nullCount > 0 && chunk.validity != null;
            const validity: Uint8Array | null = hasNulls ? chunk.validity : null;

            if (canUseValuesArray && chunk.values) {
                const vals = chunk.values;
                const isBigInt =
                    vals instanceof BigInt64Array || vals instanceof BigUint64Array;

                if (!hasNulls && !isBigInt && colOrigin === 0 && vals instanceof Float32Array) {
                    buf.set(vals.subarray(0, chunkLen), dstOffset);
                } else if (!hasNulls && !isBigInt) {
                    for (let i = 0; i < chunkLen; i++) {
                        buf[dstOffset + i] = vals[i] - colOrigin;
                    }
                } else if (!hasNulls && isBigInt) {
                    for (let i = 0; i < chunkLen; i++) {
                        const raw = vals[i];
                        buf[dstOffset + i] = hasBigIntSafeOrigin
                            ? Number(raw - originBig) - originFrac
                            : Number(raw) - colOrigin;
                    }
                } else {
                    for (let i = 0; i < chunkLen; i++) {
                        const isValid = (validity![i >> 3] & (1 << (i & 7))) !== 0;
                        if (!isValid) {
                            buf[dstOffset + i] = NaN;
                        } else {
                            const raw = vals[i];
                            if (typeof raw === 'bigint') {
                                buf[dstOffset + i] = hasBigIntSafeOrigin
                                    ? Number(raw - originBig) - originFrac
                                    : Number(raw) - colOrigin;
                            } else {
                                buf[dstOffset + i] = raw - colOrigin;
                            }
                        }
                    }
                }
            } else {
                // Bool, Float16 (precision === 0), Decimal128/256, Date32/64, Timestamp, Time
                for (let i = 0; i < chunkLen; i++) {
                    const v = chunk.at(i);
                    if (v === null || v === undefined) {
                        buf[dstOffset + i] = NaN;
                    } else if (typeof v === 'boolean') {
                        buf[dstOffset + i] = (v ? 1.0 : 0.0) - colOrigin;
                    } else if (typeof v === 'bigint') {
                        buf[dstOffset + i] = hasBigIntSafeOrigin
                            ? Number(v - originBig) - originFrac
                            : Number(v) - colOrigin;
                    } else if (typeof v === 'number') {
                        buf[dstOffset + i] = v * decMult - colOrigin;
                    } else {
                        buf[dstOffset + i] = NaN;
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
        const pooled = pooledColumns && pooledColumns[col] ? pooledColumns[col] : undefined;
        let buf: ArrayBuffer;
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

    return {
        columns,
        transferables,
        numRows,
        dictionaries: Object.keys(dictionaries).length > 0 ? dictionaries : undefined,
    };
}

const WORKER_MASK_ROWS_PER_SLICE = 262144;
const WORKER_MASK_WORDS_PER_SLICE = 8192; // 32,768 bytes (2,048 x vec4<u32>)

export interface WorkerTileRowIndex {
    rowIds: Uint32Array | BigUint64Array;
    isBigInt64: boolean;
    xU16: Uint16Array;
    yU16: Uint16Array;
    byteLength: number;
}

export interface WorkerSpatialQuerySpec {
    rect?: { minX: number; maxX: number; minY: number; maxY: number } | null;
    lasso?: ReadonlyArray<readonly [number, number]> | null;
    dataBounds?: { min_x: number; max_x: number; min_y: number; max_y: number };
    rootBounds: { minX: number; maxX: number; minY: number; maxY: number };
    effectiveFlipY: boolean;
}

/**
 * Extracts a compact per-tile `(row_id, x_u16, y_u16)` index (`Uint32Array` when all IDs fit in
 * `0 .. 2^32 - 1`, `BigUint64Array` when any ID exceeds `2^32 - 1` such as Gaia DR3 `source_id`).
 */
export function extractTileRowIndexFromTables(tables: any[]): WorkerTileRowIndex | null {
    if (!tables || tables.length === 0) return null;
    const numRows: number = tables[0].numRows;
    if (numRows <= 0) return null;

    const getCol = (name: string): any => {
        for (const t of tables) {
            const c = t.getChild(name);
            if (c) return c;
        }
        return null;
    };

    const copyU16 = (name: string): Uint16Array => {
        const out = new Uint16Array(numRows);
        const child = getCol(name);
        if (!child || !child.data) return out;
        let dst = 0;
        for (const chunk of child.data) {
            const len = Math.min(chunk.length, numRows - dst);
            if (len <= 0) break;
            if (chunk.values instanceof Uint16Array) {
                out.set(chunk.values.subarray(0, len), dst);
            } else {
                for (let i = 0; i < len; i++) {
                    out[dst + i] = Number(chunk.at(i) ?? 0);
                }
            }
            dst += len;
        }
        return out;
    };

    const xU16 = copyU16('x_u16');
    const yU16 = copyU16('y_u16');

    const rowIdCol = getCol('row_id');
    if (!rowIdCol || !rowIdCol.data || rowIdCol.data.length === 0) {
        const seq = new Uint32Array(numRows);
        for (let i = 0; i < numRows; i++) seq[i] = i;
        return {
            rowIds: seq,
            isBigInt64: false,
            xU16,
            yU16,
            byteLength: seq.byteLength + xU16.byteLength + yU16.byteLength,
        };
    }

    let needs64 = false;
    for (const chunk of rowIdCol.data) {
        const vals = chunk.values;
        if (vals instanceof BigUint64Array || vals instanceof BigInt64Array) {
            for (let i = 0; i < chunk.length; i++) {
                const v = vals[i];
                if (v < 0n || v > 0xffffffffn) {
                    needs64 = true;
                    break;
                }
            }
        } else if (vals && ArrayBuffer.isView(vals)) {
            for (let i = 0; i < chunk.length; i++) {
                const v = (vals as any)[i] as number;
                if (!Number.isInteger(v) || v < 0 || v > 0xffffffff) {
                    needs64 = true;
                    break;
                }
            }
        } else {
            for (let i = 0; i < chunk.length; i++) {
                const v = chunk.at(i);
                if (typeof v === 'bigint') {
                    if (v < 0n || v > 0xffffffffn) {
                        needs64 = true;
                        break;
                    }
                } else if (typeof v === 'number') {
                    if (!Number.isInteger(v) || v < 0 || v > 0xffffffff) {
                        needs64 = true;
                        break;
                    }
                }
            }
        }
        if (needs64) break;
    }

    if (!needs64) {
        const u32 = new Uint32Array(numRows);
        let dst = 0;
        for (const chunk of rowIdCol.data) {
            const len = Math.min(chunk.length, numRows - dst);
            if (len <= 0) break;
            const vals = chunk.values;
            if (vals instanceof Uint32Array) {
                u32.set(vals.subarray(0, len), dst);
            } else if (vals instanceof BigUint64Array || vals instanceof BigInt64Array) {
                for (let i = 0; i < len; i++) u32[dst + i] = Number(vals[i]);
            } else {
                for (let i = 0; i < len; i++) u32[dst + i] = Number(chunk.at(i) ?? 0) >>> 0;
            }
            dst += len;
        }
        return {
            rowIds: u32,
            isBigInt64: false,
            xU16,
            yU16,
            byteLength: u32.byteLength + xU16.byteLength + yU16.byteLength,
        };
    }

    const u64 = new BigUint64Array(numRows);
    let dst = 0;
    for (const chunk of rowIdCol.data) {
        const len = Math.min(chunk.length, numRows - dst);
        if (len <= 0) break;
        const vals = chunk.values;
        if (vals instanceof BigUint64Array) {
            u64.set(vals.subarray(0, len), dst);
        } else if (vals instanceof BigInt64Array) {
            for (let i = 0; i < len; i++) u64[dst + i] = BigInt.asUintN(64, vals[i]);
        } else {
            for (let i = 0; i < len; i++) {
                const v = chunk.at(i);
                if (typeof v === 'bigint') {
                    u64[dst + i] = BigInt.asUintN(64, v);
                } else if (typeof v === 'number' && Number.isFinite(v)) {
                    u64[dst + i] = BigInt.asUintN(64, BigInt(Math.trunc(v)));
                }
            }
        }
        dst += len;
    }
    return {
        rowIds: u64,
        isBigInt64: true,
        xU16,
        yU16,
        byteLength: u64.byteLength + xU16.byteLength + yU16.byteLength,
    };
}

function binarySearchU32(sorted: Uint32Array, target: number): boolean {
    let lo = 0;
    let hi = sorted.length - 1;
    while (lo <= hi) {
        const mid = (lo + hi) >>> 1;
        const v = sorted[mid];
        if (v === target) return true;
        if (v < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return false;
}

function binarySearchU64(sorted: BigUint64Array, target: bigint): boolean {
    let lo = 0;
    let hi = sorted.length - 1;
    while (lo <= hi) {
        const mid = (lo + hi) >>> 1;
        const v = sorted[mid];
        if (v === target) return true;
        if (v < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return false;
}

/**
 * Compiles `1 .. ceil(numRows / 262144)` `32,768`-byte `SlotVisibilityMask` slices
 * (`vec4<u32>`-aligned in WGSL) by binary-searching a sorted `Uint32Array` or `BigUint64Array`.
 */
export function compileSlotBitmaskSlices(
    index: WorkerTileRowIndex,
    sortedU32: Uint32Array | null,
    sortedU64: BigUint64Array | null
): ArrayBuffer[] {
    const numRows = index.rowIds.length;
    const numSlices = Math.max(1, Math.ceil(numRows / WORKER_MASK_ROWS_PER_SLICE));
    const slices: ArrayBuffer[] = [];
    const hasU32Set = Boolean(sortedU32 && sortedU32.length > 0);
    const hasU64Set = Boolean(sortedU64 && sortedU64.length > 0);

    for (let s = 0; s < numSlices; s++) {
        const buf = new ArrayBuffer(WORKER_MASK_WORDS_PER_SLICE * 4);
        const words = new Uint32Array(buf);
        if (hasU32Set || hasU64Set) {
            const rowBase = s * WORKER_MASK_ROWS_PER_SLICE;
            const count = Math.min(WORKER_MASK_ROWS_PER_SLICE, numRows - rowBase);
            if (!index.isBigInt64) {
                const ids32 = index.rowIds as Uint32Array;
                if (hasU32Set) {
                    const set32 = sortedU32!;
                    for (let localI = 0; localI < count; localI++) {
                        if (binarySearchU32(set32, ids32[rowBase + localI])) {
                            words[localI >>> 5] |= (1 << (localI & 31)) >>> 0;
                        }
                    }
                } else {
                    const set64 = sortedU64!;
                    for (let localI = 0; localI < count; localI++) {
                        if (binarySearchU64(set64, BigInt(ids32[rowBase + localI]))) {
                            words[localI >>> 5] |= (1 << (localI & 31)) >>> 0;
                        }
                    }
                }
            } else {
                const ids64 = index.rowIds as BigUint64Array;
                if (hasU64Set) {
                    const set64 = sortedU64!;
                    for (let localI = 0; localI < count; localI++) {
                        if (binarySearchU64(set64, ids64[rowBase + localI])) {
                            words[localI >>> 5] |= (1 << (localI & 31)) >>> 0;
                        }
                    }
                } else {
                    const set32 = sortedU32!;
                    for (let localI = 0; localI < count; localI++) {
                        const v = ids64[rowBase + localI];
                        if (v <= 0xffffffffn && binarySearchU32(set32, Number(v))) {
                            words[localI >>> 5] |= (1 << (localI & 31)) >>> 0;
                        }
                    }
                }
            }
        }
        slices.push(buf);
    }
    return slices;
}

function pointInPolyWorker(
    px: number,
    py: number,
    verts: ReadonlyArray<readonly [number, number]>
): boolean {
    const n = verts.length;
    if (n < 3) return false;
    let inside = false;
    for (let i = 0, j = n - 1; i < n; j = i++) {
        const xi = verts[i][0];
        const yi = verts[i][1];
        const xj = verts[j][0];
        const yj = verts[j][1];
        if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
            inside = !inside;
        }
    }
    return inside;
}

function pcgHashWorker(vIn: number): number {
    const state = (Math.imul(vIn >>> 0, 747796405) + 2891336453) >>> 0;
    const shift = ((state >>> 28) + 4) >>> 0;
    const word = Math.imul((state >>> shift) ^ state, 277803737) >>> 0;
    const res = ((word >>> 22) ^ word) >>> 0;
    return res / 4294967295.0;
}

/**
 * Binary-searches a small probe vector (`<= 64` pinned `HighlightOverlay` markers) against
 * the worker's sorted `row_id` filter without allocating any main-thread `Set<bigint>`.
 */
export function testProbeRowIdsInFilter(
    probeIds: BigUint64Array,
    sortedU32: Uint32Array | null,
    sortedU64: BigUint64Array | null
): Uint8Array {
    const out = new Uint8Array(probeIds.length);
    const hasU32 = Boolean(sortedU32 && sortedU32.length > 0);
    const hasU64 = Boolean(sortedU64 && sortedU64.length > 0);
    if (!hasU32 && !hasU64) return out;

    for (let i = 0; i < probeIds.length; i++) {
        const v = probeIds[i];
        if (hasU64) {
            out[i] = binarySearchU64(sortedU64!, v) ? 1 : 0;
        } else if (v <= 0xffffffffn) {
            out[i] = binarySearchU32(sortedU32!, Number(v)) ? 1 : 0;
        }
    }
    return out;
}

/** Evaluates a committed spatial `rect` or `lasso` query against a cached tile's `row_id`s in a worker. */
export function queryTileSpatialSelection(
    tileKey: string,
    index: WorkerTileRowIndex,
    spec: WorkerSpatialQuerySpec,
    hasActiveFilter = false,
    sortedU32: Uint32Array | null = null,
    sortedU64: BigUint64Array | null = null
): BigUint64Array {
    const [zStr, txStr, tyStr] = tileKey.split('/');
    const z = parseInt(zStr, 10);
    const tx = parseInt(txStr, 10);
    const ty = parseInt(tyStr, 10);
    const scale = Math.pow(0.5, z);
    const tileOffX = tx * scale;
    const tileOffY = ty * scale;

    const db = spec.dataBounds;
    const rb = spec.rootBounds;
    const effFlip = spec.effectiveFlipY;

    const mapX = (normX: number): number =>
        db ? db.min_x + normX * (db.max_x - db.min_x) : rb.minX + normX * (rb.maxX - rb.minX);
    const mapY = (normY: number): number =>
        db
            ? db.max_y - normY * (db.max_y - db.min_y)
            : effFlip
              ? rb.minY + normY * (rb.maxY - rb.minY)
              : rb.maxY - normY * (rb.maxY - rb.minY);

    let aabbMinX = -Infinity;
    let aabbMaxX = Infinity;
    let aabbMinY = -Infinity;
    let aabbMaxY = Infinity;

    if (spec.rect) {
        aabbMinX = Math.max(aabbMinX, Math.min(spec.rect.minX, spec.rect.maxX));
        aabbMaxX = Math.min(aabbMaxX, Math.max(spec.rect.minX, spec.rect.maxX));
        aabbMinY = Math.max(aabbMinY, Math.min(spec.rect.minY, spec.rect.maxY));
        aabbMaxY = Math.min(aabbMaxY, Math.max(spec.rect.minY, spec.rect.maxY));
    }

    const lasso = spec.lasso && spec.lasso.length >= 3 ? spec.lasso : null;
    if (lasso) {
        let lMinX = Infinity;
        let lMaxX = -Infinity;
        let lMinY = Infinity;
        let lMaxY = -Infinity;
        for (const [lx, ly] of lasso) {
            if (lx < lMinX) lMinX = lx;
            if (lx > lMaxX) lMaxX = lx;
            if (ly < lMinY) lMinY = ly;
            if (ly > lMaxY) lMaxY = ly;
        }
        aabbMinX = Math.max(aabbMinX, lMinX);
        aabbMaxX = Math.min(aabbMaxX, lMaxX);
        aabbMinY = Math.max(aabbMinY, lMinY);
        aabbMaxY = Math.min(aabbMaxY, lMaxY);
    }

    // Tile-level AABB early-out (with 2-ulp u16 jitter margin)
    const padNorm = (2.0 / 65535.0) * scale;
    const tX0 = mapX(tileOffX - padNorm);
    const tX1 = mapX(tileOffX + scale + padNorm);
    const tY0 = mapY(tileOffY - padNorm);
    const tY1 = mapY(tileOffY + scale + padNorm);
    if (
        Math.max(tX0, tX1) < aabbMinX ||
        Math.min(tX0, tX1) > aabbMaxX ||
        Math.max(tY0, tY1) < aabbMinY ||
        Math.min(tY0, tY1) > aabbMaxY
    ) {
        return new BigUint64Array(0);
    }

    const hits: bigint[] = [];
    const n = index.rowIds.length;
    const xArr = index.xU16;
    const yArr = index.yU16;
    const is64 = index.isBigInt64;
    const ids32 = !is64 ? (index.rowIds as Uint32Array) : null;
    const ids64 = is64 ? (index.rowIds as BigUint64Array) : null;
    const hasU32Set = Boolean(sortedU32 && sortedU32.length > 0);
    const hasU64Set = Boolean(sortedU64 && sortedU64.length > 0);

    if (hasActiveFilter && !hasU32Set && !hasU64Set) {
        return new BigUint64Array(0);
    }

    for (let i = 0; i < n; i++) {
        if (hasActiveFilter) {
            if (!is64) {
                const rid32 = ids32![i];
                const ok = hasU32Set
                    ? binarySearchU32(sortedU32!, rid32)
                    : binarySearchU64(sortedU64!, BigInt(rid32));
                if (!ok) continue;
            } else {
                const rid64 = ids64![i];
                const ok = hasU64Set
                    ? binarySearchU64(sortedU64!, rid64)
                    : rid64 <= 0xffffffffn && binarySearchU32(sortedU32!, Number(rid64));
                if (!ok) continue;
            }
        }
        const subX = (xArr[i] + (pcgHashWorker(i) - 0.5)) / 65535.0;
        const subY = (yArr[i] + (pcgHashWorker(i + 12345) - 0.5)) / 65535.0;
        const dx = mapX(tileOffX + subX * scale);
        if (dx < aabbMinX || dx > aabbMaxX) continue;
        const dy = mapY(tileOffY + subY * scale);
        if (dy < aabbMinY || dy > aabbMaxY) continue;
        if (lasso && !pointInPolyWorker(dx, dy, lasso)) continue;
        hits.push(is64 ? ids64![i] : BigInt(ids32![i]));
    }

    const out = new BigUint64Array(hits.length);
    for (let i = 0; i < hits.length; i++) out[i] = hits[i];
    return out;
}
