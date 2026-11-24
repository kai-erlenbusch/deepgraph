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
