import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
    columnFromArray,
    dateDay,
    dateMillisecond,
    decimal,
    dictionary,
    float16,
    int8,
    tableFromArrays,
    tableFromColumns,
    tableToIPC,
    timestamp,
    TimeUnit,
    utf8,
} from '@uwdata/flechette';
import {
    decodeRowFromTables,
    extractTileColumnsFromTables,
    MAX_ZSTD_UNCOMPRESSED_BYTES,
    parseArrowSchemaFields,
    parseTileTablesFromPayloads,
    safeZstdDecompress,
    validateZstdFrameHeader,
} from '../../src/workerCodec.ts';
import { isGaiaSourceHint } from '../../src/demo/gaiaPreset.ts';

// Canonical 3-row PyArrow IPC stream with Utf8View (`pa.string_view()`):
// row_id=[0,1,2], x_u16=[100,200,300], y_u16=[400,500,600], z=[0,0,0],
// uv=['hi', null, 'a_very_long_string_view_over_12_bytes']
const UTF8VIEW_IPC_B64 =
    '/////zABAAAQAAAAAAAKAAwABgAFAAgACgAAAAABBAAMAAAACAAIAAAABAAIAAAABAAAAAUAAADMAAAAjAAAAFwAAAAwAAAABAAAAFj///8AAAEYEAAAABgAAAAEAAAAAAAAAAIAAAB1dgAABAAEAAQAAACA////AAABAhAAAAAUAAAABAAAAAAAAAABAAAAegAAAGr///8IAAAAqP///wAAAQIQAAAAGAAAAAQAAAAAAAAABQAAAHlfdTE2AAAAlv///xAAAADU////AAABAhAAAAAYAAAABAAAAAAAAAAFAAAAeF91MTYAAADC////EAAAABAAFAAIAAYABwAMAAAAEAAQAAAAAAABAhAAAAAgAAAABAAAAAAAAAAGAAAAcm93X2lkAAAAAAYACAAEAAYAAABAAAAA/////3ABAAAUAAAAAAAAAAwAFgAGAAUACAAMAAwAAAAAAwQAHAAAAJAAAAAAAAAAAAAOABwAEAAEAAgAAAAMAA4AAADgAAAAJAAAABAAAAADAAAAAAAAAAAAAAABAAAAAQAAAAAAAAAAAAAACwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABgAAAAAAAAAGAAAAAAAAAAAAAAAAAAAABgAAAAAAAAABgAAAAAAAAAgAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAGAAAAAAAAACgAAAAAAAAAAAAAAAAAAAAoAAAAAAAAAAMAAAAAAAAAMAAAAAAAAAABAAAAAAAAADgAAAAAAAAAMAAAAAAAAABoAAAAAAAAACUAAAAAAAAAAAAAAAUAAAADAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAACAAAAAAAAAGQAyAAsAQAAkAH0AVgCAAAAAAAAAAAAAAUAAAAAAAAAAgAAAGhpAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlAAAAYV92ZQAAAAAAAAAAYV92ZXJ5X2xvbmdfc3RyaW5nX3ZpZXdfb3Zlcl8xMl9ieXRlcwAAAP////8AAAAA';

function b64ToBytes(b64: string): Uint8Array {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
}

describe('workerCodec — Flechette IPC, 1D RTO & Zstd Bomb Guard', () => {
    it('decodes Utf8View IPC streams (short inline, null, and >12B out-of-line view)', () => {
        const bytes = b64ToBytes(UTF8VIEW_IPC_B64);
        const tables = parseTileTablesFromPayloads([bytes], null, (b) => b);
        assert.equal(tables.length, 1);
        assert.equal(tables[0].numRows, 3);

        const row0 = decodeRowFromTables(tables, 0);
        const row1 = decodeRowFromTables(tables, 1);
        const row2 = decodeRowFromTables(tables, 2);
        assert.equal(row0?.attributes.uv, 'hi');
        assert.equal(row1?.attributes.uv, null);
        assert.equal(row2?.attributes.uv, 'a_very_long_string_view_over_12_bytes');
    });

    it('decodes Decimal128, Date32/Date64, Timestamp(s/ms/us/ns), Float16, Dictionary, and 3-row padded Uint16', () => {
        const table = tableFromColumns({
            row_id: columnFromArray([0n, 2_147_483_648n, 9_007_199_254_740_995n]),
            x_u16: columnFromArray(new Uint16Array([10, 32768, 65535])),
            y_u16: columnFromArray(new Uint16Array([20, 16384, 60000])),
            z: columnFromArray(new Uint8Array([0, 0, 0])),
            d32: columnFromArray([1704067200000, null, 1704153600000], dateDay()),
            d64: columnFromArray([1704067200000, null, 1704153600000], dateMillisecond()),
            ts_s: columnFromArray([1704067200000, null, 1704067201000], timestamp(TimeUnit.SECOND)),
            ts_us: columnFromArray(
                [1704067200000, null, 1704067200500],
                timestamp(TimeUnit.MICROSECOND)
            ),
            ts_ns: columnFromArray(
                [1704067200000, null, 1704067200500],
                timestamp(TimeUnit.NANOSECOND)
            ),
            dec: columnFromArray([123.45, null, -67.89], decimal(10, 2)),
            f16: columnFromArray([1.5, null, -2.25], float16()),
            legacy_cat: columnFromArray(['alpha', null, 'beta'], dictionary(utf8(), int8())),
        });

        const ipc = tableToIPC(table, { format: 'stream' })!;
        const schemaFields = parseArrowSchemaFields(ipc);
        const typeMap = Object.fromEntries(schemaFields.map((f) => [f.name, f.type]));
        assert.equal(typeMap.row_id, 'int64');
        assert.equal(typeMap.d32, 'date32');
        assert.equal(typeMap.ts_us, 'timestamp');
        assert.equal(typeMap.dec, 'decimal128');
        assert.equal(typeMap.f16, 'float16');
        assert.equal(typeMap.legacy_cat, 'dictionary');

        const parsed = parseTileTablesFromPayloads([ipc], null, (b) => b);
        const colOrigins = {
            d32: 1704067200000,
            ts_us: 1704067200000,
            ts_ns: 1704067200000,
        };
        const extracted = extractTileColumnsFromTables(
            parsed,
            ['xy_u16', 'x_u16', 'y_u16', 'd32', 'ts_us', 'ts_ns', 'dec', 'f16', 'legacy_cat'],
            undefined,
            colOrigins
        );

        // 3-row 8-byte padded Uint16 buffers MUST slice strictly to numRows = 3 (6 bytes, 12 bytes for xy_u16)
        assert.equal(extracted.numRows, 3);
        assert.equal(extracted.columns.x_u16.byteLength, 6);
        assert.equal(extracted.columns.xy_u16.byteLength, 12);

        // 1D RTO preserves sub-second precision in Float32 on modern epoch-ms timestamps!
        const tsUsF32 = new Float32Array(extracted.columns.ts_us);
        assert.equal(tsUsF32[0], 0);
        assert.ok(Number.isNaN(tsUsF32[1]));
        assert.equal(tsUsF32[2], 500);

        const d32F32 = new Float32Array(extracted.columns.d32);
        assert.equal(d32F32[0], 0);
        assert.ok(Number.isNaN(d32F32[1]));
        assert.equal(d32F32[2], 86_400_000);

        const decF32 = new Float32Array(extracted.columns.dec);
        assert.ok(Math.abs(decF32[0] - 123.45) < 1e-4);
        assert.ok(Number.isNaN(decF32[1]));
        assert.ok(Math.abs(decF32[2] - -67.89) < 1e-4);

        const f16F32 = new Float32Array(extracted.columns.f16);
        assert.equal(f16F32[0], 1.5);
        assert.ok(Number.isNaN(f16F32[1]));
        assert.equal(f16F32[2], -2.25);

        // Legacy v0.0 Dictionary extracts integer codes + dictionary strings (not NaN!)
        assert.deepEqual(extracted.dictionaries?.legacy_cat, ['alpha', 'beta']);
        const catF32 = new Float32Array(extracted.columns.legacy_cat);
        assert.equal(catF32[0], 0);
        assert.ok(Number.isNaN(catF32[1]));
        assert.equal(catF32[2], 1);

        // Row decoding preserves > i32::MAX and > 2^53 - 1 BigInt IDs & unshifted attributes
        const r1 = decodeRowFromTables(parsed, 1);
        assert.equal(r1?.rowId, 2_147_483_648);
        assert.equal(r1?.rowIdBigIntStr, undefined);

        const r2 = decodeRowFromTables(parsed, 2);
        assert.equal(r2?.rowIdBigIntStr, '9007199254740995');
        assert.equal(r2?.attributes.ts_us, 1704067200500);
        assert.equal(r2?.attributes.legacy_cat, 'beta');
    });

    it('subtracts colOrigin in BigInt before Float32 cast on high-offset Int64 columns', () => {
        const base = 1_710_000_000_000n;
        const ipc = tableToIPC(
            tableFromArrays({
                row_id: new BigUint64Array([1n, 2n]),
                x_u16: new Uint16Array([0, 65535]),
                y_u16: new Uint16Array([0, 65535]),
                z: new Uint8Array([0, 0]),
                ns_off: new BigInt64Array([base, base + 250n]),
            }),
            { format: 'stream' }
        )!;
        const parsed = parseTileTablesFromPayloads([ipc], null, (b) => b);
        const extracted = extractTileColumnsFromTables(parsed, ['ns_off'], undefined, {
            ns_off: Number(base),
        });
        const f32 = new Float32Array(extracted.columns.ns_off);
        assert.equal(f32[0], 0);
        assert.equal(f32[1], 250);
    });

    it('rejects Zstd frame headers declaring > 256 MiB before invoking WASM decompress', () => {
        // RFC 8878: Magic (4B) + FHD (FCS_Flag=3 -> 8-byte contentSize, Single_Segment=1 -> 0xE0) + 8-byte LE uint64
        const bomb = new Uint8Array(13);
        bomb.set([0x28, 0xb5, 0x2f, 0xfd, 0xe0], 0);
        const dv = new DataView(bomb.buffer);
        dv.setBigUint64(5, BigInt(MAX_ZSTD_UNCOMPRESSED_BYTES) + 1n, true);

        let wasmCalled = false;
        assert.throws(() => {
            safeZstdDecompress(bomb, () => {
                wasmCalled = true;
                return new Uint8Array(0);
            });
        }, /exceeds 256 MiB safety limit/);
        assert.equal(wasmCalled, false);

        // Valid <= 256 MiB header passes pre-check
        dv.setBigUint64(5, 1024n, true);
        validateZstdFrameHeader(bomb);
    });

    it('decodes multi-batch negative-scale Decimal128 (scale=-2), extreme negative Decimal128, and TZ Timestamps', () => {
        // Real 2-batch PyArrow 24.0 IPC stream + schema-stripped v0.1 tile payload with:
        // dec_neg_scale: pa.decimal128(10, -2) = [1200, None, -3400, 0]
        // dec_huge_neg:  pa.decimal128(38, 2)  = [-1.23456789e29, 0.05, None, -1e35]
        // ts_ns_tz:      pa.timestamp('ns', tz='America/Chicago') = [1767312000123456789, None, +2s, +3s]
        // ts_us_utc:     pa.timestamp('us', tz='UTC') = [1767312000500000, +1.5s, None, +3.5s]
        // dict_col:      2-batch Dictionary(int8, utf8) = ['catA', None, 'catB', 'catA']
        const schemaB64 =
            '/////5ACAAAQAAAAAAAKAAwABgAFAAgACgAAAAABBAAEAAAAlP///wQAAAAJAAAAMAIAAPABAADAAQAAlAEAAFABAAAEAQAAsAAAAHAAAAAUAAAAEAAYAAgABgAHAAwAEAAUABAAAAAAAAEFFAAAAEgAAAAkAAAABAAAAAAAAAAIAAAAZGljdF9jb2wAAAAACAAIAAAABAAIAAAADAAAAAgADAAIAAcACAAAAAAAAAEIAAAABAAEAAQAAAA8////AAABChAAAAAcAAAABAAAAAAAAAAJAAAAdHNfdXNfdXRjAAAAxP///wAAAgAEAAAAAwAAAFVUQwB4////AAABChAAAAAkAAAABAAAAAAAAAAIAAAAdHNfbnNfdHoAAAAACAAMAAYACAAIAAAAAAADAAQAAAAPAAAAQW1lcmljYS9DaGljYWdvAMj///8AAAEHEAAAACAAAAAEAAAAAAAAAAwAAABkZWNfaHVnZV9uZWcAAAAAuP///yYAAAACAAAAEAAUAAgABgAHAAwAAAAQABAAAAAAAAEHEAAAACgAAAAEAAAAAAAAAA0AAABkZWNfbmVnX3NjYWxlAAAACAAMAAQACAAIAAAACgAAAP7///+A////AAAAAhAAAAAUAAAABAAAAAAAAAABAAAAegAAAGr///8IAAAAqP///wAAAAIQAAAAGAAAAAQAAAAAAAAABQAAAHlfdTE2AAAAlv///xAAAADU////AAAAAhAAAAAYAAAABAAAAAAAAAAFAAAAeF91MTYAAADC////EAAAABAAFAAIAAAABwAMAAAAEAAQAAAAAAAAAhAAAAAgAAAABAAAAAAAAAAGAAAAcm93X2lkAAAAAAYACAAEAAYAAABAAAAAAAAAAA==';
        const full2BatchB64 =
            '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IAAAAAAAAABIAAAAAAAAAAEAAAAAAAAAUAAAAAAAAAAgAAAAAAAAAHAAAAAAAAAAAAAAAAAAAABwAAAAAAAAABAAAAAAAAAAgAAAAAAAAAABAAAAAAAAAIgAAAAAAAAAEAAAAAAAAACYAAAAAAAAAAAAAAAAAAAAmAAAAAAAAAACAAAAAAAAAAAAAAAJAAAAAgAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAACAAAAAAAAAAAAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAACAAAAAAAAAAEAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAQAAAAAAAAACAAAAAAAAAAAAAAAAAAAADAAAAAAAAAANAAAAAAAAAPQBWAIAAAAAvAIgAwAAAAAAAAAAAAAAAN7///////////////////8AAAAAAAAAAAAAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAGDJC/8muSUq7xF6+ACUfvbmwIYYAF4ZMufAhhgCAAAAAAAAAAAAAAAAAAAA4ActZFxHBgABAAAAAAAAAP////8AAAAA';
        const v01StrippedB64 =
            '/////6gAAAAUAAAAAAAAAAwAFAAGAAUACAAMAAwAAAAAAgQAFAAAABgAAAAAAAAACAAKAAAABAAIAAAAEAAAAAAACgAYAAwABAAIAAoAAABMAAAAEAAAAAIAAAAAAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAAAAAAABAAAAAAAAAACAAAAAAAAAAAAAAAAQAAAAIAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAgAAAAAAAAAY2F0QWNhdEL/////CAIAABQAAAAAAAAADAAWAAYABQAIAAwADAAAAAADBAAYAAAAqAAAAAAAAAAAAAoAGAAMAAQACAAKAAAAPAEAABAAAAACAAAAAAAAAAAAAAASAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAEAAAAAAAAABgAAAAAAAAAAAAAAAAAAAAYAAAAAAAAAAQAAAAAAAAAIAAAAAAAAAAAAAAAAAAAACAAAAAAAAAAAgAAAAAAAAAoAAAAAAAAAAEAAAAAAAAAMAAAAAAAAAAgAAAAAAAAAFAAAAAAAAAAAAAAAAAAAABQAAAAAAAAACAAAAAAAAAAcAAAAAAAAAABAAAAAAAAAHgAAAAAAAAAEAAAAAAAAACIAAAAAAAAAAAAAAAAAAAAiAAAAAAAAAAQAAAAAAAAAJgAAAAAAAAAAQAAAAAAAACgAAAAAAAAAAIAAAAAAAAAAAAAAAkAAAACAAAAAAAAAAAAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAACAAAAAAAAAAAAAAAAAAAAAgAAAAAAAAABAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAACAAAAAAAAAAEAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAQAAAAAAAAAKAAAAAAAAAAsAAAAAAAAAZADIAAAAAAAsAZABAAAAAAAAAAAAAAAAAQAAAAAAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADfxV9v6SK8ppvD9Sxk////BQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAFc2khubAhhgAAAAAAAAAACBB/2NcRwYAYIMOZFxHBgABAAAAAAAAAAAAAAAAAAAD/////AAAAAA==';

        const schemaBytes = b64ToBytes(schemaB64);
        const full2Bytes = b64ToBytes(full2BatchB64);

        // Verify schema-stripped v0.1 payload with negative-scale Decimal128
        const singleBatchFull = b64ToBytes(full2BatchB64);
        const v01Tables = parseTileTablesFromPayloads(
            [singleBatchFull.subarray(schemaBytes.byteLength)],
            schemaBytes,
            (x) => x
        );
        assert.equal(v01Tables[0].numRows, 4);
        void v01StrippedB64;

        // Verify self-contained 2-batch stream with globalSchemaBuffer === null AND with globalSchemaBuffer
        for (const gs of [null, schemaBytes]) {
            const tables = parseTileTablesFromPayloads([full2Bytes], gs, (x) => x);
            assert.equal(tables[0].numRows, 4);
            const extracted = extractTileColumnsFromTables(
                tables,
                ['xy_u16', 'dec_neg_scale', 'dec_huge_neg', 'ts_ns_tz', 'ts_us_utc', 'dict_col'],
                undefined,
                { ts_ns_tz: 1767312000000, ts_us_utc: 1767312000000 }
            );
            const negScaleF32 = new Float32Array(extracted.columns.dec_neg_scale);
            assert.equal(negScaleF32[0], 1200);
            assert.ok(Number.isNaN(negScaleF32[1]));
            assert.equal(negScaleF32[2], -3400);
            assert.equal(negScaleF32[3], 0);

            const hugeNegF32 = new Float32Array(extracted.columns.dec_huge_neg);
            assert.ok(hugeNegF32[0] < -1e29);
            assert.ok(Math.abs(hugeNegF32[1] - 0.05) < 1e-6);
            assert.ok(Number.isNaN(hugeNegF32[2]));
            assert.ok(hugeNegF32[3] < -9.9e34);

            const tsNsF32 = new Float32Array(extracted.columns.ts_ns_tz);
            assert.ok(Math.abs(tsNsF32[0] - 123.456789) < 1e-3);
            assert.ok(Number.isNaN(tsNsF32[1]));
            assert.equal(tsNsF32[2], 2000);
            assert.equal(tsNsF32[3], 3000);

            const row0 = decodeRowFromTables(tables, 0)!;
            const row2 = decodeRowFromTables(tables, 2)!;
            assert.equal(row0.attributes.dec_neg_scale, 1200);
            assert.equal(row2.attributes.dec_neg_scale, -3400);
            assert.equal(row0.attributes.dict_col, 'catA');
            assert.equal(row2.attributes.dict_col, 'catB');
        }

        // Also verify CustomRangeSource.getKey() hint detection in gaiaPreset
        assert.equal(
            isGaiaSourceHint({
                getKey: () => 'mem://gaia_v011_full.arrowtiles',
                getBytes: async () => ({ data: new ArrayBuffer(0) }),
            }),
            true
        );
    });

    it('imports pmtiles.worker.ts cleanly in Node v24 without ReferenceError on self or eager WASM fetch', async () => {
        const mod = await import('../../src/pmtiles.worker.ts');
        assert.ok(mod !== null && typeof mod === 'object');
    });
});
