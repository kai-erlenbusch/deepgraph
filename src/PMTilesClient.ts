import {
    PMTiles,
    Compression,
    FileSource,
    FetchSource,
    SharedPromiseCache,
    type Source,
    type RangeResponse,
} from 'pmtiles';
import PQueue from 'p-queue';
import { decompressSync } from 'fflate';
import type { BoundingBox, CustomRangeSource, TileSource } from './lib/types.ts';
import type { WorkerSpatialQuerySpec } from './workerCodec.ts';

const MAX_COL_POOL_PER_COLUMN = 16;
const MAX_RAW_PAYLOAD_POOL = 16;
const MAX_RAW_PICK_CACHE_ENTRIES = 8;
const MAX_RAW_PICK_CACHE_BYTES = 24 * 1024 * 1024; // 24 MiB hard cap for compressed pick tile LRU
const MAX_TILE_FAILURE_ATTEMPTS = 4;
const BASE_RETRY_DELAY_MS = 500;
const MIN_TILE_CACHE_SIZE = 32; // >= 21 pinned Z=0..2 overview tiles + working budget

function isCustomRangeSource(src: TileSource): src is CustomRangeSource {
    return (
        typeof src === 'object' &&
        src !== null &&
        !(typeof File !== 'undefined' && src instanceof File) &&
        typeof (src as CustomRangeSource).getBytes === 'function'
    );
}

/**
 * Wraps an HTTP `FetchSource` or `CustomRangeSource` to pre-cache the PMTiles leaf directory
 * region in RAM (only 367 KB for 193,246 tiles in `gaia_full.arrowtiles`), eliminating
 * secondary Range round-trips when zooming/panning into Z=5..8 subtrees, while cancelling
 * any in-flight prefetch cleanly on `destroy()`.
 */
export class FastFetchSource implements Source {
    private inner: Source;
    private maxLeafBytes: number;
    private prefetchController = new AbortController();
    private leafOffset = -1;
    private leafLength = 0;
    private leafBuffer: ArrayBuffer | null = null;
    private leafPromise: Promise<ArrayBuffer | null> | null = null;

    constructor(urlOrSource: string | Source, maxLeafBytes = 16 * 1024 * 1024) {
        if (typeof urlOrSource === 'string') {
            const fetchInner = new FetchSource(urlOrSource);
            (fetchInner as any).chromeWindowsNoCache = false;
            this.inner = fetchInner;
        } else {
            this.inner = urlOrSource;
        }
        this.maxLeafBytes = Math.max(0, maxLeafBytes);
    }

    public getKey(): string {
        return this.inner.getKey();
    }

    public abortPrefetch(): void {
        this.prefetchController.abort();
    }

    public prefetchLeafDirectories(leafOffset: number, leafLength: number): void {
        if (
            this.prefetchController.signal.aborted ||
            this.maxLeafBytes <= 0 ||
            !leafLength ||
            leafLength <= 0 ||
            leafLength > this.maxLeafBytes ||
            this.leafPromise
        ) {
            return;
        }
        this.leafOffset = leafOffset;
        this.leafLength = leafLength;
        this.leafPromise = this.inner
            .getBytes(leafOffset, leafLength, this.prefetchController.signal)
            .then((res) => {
                if (!this.prefetchController.signal.aborted) {
                    this.leafBuffer = res.data;
                    return res.data;
                }
                return null;
            })
            .catch(() => null);
    }

    public async getBytes(
        offset: number,
        length: number,
        signal?: AbortSignal,
        etag?: string
    ): Promise<RangeResponse> {
        if (
            this.leafOffset >= 0 &&
            offset >= this.leafOffset &&
            offset + length <= this.leafOffset + this.leafLength
        ) {
            const buf = this.leafBuffer ?? (this.leafPromise ? await this.leafPromise : null);
            if (buf) {
                const relStart = offset - this.leafOffset;
                return { data: buf.slice(relStart, relStart + length) };
            }
        }
        return this.inner.getBytes(
            offset,
            length,
            signal ?? this.prefetchController.signal,
            etag
        );
    }
}

export type { BoundingBox };

export interface TileData {
    key: string; // "z/x/y"
    columns: Record<string, ArrayBuffer>;
    numRows: number;
    needsUpdate?: boolean;
    uploadedToGpu?: boolean;
    rowMaskSlices?: ArrayBuffer[];
}

export interface SchemaField {
    name: string;
    type: string;
}

export interface DecodedTileRow {
    rowId: number;
    rowIdBigInt?: bigint;
    xU16: number;
    yU16: number;
    attributes: Record<string, unknown>;
}

export class PMTilesClient {
    private static nextClientSeq = 0;
    private readonly clientSeq = ++PMTilesClient.nextClientSeq;
    private rootBounds: BoundingBox;
    public cacheChanged = true;
    private activeTiles = new Map<string, TileData>();
    public onTileLoaded: ((tile: TileData) => void) | null = null;
    public onTileUnloaded: ((tileId: string) => void) | null = null;
    public onDictionariesLoaded: ((dictionaries: Record<string, string[]>) => void) | null = null;
    private loadingTiles = new Set<string>();
    private currentlyVisibleIds = new Set<string>();
    private lastSeenTimestamp = new Map<string, number>();
    private knownEmptyTiles = new Set<string>();
    private failedTileBackoff = new Map<string, { attempts: number; nextRetryAt: number }>();
    private permanentlyFailedTiles = new Set<string>();
    private retryTimers = new Set<ReturnType<typeof setTimeout>>();
    private maxCacheSize: number;
    private queue = new PQueue({ concurrency: 6 });
    private allWorkers: Worker[] = [];
    private idleWorkers: Worker[] = [];
    private pendingTasks: { payload: any; transfer: any[]; abortSignal?: AbortSignal }[] = [];
    private abortControllers = new Map<string, AbortController>();
    private resolvers = new Map<
        string,
        {
            resolve: Function;
            reject: Function;
            t0: number;
            tNetEnd: number;
            tWorkerStart: number;
            wasDirectoryWarm: boolean;
            signal?: AbortSignal;
            onAbort?: () => void;
        }
    >();
    private decompressResolvers = new Map<string, { resolve: Function; reject: Function }>();
    private schemaResolvers = new Map<
        string,
        { resolve: (fields: SchemaField[]) => void; reject: (err: Error) => void }
    >();
    private rowDecodeResolvers = new Map<
        string,
        { resolve: Function; reject: Function; onCacheMiss?: () => void }
    >();
    private workerRpcResolvers = new Map<
        string,
        { resolve: (data: any) => void; reject: (err: Error) => void }
    >();
    private retainRowIds = false;
    private hasActiveRowIdFilter = false;
    private indexedTileKeys = new Set<string>();
    private tileRowIndexBytes = new Map<string, number>();
    private rawPickTileCache = new Map<string, { slices: (ArrayBuffer | null)[]; byteLength: number }>();
    private rawPickTileCacheBytes = 0;
    private workerCachedRowKey = new Map<Worker, string>();
    private inFlightPickTileFetches = new Map<
        string,
        {
            promise: Promise<(ArrayBuffer | null)[] | null>;
            controller: AbortController;
            waiters: Set<object>;
        }
    >();
    private schemaBuffer: Uint8Array | null = null;
    private nextReqId = 0;
    private destroyed = false;
    private directoryWarm = false;
    private fastFetchSource: FastFetchSource | null = null;

    private bufferPool: {
        columns: Map<string, ArrayBuffer[]>;
        rawPayload: ArrayBuffer[];
    } = { columns: new Map(), rawPayload: [] };

    private requestedColumns: string[] = ['xy_u16'];
    private colOrigins: Record<string, number> = {};

    public fetchTelemetry = {
        net: [] as number[],
        worker: [] as number[],
        steadyTotals: [] as number[],
        coldBootMs: 0,
        p95: 0,
        sloBudgetMs: 50,
        isSloViolated: false,
    };

    private corePmtiles: PMTiles;
    private layerPmtiles: PMTiles[] = [];
    public currentMaxZ = 0;
    public archiveMaxZoom = 14;

    constructor(
        coreSource: TileSource,
        layerUrls: string[],
        rootBounds: BoundingBox,
        maxCacheSize = 250,
        schemaBuffer: Uint8Array | null = null,
        archiveMaxZoom = 14
    ) {
        this.rootBounds = rootBounds;
        this.maxCacheSize = Math.max(MIN_TILE_CACHE_SIZE, maxCacheSize);
        this.schemaBuffer = schemaBuffer;
        this.archiveMaxZoom = Math.max(0, Math.min(20, archiveMaxZoom));

        const hwCores = navigator.hardwareConcurrency || 8;
        const numWorkers = Math.min(6, Math.max(2, Math.floor(hwCores / 2)));
        for (let i = 0; i < numWorkers; i++) {
            const worker = new Worker(new URL('./pmtiles.worker.ts', import.meta.url), {
                type: 'module',
            });
            worker.onmessage = (e) => this.handleWorkerMessage(e, worker);

            if (schemaBuffer) {
                const schemaCopy = schemaBuffer.slice();
                worker.postMessage(
                    { action: 'init_schema', schemaBuffer: schemaCopy },
                    [schemaCopy.buffer]
                );
            }

            this.allWorkers.push(worker);
            this.idleWorkers.push(worker);
        }

        const customDecompress = async (
            buf: ArrayBuffer,
            compression: Compression
        ): Promise<ArrayBuffer> => {
            const compCode = compression as number;
            if (compression === Compression.None || compCode === 1) return buf;
            if (compression === Compression.Gzip || compCode === 2) {
                const out = decompressSync(new Uint8Array(buf));
                return out.byteOffset === 0 && out.byteLength === out.buffer.byteLength
                    ? (out.buffer as ArrayBuffer)
                    : (out.slice().buffer as ArrayBuffer);
            }
            if (compCode === 4) {
                const copy = buf.slice(0);
                return new Promise((resolve, reject) => {
                    if (this.destroyed) {
                        reject(new Error('PMTilesClient destroyed'));
                        return;
                    }
                    const id = `decomp_${++this.nextReqId}`;
                    this.decompressResolvers.set(id, { resolve, reject });
                    this.dispatchToWorker({ action: 'decompress', id, buffer: copy }, [copy]);
                });
            }
            throw new Error(`Unsupported compression: ${compression}`);
        };

        const tileDecompress = async (
            buf: ArrayBuffer,
            compression: Compression
        ): Promise<ArrayBuffer> => {
            const compCode = compression as number;
            if (compression === Compression.None || compCode === 1 || compCode === 4) return buf;
            return customDecompress(buf, compression);
        };

        const sharedCache = new SharedPromiseCache(1000, true, customDecompress);
        const createPmtiles = (src: TileSource, isCore = false) => {
            let source: Source;
            let fastFetch: FastFetchSource | null = null;
            if (typeof src === 'string') {
                fastFetch = new FastFetchSource(src, 16 * 1024 * 1024);
                if (isCore) this.fastFetchSource = fastFetch;
                source = fastFetch;
            } else if (isCustomRangeSource(src)) {
                const resolvedKey = src.key ?? src.getKey?.() ?? `custom_${this.clientSeq}`;
                const adapted: Source = {
                    getKey: () => resolvedKey,
                    getBytes: (offset, length, signal, etag) =>
                        src.getBytes(offset, length, signal, etag),
                };
                const maxPrefetch = src.maxLeafPrefetchBytes ?? 1 * 1024 * 1024;
                fastFetch = new FastFetchSource(adapted, maxPrefetch);
                if (isCore) this.fastFetchSource = fastFetch;
                source = fastFetch;
            } else {
                source = new FileSource(src);
            }
            return new PMTiles(source, sharedCache, tileDecompress);
        };

        this.corePmtiles = createPmtiles(coreSource, true);
        for (const lUrl of layerUrls) {
            this.layerPmtiles.push(createPmtiles(lUrl, false));
        }
    }

    /** Reads header and metadata from the single owned `PMTiles` instance (zero duplicate instances). */
    public async getHeaderAndMetadata(): Promise<{ header: any; metadata: any }> {
        const header = await this.corePmtiles.getHeader();
        if (this.fastFetchSource && header.leafDirectoryOffset && header.leafDirectoryLength) {
            this.fastFetchSource.prefetchLeafDirectories(
                header.leafDirectoryOffset,
                header.leafDirectoryLength
            );
        }
        const metadata = await this.corePmtiles.getMetadata();
        return { header, metadata };
    }

    /** Applies resolved archive schema, maxZoom, and rootBounds onto this single `PMTilesClient`. */
    public configureArchive(
        schemaBuffer: Uint8Array | null,
        archiveMaxZoom: number,
        rootBounds: BoundingBox
    ): void {
        this.rootBounds = rootBounds;
        this.archiveMaxZoom = Math.max(0, Math.min(20, archiveMaxZoom));
        this.clearRawPickTileCache(false);
        if (schemaBuffer) {
            this.schemaBuffer = schemaBuffer;
            for (const worker of this.allWorkers) {
                const schemaCopy = schemaBuffer.slice();
                worker.postMessage(
                    { action: 'init_schema', schemaBuffer: schemaCopy },
                    [schemaCopy.buffer]
                );
            }
        }
    }

    private cacheRawPickTile(
        key: string,
        rawItems: (ArrayBuffer | Uint8Array | null | undefined)[]
    ): (ArrayBuffer | null)[] | null {
        if (!rawItems || rawItems.length === 0 || !rawItems[0]) return null;
        const existing = this.rawPickTileCache.get(key);
        if (existing) {
            this.rawPickTileCache.delete(key);
            this.rawPickTileCache.set(key, existing);
            return existing.slices;
        }

        let totalBytes = 0;
        for (const item of rawItems) {
            if (!item) continue;
            totalBytes += item.byteLength;
        }
        if (totalBytes <= 0 || totalBytes > MAX_RAW_PICK_CACHE_BYTES) {
            return null;
        }

        const slices: (ArrayBuffer | null)[] = rawItems.map((item) => {
            if (!item || item.byteLength === 0) return null;
            if (item instanceof ArrayBuffer) return item.slice(0);
            const u8 =
                item instanceof Uint8Array
                    ? item
                    : new Uint8Array(
                          (item as any).buffer,
                          (item as any).byteOffset,
                          (item as any).byteLength
                      );
            return u8.slice().buffer as ArrayBuffer;
        });

        while (
            this.rawPickTileCache.size >= MAX_RAW_PICK_CACHE_ENTRIES ||
            (this.rawPickTileCache.size > 0 &&
                this.rawPickTileCacheBytes + totalBytes > MAX_RAW_PICK_CACHE_BYTES)
        ) {
            const oldestKey = this.rawPickTileCache.keys().next().value;
            if (oldestKey === undefined) break;
            const evicted = this.rawPickTileCache.get(oldestKey);
            this.rawPickTileCache.delete(oldestKey);
            if (evicted) {
                this.rawPickTileCacheBytes = Math.max(
                    0,
                    this.rawPickTileCacheBytes - evicted.byteLength
                );
            }
        }

        this.rawPickTileCache.set(key, { slices, byteLength: totalBytes });
        this.rawPickTileCacheBytes += totalBytes;
        return slices;
    }

    private getRawPickTile(key: string): (ArrayBuffer | null)[] | null {
        const entry = this.rawPickTileCache.get(key);
        if (!entry) return null;
        this.rawPickTileCache.delete(key);
        this.rawPickTileCache.set(key, entry);
        return entry.slices;
    }

    private clearRawPickTileCache(notifyWorkers = true, clearRowIds = false): void {
        this.rawPickTileCache.clear();
        this.rawPickTileCacheBytes = 0;
        this.workerCachedRowKey.clear();
        if (clearRowIds) {
            this.indexedTileKeys.clear();
            this.tileRowIndexBytes.clear();
            this.hasActiveRowIdFilter = false;
        }
        for (const inflight of this.inFlightPickTileFetches.values()) {
            inflight.controller.abort();
        }
        this.inFlightPickTileFetches.clear();
        if (notifyWorkers) {
            for (const worker of this.allWorkers) {
                worker.postMessage({ action: 'clear_row_cache', clearRowIds });
            }
        }
    }

    private notifyWorkersDropTile(key: string): void {
        if (!this.indexedTileKeys.has(key) && !this.tileRowIndexBytes.has(key)) return;
        this.indexedTileKeys.delete(key);
        this.tileRowIndexBytes.delete(key);
        for (const worker of this.allWorkers) {
            worker.postMessage({ action: 'drop_tile', key });
        }
    }

    public setRetainRowIds(enabled: boolean): void {
        if (this.destroyed || this.retainRowIds === enabled) return;
        this.retainRowIds = enabled;
        if (!enabled && !this.hasActiveRowIdFilter) {
            this.indexedTileKeys.clear();
            this.tileRowIndexBytes.clear();
        }
        for (const worker of this.allWorkers) {
            worker.postMessage({ action: 'set_retain_row_ids', enabled });
        }
    }

    private async ensureResidentTilesIndexed(): Promise<void> {
        if (this.destroyed) return;
        const missingKeys: string[] = [];
        for (const key of this.activeTiles.keys()) {
            if (!this.indexedTileKeys.has(key)) {
                missingKeys.push(key);
            }
        }
        if (missingKeys.length === 0) return;

        await Promise.all(
            missingKeys.map(async (key) => {
                if (this.destroyed || !this.activeTiles.has(key)) return;
                const [zStr, xStr, yStr] = key.split('/');
                const z = parseInt(zStr, 10);
                const x = parseInt(xStr, 10);
                const y = parseInt(yStr, 10);
                const rawSlices = await this.fetchOrGetRawPickTile(z, x, y, key);
                if (!rawSlices || this.destroyed || !this.activeTiles.has(key)) return;

                const transfers: ArrayBuffer[] = [];
                const copies = rawSlices.map((s) => {
                    if (!s || s.byteLength === 0) return null;
                    const c = s.slice(0);
                    transfers.push(c);
                    return c;
                });

                await new Promise<void>((resolve, reject) => {
                    if (this.destroyed) {
                        resolve();
                        return;
                    }
                    const id = `idx_${++this.nextReqId}`;
                    this.workerRpcResolvers.set(id, {
                        resolve: (res) => {
                            if (this.activeTiles.has(key)) {
                                this.indexedTileKeys.add(key);
                                if (typeof res.rowIndexBytes === 'number' && res.rowIndexBytes > 0) {
                                    this.tileRowIndexBytes.set(key, res.rowIndexBytes);
                                }
                                if (Array.isArray(res.slices) && res.slices.length > 0) {
                                    const t = this.activeTiles.get(key);
                                    if (t) t.rowMaskSlices = res.slices;
                                }
                            } else {
                                this.notifyWorkersDropTile(key);
                            }
                            resolve();
                        },
                        reject,
                    });
                    this.dispatchToWorker(
                        { action: 'index_tile_rowids', id, key, buffers: copies },
                        transfers
                    );
                });
            })
        );
    }

    /**
     * Compiles `32 KiB` (`262,144`-bit) per-slot visibility masks across all workers off the main thread.
     * Returns a map of `tileKey -> ArrayBuffer[]` slices for currently resident tiles.
     */
    public async setRowIdFilter(
        ids: Iterable<number | bigint | string> | Uint32Array | BigUint64Array | null,
        probeRowIds?: BigUint64Array | null
    ): Promise<{ masks: Map<string, ArrayBuffer[]>; probeMatches: Uint8Array | null }> {
        const result = new Map<string, ArrayBuffer[]>();
        if (this.destroyed) return { masks: result, probeMatches: null };

        if (ids === null) {
            this.hasActiveRowIdFilter = false;
            for (const tile of this.activeTiles.values()) {
                delete tile.rowMaskSlices;
            }
            await Promise.all(
                this.allWorkers.map(
                    (worker) =>
                        new Promise<void>((resolve, reject) => {
                            const id = `rowfilt_${++this.nextReqId}`;
                            this.workerRpcResolvers.set(id, {
                                resolve: () => resolve(),
                                reject,
                            });
                            worker.postMessage({
                                action: 'set_rowid_filter',
                                id,
                                hasFilter: false,
                            });
                        })
                )
            );
            if (!this.retainRowIds) {
                this.indexedTileKeys.clear();
                this.tileRowIndexBytes.clear();
            }
            return { masks: result, probeMatches: null };
        }

        // Pack into raw Uint32Array or BigUint64Array WITHOUT main-thread .sort() —
        // when callers pass Uint32Array / BigUint64Array (Mosaic Arrow IPC / pydeepgraph),
        // zero main-thread O(N) element loops or sorts occur (`AGENTS.md` §3.3)!
        let rawU32: Uint32Array | null = null;
        let rawU64: BigUint64Array | null = null;

        if (ids instanceof Uint32Array) {
            rawU32 = ids;
        } else if (ids instanceof BigUint64Array) {
            rawU64 = ids;
        } else {
            let needs64 = false;
            const tempNum: number[] = [];
            const tempBig: bigint[] = [];
            for (const item of ids) {
                if (typeof item === 'bigint') {
                    tempBig.push(BigInt.asUintN(64, item));
                    if (item < 0n || item > 0xffffffffn) needs64 = true;
                } else if (typeof item === 'string') {
                    const trimmed = item.trim();
                    if (!trimmed) continue;
                    const b = BigInt.asUintN(64, BigInt(trimmed));
                    tempBig.push(b);
                    if (b > 0xffffffffn) needs64 = true;
                } else if (typeof item === 'number' && Number.isFinite(item)) {
                    if (item >= 0 && item <= 0xffffffff && Number.isInteger(item)) {
                        tempNum.push(item >>> 0);
                    } else {
                        tempBig.push(BigInt.asUintN(64, BigInt(Math.trunc(item))));
                        needs64 = true;
                    }
                }
            }

            if (!needs64 && tempBig.length === 0) {
                rawU32 = new Uint32Array(tempNum);
            } else if (!needs64) {
                rawU32 = new Uint32Array(tempNum.length + tempBig.length);
                for (let i = 0; i < tempNum.length; i++) rawU32[i] = tempNum[i];
                for (let i = 0; i < tempBig.length; i++) {
                    rawU32[tempNum.length + i] = Number(tempBig[i]);
                }
            } else {
                rawU64 = new BigUint64Array(tempNum.length + tempBig.length);
                for (let i = 0; i < tempNum.length; i++) rawU64[i] = BigInt(tempNum[i]);
                for (let i = 0; i < tempBig.length; i++) {
                    rawU64[tempNum.length + i] = tempBig[i];
                }
            }
        }

        this.hasActiveRowIdFilter = true;
        this.setRetainRowIds(true);
        await this.ensureResidentTilesIndexed();
        if (this.destroyed) return { masks: result, probeMatches: null };

        let probeMatches: Uint8Array | null = null;
        const workerResponses = await Promise.all(
            this.allWorkers.map(
                (worker, workerIdx) =>
                    new Promise<{ key: string; slices: ArrayBuffer[] }[]>((resolve, reject) => {
                        const id = `rowfilt_${++this.nextReqId}`;
                        const u32Copy = rawU32 ? (rawU32.slice().buffer as ArrayBuffer) : null;
                        const u64Copy = rawU64 ? (rawU64.slice().buffer as ArrayBuffer) : null;
                        const probeCopy =
                            workerIdx === 0 && probeRowIds && probeRowIds.length > 0
                                ? (probeRowIds.slice().buffer as ArrayBuffer)
                                : null;
                        const transfers: ArrayBuffer[] = [];
                        if (u32Copy) transfers.push(u32Copy);
                        if (u64Copy) transfers.push(u64Copy);
                        if (probeCopy) transfers.push(probeCopy);

                        this.workerRpcResolvers.set(id, {
                            resolve: (res) => {
                                if (workerIdx === 0 && res.probeMaskBuffer instanceof ArrayBuffer) {
                                    probeMatches = new Uint8Array(res.probeMaskBuffer);
                                }
                                resolve(res.masks ?? []);
                            },
                            reject,
                        });
                        worker.postMessage(
                            {
                                action: 'set_rowid_filter',
                                id,
                                hasFilter: true,
                                isSorted: false,
                                u32Buffer: u32Copy,
                                u64Buffer: u64Copy,
                                probeRowIdsBuffer: probeCopy,
                            },
                            transfers
                        );
                    })
            )
        );

        for (const masks of workerResponses) {
            for (const m of masks) {
                result.set(m.key, m.slices);
                const tile = this.activeTiles.get(m.key);
                if (tile && !tile.uploadedToGpu) {
                    tile.rowMaskSlices = m.slices;
                }
            }
        }
        return { masks: result, probeMatches };
    }

    /** Queries loaded visible tiles in workers for `row_id`s matching a spatial `rect` or `lasso`. */
    public async queryLoadedSpatialSelection(
        spec: WorkerSpatialQuerySpec
    ): Promise<{ rowIds: number[]; rowIdsBigInt: bigint[] }> {
        if (this.destroyed) return { rowIds: [], rowIdsBigInt: [] };
        this.setRetainRowIds(true);
        await this.ensureResidentTilesIndexed();
        if (this.destroyed) return { rowIds: [], rowIdsBigInt: [] };

        const visibleKeys = Array.from(this.currentlyVisibleIds);
        const buffers = await Promise.all(
            this.allWorkers.map(
                (worker) =>
                    new Promise<ArrayBuffer>((resolve, reject) => {
                        const id = `selq_${++this.nextReqId}`;
                        this.workerRpcResolvers.set(id, {
                            resolve: (res) => resolve(res.idsBuffer as ArrayBuffer),
                            reject,
                        });
                        worker.postMessage({
                            action: 'query_loaded_selection',
                            id,
                            visibleKeys,
                            spec,
                        });
                    })
            )
        );

        const seen = new Set<bigint>();
        const rowIds: number[] = [];
        const rowIdsBigInt: bigint[] = [];
        for (const buf of buffers) {
            const arr = new BigUint64Array(buf);
            for (let i = 0; i < arr.length; i++) {
                const v = arr[i];
                if (seen.has(v)) continue;
                seen.add(v);
                rowIdsBigInt.push(v);
                rowIds.push(Number(v));
            }
        }
        return { rowIds, rowIdsBigInt };
    }

    public setRootBounds(rootBounds: BoundingBox): void {
        this.rootBounds = rootBounds;
    }

    public setColOrigins(origins: Record<string, number>): void {
        this.colOrigins = { ...origins };
    }

    private recycleColumnMap(columns: Record<string, ArrayBuffer> | undefined): void {
        if (!columns) return;
        for (const colName of Object.keys(columns)) {
            const buf = columns[colName];
            if (!buf || buf.byteLength === 0) continue;
            if (!this.requestedColumns.includes(colName)) continue;
            let colPool = this.bufferPool.columns.get(colName);
            if (!colPool) {
                colPool = [];
                this.bufferPool.columns.set(colName, colPool);
            }
            if (colPool.length < MAX_COL_POOL_PER_COLUMN) {
                colPool.push(buf);
            }
        }
    }

    /**
     * Releases a tile's CPU `ArrayBuffer`s immediately after they have been uploaded to `GPUBuffer`s,
     * eliminating duplicate CPU TypedArray retention across cached tiles.
     */
    public releaseTileCpuBuffers(tile: TileData): void {
        if (tile.columns) {
            this.recycleColumnMap(tile.columns);
            tile.columns = {};
        }
        delete tile.rowMaskSlices;
        tile.uploadedToGpu = true;
    }

    /** Returns exact CPU `ArrayBuffer` retention & recycling pool metrics for Phase 6 lifecycle audits. */
    public getCpuMemoryStats(): {
        activeTileCount: number;
        uploadedTileCount: number;
        retainedTileColumnArrays: number;
        retainedTileCpuBytes: number;
        pooledColumnBuffers: number;
        pooledColumnBytes: number;
        pooledRawPayloadBuffers: number;
        pooledRawPayloadBytes: number;
        rawPickCacheEntries: number;
        rawPickCacheBytes: number;
        totalRetainedCpuBytes: number;
    } {
        let uploadedTileCount = 0;
        let retainedTileColumnArrays = 0;
        let retainedTileCpuBytes = 0;
        for (const tile of this.activeTiles.values()) {
            if (tile.uploadedToGpu) uploadedTileCount++;
            if (tile.columns) {
                for (const k of Object.keys(tile.columns)) {
                    const b = tile.columns[k];
                    if (b) {
                        retainedTileColumnArrays++;
                        retainedTileCpuBytes += b.byteLength;
                    }
                }
            }
        }
        let pooledColumnBuffers = 0;
        let pooledColumnBytes = 0;
        for (const arr of this.bufferPool.columns.values()) {
            for (const b of arr) {
                pooledColumnBuffers++;
                pooledColumnBytes += b.byteLength;
            }
        }
        let pooledRawPayloadBuffers = 0;
        let pooledRawPayloadBytes = 0;
        for (const b of this.bufferPool.rawPayload) {
            pooledRawPayloadBuffers++;
            pooledRawPayloadBytes += b.byteLength;
        }
        let workerRowIndexBytes = 0;
        for (const b of this.tileRowIndexBytes.values()) {
            workerRowIndexBytes += b;
        }
        return {
            activeTileCount: this.activeTiles.size,
            uploadedTileCount,
            retainedTileColumnArrays,
            retainedTileCpuBytes,
            pooledColumnBuffers,
            pooledColumnBytes,
            pooledRawPayloadBuffers,
            pooledRawPayloadBytes,
            rawPickCacheEntries: this.rawPickTileCache.size,
            rawPickCacheBytes: this.rawPickTileCacheBytes,
            totalRetainedCpuBytes:
                retainedTileCpuBytes +
                pooledColumnBytes +
                pooledRawPayloadBytes +
                this.rawPickTileCacheBytes +
                workerRowIndexBytes,
        };
    }

    private recycleAbortedPayload(payload: any) {
        if (payload?.buffers) {
            for (const rawBuf of payload.buffers) {
                if (rawBuf && this.bufferPool.rawPayload.length < MAX_RAW_PAYLOAD_POOL) {
                    this.bufferPool.rawPayload.push(rawBuf);
                }
            }
        }
        if (payload?.pooledColumns) {
            this.recycleColumnMap(payload.pooledColumns);
        }
    }

    private dispatchToWorker(payload: any, transfer: any[], abortSignal?: AbortSignal) {
        if (this.destroyed || abortSignal?.aborted) {
            this.recycleAbortedPayload(payload);
            return;
        }
        const worker = this.idleWorkers.shift();
        if (worker) {
            if (payload.reqId) {
                const req = this.resolvers.get(payload.reqId);
                if (req) req.tWorkerStart = performance.now();
            }
            worker.postMessage(payload, transfer);
        } else if (payload.action === 'decompress' || payload.action === 'get_schema') {
            this.pendingTasks.unshift({ payload, transfer, abortSignal });
        } else {
            this.pendingTasks.push({ payload, transfer, abortSignal });
        }
    }

    private checkPendingTasks(worker: Worker) {
        if (this.destroyed) return;
        while (this.pendingTasks.length > 0) {
            const task = this.pendingTasks.shift()!;
            if (task.abortSignal?.aborted) {
                this.recycleAbortedPayload(task.payload);
                continue;
            }
            if (task.payload.reqId) {
                const req = this.resolvers.get(task.payload.reqId);
                if (req) req.tWorkerStart = performance.now();
            }
            worker.postMessage(task.payload, task.transfer);
            return;
        }
        this.idleWorkers.push(worker);
    }

    public setRequestedColumns(cols: string[]) {
        const normalized = cols.includes('xy_u16') ? [...cols] : ['xy_u16', ...cols];
        const alreadyLoaded =
            (this.activeTiles.size > 0 || this.loadingTiles.size > 0) &&
            normalized.every((c) => this.requestedColumns.includes(c));

        if (alreadyLoaded) {
            return;
        }
        if (
            this.activeTiles.size === 0 &&
            this.loadingTiles.size === 0 &&
            normalized.length === this.requestedColumns.length &&
            normalized.every((c, i) => c === this.requestedColumns[i])
        ) {
            return;
        }

        // Prune buffer pools for columns that are no longer active when resetting tiles
        const nextSet = new Set(normalized);
        for (const pooledCol of Array.from(this.bufferPool.columns.keys())) {
            if (!nextSet.has(pooledCol)) {
                this.bufferPool.columns.delete(pooledCol);
            }
        }

        // F-09: Notify GPU store to unload every active tile BEFORE recycling/clearing buffers!
        this.requestedColumns = normalized;
        if (this.onTileUnloaded) {
            for (const key of this.activeTiles.keys()) {
                this.onTileUnloaded(key);
            }
        }

        for (const [key, tile] of this.activeTiles.entries()) {
            this.notifyWorkersDropTile(key);
            this.recycleColumnMap(tile.columns);
            tile.columns = {};
        }

        this.activeTiles.clear();
        this.lastSeenTimestamp.clear();
        this.failedTileBackoff.clear();
        for (const timer of this.retryTimers) clearTimeout(timer);
        this.retryTimers.clear();
        this.cacheChanged = true;
        this.queue.clear();
        this.loadingTiles.clear();
        for (const abort of this.abortControllers.values()) {
            abort.abort();
        }
        this.abortControllers.clear();
        this.clearRawPickTileCache(true, false);
    }

    public async getSchema(): Promise<SchemaField[]> {
        let bytes = this.schemaBuffer;
        if (!bytes) {
            const metadata: any = await this.corePmtiles.getMetadata();
            if (!metadata || !metadata.arrow_schema) {
                throw new Error('No arrow_schema found in PMTiles metadata');
            }
            const binaryString = atob(metadata.arrow_schema);
            bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            this.schemaBuffer = bytes;
        }

        const schemaCopy = bytes.slice();
        return new Promise<SchemaField[]>((resolve, reject) => {
            if (this.destroyed) {
                reject(new Error('PMTilesClient destroyed'));
                return;
            }
            const id = `schema_${++this.nextReqId}`;
            this.schemaResolvers.set(id, { resolve, reject });
            this.dispatchToWorker(
                { action: 'get_schema', id, schemaBuffer: schemaCopy.buffer },
                [schemaCopy.buffer]
            );
        });
    }

    private async fetchOrGetRawPickTile(
        z: number,
        x: number,
        y: number,
        tileKey: string,
        signal?: AbortSignal
    ): Promise<(ArrayBuffer | null)[] | null> {
        if (this.destroyed) return null;
        if (signal?.aborted) {
            throw new DOMException('Hover decode aborted', 'AbortError');
        }

        const cached = this.getRawPickTile(tileKey);
        if (cached) return cached;

        let inflight = this.inFlightPickTileFetches.get(tileKey);
        if (!inflight) {
            const controller = new AbortController();
            const waiters = new Set<object>();
            const promise = (async (): Promise<(ArrayBuffer | null)[] | null> => {
                try {
                    const promises = [this.corePmtiles.getZxy(z, x, y, controller.signal)];
                    for (const layer of this.layerPmtiles) {
                        promises.push(layer.getZxy(z, x, y, controller.signal));
                    }
                    const tiles = await Promise.all(promises);
                    if (this.destroyed || controller.signal.aborted || !tiles[0]?.data) {
                        return null;
                    }
                    const rawItems = tiles.map((t) => (t?.data ? (t.data as any) : null));
                    const stored = this.cacheRawPickTile(tileKey, rawItems);
                    if (stored) return stored;
                    return rawItems.map((item) => {
                        if (!item || item.byteLength === 0) return null;
                        if (item instanceof ArrayBuffer) return item.slice(0);
                        const u8 =
                            item instanceof Uint8Array
                                ? item
                                : new Uint8Array(item.buffer, item.byteOffset, item.byteLength);
                        return u8.slice().buffer as ArrayBuffer;
                    });
                } finally {
                    if (this.inFlightPickTileFetches.get(tileKey)?.controller === controller) {
                        this.inFlightPickTileFetches.delete(tileKey);
                    }
                }
            })();
            inflight = { promise, controller, waiters };
            this.inFlightPickTileFetches.set(tileKey, inflight);
        }

        const waiterToken = {};
        inflight.waiters.add(waiterToken);
        const capturedInflight = inflight;
        const onAbort = () => {
            capturedInflight.waiters.delete(waiterToken);
            // Defer underlying HTTP abort by one microtask so an immediate intra-tile hover
            // replacement (`prevController.abort()` -> `decodeTileRow` on the same `z/x/y`)
            // seamlessly reattaches to the in-flight download instead of abort-restarting it,
            // while cross-tile moves and `pointerleave` still abort stale tile downloads in <1 us!
            queueMicrotask(() => {
                if (capturedInflight.waiters.size === 0) {
                    capturedInflight.controller.abort();
                    if (this.inFlightPickTileFetches.get(tileKey) === capturedInflight) {
                        this.inFlightPickTileFetches.delete(tileKey);
                    }
                }
            });
        };
        signal?.addEventListener('abort', onAbort, { once: true });

        try {
            const slices = await inflight.promise;
            if (signal?.aborted) {
                throw new DOMException('Hover decode aborted', 'AbortError');
            }
            return slices;
        } finally {
            signal?.removeEventListener('abort', onAbort);
            inflight.waiters.delete(waiterToken);
        }
    }

    private dispatchDecodeRowWithBuffers(
        workerCacheKey: string,
        rawSlices: (ArrayBuffer | null)[],
        rowIndex: number,
        categories: Record<string, string[]>,
        signal?: AbortSignal
    ): Promise<DecodedTileRow | null> {
        if (this.destroyed) return Promise.resolve(null);
        if (signal?.aborted) {
            return Promise.reject(new DOMException('Hover decode aborted', 'AbortError'));
        }

        const transfers: ArrayBuffer[] = [];
        const copies = rawSlices.map((s) => {
            if (!s || s.byteLength === 0) return null;
            const c = s.slice(0);
            transfers.push(c);
            return c;
        });

        return new Promise((resolve, reject) => {
            if (this.destroyed) {
                resolve(null);
                return;
            }
            const id = `pickrow_${++this.nextReqId}`;
            const onAbort = () => {
                if (this.rowDecodeResolvers.delete(id)) {
                    reject(new DOMException('Hover decode aborted', 'AbortError'));
                }
            };
            signal?.addEventListener('abort', onAbort, { once: true });

            this.rowDecodeResolvers.set(id, {
                resolve: (val: DecodedTileRow | null) => {
                    signal?.removeEventListener('abort', onAbort);
                    resolve(val);
                },
                reject: (err: any) => {
                    signal?.removeEventListener('abort', onAbort);
                    reject(err);
                },
            });
            this.dispatchToWorker(
                {
                    action: 'decode_row',
                    id,
                    key: workerCacheKey,
                    buffers: copies,
                    rowIndex,
                    categories,
                },
                transfers,
                signal
            );
        });
    }

    /**
     * Decodes `row_id` and all retained columns for a single picked row `(z, x, y, rowIndex)`
     * directly from `.arrowtiles` on demand (`D8(2)`), backed by:
     * 1. A 1-tile parsed Arrow `Table` cache per worker (`O(1)` consecutive hovers in the same tile), and
     * 2. A byte-capped (`<= 8` tiles / `<= 24 MiB`) compressed-tile LRU seeded by `loadTile`.
     */
    public async decodeTileRow(
        z: number,
        x: number,
        y: number,
        rowIndex: number,
        categories: Record<string, string[]> = {},
        signal?: AbortSignal
    ): Promise<DecodedTileRow | null> {
        if (this.destroyed || rowIndex < 0) return null;
        if (signal?.aborted) {
            throw new DOMException('Hover decode aborted', 'AbortError');
        }

        const tileKey = `${z}/${x}/${y}`;
        const workerCacheKey = `${this.clientSeq}:${tileKey}`;

        // Fast Path 1: An idle worker already holds the parsed Arrow Table for this tile (`0` copies, `0` Zstd)
        const warmIdx = this.idleWorkers.findIndex(
            (w) => this.workerCachedRowKey.get(w) === workerCacheKey
        );
        if (warmIdx >= 0) {
            this.getRawPickTile(tileKey); // Refresh compressed LRU recency
            const warmWorker = this.idleWorkers.splice(warmIdx, 1)[0];
            const warmResult = await new Promise<{ hit: DecodedTileRow | null; missed: boolean }>(
                (resolve, reject) => {
                    if (this.destroyed) {
                        resolve({ hit: null, missed: false });
                        return;
                    }
                    const id = `pickrow_${++this.nextReqId}`;
                    const onAbort = () => {
                        if (this.rowDecodeResolvers.delete(id)) {
                            reject(new DOMException('Hover decode aborted', 'AbortError'));
                        }
                    };
                    signal?.addEventListener('abort', onAbort, { once: true });

                    this.rowDecodeResolvers.set(id, {
                        resolve: (val: DecodedTileRow | null) => {
                            signal?.removeEventListener('abort', onAbort);
                            resolve({ hit: val, missed: false });
                        },
                        reject: (err: any) => {
                            signal?.removeEventListener('abort', onAbort);
                            reject(err);
                        },
                        onCacheMiss: () => {
                            signal?.removeEventListener('abort', onAbort);
                            resolve({ hit: null, missed: true });
                        },
                    });
                    warmWorker.postMessage({
                        action: 'decode_row',
                        id,
                        key: workerCacheKey,
                        buffers: null,
                        rowIndex,
                        categories,
                    });
                }
            );
            if (!warmResult.missed) {
                return warmResult.hit;
            }
        }

        // Fast Path 2 (compressed LRU hit) or Miss Path 3 (coalesced Range fetch with AbortSignal)
        const rawSlices = await this.fetchOrGetRawPickTile(z, x, y, tileKey, signal);
        if (!rawSlices || this.destroyed) return null;
        return this.dispatchDecodeRowWithBuffers(
            workerCacheKey,
            rawSlices,
            rowIndex,
            categories,
            signal
        );
    }

    public destroy() {
        if (this.destroyed) return;
        this.destroyed = true;
        this.fastFetchSource?.abortPrefetch();
        this.clearRawPickTileCache(false);
        for (const timer of this.retryTimers) {
            clearTimeout(timer);
        }
        this.retryTimers.clear();
        this.queue.clear();
        for (const abort of this.abortControllers.values()) {
            abort.abort();
        }
        this.abortControllers.clear();
        for (const [, dec] of this.decompressResolvers) {
            dec.reject(new Error('PMTilesClient destroyed'));
        }
        this.decompressResolvers.clear();
        for (const [, sch] of this.schemaResolvers) {
            sch.reject(new Error('PMTilesClient destroyed'));
        }
        this.schemaResolvers.clear();
        for (const [, rowReq] of this.rowDecodeResolvers) {
            rowReq.reject(new Error('PMTilesClient destroyed'));
        }
        this.rowDecodeResolvers.clear();
        for (const [, rpcReq] of this.workerRpcResolvers) {
            rpcReq.reject(new Error('PMTilesClient destroyed'));
        }
        this.workerRpcResolvers.clear();
        this.indexedTileKeys.clear();
        this.tileRowIndexBytes.clear();
        for (const worker of this.allWorkers) {
            worker.terminate();
        }
        this.allWorkers = [];
        this.idleWorkers = [];
        this.pendingTasks = [];
        this.activeTiles.clear();
        this.loadingTiles.clear();
        this.bufferPool.columns.clear();
        this.bufferPool.rawPayload = [];
        this.failedTileBackoff.clear();
        this.permanentlyFailedTiles.clear();
    }

    private handleWorkerMessage(e: MessageEvent, worker: Worker) {
        if (this.destroyed) return;
        const data = e.data;
        if (
            data.action === 'set_rowid_filter' ||
            data.action === 'query_loaded_selection' ||
            data.action === 'index_tile_rowids'
        ) {
            const resolver = this.workerRpcResolvers.get(data.id);
            if (resolver) {
                this.workerRpcResolvers.delete(data.id);
                if (data.error) resolver.reject(new Error(data.error));
                else resolver.resolve(data);
            }
            if (data.action === 'index_tile_rowids') {
                this.checkPendingTasks(worker);
            }
            return;
        }

        if (data.action === 'decompress') {
            const resolver = this.decompressResolvers.get(data.id);
            if (resolver) {
                this.decompressResolvers.delete(data.id);
                if (data.error) resolver.reject(new Error(data.error));
                else resolver.resolve(data.buffer);
            }
            this.checkPendingTasks(worker);
            return;
        }

        if (data.action === 'get_schema') {
            const resolver = this.schemaResolvers.get(data.id);
            if (resolver) {
                this.schemaResolvers.delete(data.id);
                if (data.error) resolver.reject(new Error(data.error));
                else resolver.resolve(data.fields || []);
            }
            this.checkPendingTasks(worker);
            return;
        }

        if (data.action === 'decode_row') {
            if (!data.error && !data.cacheMiss && typeof data.cachedKey === 'string') {
                this.workerCachedRowKey.set(worker, data.cachedKey);
            } else if (data.cacheMiss) {
                this.workerCachedRowKey.delete(worker);
            }
            const resolver = this.rowDecodeResolvers.get(data.id);
            if (resolver) {
                this.rowDecodeResolvers.delete(data.id);
                if (data.cacheMiss && resolver.onCacheMiss) {
                    resolver.onCacheMiss();
                } else if (data.error) {
                    resolver.reject(new Error(data.error));
                } else if (!data.result) {
                    resolver.resolve(null);
                } else {
                    const r = data.result;
                    resolver.resolve({
                        rowId: r.rowId,
                        ...(r.rowIdBigIntStr ? { rowIdBigInt: BigInt(r.rowIdBigIntStr) } : {}),
                        xU16: r.xU16,
                        yU16: r.yU16,
                        attributes: r.attributes || {},
                    } as DecodedTileRow);
                }
            }
            this.checkPendingTasks(worker);
            return;
        }

        const resolverKey = data.reqId || data.key;
        if (data.error) {
            console.error('Worker error:', data.error);
        }
        if (!data.columns || !data.columns['xy_u16']) {
            const resolver = this.resolvers.get(resolverKey);
            if (resolver) {
                this.resolvers.delete(resolverKey);
                if (resolver.signal && resolver.onAbort) {
                    resolver.signal.removeEventListener('abort', resolver.onAbort);
                }
                resolver.reject(new Error(data.error || 'No xBuffer/yBuffer'));
            }
            this.checkPendingTasks(worker);
            return;
        }

        const numRows = data.numRows;
        const req = this.resolvers.get(resolverKey);
        if (req) {
            this.resolvers.delete(resolverKey);
            if (req.signal && req.onAbort) {
                req.signal.removeEventListener('abort', req.onAbort);
            }
            if (data.error) {
                req.reject(new Error(data.error));
            } else {
                if (data.dictionaries && this.onDictionariesLoaded) {
                    this.onDictionariesLoaded(data.dictionaries);
                }
                if (typeof data.rowIndexBytes === 'number' && data.rowIndexBytes > 0) {
                    this.indexedTileKeys.add(data.key);
                    this.tileRowIndexBytes.set(data.key, data.rowIndexBytes);
                }
                const tileData: TileData = {
                    key: data.key,
                    columns: data.columns,
                    numRows,
                    needsUpdate: true,
                    uploadedToGpu: false,
                    ...(Array.isArray(data.rowMaskSlices) && data.rowMaskSlices.length > 0
                        ? { rowMaskSlices: data.rowMaskSlices }
                        : {}),
                };
                req.resolve(tileData);

                if (data.rawPayloads) {
                    for (const rawBuf of data.rawPayloads) {
                        if (rawBuf && this.bufferPool.rawPayload.length < MAX_RAW_PAYLOAD_POOL) {
                            this.bufferPool.rawPayload.push(rawBuf);
                        }
                    }
                }

                const tWorkerEnd = performance.now();
                const netMs = Math.max(0, req.tNetEnd - req.t0);
                // Prefer pure synchronous worker decode time (`data.decodeMs`) to exclude main-thread queue & cold WASM compile
                const workerMs =
                    typeof data.decodeMs === 'number' && Number.isFinite(data.decodeMs)
                        ? Math.max(0, data.decodeMs)
                        : Math.max(0, tWorkerEnd - (req.tWorkerStart || req.tNetEnd));

                const isSteadySample = Boolean(data.wasWarm) && req.wasDirectoryWarm;
                if (!isSteadySample) {
                    this.fetchTelemetry.coldBootMs = Math.max(
                        this.fetchTelemetry.coldBootMs,
                        netMs + workerMs
                    );
                } else {
                    this.fetchTelemetry.steadyTotals.push(netMs + workerMs);
                    if (this.fetchTelemetry.steadyTotals.length > 50) {
                        this.fetchTelemetry.steadyTotals.shift();
                    }
                }

                this.fetchTelemetry.net.push(netMs);
                this.fetchTelemetry.worker.push(workerMs);
                if (this.fetchTelemetry.net.length > 50) {
                    this.fetchTelemetry.net.shift();
                    this.fetchTelemetry.worker.shift();
                }

                // Compute SLO P95 over steady-state samples once available (so cold WASM/directory boot never pins SLO red)
                const sloSource =
                    this.fetchTelemetry.steadyTotals.length > 0
                        ? this.fetchTelemetry.steadyTotals
                        : this.fetchTelemetry.net.map(
                              (n, i) => n + (this.fetchTelemetry.worker[i] ?? 0)
                          );
                const sortedTotals = [...sloSource].sort((a, b) => a - b);
                const p95Idx = Math.min(
                    sortedTotals.length - 1,
                    Math.floor(sortedTotals.length * 0.95)
                );
                this.fetchTelemetry.p95 = sortedTotals[p95Idx] ?? 0;
                this.fetchTelemetry.isSloViolated =
                    this.fetchTelemetry.steadyTotals.length > 0 &&
                    this.fetchTelemetry.p95 > this.fetchTelemetry.sloBudgetMs;
            }
        } else {
            // Request was aborted before worker responded — recycle within tight cap
            if (typeof data.key === 'string') {
                this.notifyWorkersDropTile(data.key);
            }
            this.recycleColumnMap(data.columns);
            if (data.rawPayloads) {
                for (const rawBuf of data.rawPayloads) {
                    if (rawBuf && this.bufferPool.rawPayload.length < MAX_RAW_PAYLOAD_POOL) {
                        this.bufferPool.rawPayload.push(rawBuf);
                    }
                }
            }
        }
        this.checkPendingTasks(worker);
    }

    private visibleTilesBuf: TileData[] = [];

    public getVisibleTiles(
        worldFrustum: BoundingBox,
        z: number,
        flipY = false,
        isSpatialX = true,
        isSpatialY = true
    ): TileData[] {
        if (this.destroyed) return [];
        this.currentlyVisibleIds.clear();

        const effectiveZ =
            !isSpatialX && !isSpatialY
                ? Math.min(z, 2, this.archiveMaxZoom)
                : Math.min(z, this.archiveMaxZoom);

        const candidateTiles: {
            cz: number;
            x: number;
            y: number;
            key: string;
            score: number;
        }[] = [];

        const centerX = (worldFrustum.minX + worldFrustum.maxX) / 2.0;
        const centerY = (worldFrustum.minY + worldFrustum.maxY) / 2.0;

        const traverse = (cz: number, x: number, y: number) => {
            if (cz > effectiveZ) return;
            const key = `${cz}/${x}/${y}`;

            if (this.knownEmptyTiles.has(key)) return;

            const numTiles = 1 << cz;
            const width = (this.rootBounds.maxX - this.rootBounds.minX) / numTiles;
            const height = (this.rootBounds.maxY - this.rootBounds.minY) / numTiles;

            const minX = this.rootBounds.minX + x * width;
            const maxX = this.rootBounds.minX + (x + 1) * width;
            const minY = flipY
                ? this.rootBounds.minY + y * height
                : this.rootBounds.maxY - (y + 1) * height;
            const maxY = flipY
                ? this.rootBounds.minY + (y + 1) * height
                : this.rootBounds.maxY - y * height;

            const cullX = isSpatialX && (maxX < worldFrustum.minX || minX > worldFrustum.maxX);
            const cullY = isSpatialY && (maxY < worldFrustum.minY || minY > worldFrustum.maxY);
            if (cullX || cullY) {
                return;
            }

            const tileCenterX = (minX + maxX) / 2.0;
            const tileCenterY = (minY + maxY) / 2.0;

            const dX = (tileCenterX - centerX) / (this.rootBounds.maxX - this.rootBounds.minX);
            const dY = (tileCenterY - centerY) / (this.rootBounds.maxY - this.rootBounds.minY);
            const distSq = dX * dX + dY * dY;

            const score = distSq + cz * 0.25;
            candidateTiles.push({ cz, x, y, key, score });

            if (cz < effectiveZ) {
                traverse(cz + 1, x * 2, y * 2);
                traverse(cz + 1, x * 2 + 1, y * 2);
                traverse(cz + 1, x * 2, y * 2 + 1);
                traverse(cz + 1, x * 2 + 1, y * 2 + 1);
            }
        };

        traverse(0, 0, 0);
        candidateTiles.sort((a, b) => a.score - b.score);

        const budget = Math.max(1, this.maxCacheSize - 10);
        const selectedTiles = candidateTiles.slice(0, budget);
        this.currentMaxZ = effectiveZ;

        const now = performance.now();
        for (let i = 0; i < selectedTiles.length; i++) {
            const { cz, x, y, key } = selectedTiles[i];
            if (this.permanentlyFailedTiles.has(key)) continue;

            this.currentlyVisibleIds.add(key);
            this.lastSeenTimestamp.set(key, now);

            if (!this.activeTiles.has(key) && !this.loadingTiles.has(key)) {
                const backoff = this.failedTileBackoff.get(key);
                if (backoff && now < backoff.nextRetryAt) {
                    continue;
                }
                const abortController = new AbortController();
                this.loadingTiles.add(key);
                this.abortControllers.set(key, abortController);
                const queuePriority = budget - i;
                this.loadTile(cz, x, y, key, abortController, queuePriority);
            }
        }

        for (const key of Array.from(this.loadingTiles)) {
            if (!this.currentlyVisibleIds.has(key)) {
                this.abortControllers.get(key)?.abort();
                this.abortControllers.delete(key);
                this.loadingTiles.delete(key);
            }
        }

        // Release any tile that finished worker decoding after the camera moved off-screen
        // before it was ever uploaded to a GPU slot (`0 B` duplicate/orphaned CPU column retention).
        for (const [key, tile] of Array.from(this.activeTiles.entries())) {
            if (!this.currentlyVisibleIds.has(key) && !tile.uploadedToGpu) {
                this.notifyWorkersDropTile(key);
                this.recycleColumnMap(tile.columns);
                tile.columns = {};
                this.activeTiles.delete(key);
                this.lastSeenTimestamp.delete(key);
            }
        }

        // Evict LRU off-screen tiles if we exceed maxCacheSize (pin Z=0..2 overview foundation)
        let totalTiles = this.activeTiles.size + this.loadingTiles.size;
        while (totalTiles > this.maxCacheSize) {
            let oldestKey: string | null = null;
            let oldestTime = Infinity;

            for (const key of this.activeTiles.keys()) {
                if (
                    !this.currentlyVisibleIds.has(key) &&
                    !key.startsWith('0/') &&
                    !key.startsWith('1/') &&
                    !key.startsWith('2/')
                ) {
                    const time = this.lastSeenTimestamp.get(key) || 0;
                    if (time < oldestTime) {
                        oldestTime = time;
                        oldestKey = key;
                    }
                }
            }

            if (oldestKey) {
                const evictedTile = this.activeTiles.get(oldestKey);
                if (evictedTile) {
                    this.recycleColumnMap(evictedTile.columns);
                    evictedTile.columns = {};
                }
                this.notifyWorkersDropTile(oldestKey);
                this.activeTiles.delete(oldestKey);
                this.lastSeenTimestamp.delete(oldestKey);
                if (this.onTileUnloaded) this.onTileUnloaded(oldestKey);
                totalTiles--;
            } else {
                break;
            }
        }

        this.visibleTilesBuf.length = 0;
        for (const key of this.currentlyVisibleIds) {
            const tile = this.activeTiles.get(key);
            if (tile) this.visibleTilesBuf.push(tile);
        }
        return this.visibleTilesBuf;
    }

    private parseArrowInWorker(
        key: string,
        dataArray: (Uint8Array | ArrayBuffer | null)[],
        t0: number,
        tNetEnd: number,
        wasDirectoryWarm: boolean,
        signal: AbortSignal
    ): Promise<TileData> {
        return new Promise((resolve, reject) => {
            const reqId = `req_${++this.nextReqId}`;
            let timeoutId: ReturnType<typeof setTimeout>;

            const cleanup = () => {
                clearTimeout(timeoutId);
                this.resolvers.delete(reqId);
            };

            const onAbort = () => {
                cleanup();
                reject(new DOMException('Aborted', 'AbortError'));
            };

            if (signal.aborted) {
                onAbort();
                return;
            }
            signal.addEventListener('abort', onAbort);

            timeoutId = setTimeout(() => {
                signal.removeEventListener('abort', onAbort);
                cleanup();
                reject(new Error(`Worker decode timeout for tile ${key}`));
            }, 15000);

            const wrappedResolve = (val: any) => {
                clearTimeout(timeoutId);
                resolve(val);
            };
            const wrappedReject = (err: any) => {
                clearTimeout(timeoutId);
                reject(err);
            };

            this.resolvers.set(reqId, {
                resolve: wrappedResolve,
                reject: wrappedReject,
                t0,
                tNetEnd,
                tWorkerStart: tNetEnd,
                wasDirectoryWarm,
                signal,
                onAbort,
            });

            const rawPayloadsToTransfer: ArrayBuffer[] = [];
            const copies = dataArray.map((d) => {
                if (!d) return null;
                if (d instanceof ArrayBuffer) {
                    rawPayloadsToTransfer.push(d);
                    return d;
                }
                if (
                    d instanceof Uint8Array &&
                    d.byteOffset === 0 &&
                    d.byteLength === d.buffer.byteLength
                ) {
                    rawPayloadsToTransfer.push(d.buffer as ArrayBuffer);
                    return d.buffer as ArrayBuffer;
                }
                const sourceData =
                    d instanceof Uint8Array
                        ? d
                        : new Uint8Array(
                              (d as any).buffer,
                              (d as any).byteOffset,
                              (d as any).byteLength
                          );
                const buf = sourceData.slice().buffer as ArrayBuffer;
                rawPayloadsToTransfer.push(buf);
                return buf;
            });

            const pooledColumns: Record<string, ArrayBuffer> = {};
            for (const col of this.requestedColumns) {
                const p = this.bufferPool.columns.get(col)?.pop();
                if (p) pooledColumns[col] = p;
            }

            const transferables = [...rawPayloadsToTransfer];
            for (const col of Object.keys(pooledColumns)) {
                transferables.push(pooledColumns[col]);
            }

            const payload = {
                action: 'decode',
                reqId,
                key,
                buffers: copies,
                requestedColumns: this.requestedColumns,
                pooledColumns,
                colOrigins: this.colOrigins,
                retainRowIds: this.retainRowIds || this.hasActiveRowIdFilter,
            };
            this.dispatchToWorker(payload, transferables, signal);
        });
    }

    private async loadTile(
        z: number,
        x: number,
        y: number,
        key: string,
        abortController: AbortController,
        priority: number
    ) {
        try {
            const netResult = await this.queue.add(
                async () => {
                    if (this.destroyed || abortController.signal.aborted) return null;
                    if (
                        !this.currentlyVisibleIds.has(key) &&
                        this.activeTiles.size + this.loadingTiles.size >= this.maxCacheSize
                    ) {
                        return null;
                    }
                    const wasDirectoryWarm = this.directoryWarm;
                    const t0 = performance.now();
                    const promises = [this.corePmtiles.getZxy(z, x, y, abortController.signal)];
                    for (const layer of this.layerPmtiles) {
                        promises.push(layer.getZxy(z, x, y, abortController.signal));
                    }
                    const tiles = await Promise.all(promises);
                    if (this.destroyed || abortController.signal.aborted) return null;
                    this.directoryWarm = true;
                    const tNetEnd = performance.now();
                    return { tiles, t0, tNetEnd, wasDirectoryWarm };
                },
                { priority }
            );

            if (!netResult || this.destroyed || abortController.signal.aborted) {
                return;
            }

            const { tiles, t0, tNetEnd, wasDirectoryWarm } = netResult;
            if (tiles[0]) {
                const payloads = tiles.map((t) => (t ? t.data : null));
                // Seed byte-capped compressed pick LRU before transferring payloads to worker
                this.cacheRawPickTile(key, payloads);
                const tileData = await this.parseArrowInWorker(
                    key,
                    payloads,
                    t0,
                    tNetEnd,
                    wasDirectoryWarm,
                    abortController.signal
                );
                if (this.destroyed || abortController.signal.aborted) {
                    this.recycleColumnMap(tileData.columns);
                    return;
                }
                this.failedTileBackoff.delete(key);
                this.activeTiles.set(key, tileData);
                this.cacheChanged = true;
                if (this.onTileLoaded) this.onTileLoaded(tileData);
            } else {
                this.failedTileBackoff.delete(key);
                this.knownEmptyTiles.add(key);
            }
        } catch (e: any) {
            if (this.destroyed || e?.name === 'AbortError' || abortController.signal.aborted) {
                return;
            }
            const prev = this.failedTileBackoff.get(key);
            const attempts = (prev?.attempts ?? 0) + 1;
            if (attempts >= MAX_TILE_FAILURE_ATTEMPTS) {
                this.failedTileBackoff.delete(key);
                this.permanentlyFailedTiles.add(key);
                console.warn(
                    `[ArrowTiles] Tile ${key} failed ${attempts} times; added to negative cache:`,
                    e
                );
            } else {
                const delayMs = Math.min(16_000, BASE_RETRY_DELAY_MS * Math.pow(2, attempts - 1));
                this.failedTileBackoff.set(key, {
                    attempts,
                    nextRetryAt: performance.now() + delayMs,
                });
                console.warn(
                    `[ArrowTiles] Transient failure loading tile ${key} (attempt ${attempts}/${MAX_TILE_FAILURE_ATTEMPTS}, retry in ${delayMs}ms):`,
                    e
                );
                const timer = setTimeout(() => {
                    this.retryTimers.delete(timer);
                    if (!this.destroyed) {
                        this.cacheChanged = true;
                    }
                }, delayMs);
                this.retryTimers.add(timer);
            }
        } finally {
            if (this.abortControllers.get(key) === abortController) {
                this.loadingTiles.delete(key);
                this.abortControllers.delete(key);
            }
        }
    }
}
