import {
    MosaicClient,
    clauseIntervals,
    clausePoint,
    coordinator as defaultCoordinator,
} from '@uwdata/mosaic-core';
import { Query, asNode, sql, verbatim } from '@uwdata/mosaic-sql';
import { simplifyRdpPolygon } from './core/FilterGpuResources.ts';
import type {
    DeepgraphFilter,
    DeepgraphPlot,
    LassoPolygonClause,
    PointSelectEvent,
    RectBrushClause,
    SelectionGeometryClause,
} from './lib/types.ts';

/** Structural representation of a UW Mosaic `SelectionClause` (zero external `.d.ts` dependencies). */
export interface MosaicSelectionClauseLike {
    meta?: { type?: string; [key: string]: unknown };
    source: unknown;
    clients?: Set<unknown>;
    fields?: ReadonlyArray<unknown>;
    value: unknown;
    predicate: unknown;
}

/** Structural interface compatible with UW Mosaic's `Selection` (`crossfilter`, `intersect`, `union`, `single`). */
export interface MosaicSelectionLike {
    readonly clauses: ReadonlyArray<MosaicSelectionClauseLike> & {
        active?: MosaicSelectionClauseLike;
    };
    readonly active?: MosaicSelectionClauseLike;
    readonly resolver?: {
        union?: boolean;
        cross?: boolean;
        single?: boolean;
        empty?: boolean;
    };
    update(clause: MosaicSelectionClauseLike): unknown;
    activate?(clause: MosaicSelectionClauseLike): unknown;
    reset?(clauses?: ReadonlyArray<MosaicSelectionClauseLike>): unknown;
    skip?(client: unknown, clause?: MosaicSelectionClauseLike): boolean;
    predicate(client?: unknown, noSkip?: boolean): unknown;
    addEventListener(type: string, callback: (value: unknown) => void): void;
    removeEventListener(type: string, callback: (value: unknown) => void): void;
}

/** Structural interface compatible with UW Mosaic's `Coordinator`. */
export interface MosaicCoordinatorLike {
    connect(client: MosaicClientLike): void;
    disconnect(client: MosaicClientLike): void;
    requestQuery(client: MosaicClientLike, query?: unknown): Promise<unknown>;
}

/** Structural base interface compatible with UW Mosaic's `MosaicClient`. */
export interface MosaicClientLike {
    coordinator: MosaicCoordinatorLike | null;
    enabled: boolean;
    readonly initialized: boolean;
    readonly pending: Promise<unknown>;
    readonly filterBy: MosaicSelectionLike | undefined;
    readonly filterStable: boolean;
    prepare(): Promise<void>;
    query(filter?: unknown): unknown;
    queryPending(): this;
    queryResult(data: unknown): this;
    queryError(error: Error): this;
    requestQuery(query?: unknown): Promise<unknown> | null;
    requestUpdate(): void;
    update(): this;
    initialize(): void;
}

export interface DeepgraphMosaicOptions {
    /** Live `DeepgraphPlot` instance to link with UW Mosaic. */
    plot: DeepgraphPlot;
    /** Backing DuckDB table or view name queried when Path B (`SELECT row_id`) runs. */
    table: string;
    /** Primary key / row identifier column in `table` matching `.arrowtiles` `row_id` (`'row_id'` default). */
    rowIdColumn?: string;
    /** Backing DuckDB column corresponding to spatial X when `plot` is in spatial mode (`'x'` default). */
    xColumn?: string;
    /** Backing DuckDB column corresponding to spatial Y when `plot` is in spatial mode (`'y'` default). */
    yColumn?: string;
    /** Outbound Mosaic `Selection` updated when brushing, lassoing, or clicking in `Deepgraph`. */
    as?: MosaicSelectionLike;
    /** Inbound Mosaic `Selection` cross-filtering `Deepgraph` via Path A (`0 ms` GPU) or Path B (`SELECT row_id`). */
    filterBy?: MosaicSelectionLike;
    /** Optional Mosaic `Coordinator` to connect immediately (defaults to `coordinator()` singleton when `autoConnect !== false`). */
    coordinator?: MosaicCoordinatorLike | null;
    /** Automatically connect to `coordinator` on construction (`true` default when `coordinator` is provided). */
    autoConnect?: boolean;
    /** Publish outbound brush/lasso clauses live during drag as well as on pointerup (`true` default). */
    publishLiveBrush?: boolean;
    /** Optional `unselectedOpacity` (`0..1`) applied to Path A GPU filters (`0.0` hard-cull default). */
    unselectedOpacity?: number;
}

export type MosaicExecutionPath = 'idle' | 'gpu' | 'sql';

export interface DeepgraphMosaicClientInstance extends MosaicClientLike {
    readonly filterSelection: MosaicSelectionLike | undefined;
    /** Last inbound filter execution path (`'idle'`, `'gpu'` for 0-SQL Path A, or `'sql'` for Path B escape hatch). */
    readonly lastExecutionPath: MosaicExecutionPath;
    /** Total number of Path B DuckDB `SELECT row_id` queries dispatched through the Mosaic `Coordinator`. */
    readonly sqlQueryCount: number;
    /** Promise resolving once the latest inbound GPU or async `row_id` bitmask update settles. */
    readonly filterSettled: Promise<void>;
    /** Clears any active selection on the linked `DeepgraphPlot`. */
    reset(): void;
    /** Disconnects from the Mosaic `Coordinator`, removes selection listeners, and detaches plot hooks. */
    destroy(): void;
}

export interface DeepgraphMosaicClientConstructor {
    new (options: DeepgraphMosaicOptions): DeepgraphMosaicClientInstance;
}

function quoteSqlIdent(ident: string): string {
    return `"${String(ident).replace(/"/g, '""')}"`;
}

/**
 * Generates a pure DuckDB SQL `WHERE` predicate for a 2D concave lasso polygon combining:
 * 1. Ramer-Douglas-Peucker simplification matching WebGPU `FilterGpuResources` (`<= 64` vertices),
 * 2. An axis-aligned bounding-box `BETWEEN` guard for Parquet/table zonemap pushdown, and
 * 3. An exact division-free inline even-odd ray-cast test (zero DuckDB C++ extensions or UDFs).
 */
const MAX_LASSO_COORD_ABS = 1e150; // Prevents DuckDB SIMD `DOUBLE` multiplication overflow (`< 1.79e308`)

export function buildLassoDuckDbSql(
    xColumn: string,
    yColumn: string,
    points: ReadonlyArray<readonly [number, number]>,
    maxVertices = 64
): string {
    const verts = simplifyRdpPolygon(points, maxVertices);
    if (verts.length < 3) {
        return 'FALSE';
    }

    const qx = quoteSqlIdent(xColumn);
    const qy = quoteSqlIdent(yColumn);

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const [vx, vy] of verts) {
        if (vx < minX) minX = vx;
        if (vx > maxX) maxX = vx;
        if (vy < minY) minY = vy;
        if (vy > maxY) maxY = vy;
    }
    if (
        !Number.isFinite(minX) ||
        !Number.isFinite(maxX) ||
        !Number.isFinite(minY) ||
        !Number.isFinite(maxY) ||
        Math.abs(minX) > MAX_LASSO_COORD_ABS ||
        Math.abs(maxX) > MAX_LASSO_COORD_ABS ||
        Math.abs(minY) > MAX_LASSO_COORD_ABS ||
        Math.abs(maxY) > MAX_LASSO_COORD_ABS
    ) {
        return 'FALSE';
    }

    const edgeTerms: string[] = [];
    const n = verts.length;
    for (let i = 0, j = n - 1; i < n; j = i++) {
        const [xi, yi] = verts[i];
        const [xj, yj] = verts[j];
        const dy = yj - yi;
        // Horizontal edges (`yi === yj`) can never satisfy `(yi > y) <> (yj > y)`; skipping them
        // precomputes `slope = (xj - xi) / (yj - yi)` in Float64 with zero runtime SQL divisions!
        if (!Number.isFinite(dy) || Math.abs(dy) < 1e-15) continue;
        const slope = (xj - xi) / dy;
        if (!Number.isFinite(slope) || Math.abs(slope) > MAX_LASSO_COORD_ABS) {
            return 'FALSE';
        }
        edgeTerms.push(
            `CASE WHEN ((${yi} > ${qy}) <> (${yj} > ${qy})) AND (${qx} < ((${qy} - ${yi}) * ${slope} + ${xi})) THEN 1 ELSE 0 END`
        );
    }

    if (edgeTerms.length === 0) {
        return 'FALSE';
    }

    const aabbSql = `(${qx} BETWEEN ${minX} AND ${maxX}) AND (${qy} BETWEEN ${minY} AND ${maxY})`;
    const paritySql = `((${edgeTerms.join(' + ')}) % 2) = 1`;
    return `(${aabbSql} AND ${paritySql})`;
}

function extractSimpleColumnName(fieldNode: unknown): string | null {
    if (typeof fieldNode === 'string') {
        const trimmed = fieldNode.trim();
        return trimmed.length > 0 ? trimmed : null;
    }
    if (!fieldNode || typeof fieldNode !== 'object') return null;
    const obj = fieldNode as Record<string, unknown>;
    if (obj.type === 'COLUMN_REF' && typeof obj.column === 'string' && !obj.table) {
        return obj.column;
    }
    if (typeof obj.column === 'string' && !('expr' in obj) && !('args' in obj)) {
        return obj.column;
    }
    return null;
}

function extractRowIdsFromQueryResult(
    data: unknown,
    rowIdKey = 'row_id'
): Uint32Array | BigUint64Array | (number | bigint)[] {
    if (!data) return new Uint32Array(0);
    if (data instanceof Uint32Array || data instanceof BigUint64Array) return data;
    if (Array.isArray(data)) {
        const out: (number | bigint)[] = [];
        for (const row of data) {
            if (typeof row === 'number' || typeof row === 'bigint') {
                out.push(row);
            } else if (row && typeof row === 'object' && rowIdKey in (row as any)) {
                const v = (row as any)[rowIdKey];
                if (typeof v === 'number' || typeof v === 'bigint') out.push(v);
                else if (typeof v === 'string' && v.trim()) out.push(BigInt(v.trim()));
            }
        }
        return out;
    }

    const tbl = data as any;
    if (typeof tbl.getChild === 'function') {
        const col = tbl.getChild(rowIdKey) ?? tbl.getChildAt?.(0);
        if (!col) return new Uint32Array(0);
        const chunks: any[] = Array.isArray(col.data) ? col.data : [];
        if (chunks.length > 0) {
            const allZeroNulls = chunks.every(
                (ch) => ch?.values && (ch.nullCount === undefined || ch.nullCount === 0)
            );
            if (allZeroNulls) {
                const totalLen = chunks.reduce(
                    (acc, ch) => acc + Math.min(ch.length ?? ch.values.length, ch.values.length),
                    0
                );
                if (chunks.length === 1) {
                    const vals = chunks[0].values;
                    const len = Math.min(tbl.numRows ?? totalLen, totalLen);
                    if (vals instanceof Uint32Array) return vals.subarray(0, len);
                    if (vals instanceof BigUint64Array) return vals.subarray(0, len);
                    if (vals instanceof BigInt64Array) {
                        return new BigUint64Array(vals.buffer, vals.byteOffset, len);
                    }
                }
                const firstVals = chunks[0].values;
                if (firstVals instanceof Uint32Array) {
                    const outU32 = new Uint32Array(totalLen);
                    let off = 0;
                    for (const ch of chunks) {
                        const len = Math.min(ch.length ?? ch.values.length, ch.values.length);
                        outU32.set((ch.values as Uint32Array).subarray(0, len), off);
                        off += len;
                    }
                    return outU32;
                }
                if (firstVals instanceof BigUint64Array || firstVals instanceof BigInt64Array) {
                    const outU64 = new BigUint64Array(totalLen);
                    let off = 0;
                    for (const ch of chunks) {
                        const raw = ch.values as BigUint64Array | BigInt64Array;
                        const len = Math.min(ch.length ?? raw.length, raw.length);
                        const view =
                            raw instanceof BigUint64Array
                                ? raw.subarray(0, len)
                                : new BigUint64Array(raw.buffer, raw.byteOffset, len);
                        outU64.set(view, off);
                        off += len;
                    }
                    return outU64;
                }
                if (firstVals instanceof Int32Array) {
                    let hasNeg = false;
                    for (const ch of chunks) {
                        const arr = ch.values as Int32Array;
                        const len = Math.min(ch.length ?? arr.length, arr.length);
                        for (let i = 0; i < len; i++) {
                            if (arr[i] < 0) {
                                hasNeg = true;
                                break;
                            }
                        }
                        if (hasNeg) break;
                    }
                    if (!hasNeg) {
                        const outU32 = new Uint32Array(totalLen);
                        let off = 0;
                        for (const ch of chunks) {
                            const arr = ch.values as Int32Array;
                            const len = Math.min(ch.length ?? arr.length, arr.length);
                            outU32.set(
                                new Uint32Array(arr.buffer, arr.byteOffset, len),
                                off
                            );
                            off += len;
                        }
                        return outU32;
                    }
                    const outU64 = new BigUint64Array(totalLen);
                    let off = 0;
                    for (const ch of chunks) {
                        const arr = ch.values as Int32Array;
                        const len = Math.min(ch.length ?? arr.length, arr.length);
                        for (let i = 0; i < len; i++) {
                            outU64[off++] = BigInt.asUintN(64, BigInt(arr[i]));
                        }
                    }
                    return outU64;
                }
            }
        }
        const n = Number(tbl.numRows ?? col.length ?? 0);
        const out: (number | bigint)[] = [];
        for (let i = 0; i < n; i++) {
            const v = col.at ? col.at(i) : col.get?.(i);
            if (typeof v === 'number' || typeof v === 'bigint') {
                out.push(v);
            }
        }
        return out;
    }

    return new Uint32Array(0);
}

const RuntimeMosaicClientBase = MosaicClient as unknown as {
    new (filterSelection?: unknown): MosaicClientLike;
};

class DeepgraphMosaicClientImpl
    extends RuntimeMosaicClientBase
    implements DeepgraphMosaicClientInstance
{
    private plot: DeepgraphPlot;
    private table: string;
    private rowIdColumn: string;
    private xColumn: string;
    private yColumn: string;
    private outSelection?: MosaicSelectionLike;
    public readonly filterSelection: MosaicSelectionLike | undefined;
    private publishLiveBrush: boolean;
    private unselectedOpacity: number;

    public lastExecutionPath: MosaicExecutionPath = 'idle';
    public sqlQueryCount = 0;

    private _latestEval: Promise<void> = Promise.resolve();
    private evalSeq = 0;
    private activeSqlQuerySeq = 0;
    private hasActiveSqlMask = false;
    private unsubBrush: (() => void) | null = null;
    private unsubSelect: (() => void) | null = null;
    private boundOnSelectionValue: (() => Promise<void>) | null = null;
    private destroyed = false;

    public get filterSettled(): Promise<void> {
        return (async () => {
            const sel = this.filterSelection as any;
            if (sel && typeof sel.pending === 'function') {
                for (let guard = 0; guard < 16; guard++) {
                    const inflight = sel._callbacks?.get?.('value')?.pending;
                    if (!inflight) break;
                    await sel.pending('value');
                }
            }
            await this._latestEval;
        })();
    }

    constructor(options: DeepgraphMosaicOptions) {
        // Pass `undefined` to `super` so `Coordinator.updateSelection` never issues `sql = "null"`
        // on Path A (`0`-SQL GPU updates); Path B explicitly dispatches via `this.requestQuery(q)`!
        super(undefined);

        this.plot = options.plot;
        this.table = options.table;
        this.rowIdColumn = options.rowIdColumn ?? 'row_id';
        this.xColumn = options.xColumn ?? 'x';
        this.yColumn = options.yColumn ?? 'y';
        this.outSelection = options.as;
        this.filterSelection = options.filterBy;
        this.publishLiveBrush = options.publishLiveBrush ?? true;
        this.unselectedOpacity = options.unselectedOpacity ?? 0.0;

        if (this.outSelection) {
            if (this.publishLiveBrush) {
                this.unsubBrush = this.plot.on('brush', (clause) => {
                    this.publishGeometryClause(clause);
                });
            }
            this.unsubSelect = this.plot.on('select', (ev) => {
                this.publishSelectEvent(ev);
            });
        }

        if (this.filterSelection) {
            this.boundOnSelectionValue = () => {
                const p = this.evaluateInboundSelection();
                this._latestEval = p;
                return p;
            };
            this.filterSelection.addEventListener('value', this.boundOnSelectionValue);
        }

        const shouldConnect =
            options.coordinator !== null &&
            (options.coordinator !== undefined || options.autoConnect === true);
        if (shouldConnect) {
            const mc = (options.coordinator ??
                (defaultCoordinator() as unknown as MosaicCoordinatorLike));
            mc.connect(this);
        } else if (this.filterSelection && this.filterSelection.clauses.length > 0) {
            this._latestEval = this.evaluateInboundSelection();
        }
    }

    public override async prepare(): Promise<void> {
        if (this.filterSelection && this.filterSelection.clauses.length > 0) {
            const p = this.evaluateInboundSelection();
            this._latestEval = p;
            await p;
        }
    }

    public override query(_filter?: unknown): unknown {
        return null;
    }

    public override queryResult(data: unknown): this {
        if (this.destroyed || this.activeSqlQuerySeq !== this.evalSeq) return this;
        const ids = extractRowIdsFromQueryResult(data, 'row_id');
        this.hasActiveSqlMask = true;
        this._latestEval = this.plot.setFilteredRowIds(ids);
        return this;
    }

    public reset(): void {
        if (this.destroyed) return;
        this.plot.clearSelection();
    }

    private resolveSqlAxes(clause?: SelectionGeometryClause): { xCol: string; yCol: string } {
        const vs = this.plot.getViewState();
        const xField = clause?.xField !== undefined ? clause.xField : vs.xField;
        const yField = clause?.yField !== undefined ? clause.yField : vs.yField;
        return {
            xCol: xField ?? this.xColumn,
            yCol: yField ?? this.yColumn,
        };
    }

    private publishGeometryClause(clause: SelectionGeometryClause): void {
        if (!this.outSelection || this.destroyed) return;
        const { xCol, yCol } = this.resolveSqlAxes(clause);
        const clients = new Set<unknown>([this]);

        if (clause.kind === 'rect' && clause.rect) {
            const r = clause.rect;
            const c = clauseIntervals(
                [xCol, yCol],
                [
                    [Math.min(r.minX, r.maxX), Math.max(r.minX, r.maxX)],
                    [Math.min(r.minY, r.maxY), Math.max(r.minY, r.maxY)],
                ],
                { source: this as any, clients: clients as any }
            );
            this.outSelection.update(c as unknown as MosaicSelectionClauseLike);
            return;
        }

        if (clause.kind === 'lasso' && clause.lasso && clause.lasso.points.length >= 3) {
            const sqlPred = buildLassoDuckDbSql(xCol, yCol, clause.lasso.points);
            this.outSelection.update({
                meta: { type: 'lasso' },
                source: this,
                clients,
                fields: [asNode(xCol), asNode(yCol)],
                value: clause.lasso.points,
                predicate: verbatim(sqlPred),
            });
            return;
        }

        if (clause.kind === 'none') {
            this.outSelection.update({
                meta: { type: 'none' },
                source: this,
                clients,
                value: null,
                predicate: null,
            });
        }
    }

    private publishSelectEvent(ev: PointSelectEvent): void {
        if (!this.outSelection || this.destroyed) return;
        const clients = new Set<unknown>([this]);

        if (ev.points.length > 0) {
            const hit = ev.points[0];
            const val = hit.rowIdBigInt !== undefined ? hit.rowIdBigInt : hit.rowId;
            const c = clausePoint(this.rowIdColumn, val, {
                source: this as any,
                clients: clients as any,
            });
            this.outSelection.update(c as unknown as MosaicSelectionClauseLike);
            return;
        }

        if (ev.clause) {
            this.publishGeometryClause(ev.clause);
        }
    }

    private isSelfClause(sel: MosaicSelectionLike, clause: MosaicSelectionClauseLike): boolean {
        if (clause.source === this || clause.clients?.has?.(this)) return true;
        if (typeof sel.skip === 'function' && sel.skip(this, clause)) return true;
        return false;
    }

    private mergeWithSelfBrush(inbound: DeepgraphFilter | null): DeepgraphFilter | null {
        const cur = this.plot.getFilter();
        const selfRect: RectBrushClause | null | undefined = cur?.rect;
        const selfLasso: LassoPolygonClause | null | undefined = cur?.lasso;

        if (!inbound) {
            if (selfRect || selfLasso) {
                return {
                    ...(selfRect ? { rect: selfRect } : {}),
                    ...(selfLasso ? { lasso: selfLasso } : {}),
                    ...(this.unselectedOpacity > 0
                        ? { unselectedOpacity: this.unselectedOpacity }
                        : {}),
                };
            }
            return null;
        }

        const merged: DeepgraphFilter = {
            ...inbound,
            ...(inbound.rect === undefined && selfRect ? { rect: selfRect } : {}),
            ...(inbound.lasso === undefined && selfLasso ? { lasso: selfLasso } : {}),
            ...(inbound.unselectedOpacity === undefined && this.unselectedOpacity > 0
                ? { unselectedOpacity: this.unselectedOpacity }
                : {}),
        };
        return merged;
    }

    private tryCompileGpuFilter(
        clauses: ReadonlyArray<MosaicSelectionClauseLike>
    ):
        | { kind: 'gpu'; filter: DeepgraphFilter }
        | { kind: 'direct_rowids'; ids: (number | bigint)[] }
        | null {
        const ranges: Record<string, [number, number]> = {};
        const categories: Record<string, (string | number)[]> = {};
        let rect: RectBrushClause | undefined;
        let lasso: LassoPolygonClause | undefined;
        let directRowIds: (number | bigint)[] | null = null;

        const { xCol, yCol } = this.resolveSqlAxes();
        const vs = this.plot.getViewState();
        const meta = this.plot.getMetadata();
        const schemaSet = new Set(meta?.schemaFields ?? []);

        const mapSqlColToArchiveField = (col: string): string => {
            if (vs.xField === null && col === this.xColumn && !schemaSet.has(col)) {
                return 'x_u16';
            }
            if (vs.yField === null && col === this.yColumn && !schemaSet.has(col)) {
                return 'y_u16';
            }
            return col;
        };

        for (const c of clauses) {
            const mType = c.meta?.type;
            const fields = c.fields ?? [];

            if (mType === 'interval' && fields.length === 1 && Array.isArray(c.value)) {
                const rawCol = extractSimpleColumnName(fields[0]);
                const pair =
                    c.value.length === 1 && Array.isArray(c.value[0]) ? c.value[0] : c.value;
                if (!rawCol || pair.length < 2) return null;
                const lo = Number(pair[0]);
                const hi = Number(pair[1]);
                if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
                const archiveCol = mapSqlColToArchiveField(rawCol);
                const prev = ranges[archiveCol];
                ranges[archiveCol] = prev
                    ? [Math.max(prev[0], Math.min(lo, hi)), Math.min(prev[1], Math.max(lo, hi))]
                    : [Math.min(lo, hi), Math.max(lo, hi)];
                continue;
            }

            if (mType === 'interval' && fields.length === 2 && Array.isArray(c.value)) {
                const col0 = extractSimpleColumnName(fields[0]);
                const col1 = extractSimpleColumnName(fields[1]);
                if (
                    !col0 ||
                    !col1 ||
                    c.value.length < 2 ||
                    !Array.isArray(c.value[0]) ||
                    !Array.isArray(c.value[1])
                ) {
                    return null;
                }
                const x0 = Number(c.value[0][0]);
                const x1 = Number(c.value[0][1]);
                const y0 = Number(c.value[1][0]);
                const y1 = Number(c.value[1][1]);
                if (
                    !Number.isFinite(x0) ||
                    !Number.isFinite(x1) ||
                    !Number.isFinite(y0) ||
                    !Number.isFinite(y1)
                ) {
                    return null;
                }
                if (col0 === xCol && col1 === yCol) {
                    rect = {
                        minX: Math.min(x0, x1),
                        maxX: Math.max(x0, x1),
                        minY: Math.min(y0, y1),
                        maxY: Math.max(y0, y1),
                    };
                } else {
                    ranges[mapSqlColToArchiveField(col0)] = [Math.min(x0, x1), Math.max(x0, x1)];
                    ranges[mapSqlColToArchiveField(col1)] = [Math.min(y0, y1), Math.max(y0, y1)];
                }
                continue;
            }

            if (mType === 'lasso' && fields.length === 2 && Array.isArray(c.value)) {
                const col0 = extractSimpleColumnName(fields[0]);
                const col1 = extractSimpleColumnName(fields[1]);
                if (col0 === xCol && col1 === yCol && c.value.length >= 3) {
                    lasso = { points: c.value as [number, number][] };
                    continue;
                }
                return null;
            }

            if (mType === 'point' && fields.length === 1) {
                const rawCol = extractSimpleColumnName(fields[0]);
                if (!rawCol || c.value === undefined || c.value === null) return null;

                // Multi-point single-column `clausePoints([col], [[v1], [v2]])` vs `clausePoint(col, v)`
                const rawVals: unknown[] = Array.isArray(c.value)
                    ? c.value.map((item) => (Array.isArray(item) ? item[0] : item))
                    : [c.value];

                if (rawCol === this.rowIdColumn && clauses.length === 1) {
                    const parsedIds: (number | bigint)[] = [];
                    for (const rv of rawVals) {
                        if (typeof rv === 'number' || typeof rv === 'bigint') parsedIds.push(rv);
                        else if (typeof rv === 'string' && rv.trim()) parsedIds.push(BigInt(rv.trim()));
                    }
                    directRowIds = parsedIds;
                    continue;
                }

                const dict = this.plot.getCategories(rawCol);
                if (dict) {
                    const allowed: (string | number)[] = [];
                    for (const rv of rawVals) {
                        if (typeof rv === 'string' || typeof rv === 'number') {
                            allowed.push(rv);
                        }
                    }
                    categories[rawCol] = categories[rawCol]
                        ? categories[rawCol].filter((v) => allowed.includes(v))
                        : allowed;
                    continue;
                }

                if (rawVals.length === 1 && typeof rawVals[0] === 'number' && Number.isFinite(rawVals[0])) {
                    const v = rawVals[0];
                    ranges[mapSqlColToArchiveField(rawCol)] = [v, v];
                    continue;
                }

                return null;
            }

            return null;
        }

        if (directRowIds !== null) {
            return { kind: 'direct_rowids', ids: directRowIds };
        }

        const candidate: DeepgraphFilter = {
            ...(rect ? { rect } : {}),
            ...(lasso ? { lasso } : {}),
            ...(Object.keys(ranges).length > 0 ? { ranges } : {}),
            ...(Object.keys(categories).length > 0 ? { categories } : {}),
            ...(this.unselectedOpacity > 0 ? { unselectedOpacity: this.unselectedOpacity } : {}),
        };

        if (!this.plot.canApplyFilterOnGpu(candidate)) {
            return null;
        }
        return { kind: 'gpu', filter: candidate };
    }

    private async evaluateInboundSelection(): Promise<void> {
        const sel = this.filterSelection;
        if (!sel || this.destroyed) return;

        // Cross-filter skip: if our own outbound interaction triggered this active update, do nothing
        if (sel.active && sel.resolver?.cross && this.isSelfClause(sel, sel.active)) {
            return;
        }

        const mySeq = ++this.evalSeq;
        const externalClauses = sel.clauses.filter(
            (c) => c.predicate !== null && c.predicate !== undefined && !this.isSelfClause(sel, c)
        );

        if (externalClauses.length === 0) {
            this.plot.setFilter(this.mergeWithSelfBrush(null));
            if (sel.resolver?.empty && sel.clauses.length === 0) {
                this.hasActiveSqlMask = true;
                this.lastExecutionPath = 'gpu';
                await this.plot.setFilteredRowIds([]);
                return;
            }
            if (this.hasActiveSqlMask) {
                this.hasActiveSqlMask = false;
                await this.plot.setFilteredRowIds(null);
            }
            this.lastExecutionPath = 'idle';
            return;
        }

        const isUnion = Boolean(sel.resolver?.union && externalClauses.length > 1);
        const compiled = !isUnion ? this.tryCompileGpuFilter(externalClauses) : null;

        if (compiled?.kind === 'gpu') {
            const merged = this.mergeWithSelfBrush(compiled.filter);
            if (this.plot.canApplyFilterOnGpu(merged)) {
                this.lastExecutionPath = 'gpu';
                this.plot.setFilter(merged);
                if (this.hasActiveSqlMask) {
                    this.hasActiveSqlMask = false;
                    await this.plot.setFilteredRowIds(null);
                }
                return;
            }
        }

        if (compiled?.kind === 'direct_rowids') {
            this.plot.setFilter(this.mergeWithSelfBrush(null));
            this.hasActiveSqlMask = true;
            this.lastExecutionPath = 'gpu';
            await this.plot.setFilteredRowIds(compiled.ids);
            return;
        }

        // Path B: DuckDB `SELECT row_id` SQL Escape Hatch via Mosaic Coordinator
        this.plot.setFilter(this.mergeWithSelfBrush(null));

        const filterPred = sel.predicate(this);
        if (!filterPred || (Array.isArray(filterPred) && filterPred.length === 0)) {
            if (this.hasActiveSqlMask) {
                this.hasActiveSqlMask = false;
                await this.plot.setFilteredRowIds(null);
            }
            this.lastExecutionPath = 'idle';
            return;
        }

        if (!this.coordinator) {
            return;
        }

        const ridColNode = asNode(this.rowIdColumn);
        const q = Query.from(this.table)
            .select({ row_id: ridColNode })
            .where(sql`${ridColNode} IS NOT NULL`, filterPred as any);

        this.activeSqlQuerySeq = mySeq;
        this.lastExecutionPath = 'sql';
        this.sqlQueryCount++;
        await this.requestQuery(q);
        await this._latestEval;
    }

    public destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        if (this.unsubBrush) {
            this.unsubBrush();
            this.unsubBrush = null;
        }
        if (this.unsubSelect) {
            this.unsubSelect();
            this.unsubSelect = null;
        }
        if (this.filterSelection && this.boundOnSelectionValue) {
            this.filterSelection.removeEventListener('value', this.boundOnSelectionValue);
            this.boundOnSelectionValue = null;
        }
        if (this.coordinator) {
            this.coordinator.disconnect(this);
        }
    }
}

export type DeepgraphMosaicClient = DeepgraphMosaicClientInstance;

/**
 * Hybrid UW Mosaic Adapter for `Deepgraph` (`deepgraph/mosaic`):
 * - **Path A (`0` SQL queries, `< 0.25 ms`):** Evaluates interval, category, and 2D brush/lasso
 *   clauses directly on GPU uniforms (`@binding(10)`).
 * - **Path B (DuckDB `row_id` Escape Hatch):** Routes cold-column, `ILIKE` text, `OR`-union, or
 *   `> 512`-category SQL predicates through the Mosaic `Coordinator` into `@binding(11)` `32 KiB`
 *   worker-compiled slot bitmasks.
 */
export const DeepgraphMosaicClient: DeepgraphMosaicClientConstructor =
    DeepgraphMosaicClientImpl as unknown as DeepgraphMosaicClientConstructor;
