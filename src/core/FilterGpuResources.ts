import type { BoundingBox, DeepgraphFilter } from '../lib/types.ts';
import type { Camera2D } from './Renderer.ts';
import type { InternalAxisConfig, TileGpuStore } from './TileGpuStore.ts';

export const GLOBAL_FILTER_UNIFORM_BYTES = 1024;
export const SLOT_MASK_SLICE_BYTES = 32768; // 2,048 x vec4<u32> = 8,192 u32 = 262,144 bits (16B & 256B aligned)
export const SLOT_MASK_ROWS_PER_SLICE = 262144;
export const MAX_LASSO_GPU_VERTICES = 64;
export const MAX_GPU_CATEGORY_CODES = 512;

export const FILTER_FLAG_HAS_GLOBAL = 1;
export const FILTER_FLAG_HAS_SLOT_MASK = 2;
export const FILTER_FLAG_HAS_RECT = 4;
export const FILTER_FLAG_HAS_LASSO = 8;
export const FILTER_FLAG_HAS_CAT0 = 16;
export const FILTER_FLAG_HAS_CAT1 = 32;

export interface AxisProjectionContext {
    xAxis: InternalAxisConfig;
    yAxis: InternalAxisConfig;
    rootBounds: BoundingBox;
    dataBounds?: { min_x: number; max_x: number; min_y: number; max_y: number };
    effectiveFlipY: boolean;
    colOrigins: Record<string, number>;
}

export interface GpuFilterPlan {
    supported: boolean;
    hasActivePredicates: boolean;
    extraFilterField1: string;
    extraFilterField2: string;
    reason?: string;
}

export interface GpuFilterCapabilityContext {
    isXSpatial: boolean;
    isYSpatial: boolean;
    xField: string;
    yField: string;
    colorField: string;
    sizeField: string;
    auxField1: string;
    auxField2: string;
    schemaFields: ReadonlyArray<string>;
    categories: Record<string, string[]>;
}

function sqSegDist(
    px: number,
    py: number,
    x1: number,
    y1: number,
    x2: number,
    y2: number
): number {
    let x = x1;
    let y = y1;
    const dx = x2 - x1;
    const dy = y2 - y1;
    if (dx !== 0 || dy !== 0) {
        const t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy);
        if (t > 1) {
            x = x2;
            y = y2;
        } else if (t > 0) {
            x += dx * t;
            y += dy * t;
        }
    }
    const ex = px - x;
    const ey = py - y;
    return ex * ex + ey * ey;
}

function rdpPass(
    pts: ReadonlyArray<readonly [number, number]>,
    sqTolerance: number
): [number, number][] {
    const n = pts.length;
    if (n <= 2) {
        return pts.map((p) => [p[0], p[1]]);
    }
    const keep = new Uint8Array(n);
    keep[0] = 1;
    keep[n - 1] = 1;
    const stack: [number, number][] = [[0, n - 1]];

    while (stack.length > 0) {
        const [first, last] = stack.pop()!;
        let maxSq = 0;
        let idx = -1;
        const [x1, y1] = pts[first];
        const [x2, y2] = pts[last];
        for (let i = first + 1; i < last; i++) {
            const dSq = sqSegDist(pts[i][0], pts[i][1], x1, y1, x2, y2);
            if (dSq > maxSq) {
                maxSq = dSq;
                idx = i;
            }
        }
        if (maxSq > sqTolerance && idx !== -1) {
            keep[idx] = 1;
            stack.push([first, idx], [idx, last]);
        }
    }

    const out: [number, number][] = [];
    for (let i = 0; i < n; i++) {
        if (keep[i]) {
            out.push([pts[i][0], pts[i][1]]);
        }
    }
    return out;
}

/**
 * Simplifies a closed 2D polygon via Ramer-Douglas-Peucker to `<= maxVertices` vertices
 * while preserving `>= 3` non-degenerate vertices.
 */
export function simplifyRdpPolygon(
    points: ReadonlyArray<readonly [number, number]>,
    maxVertices = MAX_LASSO_GPU_VERTICES
): [number, number][] {
    const nIn = points.length;
    const strideLen = nIn > 16384 ? 16384 : nIn;
    const cleaned: [number, number][] = [];
    for (let k = 0; k < strideLen; k++) {
        const idx = nIn > 16384 ? Math.floor((k * nIn) / 16384) : k;
        const x = Number(points[idx][0]);
        const y = Number(points[idx][1]);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        if (cleaned.length > 0) {
            const prev = cleaned[cleaned.length - 1];
            if (prev[0] === x && prev[1] === y) continue;
        }
        cleaned.push([x, y]);
    }
    if (
        cleaned.length > 3 &&
        cleaned[0][0] === cleaned[cleaned.length - 1][0] &&
        cleaned[0][1] === cleaned[cleaned.length - 1][1]
    ) {
        cleaned.pop();
    }
    if (cleaned.length <= maxVertices) {
        return cleaned;
    }
    const cappedInput: [number, number][] =
        cleaned.length > 4096
            ? Array.from(
                  { length: 4096 },
                  (_, k) => cleaned[Math.floor((k * cleaned.length) / 4096)]
              )
            : cleaned;

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const [x, y] of cappedInput) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
    }
    const spanX = maxX - minX;
    const spanY = maxY - minY;
    const rawDiagSq = spanX * spanX + spanY * spanY;

    if (Number.isFinite(rawDiagSq)) {
        const diagSq = Math.max(1e-24, rawDiagSq);
        // Close seam temporarily during RDP so start/end anchor preserves loop topology
        const closedInput: [number, number][] = [
            ...cappedInput,
            [cappedInput[0][0], cappedInput[0][1]],
        ];
        let lo = 0.0;
        let hi = diagSq * 0.25;
        let best: [number, number][] | null = null;

        for (let iter = 0; iter < 24; iter++) {
            const mid = iter === 0 ? diagSq * 1e-8 : 0.5 * (lo + hi);
            const cand = rdpPass(closedInput, mid);
            const openCand = cand.length > 1 ? cand.slice(0, cand.length - 1) : cand;
            if (openCand.length <= maxVertices) {
                if (openCand.length >= 3) {
                    best = openCand;
                }
                hi = mid;
            } else {
                lo = mid;
            }
        }

        if (best && best.length >= 3 && best.length <= maxVertices) {
            return best;
        }
    }

    // Deterministic uniform stride fallback if colinear/degenerate or extreme float span
    const fallback: [number, number][] = [];
    const target = Math.max(3, maxVertices);
    for (let k = 0; k < target; k++) {
        const idx = Math.min(
            cappedInput.length - 1,
            Math.floor((k * cappedInput.length) / target)
        );
        fallback.push(cappedInput[idx]);
    }
    return fallback;
}

/** Exact CPU even-odd horizontal ray-cast matching WGSL `eval_lasso_contains`. */
export function pointInPolygon2D(
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
        const intersects =
            yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi;
        if (intersects) {
            inside = !inside;
        }
    }
    return inside;
}

/** Projects a single active-axis data coordinate `(dataX, dataY)` into world space `(worldX, worldY)` in Float64. */
export function projectDataToWorld64(
    dataX: number,
    dataY: number,
    ctx: AxisProjectionContext
): { worldX: number; worldY: number } {
    const { xAxis, yAxis, rootBounds: rb, dataBounds: db, effectiveFlipY, colOrigins } = ctx;

    const mapSpatialNorm = (isX: boolean, dataVal: number, fieldName: string): number => {
        if (fieldName === 'y_u16') {
            if (db) {
                const span = Math.max(1e-12, db.max_y - db.min_y);
                return (db.max_y - dataVal) / span;
            }
            const span = Math.max(1e-12, rb.maxY - rb.minY);
            return effectiveFlipY ? (dataVal - rb.minY) / span : (rb.maxY - dataVal) / span;
        }
        if (db && isX) {
            const span = Math.max(1e-12, db.max_x - db.min_x);
            return (dataVal - db.min_x) / span;
        }
        const span = Math.max(1e-12, rb.maxX - rb.minX);
        return (dataVal - rb.minX) / span;
    };

    let worldX: number;
    if (xAxis.field === 'x_u16' || xAxis.field === 'y_u16') {
        const norm = mapSpatialNorm(true, dataX, xAxis.field);
        const domSpan = xAxis.domain[1] - xAxis.domain[0];
        const safeDom = Math.abs(domSpan) < 1e-12 ? 1.0 : domSpan;
        worldX =
            xAxis.range[0] +
            ((norm - xAxis.domain[0]) / safeDom) * (xAxis.range[1] - xAxis.range[0]);
    } else {
        const shifted = dataX - (colOrigins[xAxis.field] ?? 0);
        const domSpan = xAxis.domain[1] - xAxis.domain[0];
        const safeDom = Math.abs(domSpan) < 1e-12 ? 1.0 : domSpan;
        worldX =
            xAxis.range[0] +
            ((shifted - xAxis.domain[0]) / safeDom) * (xAxis.range[1] - xAxis.range[0]);
    }

    let worldY: number;
    if (yAxis.field === 'y_u16' || yAxis.field === 'x_u16') {
        const norm = mapSpatialNorm(false, dataY, yAxis.field);
        const domSpan = yAxis.domain[1] - yAxis.domain[0];
        const safeDom = Math.abs(domSpan) < 1e-12 ? 1.0 : domSpan;
        worldY =
            yAxis.range[0] +
            ((norm - yAxis.domain[0]) / safeDom) * (yAxis.range[1] - yAxis.range[0]);
    } else {
        const shifted = dataY - (colOrigins[yAxis.field] ?? 0);
        const domSpan = yAxis.domain[1] - yAxis.domain[0];
        const safeDom = Math.abs(domSpan) < 1e-12 ? 1.0 : domSpan;
        worldY =
            yAxis.range[0] +
            ((shifted - yAxis.domain[0]) / safeDom) * (yAxis.range[1] - yAxis.range[0]);
    }

    return { worldX, worldY };
}

/** Unprojects a CSS pixel `(cssX, cssY)` on the canvas into active-axis **data coordinates** `(dataX, dataY)` in Float64. */
export function unprojectCssToData64(
    cssX: number,
    cssY: number,
    cssWidth: number,
    cssHeight: number,
    camera: Camera2D,
    ctx: AxisProjectionContext
): { dataX: number; dataY: number } {
    const safeW = Math.max(1, cssWidth);
    const safeH = Math.max(1, cssHeight);
    const camLeft = camera.position.x + camera.left / camera.zoom;
    const camRight = camera.position.x + camera.right / camera.zoom;
    const camBottom = camera.position.y + camera.bottom / camera.zoom;
    const camTop = camera.position.y + camera.top / camera.zoom;

    const worldX = camLeft + (cssX / safeW) * (camRight - camLeft);
    const worldY = camTop - (cssY / safeH) * (camTop - camBottom);

    const { xAxis, yAxis, rootBounds: rb, dataBounds: db, effectiveFlipY, colOrigins } = ctx;

    const unmapSpatialNorm = (norm: number, fieldName: string): number => {
        if (fieldName === 'y_u16') {
            if (db) {
                return db.max_y - norm * (db.max_y - db.min_y);
            }
            return effectiveFlipY
                ? rb.minY + norm * (rb.maxY - rb.minY)
                : rb.maxY - norm * (rb.maxY - rb.minY);
        }
        if (db) {
            return db.min_x + norm * (db.max_x - db.min_x);
        }
        return rb.minX + norm * (rb.maxX - rb.minX);
    };

    let dataX: number;
    {
        const rangeSpan = xAxis.range[1] - xAxis.range[0];
        const t = Math.abs(rangeSpan) < 1e-12 ? 0.0 : (worldX - xAxis.range[0]) / rangeSpan;
        const shiftedOrNorm = xAxis.domain[0] + t * (xAxis.domain[1] - xAxis.domain[0]);
        if (xAxis.field === 'x_u16' || xAxis.field === 'y_u16') {
            dataX = unmapSpatialNorm(shiftedOrNorm, xAxis.field);
        } else {
            dataX = shiftedOrNorm + (colOrigins[xAxis.field] ?? 0);
        }
    }

    let dataY: number;
    {
        const rangeSpan = yAxis.range[1] - yAxis.range[0];
        const t = Math.abs(rangeSpan) < 1e-12 ? 0.0 : (worldY - yAxis.range[0]) / rangeSpan;
        const shiftedOrNorm = yAxis.domain[0] + t * (yAxis.domain[1] - yAxis.domain[0]);
        if (yAxis.field === 'y_u16' || yAxis.field === 'x_u16') {
            dataY = unmapSpatialNorm(shiftedOrNorm, yAxis.field);
        } else {
            dataY = shiftedOrNorm + (colOrigins[yAxis.field] ?? 0);
        }
    }

    return { dataX, dataY };
}

/**
 * Determines whether `filter` can be evaluated 100% on the GPU uniform fast path (`Path A`),
 * and assigns up to 2 unmapped columns into idle `@binding(4)` / `@binding(5)` slots when
 * `isXSpatial && !auxField1` / `isYSpatial && !auxField2`.
 */
export function planGpuFilterExecution(
    filter: DeepgraphFilter | null | undefined,
    ctx: GpuFilterCapabilityContext
): GpuFilterPlan {
    if (!filter) {
        return {
            supported: true,
            hasActivePredicates: false,
            extraFilterField1: '',
            extraFilterField2: '',
        };
    }

    const schemaSet = new Set(ctx.schemaFields);
    const boundCols = new Set<string>();
    if (ctx.colorField && ctx.colorField !== 'x_u16' && ctx.colorField !== 'y_u16') {
        boundCols.add(ctx.colorField);
    }
    if (ctx.sizeField && ctx.sizeField !== 'x_u16' && ctx.sizeField !== 'y_u16') {
        boundCols.add(ctx.sizeField);
    }
    if (!ctx.isXSpatial && ctx.xField && ctx.xField !== 'x_u16' && ctx.xField !== 'y_u16') {
        boundCols.add(ctx.xField);
    }
    if (!ctx.isYSpatial && ctx.yField && ctx.yField !== 'x_u16' && ctx.yField !== 'y_u16') {
        boundCols.add(ctx.yField);
    }

    const neededExtraCols: string[] = [];
    const requireGpuColumn = (col: string): boolean => {
        if (col === 'x_u16' || col === 'y_u16') return true;
        if (!schemaSet.has(col) && !ctx.categories[col]) return false;
        if (!boundCols.has(col) && !neededExtraCols.includes(col)) {
            neededExtraCols.push(col);
        }
        return true;
    };

    let hasActive = false;
    if (filter.rect) {
        const r = filter.rect;
        if (
            Number.isFinite(r.minX) &&
            Number.isFinite(r.maxX) &&
            Number.isFinite(r.minY) &&
            Number.isFinite(r.maxY)
        ) {
            hasActive = true;
        }
    }
    if (filter.lasso && Array.isArray(filter.lasso.points) && filter.lasso.points.length >= 3) {
        hasActive = true;
    }

    if (filter.ranges) {
        for (const [col, pair] of Object.entries(filter.ranges)) {
            if (!pair || pair.length < 2) continue;
            if (!Number.isFinite(pair[0]) || !Number.isFinite(pair[1])) continue;
            hasActive = true;
            if (!requireGpuColumn(col)) {
                return {
                    supported: false,
                    hasActivePredicates: true,
                    extraFilterField1: '',
                    extraFilterField2: '',
                    reason: `Column '${col}' is not resident in archive schemaFields`,
                };
            }
        }
    }

    if (filter.categories) {
        const catEntries = Object.entries(filter.categories).filter(
            ([, vals]) => Array.isArray(vals)
        );
        if (catEntries.length > 2) {
            return {
                supported: false,
                hasActivePredicates: true,
                extraFilterField1: '',
                extraFilterField2: '',
                reason: 'More than 2 simultaneous categorical column filters',
            };
        }
        for (const [col, vals] of catEntries) {
            hasActive = true;
            if (col === 'x_u16' || col === 'y_u16' || !requireGpuColumn(col)) {
                return {
                    supported: false,
                    hasActivePredicates: true,
                    extraFilterField1: '',
                    extraFilterField2: '',
                    reason: `Categorical column '${col}' is not GPU-resident`,
                };
            }
            const dict = ctx.categories[col];
            if (dict && dict.length > MAX_GPU_CATEGORY_CODES) {
                return {
                    supported: false,
                    hasActivePredicates: true,
                    extraFilterField1: '',
                    extraFilterField2: '',
                    reason: `Categorical column '${col}' has ${dict.length} > ${MAX_GPU_CATEGORY_CODES} categories`,
                };
            }
            for (const v of vals) {
                if (typeof v === 'number') {
                    if (!Number.isInteger(v) || v < 0 || v >= MAX_GPU_CATEGORY_CODES) {
                        return {
                            supported: false,
                            hasActivePredicates: true,
                            extraFilterField1: '',
                            extraFilterField2: '',
                            reason: `Category code '${v}' out of [0, ${MAX_GPU_CATEGORY_CODES - 1}]`,
                        };
                    }
                } else if (typeof v === 'string') {
                    if (!dict) {
                        return {
                            supported: false,
                            hasActivePredicates: true,
                            extraFilterField1: '',
                            extraFilterField2: '',
                            reason: `Missing dictionary for categorical column '${col}'`,
                        };
                    }
                    const code = dict.indexOf(v);
                    if (code >= MAX_GPU_CATEGORY_CODES) {
                        return {
                            supported: false,
                            hasActivePredicates: true,
                            extraFilterField1: '',
                            extraFilterField2: '',
                            reason: `Category '${v}' exceeds 512-code GPU fast-path cap`,
                        };
                    }
                }
            }
        }
    }

    const slot1Free = ctx.isXSpatial && !ctx.auxField1;
    const slot2Free = ctx.isYSpatial && !ctx.auxField2;
    const freeCount = (slot1Free ? 1 : 0) + (slot2Free ? 1 : 0);

    if (neededExtraCols.length > freeCount) {
        return {
            supported: false,
            hasActivePredicates: hasActive,
            extraFilterField1: '',
            extraFilterField2: '',
            reason: `Filter needs ${neededExtraCols.length} extra GPU columns, only ${freeCount} idle axis slot(s) available`,
        };
    }

    let extraFilterField1 = '';
    let extraFilterField2 = '';
    let idx = 0;
    if (slot1Free && idx < neededExtraCols.length) {
        extraFilterField1 = neededExtraCols[idx++];
    }
    if (slot2Free && idx < neededExtraCols.length) {
        extraFilterField2 = neededExtraCols[idx++];
    }

    return {
        supported: true,
        hasActivePredicates: hasActive,
        extraFilterField1,
        extraFilterField2,
    };
}

export interface PackFilterUniformParams {
    filter: DeepgraphFilter | null;
    hasSlotBitmask: boolean;
    camLeft: number;
    camTop: number;
    projCtx: AxisProjectionContext;
    /** Active storage channel names `[colorCol, sizeCol, xOrExtra1Col, yOrExtra2Col]`. */
    boundChannelCols: [string, string, string, string];
    categories: Record<string, string[]>;
}

/**
 * Packs the 1,024-byte `GlobalFilterUniforms` buffer in exact Float64 camera-relative `(relWorldX, fromTopY)`
 * and `1D RTO`-shifted column space.
 */
export function packGlobalFilterUniforms(
    outBuffer: ArrayBuffer,
    params: PackFilterUniformParams
): { activeFlags: number; unselectedOpacity: number; simplifiedLasso: [number, number][] | null } {
    const u32 = new Uint32Array(outBuffer, 0, GLOBAL_FILTER_UNIFORM_BYTES / 4);
    const f32 = new Float32Array(outBuffer, 0, GLOBAL_FILTER_UNIFORM_BYTES / 4);
    u32.fill(0);

    const { filter, hasSlotBitmask, camLeft, camTop, projCtx, boundChannelCols, categories } =
        params;

    let activeFlags = hasSlotBitmask ? FILTER_FLAG_HAS_SLOT_MASK : 0;
    let rangeActiveMask = 0;
    let lassoVertCount = 0;
    let catChannels = 0;
    let simplifiedLasso: [number, number][] | null = null;

    const unselectedOpacity =
        filter?.unselectedOpacity !== undefined && Number.isFinite(filter.unselectedOpacity)
            ? Math.max(0.0, Math.min(1.0, filter.unselectedOpacity))
            : 0.0;

    if (filter) {
        // 1. 2D Rectangle Brush in active data coordinates -> target camera-relative world bounds
        if (filter.rect) {
            const r = filter.rect;
            if (
                Number.isFinite(r.minX) &&
                Number.isFinite(r.maxX) &&
                Number.isFinite(r.minY) &&
                Number.isFinite(r.maxY)
            ) {
                const p0 = projectDataToWorld64(r.minX, r.minY, projCtx);
                const p1 = projectDataToWorld64(r.maxX, r.maxY, projCtx);
                const relX0 = p0.worldX - camLeft;
                const relX1 = p1.worldX - camLeft;
                const topY0 = camTop - p0.worldY;
                const topY1 = camTop - p1.worldY;

                f32[4] = Math.min(relX0, relX1);
                f32[5] = Math.max(relX0, relX1);
                f32[6] = Math.min(topY0, topY1);
                f32[7] = Math.max(topY0, topY1);
                activeFlags |= FILTER_FLAG_HAS_GLOBAL | FILTER_FLAG_HAS_RECT;
            }
        }

        // 2. 2D Freehand Polygon Lasso (RDP <= 64 vertices) -> target camera-relative world space
        if (filter.lasso && Array.isArray(filter.lasso.points) && filter.lasso.points.length >= 3) {
            const simp = simplifyRdpPolygon(filter.lasso.points, MAX_LASSO_GPU_VERTICES);
            if (simp.length >= 3) {
                simplifiedLasso = simp;
                lassoVertCount = simp.length;
                let minRelX = Infinity;
                let maxRelX = -Infinity;
                let minTopY = Infinity;
                let maxTopY = -Infinity;
                // lassoVerts starts at byte 288 -> float index 72
                const vertBase = 72;
                for (let i = 0; i < simp.length; i++) {
                    const pw = projectDataToWorld64(simp[i][0], simp[i][1], projCtx);
                    const rx = pw.worldX - camLeft;
                    const ty = camTop - pw.worldY;
                    if (rx < minRelX) minRelX = rx;
                    if (rx > maxRelX) maxRelX = rx;
                    if (ty < minTopY) minTopY = ty;
                    if (ty > maxTopY) maxTopY = ty;
                    f32[vertBase + i * 2] = rx;
                    f32[vertBase + i * 2 + 1] = ty;
                }
                f32[8] = minRelX;
                f32[9] = maxRelX;
                f32[10] = minTopY;
                f32[11] = maxTopY;
                activeFlags |= FILTER_FLAG_HAS_GLOBAL | FILTER_FLAG_HAS_LASSO;
            }
        }

        // 3. Numeric Range Predicates (1D-RTO shifted into GPU column space)
        if (filter.ranges) {
            const chanRangesBase = 16; // byte 64 / 4 = 16
            for (const [col, pair] of Object.entries(filter.ranges)) {
                if (!pair || pair.length < 2) continue;
                const rawLo = Math.min(pair[0], pair[1]);
                const rawHi = Math.max(pair[0], pair[1]);
                if (!Number.isFinite(rawLo) || !Number.isFinite(rawHi)) continue;

                if (col === 'x_u16') {
                    const db = projCtx.dataBounds;
                    const rb = projCtx.rootBounds;
                    const minD = db ? db.min_x : rb.minX;
                    const spanD = Math.max(1e-12, (db ? db.max_x : rb.maxX) - minD);
                    const g0 = (rawLo - minD) / spanD;
                    const g1 = (rawHi - minD) / spanD;
                    f32[chanRangesBase + 4 * 4 + 0] = Math.min(g0, g1);
                    f32[chanRangesBase + 4 * 4 + 1] = Math.max(g0, g1);
                    rangeActiveMask |= 1 << 4;
                    activeFlags |= FILTER_FLAG_HAS_GLOBAL;
                    continue;
                }
                if (col === 'y_u16') {
                    const db = projCtx.dataBounds;
                    const rb = projCtx.rootBounds;
                    const g0 = db
                        ? (db.max_y - rawLo) / Math.max(1e-12, db.max_y - db.min_y)
                        : projCtx.effectiveFlipY
                          ? (rawLo - rb.minY) / Math.max(1e-12, rb.maxY - rb.minY)
                          : (rb.maxY - rawLo) / Math.max(1e-12, rb.maxY - rb.minY);
                    const g1 = db
                        ? (db.max_y - rawHi) / Math.max(1e-12, db.max_y - db.min_y)
                        : projCtx.effectiveFlipY
                          ? (rawHi - rb.minY) / Math.max(1e-12, rb.maxY - rb.minY)
                          : (rb.maxY - rawHi) / Math.max(1e-12, rb.maxY - rb.minY);
                    f32[chanRangesBase + 5 * 4 + 0] = Math.min(g0, g1);
                    f32[chanRangesBase + 5 * 4 + 1] = Math.max(g0, g1);
                    rangeActiveMask |= 1 << 5;
                    activeFlags |= FILTER_FLAG_HAS_GLOBAL;
                    continue;
                }

                const origin = projCtx.colOrigins[col] ?? 0;
                const shiftedLo = rawLo - origin;
                const shiftedHi = rawHi - origin;
                for (let ch = 0; ch < 4; ch++) {
                    if (boundChannelCols[ch] === col) {
                        f32[chanRangesBase + ch * 4 + 0] = shiftedLo;
                        f32[chanRangesBase + ch * 4 + 1] = shiftedHi;
                        rangeActiveMask |= 1 << ch;
                        activeFlags |= FILTER_FLAG_HAS_GLOBAL;
                        break;
                    }
                }
            }
        }

        // 4. Up to 2 x 512-code Categorical Bitsets (catMask0 @ u32[40..55], catMask1 @ u32[56..71])
        if (filter.categories) {
            let catSlot = 0;
            for (const [col, allowed] of Object.entries(filter.categories)) {
                if (!Array.isArray(allowed) || catSlot >= 2) continue;
                const chIdx = boundChannelCols.indexOf(col);
                if (chIdx < 0) continue;

                const maskBase = catSlot === 0 ? 40 : 56;
                const dict = categories[col];
                for (const item of allowed) {
                    let code = -1;
                    if (typeof item === 'number' && Number.isInteger(item)) {
                        code = item;
                    } else if (typeof item === 'string' && dict) {
                        code = dict.indexOf(item);
                    }
                    if (code >= 0 && code < MAX_GPU_CATEGORY_CODES) {
                        u32[maskBase + (code >>> 5)] |= (1 << (code & 31)) >>> 0;
                    }
                }
                if (catSlot === 0) {
                    catChannels |= chIdx & 0xff;
                    activeFlags |= FILTER_FLAG_HAS_GLOBAL | FILTER_FLAG_HAS_CAT0;
                } else {
                    catChannels |= (chIdx & 0xff) << 8;
                    activeFlags |= FILTER_FLAG_HAS_GLOBAL | FILTER_FLAG_HAS_CAT1;
                }
                catSlot++;
            }
        }
    }

    u32[0] = activeFlags >>> 0;
    u32[1] = lassoVertCount >>> 0;
    u32[2] = rangeActiveMask >>> 0;
    u32[3] = catChannels >>> 0;
    f32[12] = unselectedOpacity;

    return { activeFlags, unselectedOpacity, simplifiedLasso };
}

/** Evaluates a `DeepgraphFilter` on CPU for a single decoded marker/row (used by `HighlightOverlay` & tests). */
export function evaluatePointMatchesFilterCpu(
    dataX: number,
    dataY: number,
    attributes: Record<string, unknown>,
    filter: DeepgraphFilter | null,
    categories: Record<string, string[]>,
    activeFilteredRowIds?: ReadonlySet<bigint> | null,
    rowIdBigInt?: bigint
): boolean {
    if (activeFilteredRowIds && rowIdBigInt !== undefined) {
        if (!activeFilteredRowIds.has(rowIdBigInt)) return false;
    }
    if (!filter) return true;

    if (filter.rect) {
        const loX = Math.min(filter.rect.minX, filter.rect.maxX);
        const hiX = Math.max(filter.rect.minX, filter.rect.maxX);
        const loY = Math.min(filter.rect.minY, filter.rect.maxY);
        const hiY = Math.max(filter.rect.minY, filter.rect.maxY);
        if (dataX < loX || dataX > hiX || dataY < loY || dataY > hiY) return false;
    }

    if (filter.lasso && filter.lasso.points.length >= 3) {
        const simp = simplifyRdpPolygon(filter.lasso.points, MAX_LASSO_GPU_VERTICES);
        if (simp.length >= 3 && !pointInPolygon2D(dataX, dataY, simp)) {
            return false;
        }
    }

    if (filter.ranges) {
        for (const [col, pair] of Object.entries(filter.ranges)) {
            if (!pair || pair.length < 2) continue;
            const lo = Math.min(pair[0], pair[1]);
            const hi = Math.max(pair[0], pair[1]);
            const raw = attributes[col];
            const v = typeof raw === 'number' ? raw : Number(raw);
            if (!Number.isFinite(v) || v < lo || v > hi) return false;
        }
    }

    if (filter.categories) {
        for (const [col, allowed] of Object.entries(filter.categories)) {
            if (!Array.isArray(allowed)) continue;
            if (allowed.length === 0) return false;
            const raw = attributes[col];
            if (raw === null || raw === undefined) return false;
            const dict = categories[col];
            let matched = false;
            for (const item of allowed) {
                if (raw === item) {
                    matched = true;
                    break;
                }
                if (typeof raw === 'number' && typeof item === 'string' && dict) {
                    if (dict[raw] === item) {
                        matched = true;
                        break;
                    }
                } else if (typeof raw === 'string' && typeof item === 'number' && dict) {
                    if (dict[item] === raw) {
                        matched = true;
                        break;
                    }
                }
            }
            if (!matched) return false;
        }
    }

    return true;
}

/**
 * Shared WGSL struct definitions & predicates injected into both `ComputeRasterizer`
 * (`@binding(10)`, `@binding(11)`) and `PickingLayer` (`@binding(9)`, `@binding(10)`).
 */
export function buildSharedFilterWGSL(filterBinding: number, maskBinding: number): string {
    return /* wgsl */ `
struct GlobalFilterUniforms {
    header: vec4<u32>,
    rectRelBounds: vec4<f32>,
    lassoRelAabb: vec4<f32>,
    styleParams: vec4<f32>,
    chanRanges: array<vec4<f32>, 6>,
    catMask0: array<vec4<u32>, 4>,
    catMask1: array<vec4<u32>, 4>,
    lassoVerts: array<vec4<f32>, 32>,
    _tail: array<vec4<f32>, 14>,
};

struct SlotVisibilityMask {
    words: array<vec4<u32>, 2048>,
};

@group(0) @binding(${filterBinding}) var<uniform> filterU: GlobalFilterUniforms;
@group(0) @binding(${maskBinding}) var<uniform> rowMaskU: SlotVisibilityMask;

fn get_lasso_vert(idx: u32) -> vec2<f32> {
    let pair = filterU.lassoVerts[idx >> 1u];
    if ((idx & 1u) == 0u) {
        return pair.xy;
    }
    return pair.zw;
}

fn eval_lasso_contains(px: f32, py: f32, nVerts: u32) -> bool {
    if (nVerts < 3u) {
        return false;
    }
    var inside = false;
    var j = nVerts - 1u;
    for (var k = 0u; k < nVerts; k = k + 1u) {
        let vk = get_lasso_vert(k);
        let vj = get_lasso_vert(j);
        let condY = (vk.y > py) != (vj.y > py);
        if (condY) {
            let xCross = ((vj.x - vk.x) * (py - vk.y)) / (vj.y - vk.y) + vk.x;
            if (px < xCross) {
                inside = !inside;
            }
        }
        j = k;
    }
    return inside;
}

fn eval_cat_channel_val(
    ch: u32,
    cVal: f32, hasC: bool,
    sVal: f32, hasS: bool,
    xVal: f32, hasX: bool,
    yVal: f32, hasY: bool,
    outVal: ptr<function, f32>
) -> bool {
    if (ch == 0u) { (*outVal) = cVal; return hasC; }
    if (ch == 1u) { (*outVal) = sVal; return hasS; }
    if (ch == 2u) { (*outVal) = xVal; return hasX; }
    (*outVal) = yVal;
    return hasY;
}

fn eval_filter_selected(
    localMaskIdx: u32,
    targetRelX: f32,
    targetFromTopY: f32,
    globalX: f32,
    globalY: f32,
    cVal: f32, hasC: bool,
    sVal: f32, hasS: bool,
    xVal: f32, hasX: bool,
    yVal: f32, hasY: bool
) -> bool {
    let activeFlags = filterU.header.x;
    if (activeFlags == 0u) {
        return true;
    }

    if ((activeFlags & 2u) != 0u) {
        if (localMaskIdx >= 262144u) {
            return false;
        }
        let w = rowMaskU.words[localMaskIdx >> 7u][(localMaskIdx >> 5u) & 3u];
        if (((w >> (localMaskIdx & 31u)) & 1u) == 0u) {
            return false;
        }
    }

    if ((activeFlags & 1u) == 0u) {
        return true;
    }

    if ((activeFlags & 4u) != 0u) {
        if (targetRelX < filterU.rectRelBounds.x || targetRelX > filterU.rectRelBounds.y ||
            targetFromTopY < filterU.rectRelBounds.z || targetFromTopY > filterU.rectRelBounds.w) {
            return false;
        }
    }

    if ((activeFlags & 8u) != 0u) {
        if (targetRelX < filterU.lassoRelAabb.x || targetRelX > filterU.lassoRelAabb.y ||
            targetFromTopY < filterU.lassoRelAabb.z || targetFromTopY > filterU.lassoRelAabb.w) {
            return false;
        }
        if (!eval_lasso_contains(targetRelX, targetFromTopY, filterU.header.y)) {
            return false;
        }
    }

    let rangeMask = filterU.header.z;
    if (rangeMask != 0u) {
        if ((rangeMask & 1u) != 0u) {
            if (!hasC || !(cVal == cVal) || cVal < filterU.chanRanges[0].x || cVal > filterU.chanRanges[0].y) {
                return false;
            }
        }
        if ((rangeMask & 2u) != 0u) {
            if (!hasS || !(sVal == sVal) || sVal < filterU.chanRanges[1].x || sVal > filterU.chanRanges[1].y) {
                return false;
            }
        }
        if ((rangeMask & 4u) != 0u) {
            if (!hasX || !(xVal == xVal) || xVal < filterU.chanRanges[2].x || xVal > filterU.chanRanges[2].y) {
                return false;
            }
        }
        if ((rangeMask & 8u) != 0u) {
            if (!hasY || !(yVal == yVal) || yVal < filterU.chanRanges[3].x || yVal > filterU.chanRanges[3].y) {
                return false;
            }
        }
        if ((rangeMask & 16u) != 0u) {
            if (globalX < filterU.chanRanges[4].x || globalX > filterU.chanRanges[4].y) {
                return false;
            }
        }
        if ((rangeMask & 32u) != 0u) {
            if (globalY < filterU.chanRanges[5].x || globalY > filterU.chanRanges[5].y) {
                return false;
            }
        }
    }

    if ((activeFlags & 16u) != 0u) {
        var cv0 = 0.0;
        let ok0 = eval_cat_channel_val(
            filterU.header.w & 0xFFu,
            cVal, hasC, sVal, hasS, xVal, hasX, yVal, hasY, &cv0
        );
        if (!ok0 || !(cv0 == cv0) || cv0 < -0.25 || cv0 >= 511.5) {
            return false;
        }
        let code0 = u32(cv0 + 0.5);
        if (abs(cv0 - f32(code0)) > 0.05) {
            return false;
        }
        let w0 = filterU.catMask0[code0 >> 7u][(code0 >> 5u) & 3u];
        if (((w0 >> (code0 & 31u)) & 1u) == 0u) {
            return false;
        }
    }

    if ((activeFlags & 32u) != 0u) {
        var cv1 = 0.0;
        let ok1 = eval_cat_channel_val(
            (filterU.header.w >> 8u) & 0xFFu,
            cVal, hasC, sVal, hasS, xVal, hasX, yVal, hasY, &cv1
        );
        if (!ok1 || !(cv1 == cv1) || cv1 < -0.25 || cv1 >= 511.5) {
            return false;
        }
        let code1 = u32(cv1 + 0.5);
        if (abs(cv1 - f32(code1)) > 0.05) {
            return false;
        }
        let w1 = filterU.catMask1[code1 >> 7u][(code1 >> 5u) & 3u];
        if (((w1 >> (code1 & 31u)) & 1u) == 0u) {
            return false;
        }
    }

    return true;
}
`;
}

/**
 * Shared owner of `@binding(10)` (`GlobalFilterUniforms`, `1 KiB`) and `@binding(11)`
 * (`SlotVisibilityMask`, `32 KiB` idle / lazy `maxSlots * 32 KiB` when `setFilteredRowIds` is active).
 */
export class FilterGpuResources {
    private device: GPUDevice;
    private maxSlots: number;

    public globalFilterBuffer: GPUBuffer | null = null;
    public dummyAllOnesMaskBuffer: GPUBuffer | null = null;
    private slotMaskBuffer: GPUBuffer | null = null;
    private allocatedMaskSlices = 0;

    private stagingBuffer = new ArrayBuffer(GLOBAL_FILTER_UNIFORM_BYTES);
    private zeroSlice = new Uint32Array(SLOT_MASK_SLICE_BYTES / 4);
    private slotOverflowSlices = new Map<number, number[]>(); // slotIdx -> extra slice indices (> 0)
    private freeExtraSliceIndices: number[] = [];
    private slotCpuMaskWords = new Map<number, Uint32Array[]>();
    private nextExtraSliceIdx = 0;

    public currentFilter: DeepgraphFilter | null = null;
    public hasSlotBitmask = false;
    public lastActiveFlags = 0;
    public lastUnselectedOpacity = 0.0;
    public maskBufferGeneration = 1;

    private initialized = false;
    private disposed = false;

    constructor(device: GPUDevice, maxSlots: number) {
        this.device = device;
        this.maxSlots = Math.max(64, maxSlots);
    }

    public init(): boolean {
        if (this.disposed) return false;
        if (this.initialized) return true;

        this.globalFilterBuffer = this.device.createBuffer({
            label: 'FilterGpuResources_GlobalUniforms_1KB',
            size: GLOBAL_FILTER_UNIFORM_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.dummyAllOnesMaskBuffer = this.device.createBuffer({
            label: 'FilterGpuResources_DummyAllOnesMask_32KB',
            size: SLOT_MASK_SLICE_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const allOnes = new Uint32Array(SLOT_MASK_SLICE_BYTES / 4).fill(0xffffffff);
        this.device.queue.writeBuffer(this.dummyAllOnesMaskBuffer, 0, allOnes);

        this.initialized = true;
        return true;
    }

    /** Active GPUBuffer to bind at the dynamic `SlotVisibilityMask` uniform binding. */
    public getActiveMaskBuffer(): GPUBuffer | null {
        if (!this.init()) return null;
        return this.hasSlotBitmask && this.slotMaskBuffer
            ? this.slotMaskBuffer
            : this.dummyAllOnesMaskBuffer;
    }

    /** Dynamic byte offset into `getActiveMaskBuffer()` for `(slotIdx, chunkIdx)`. */
    public getMaskDynamicOffset(slotIdx: number, chunkIdx = 0): number {
        if (!this.hasSlotBitmask || !this.slotMaskBuffer) {
            return 0;
        }
        if (chunkIdx === 0) {
            return slotIdx * SLOT_MASK_SLICE_BYTES;
        }
        const extras = this.slotOverflowSlices.get(slotIdx);
        const sliceIdx = extras?.[chunkIdx - 1] ?? slotIdx;
        return sliceIdx * SLOT_MASK_SLICE_BYTES;
    }

    private ensureSlotMaskBufferCapacity(requiredSlices: number): void {
        if (this.disposed) return;
        this.init();
        if (this.slotMaskBuffer && this.allocatedMaskSlices >= requiredSlices) {
            return;
        }
        const targetSlices = Math.max(this.maxSlots, requiredSlices);
        const maxBuf = Number(this.device.limits?.maxBufferSize ?? 268_435_456);
        const alignedMaxBuf = Math.floor(maxBuf / 256) * 256;
        const byteSize = Math.max(
            SLOT_MASK_SLICE_BYTES,
            Math.min(alignedMaxBuf, targetSlices * SLOT_MASK_SLICE_BYTES)
        );
        const nextBuf = this.device.createBuffer({
            label: 'FilterGpuResources_SlotVisibilityMasks',
            size: byteSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        if (this.slotMaskBuffer) {
            const copyBytes = Math.min(this.slotMaskBuffer.size, byteSize);
            const enc = this.device.createCommandEncoder({
                label: 'FilterGpuResources_GrowSlotMaskCopy',
            });
            enc.copyBufferToBuffer(this.slotMaskBuffer, 0, nextBuf, 0, copyBytes);
            this.device.queue.submit([enc.finish()]);
            this.slotMaskBuffer.destroy();
        }
        this.slotMaskBuffer = nextBuf;
        this.allocatedMaskSlices = Math.floor(byteSize / SLOT_MASK_SLICE_BYTES);
        this.maskBufferGeneration++;
    }

    public enableSlotBitmasks(): void {
        if (this.disposed) return;
        this.ensureSlotMaskBufferCapacity(this.maxSlots);
        if (!this.hasSlotBitmask) {
            this.hasSlotBitmask = true;
            this.maskBufferGeneration++;
        }
    }

    public clearSlotBitmasks(): void {
        this.hasSlotBitmask = false;
        this.slotOverflowSlices.clear();
        this.freeExtraSliceIndices.length = 0;
        this.slotCpuMaskWords.clear();
        this.nextExtraSliceIdx = 0;
        if (this.slotMaskBuffer) {
            this.slotMaskBuffer.destroy();
            this.slotMaskBuffer = null;
            this.allocatedMaskSlices = 0;
            this.maskBufferGeneration++;
        }
    }

    private releaseOverflowIndices(slotIdx: number): void {
        const existing = this.slotOverflowSlices.get(slotIdx);
        if (existing) {
            for (let i = existing.length - 1; i >= 0; i--) {
                this.freeExtraSliceIndices.push(existing[i]);
            }
            this.slotOverflowSlices.delete(slotIdx);
        }
    }

    public clearSingleSlotMask(slotIdx: number): void {
        this.releaseOverflowIndices(slotIdx);
        this.slotCpuMaskWords.delete(slotIdx);
        if (
            this.hasSlotBitmask &&
            this.slotMaskBuffer &&
            (slotIdx + 1) * SLOT_MASK_SLICE_BYTES <= this.slotMaskBuffer.size
        ) {
            this.device.queue.writeBuffer(
                this.slotMaskBuffer,
                slotIdx * SLOT_MASK_SLICE_BYTES,
                this.zeroSlice
            );
        }
    }

    /** $O(1)$ CPU bit-test against the slot's active `32 KiB` visibility mask (zero `Set<bigint>` retention). */
    public isSlotRowSelected(slotIdx: number, rawRowIdx: number): boolean {
        if (!this.hasSlotBitmask) return true;
        if (rawRowIdx < 0) return false;
        const slices = this.slotCpuMaskWords.get(slotIdx);
        if (!slices || slices.length === 0) return false;
        const sliceIdx = Math.floor(rawRowIdx / SLOT_MASK_ROWS_PER_SLICE);
        const localIdx = rawRowIdx - sliceIdx * SLOT_MASK_ROWS_PER_SLICE;
        const words = slices[sliceIdx];
        if (!words || localIdx >>> 5 >= words.length) return false;
        return ((words[localIdx >>> 5] >>> (localIdx & 31)) & 1) !== 0;
    }

    /** Writes 1 or more `32 KiB` `Uint32Array` bitmask slices for `slotIdx` (supporting `> 262,144`-row oversize leaves). */
    public writeSlotMaskSlices(slotIdx: number, slices: ReadonlyArray<ArrayBuffer>): void {
        if (this.disposed || slotIdx < 0 || slotIdx >= this.maxSlots) return;
        if (slices.length === 0) {
            if (this.hasSlotBitmask) {
                this.clearSingleSlotMask(slotIdx);
            }
            return;
        }
        this.enableSlotBitmasks();
        if (!this.slotMaskBuffer) return;

        const extraCount = Math.max(0, slices.length - 1);
        let extraIndices: number[] | undefined;
        if (extraCount > 0) {
            extraIndices = this.slotOverflowSlices.get(slotIdx);
            if (!extraIndices) {
                extraIndices = [];
                this.slotOverflowSlices.set(slotIdx, extraIndices);
            }
            while (extraIndices.length > extraCount) {
                this.freeExtraSliceIndices.push(extraIndices.pop()!);
            }
            while (extraIndices.length < extraCount) {
                const reused = this.freeExtraSliceIndices.pop();
                extraIndices.push(
                    reused !== undefined ? reused : this.maxSlots + this.nextExtraSliceIdx++
                );
            }
            let maxSliceIdx = slotIdx;
            for (const idx of extraIndices) {
                if (idx > maxSliceIdx) maxSliceIdx = idx;
            }
            this.ensureSlotMaskBufferCapacity(maxSliceIdx + 1);
        } else {
            this.releaseOverflowIndices(slotIdx);
        }

        const cpuMirrors: Uint32Array[] = [];
        for (let c = 0; c < slices.length; c++) {
            const targetSlice = c === 0 ? slotIdx : extraIndices![c - 1];
            const byteOffset = targetSlice * SLOT_MASK_SLICE_BYTES;
            const buf = slices[c];
            if (buf.byteLength === SLOT_MASK_SLICE_BYTES) {
                // Keep a compact CPU copy (`32 KiB`) so `releaseTileCpuBuffers` can drop `tile.rowMaskSlices`
                const cpuCopy = new Uint32Array(buf.slice(0));
                cpuMirrors.push(cpuCopy);
                if (byteOffset + SLOT_MASK_SLICE_BYTES <= this.slotMaskBuffer.size) {
                    this.device.queue.writeBuffer(
                        this.slotMaskBuffer,
                        byteOffset,
                        cpuCopy.buffer,
                        0,
                        SLOT_MASK_SLICE_BYTES
                    );
                }
            } else {
                const padded = new Uint32Array(SLOT_MASK_SLICE_BYTES / 4);
                new Uint8Array(padded.buffer).set(
                    new Uint8Array(buf, 0, Math.min(buf.byteLength, SLOT_MASK_SLICE_BYTES))
                );
                cpuMirrors.push(padded);
                if (byteOffset + SLOT_MASK_SLICE_BYTES <= this.slotMaskBuffer.size) {
                    this.device.queue.writeBuffer(this.slotMaskBuffer, byteOffset, padded);
                }
            }
        }
        this.slotCpuMaskWords.set(slotIdx, cpuMirrors);
    }

    /** Updates `globalFilterBuffer` for the current camera & encoding frame. */
    public updateFrameUniforms(
        camera: Camera2D,
        tileStore: TileGpuStore,
        projCtx: AxisProjectionContext,
        boundChannelCols: [string, string, string, string]
    ): { isFilterActive: boolean; useTwoPassDim: boolean } {
        if (!this.init() || !this.globalFilterBuffer) {
            return { isFilterActive: false, useTwoPassDim: false };
        }

        const camLeft = camera.position.x + camera.left / camera.zoom;
        const camTop = camera.position.y + camera.top / camera.zoom;

        const packed = packGlobalFilterUniforms(this.stagingBuffer, {
            filter: this.currentFilter,
            hasSlotBitmask: this.hasSlotBitmask,
            camLeft,
            camTop,
            projCtx,
            boundChannelCols,
            categories: tileStore.categoricalDictionaries,
        });

        this.lastActiveFlags = packed.activeFlags;
        this.lastUnselectedOpacity = packed.unselectedOpacity;
        this.device.queue.writeBuffer(
            this.globalFilterBuffer,
            0,
            this.stagingBuffer,
            0,
            GLOBAL_FILTER_UNIFORM_BYTES
        );

        const isFilterActive = packed.activeFlags !== 0;
        const modeSupportsDim = tileStore.computeMode === 0 || tileStore.computeMode === 1;
        const useTwoPassDim =
            isFilterActive && modeSupportsDim && packed.unselectedOpacity > 1e-4;

        return { isFilterActive, useTwoPassDim };
    }

    public getLiveGpuBytes(): number {
        return (
            (this.globalFilterBuffer?.size ?? 0) +
            (this.dummyAllOnesMaskBuffer?.size ?? 0) +
            (this.slotMaskBuffer?.size ?? 0)
        );
    }

    public dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        if (this.globalFilterBuffer) {
            this.globalFilterBuffer.destroy();
            this.globalFilterBuffer = null;
        }
        if (this.dummyAllOnesMaskBuffer) {
            this.dummyAllOnesMaskBuffer.destroy();
            this.dummyAllOnesMaskBuffer = null;
        }
        if (this.slotMaskBuffer) {
            this.slotMaskBuffer.destroy();
            this.slotMaskBuffer = null;
        }
        this.allocatedMaskSlices = 0;
        this.slotOverflowSlices.clear();
        this.freeExtraSliceIndices.length = 0;
        this.slotCpuMaskWords.clear();
        this.initialized = false;
    }
}
