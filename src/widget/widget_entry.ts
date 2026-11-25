import {
    Deepgraph,
    type CustomRangeSource,
    type DeepgraphEncoding,
    type DeepgraphFilter,
    type DeepgraphPlot,
    type EdgeSpec,
    type EdgeStyleOptions,
    type InteractionMode,
    type PointHitEvent,
    type PointSelectEvent,
    type TileSource,
    type ViewOptionsPatch,
} from '../index.ts';

interface AnyWidgetModel {
    get(key: string): any;
    set(key: string, value: any): void;
    save_changes(): void;
    send(content: Record<string, unknown>, callbacks?: unknown, buffers?: ArrayBuffer[]): void;
    on(event: string, callback: (...args: any[]) => void): void;
    off(event: string, callback?: (...args: any[]) => void): void;
}

interface RenderContext {
    model: AnyWidgetModel;
    el: HTMLElement;
}

interface PendingRangeRequest {
    totalLength: number;
    totalParts: number;
    receivedParts: number;
    assembled: Uint8Array | null;
    singlePartFast: ArrayBuffer | null;
    resolve: (data: ArrayBuffer) => void;
    reject: (err: Error) => void;
    cleanupAbort?: () => void;
}

interface PendingRpcRequest {
    resolve: (payload: any) => void;
    reject: (err: Error) => void;
    cleanupAbort?: () => void;
}

interface PendingBinaryRowFilter {
    filterSeq: number;
    dtype: 'u32' | 'u64';
    totalBytes: number;
    totalParts: number;
    receivedParts: number;
    assembled: Uint8Array;
}

const MAX_SAFE_BIGINT = 9007199254740991n;

function toArrayBufferSlice(viewOrBuf: ArrayBuffer | ArrayBufferView | string): ArrayBuffer {
    if (typeof viewOrBuf === 'string') {
        const bin = atob(viewOrBuf);
        const out = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
        return out.buffer;
    }
    if (viewOrBuf instanceof ArrayBuffer) {
        return viewOrBuf;
    }
    if (ArrayBuffer.isView(viewOrBuf)) {
        const u8 = new Uint8Array(viewOrBuf.buffer, viewOrBuf.byteOffset, viewOrBuf.byteLength);
        if (
            u8.byteOffset === 0 &&
            u8.byteLength === u8.buffer.byteLength &&
            u8.buffer instanceof ArrayBuffer
        ) {
            return u8.buffer;
        }
        return u8.slice().buffer as ArrayBuffer;
    }
    return new ArrayBuffer(0);
}

function formatSafeRowId(rowId: number, rowIdBigInt?: bigint): number | string {
    if (rowIdBigInt !== undefined) {
        if (rowIdBigInt >= -MAX_SAFE_BIGINT && rowIdBigInt <= MAX_SAFE_BIGINT) {
            return Number(rowIdBigInt);
        }
        return rowIdBigInt.toString();
    }
    return Math.trunc(rowId);
}

async function probeLocalHttpRange(url: string, timeoutMs = 500): Promise<boolean> {
    if (!url || typeof fetch !== 'function') return false;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
        const res = await fetch(url, {
            method: 'GET',
            headers: { Range: 'bytes=0-0' },
            signal: ctrl.signal,
        });
        await res.body?.cancel().catch(() => {});
        return res.status === 206;
    } catch {
        return false;
    } finally {
        clearTimeout(timer);
    }
}

export async function render({ model, el }: RenderContext): Promise<() => void> {
    let disposed = false;
    let plot: DeepgraphPlot | null = null;
    let suppressModeSync = false;
    let nextReqSeq = 0;

    const pendingRanges = new Map<string, PendingRangeRequest>();
    const pendingRpcs = new Map<string, PendingRpcRequest>();
    let pendingRowFilter: PendingBinaryRowFilter | null = null;
    let deferredRowFilter: Uint32Array | BigUint64Array | 'clear' | null = null;
    let latestFilterSeq = 0;
    let pinnedHit: PointHitEvent | null = null;

    const applyRowIdFilter = (val: Uint32Array | BigUint64Array | 'clear') => {
        if (!plot) {
            deferredRowFilter = val;
            return;
        }
        deferredRowFilter = null;
        void plot.setFilteredRowIds(val === 'clear' ? null : val);
    };

    // Host wrapper strictly inside `el` (`AGENTS.md` §3.1)
    el.replaceChildren();
    const rootWrap = document.createElement('div');
    rootWrap.className = 'dg-anywidget-root';
    rootWrap.style.position = 'relative';
    rootWrap.style.width = String(model.get('width') || '100%');
    rootWrap.style.height = `${Number(model.get('height') || 560)}px`;
    rootWrap.style.background = '#111111';
    rootWrap.style.borderRadius = '8px';
    rootWrap.style.overflow = 'hidden';
    rootWrap.style.userSelect = 'none';
    el.appendChild(rootWrap);

    const stage = document.createElement('div');
    stage.className = 'dg-anywidget-stage';
    stage.style.width = '100%';
    stage.style.height = '100%';
    rootWrap.appendChild(stage);

    // Floating inspect pill (click / hover readout strictly inside widget container)
    const inspectBadge = document.createElement('div');
    inspectBadge.className = 'dg-anywidget-inspect';
    inspectBadge.style.cssText =
        'position:absolute;left:12px;bottom:12px;max-width:360px;padding:7px 10px;' +
        'background:rgba(15,18,24,0.86);backdrop-filter:blur(6px);color:#e5e7eb;' +
        'border:1px solid rgba(125,211,252,0.35);border-radius:6px;' +
        'font:11px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;' +
        'pointer-events:none;display:none;z-index:10;white-space:pre-wrap;word-break:break-word;';
    rootWrap.appendChild(inspectBadge);

    const renderBadge = (hit: PointHitEvent | null, prefix: string) => {
        if (!hit) {
            inspectBadge.style.display = 'none';
            return;
        }
        const idVal = formatSafeRowId(hit.rowId, hit.rowIdBigInt);
        const lines = [
            `${prefix} #${idVal} (${hit.dataX.toFixed(3)}, ${hit.dataY.toFixed(3)})`,
        ];
        const skipKeys = new Set(['row_id', 'x', 'y', 'x_u16', 'y_u16', 'z']);
        const entries = Object.entries(hit.attributes || {})
            .filter(([k]) => !skipKeys.has(k))
            .slice(0, 7);
        for (const [k, v] of entries) {
            if (v === null || v === undefined) continue;
            const s = typeof v === 'string' && v.length > 90 ? `${v.slice(0, 90)}…` : String(v);
            lines.push(`${k}: ${s}`);
        }
        inspectBadge.textContent = lines.join('\n');
        inspectBadge.style.display = 'block';
    };

    // Sleek floating mini-toolbar (`Pan` | `Brush` | `Lasso` | `Fit` | `Clear`)
    const toolbar = document.createElement('div');
    toolbar.className = 'dg-anywidget-toolbar';
    toolbar.style.cssText =
        'position:absolute;top:10px;right:10px;display:flex;gap:4px;padding:4px;' +
        'background:rgba(17,20,27,0.84);backdrop-filter:blur(6px);' +
        'border:1px solid rgba(255,255,255,0.14);border-radius:7px;z-index:10;' +
        'font:11px/1 system-ui,-apple-system,sans-serif;';
    toolbar.style.display = model.get('show_toolbar') === false ? 'none' : 'flex';
    rootWrap.appendChild(toolbar);

    const modeButtons = new Map<InteractionMode, HTMLButtonElement>();
    const styleToolbarBtn = (btn: HTMLButtonElement, active: boolean) => {
        btn.style.cssText =
            'border:none;border-radius:5px;padding:5px 9px;cursor:pointer;font:inherit;font-weight:500;' +
            (active
                ? 'background:#38bdf8;color:#082f49;'
                : 'background:transparent;color:#d1d5db;');
    };
    const updateToolbarActiveMode = (mode: InteractionMode) => {
        for (const [m, b] of modeButtons.entries()) {
            styleToolbarBtn(b, m === mode);
        }
    };

    for (const [mode, label] of [
        ['pan', 'Pan'],
        ['brush', 'Brush'],
        ['lasso', 'Lasso'],
    ] as const) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.dataset.dgMode = mode;
        btn.textContent = label;
        styleToolbarBtn(btn, (model.get('interaction_mode') || 'pan') === mode);
        btn.addEventListener('click', (ev) => {
            ev.stopPropagation();
            if (!plot) return;
            plot.setInteractionMode(mode);
            updateToolbarActiveMode(mode);
            suppressModeSync = true;
            model.set('interaction_mode', mode);
            model.save_changes();
            suppressModeSync = false;
        });
        modeButtons.set(mode, btn);
        toolbar.appendChild(btn);
    }

    const fitBtn = document.createElement('button');
    fitBtn.type = 'button';
    fitBtn.dataset.dgAction = 'fit';
    fitBtn.textContent = 'Fit';
    styleToolbarBtn(fitBtn, false);
    fitBtn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        plot?.fitView();
    });
    toolbar.appendChild(fitBtn);

    const clearBtn = document.createElement('button');
    clearBtn.type = 'button';
    clearBtn.dataset.dgAction = 'clear';
    clearBtn.textContent = 'Clear';
    styleToolbarBtn(clearBtn, false);
    clearBtn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        if (!plot) return;
        plot.clearSelection();
        pinnedHit = null;
        renderBadge(null, '');
        model.set('selected_row_ids', []);
        model.set('selected_row_ids_count', 0);
        model.set('selected_row_ids_truncated', false);
        model.set('clicked_row', null);
        model.set('selection_clause', { kind: 'none', xField: null, yField: null });
        model.save_changes();
    });
    toolbar.appendChild(clearBtn);

    const createCommRangeSource = (archiveKey: string): CustomRangeSource => ({
        key: `comm://${archiveKey}`,
        // Keep automatic PMTiles leaf directory prefetch at 256 KiB (< 512 KiB single-frame WS budget)
        maxLeafPrefetchBytes: 256 * 1024,
        getBytes: (offset: number, length: number, signal?: AbortSignal) =>
            new Promise<{ data: ArrayBuffer }>((resolve, reject) => {
                if (disposed) {
                    reject(new Error('Widget disposed'));
                    return;
                }
                if (signal?.aborted) {
                    reject(new DOMException('Aborted', 'AbortError'));
                    return;
                }
                const reqId = `rng_${++nextReqSeq}`;
                const entry: PendingRangeRequest = {
                    totalLength: Math.max(0, length),
                    totalParts: 0,
                    receivedParts: 0,
                    assembled: null,
                    singlePartFast: null,
                    resolve: (data) => resolve({ data }),
                    reject,
                };
                if (signal) {
                    const onAbort = () => {
                        if (pendingRanges.delete(reqId)) {
                            model.send({ type: 'range_abort', reqId });
                            reject(new DOMException('Aborted', 'AbortError'));
                        }
                    };
                    signal.addEventListener('abort', onAbort, { once: true });
                    entry.cleanupAbort = () => signal.removeEventListener('abort', onAbort);
                }
                pendingRanges.set(reqId, entry);
                model.send({
                    type: 'range_req',
                    reqId,
                    offset: Math.max(0, Math.trunc(offset)),
                    length: Math.max(0, Math.trunc(length)),
                });
            }),
    });

    const sendRpc = (
        payload: Record<string, unknown>,
        signal?: AbortSignal
    ): Promise<any> =>
        new Promise((resolve, reject) => {
            if (disposed) {
                reject(new Error('Widget disposed'));
                return;
            }
            if (signal?.aborted) {
                reject(new DOMException('Aborted', 'AbortError'));
                return;
            }
            const reqId = `rpc_${++nextReqSeq}`;
            const entry: PendingRpcRequest = { resolve, reject };
            if (signal) {
                const onAbort = () => {
                    if (pendingRpcs.delete(reqId)) {
                        model.send({ type: 'rpc_abort', reqId });
                        reject(new DOMException('Aborted', 'AbortError'));
                    }
                };
                signal.addEventListener('abort', onAbort, { once: true });
                entry.cleanupAbort = () => signal.removeEventListener('abort', onAbort);
            }
            pendingRpcs.set(reqId, entry);
            model.send({ ...payload, reqId });
        });

    const onCustomMessage = (msg: any, buffers?: Array<ArrayBuffer | ArrayBufferView>) => {
        if (!msg || typeof msg !== 'object' || disposed) return;

        if (msg.type === 'range_res') {
            const entry = pendingRanges.get(msg.reqId);
            if (!entry) return;
            if (msg.error) {
                entry.cleanupAbort?.();
                pendingRanges.delete(msg.reqId);
                entry.reject(new Error(String(msg.error)));
                return;
            }
            const totalParts = Math.max(1, Number(msg.totalParts || 1));
            const totalLen = Math.max(0, Number(msg.totalLength ?? entry.totalLength));
            const partIdx = Math.max(0, Number(msg.partIndex || 0));
            const rawChunk = buffers && buffers[0] ? toArrayBufferSlice(buffers[0]) : new ArrayBuffer(0);

            if (totalParts === 1) {
                entry.cleanupAbort?.();
                pendingRanges.delete(msg.reqId);
                entry.resolve(rawChunk);
                return;
            }

            if (!entry.assembled) {
                entry.totalParts = totalParts;
                entry.totalLength = totalLen;
                entry.assembled = new Uint8Array(totalLen);
            }
            const dstOff = Number(msg.byteOffset ?? partIdx * 524288);
            const srcU8 = new Uint8Array(rawChunk);
            const copyLen = Math.max(0, Math.min(srcU8.byteLength, entry.assembled.byteLength - dstOff));
            if (copyLen > 0) {
                entry.assembled.set(srcU8.subarray(0, copyLen), dstOff);
            }
            entry.receivedParts++;
            if (entry.receivedParts >= totalParts) {
                entry.cleanupAbort?.();
                pendingRanges.delete(msg.reqId);
                entry.resolve(entry.assembled.buffer as ArrayBuffer);
            }
            return;
        }

        if (msg.type === 'resolve_row_res' || msg.type === 'resolve_edges_res') {
            const entry = pendingRpcs.get(msg.reqId);
            if (!entry) return;
            entry.cleanupAbort?.();
            pendingRpcs.delete(msg.reqId);
            if (msg.error) {
                entry.reject(new Error(String(msg.error)));
            } else {
                entry.resolve(msg);
            }
            return;
        }

        if (msg.type === 'rowid_filter_bin') {
            const msgSeq = Number(msg.filterSeq ?? 0);
            if (msgSeq > 0) {
                if (msgSeq < latestFilterSeq) {
                    return; // Stale chunk from an older superseded filter call
                }
                if (msgSeq > latestFilterSeq) {
                    latestFilterSeq = msgSeq;
                    pendingRowFilter = null;
                }
            }
            if (msg.dtype === 'clear') {
                pendingRowFilter = null;
                applyRowIdFilter('clear');
                return;
            }
            const dtype: 'u32' | 'u64' = msg.dtype === 'u64' ? 'u64' : 'u32';
            const totalParts = Math.max(1, Number(msg.totalParts || 1));
            const totalBytes = Math.max(0, Number(msg.totalBytes || 0));
            if (totalBytes > 160 * 1024 * 1024) {
                pendingRowFilter = null;
                return;
            }
            const partIdx = Math.max(0, Number(msg.partIndex || 0));
            const chunkBuf = buffers && buffers[0] ? toArrayBufferSlice(buffers[0]) : new ArrayBuffer(0);

            if (totalParts === 1) {
                pendingRowFilter = null;
                const typed =
                    dtype === 'u64'
                        ? new BigUint64Array(chunkBuf, 0, Math.floor(chunkBuf.byteLength / 8))
                        : new Uint32Array(chunkBuf, 0, Math.floor(chunkBuf.byteLength / 4));
                applyRowIdFilter(typed);
                return;
            }

            if (!pendingRowFilter || pendingRowFilter.filterSeq !== msgSeq) {
                pendingRowFilter = {
                    filterSeq: msgSeq,
                    dtype,
                    totalBytes,
                    totalParts,
                    receivedParts: 0,
                    assembled: new Uint8Array(totalBytes),
                };
            }
            const dstOff = Number(msg.byteOffset ?? partIdx * 524288);
            const srcU8 = new Uint8Array(chunkBuf);
            const copyLen = Math.max(
                0,
                Math.min(srcU8.byteLength, pendingRowFilter.assembled.byteLength - dstOff)
            );
            if (copyLen > 0) {
                pendingRowFilter.assembled.set(srcU8.subarray(0, copyLen), dstOff);
            }
            pendingRowFilter.receivedParts++;
            if (pendingRowFilter.receivedParts >= pendingRowFilter.totalParts) {
                const fullBuf = pendingRowFilter.assembled.buffer as ArrayBuffer;
                const finalDtype = pendingRowFilter.dtype;
                pendingRowFilter = null;
                const typed =
                    finalDtype === 'u64'
                        ? new BigUint64Array(fullBuf, 0, Math.floor(fullBuf.byteLength / 8))
                        : new Uint32Array(fullBuf, 0, Math.floor(fullBuf.byteLength / 4));
                applyRowIdFilter(typed);
            }
            return;
        }

        if (msg.type === 'fit_view') {
            plot?.fitView();
        } else if (msg.type === 'clear_selection') {
            pinnedHit = null;
            renderBadge(null, '');
            plot?.clearSelection();
        } else if (msg.type === 'set_viewport' && msg.viewport) {
            plot?.setViewport(msg.viewport);
        }
    };

    model.on('msg:custom', onCustomMessage);
    // Request replay of any binary `row_id` filter queued before mount or across browser reloads
    model.send({ type: 'frontend_ready' });

    if (!Deepgraph.isSupported()) {
        const fallback = document.createElement('div');
        fallback.style.cssText =
            'color:#fca5a5;padding:20px;font:13px/1.5 system-ui,sans-serif;';
        fallback.textContent =
            'WebGPU is not available in this browser context. Open in Chrome/Edge 113+ with WebGPU enabled.';
        stage.appendChild(fallback);
        return () => {
            disposed = true;
            model.off('msg:custom', onCustomMessage);
        };
    }

    const resolveEffectiveSource = async (): Promise<{
        source: TileSource | null;
        transport: string;
    }> => {
        const mode = String(model.get('source_mode') || 'auto');
        const httpUrl = String(model.get('http_url') || '');
        const archiveKey = String(model.get('archive_key') || 'default');

        if (mode === 'none') return { source: null, transport: 'none' };
        if (mode === 'remote_url' || mode === 'http') {
            return { source: httpUrl || null, transport: mode };
        }
        if (mode === 'comm') {
            return { source: createCommRangeSource(archiveKey), transport: 'comm' };
        }
        // 'auto': fast 500ms localhost HTTP Range probe, falling back to <= 512 KiB CommRangeSource
        if (httpUrl && (await probeLocalHttpRange(httpUrl, 500))) {
            return { source: httpUrl, transport: 'http' };
        }
        return { source: createCommRangeSource(archiveKey), transport: 'comm' };
    };

    const { source: initialSource, transport: resolvedTransport } = await resolveEffectiveSource();
    if (disposed) {
        model.off('msg:custom', onCustomMessage);
        return () => {};
    }
    rootWrap.dataset.dgTransport = resolvedTransport;

    const initMode: InteractionMode = (model.get('interaction_mode') as InteractionMode) || 'pan';
    const initEncoding = (model.get('encoding') as DeepgraphEncoding | undefined) || undefined;
    const initFilter = (model.get('filter_spec') as DeepgraphFilter | null | undefined) ?? null;
    const initEdges = (model.get('static_edges') as EdgeSpec[] | undefined) || undefined;
    const initEdgeStyle = (model.get('edge_style') as EdgeStyleOptions | undefined) || undefined;
    const powerPref: 'high-performance' | 'low-power' =
        model.get('power_preference') === 'low-power' ? 'low-power' : 'high-performance';

    const rawMaxSel = model.get('max_selected_row_ids');
    const maxSelectedRowIds =
        typeof rawMaxSel === 'number' && Number.isFinite(rawMaxSel)
            ? Math.max(0, Math.trunc(rawMaxSel))
            : 10000;

    try {
        plot = await Deepgraph.create(stage, {
            ...(initialSource ? { url: initialSource } : {}),
            powerPreference: powerPref,
            enableRowIdIndex: maxSelectedRowIds > 0,
            maxSelectedRowIds,
            interactionMode: initMode,
            ...(initEncoding && Object.keys(initEncoding).length > 0
                ? { encoding: initEncoding }
                : {}),
            ...(initFilter ? { filter: initFilter } : {}),
            ...(initEdges && initEdges.length > 0 ? { edges: initEdges } : {}),
            ...(initEdgeStyle && Object.keys(initEdgeStyle).length > 0
                ? { edgeStyle: initEdgeStyle }
                : {}),
            resolveEdgesOnHover: model.get('resolve_edges_on_hover') !== false,
            resolveRowAttributes: async (rowId, retained, rowIdBigInt, signal) => {
                if (!model.get('has_sidecar')) return retained;
                const res = await sendRpc(
                    {
                        type: 'resolve_row',
                        rowId: formatSafeRowId(rowId, rowIdBigInt),
                    },
                    signal
                );
                if (!res?.attributes || typeof res.attributes !== 'object') {
                    return retained;
                }
                // Place high-signal cold text/summary keys first, then remaining cold sidecar attributes,
                // and finally retained GPU tile columns so inspect badge never clips summaries/prompts!
                const coldAttrs = res.attributes as Record<string, unknown>;
                const priorityColdKeys = [
                    'record_summary',
                    'summary',
                    'prompt',
                    'title',
                    'name',
                    'label',
                    'text',
                ];
                const ordered: Record<string, unknown> = {};
                for (const pk of priorityColdKeys) {
                    if (pk in coldAttrs && !(pk in retained)) {
                        ordered[pk] = coldAttrs[pk];
                    }
                }
                for (const [k, v] of Object.entries(coldAttrs)) {
                    if (!(k in retained) && !(k in ordered)) {
                        ordered[k] = v;
                    }
                }
                for (const [k, v] of Object.entries(retained)) {
                    ordered[k] = k in coldAttrs ? coldAttrs[k] : v;
                }
                return ordered;
            },
            resolveEdges: async (hit, trigger, signal) => {
                if (!model.get('has_edge_resolver')) return [];
                if (trigger === 'hover' && model.get('resolve_edges_on_hover') === false) {
                    return [];
                }
                const res = await sendRpc(
                    {
                        type: 'resolve_edges',
                        rowId: formatSafeRowId(hit.rowId, hit.rowIdBigInt),
                        trigger,
                        dataX: hit.dataX,
                        dataY: hit.dataY,
                    },
                    signal
                );
                return Array.isArray(res?.edges) ? (res.edges as EdgeSpec[]) : [];
            },
        });
    } catch (err) {
        const errBox = document.createElement('div');
        errBox.style.cssText = 'color:#fca5a5;padding:20px;font:13px/1.5 system-ui,sans-serif;';
        errBox.textContent = `Deepgraph initialization failed: ${
            err instanceof Error ? err.message : String(err)
        }`;
        stage.appendChild(errBox);
        return () => {
            disposed = true;
            model.off('msg:custom', onCustomMessage);
        };
    }

    if (disposed) {
        plot.destroy();
        model.off('msg:custom', onCustomMessage);
        return () => {};
    }

    const initViewOpts = model.get('view_options') as ViewOptionsPatch | undefined;
    if (initViewOpts && Object.keys(initViewOpts).length > 0) {
        plot.setViewOptions(initViewOpts);
    }
    if (initialSource) {
        plot.fitView();
    }
    if (deferredRowFilter !== null) {
        const pending = deferredRowFilter;
        deferredRowFilter = null;
        void plot.setFilteredRowIds(pending === 'clear' ? null : pending);
    }

    // Expose diagnostic reference on rootWrap ONLY for local E2E verification (`0` window globals!)
    (rootWrap as any).__dgPlot = plot;
    plot.on('telemetry', (snap) => {
        (rootWrap as any).__dgTelemetry = snap;
    });

    plot.on('hover', (hit) => {
        // Non-reactive UI preview ONLY (`0` model.save_changes() calls on pointermove!)
        if (!pinnedHit) {
            renderBadge(hit, 'Hover');
        }
    });

    // Committed selection handler -> single atomic `model.save_changes()` for Marimo reactivity!
    // Note: `selection_sql` is computed & validated server-side in Python (`widget.py`) from
    // `selection_clause` and `clicked_row` so untrusted client strings never enter DuckDB SQL.
    plot.on('select', (ev: PointSelectEvent) => {
        const curCapRaw = model.get('max_selected_row_ids');
        const curCap =
            typeof curCapRaw === 'number' && Number.isFinite(curCapRaw)
                ? Math.max(0, Math.trunc(curCapRaw))
                : 10000;

        let safeIds: (number | string)[] = [];
        let rawCount = 0;
        if (ev.rowIdsBigInt && ev.rowIdsBigInt.length > 0) {
            rawCount = ev.rowIdsBigInt.length;
            const slice = curCap < rawCount ? ev.rowIdsBigInt.slice(0, curCap) : ev.rowIdsBigInt;
            safeIds = slice.map((b) =>
                b >= -MAX_SAFE_BIGINT && b <= MAX_SAFE_BIGINT ? Number(b) : b.toString()
            );
        } else if (ev.rowIds && ev.rowIds.length > 0) {
            rawCount = ev.rowIds.length;
            const slice = curCap < rawCount ? ev.rowIds.slice(0, curCap) : ev.rowIds;
            safeIds = slice.map((n) => Math.trunc(n));
        }

        const totalMatched = ev.totalMatched ?? rawCount;
        const truncated = Boolean(ev.rowIdsTruncated) || totalMatched > safeIds.length;

        if (ev.points.length > 0) {
            const p = ev.points[0];
            pinnedHit = p;
            renderBadge(p, 'Selected');
            model.set('clicked_row', {
                rowId: formatSafeRowId(p.rowId, p.rowIdBigInt),
                dataX: p.dataX,
                dataY: p.dataY,
                attributes: p.attributes,
            });
        } else {
            pinnedHit = null;
            renderBadge(null, '');
            model.set('clicked_row', null);
        }

        model.set('selected_row_ids', safeIds);
        model.set('selected_row_ids_count', totalMatched);
        model.set('selected_row_ids_truncated', truncated);
        model.set('selection_clause', ev.clause ? JSON.parse(JSON.stringify(ev.clause)) : null);
        model.save_changes();
    });

    // Python -> JS live trait listeners
    const onEncodingChange = () => {
        const enc = model.get('encoding') as DeepgraphEncoding | undefined;
        if (enc && plot) plot.setEncoding(enc);
    };
    const onViewOptionsChange = () => {
        const vo = model.get('view_options') as ViewOptionsPatch | undefined;
        if (vo && plot) plot.setViewOptions(vo);
    };
    const onModeChange = () => {
        if (suppressModeSync || !plot) return;
        const m = (model.get('interaction_mode') as InteractionMode) || 'pan';
        plot.setInteractionMode(m);
        updateToolbarActiveMode(m);
    };
    const onFilterSpecChange = () => {
        if (!plot) return;
        const f = (model.get('filter_spec') as DeepgraphFilter | null | undefined) ?? null;
        plot.setFilter(f);
    };
    const onStaticEdgesChange = () => {
        if (!plot) return;
        const edges = (model.get('static_edges') as EdgeSpec[] | undefined) || [];
        const style = (model.get('edge_style') as EdgeStyleOptions | undefined) || undefined;
        plot.setEdges(edges, style);
    };
    const onEdgeStyleChange = () => {
        if (!plot) return;
        const style = model.get('edge_style') as EdgeStyleOptions | undefined;
        if (style) plot.setEdgeStyle(style);
    };
    let archiveChangeSeq = 0;
    const onArchiveChange = async () => {
        if (!plot || disposed) return;
        const mySeq = ++archiveChangeSeq;
        pinnedHit = null;
        renderBadge(null, '');
        try {
            const { source, transport } = await resolveEffectiveSource();
            if (disposed || !plot || !source || mySeq !== archiveChangeSeq) return;
            rootWrap.dataset.dgTransport = transport;
            const enc = (model.get('encoding') as DeepgraphEncoding | undefined) || undefined;
            await plot.load(source, enc && Object.keys(enc).length > 0 ? { encoding: enc } : {});
            if (disposed || !plot || mySeq !== archiveChangeSeq) return;
            const vo = model.get('view_options') as ViewOptionsPatch | undefined;
            if (vo && Object.keys(vo).length > 0) {
                plot.setViewOptions(vo);
            }
            plot.fitView();
        } catch (err: any) {
            if (err?.name === 'AbortError' || disposed) return;
            console.warn('[pydeepgraph] Failed to reload archive:', err);
        }
    };
    const onHeightChange = () => {
        rootWrap.style.height = `${Number(model.get('height') || 560)}px`;
    };
    const onToolbarVisChange = () => {
        toolbar.style.display = model.get('show_toolbar') === false ? 'none' : 'flex';
    };

    model.on('change:encoding', onEncodingChange);
    model.on('change:view_options', onViewOptionsChange);
    model.on('change:interaction_mode', onModeChange);
    model.on('change:filter_spec', onFilterSpecChange);
    model.on('change:static_edges', onStaticEdgesChange);
    model.on('change:edge_style', onEdgeStyleChange);
    model.on('change:archive_key', onArchiveChange);
    model.on('change:height', onHeightChange);
    model.on('change:show_toolbar', onToolbarVisChange);

    return () => {
        disposed = true;
        model.off('msg:custom', onCustomMessage);
        model.off('change:encoding', onEncodingChange);
        model.off('change:view_options', onViewOptionsChange);
        model.off('change:interaction_mode', onModeChange);
        model.off('change:filter_spec', onFilterSpecChange);
        model.off('change:static_edges', onStaticEdgesChange);
        model.off('change:edge_style', onEdgeStyleChange);
        model.off('change:archive_key', onArchiveChange);
        model.off('change:height', onHeightChange);
        model.off('change:show_toolbar', onToolbarVisChange);
        for (const p of pendingRanges.values()) {
            p.cleanupAbort?.();
            p.reject(new Error('Widget disposed'));
        }
        pendingRanges.clear();
        for (const r of pendingRpcs.values()) {
            r.cleanupAbort?.();
            r.reject(new Error('Widget disposed'));
        }
        pendingRpcs.clear();
        if (plot) {
            plot.destroy();
            plot = null;
        }
        delete (rootWrap as any).__dgPlot;
        delete (rootWrap as any).__dgTelemetry;
    };
}

export default { render };
