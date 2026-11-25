from __future__ import annotations

import math
import mmap
import os
import re
from pathlib import Path
from typing import Any

import anywidget
import numpy as np
import pyarrow as pa
import traitlets

from ._edges import DuckDbEdgeResolver, EdgeResolverCallable
from ._http_range import EXPLICIT_RANGE_MAX_BYTES, get_shared_range_daemon
from ._ingest import (
    DEFAULT_MAX_CACHE_BYTES,
    DEFAULT_MAX_FILTERED_ROW_IDS,
    SidecarStore,
    quote_sql_ident,
    resolve_dataset_input,
)

# Strictly <= 512 KiB per WebSocket binary frame so Marimo / `websockets` (1 MiB cap)
# and Jupyter Tornado (10 MiB cap) never drop large tile or directory reads!
COMM_CHUNK_BYTES = 512 * 1024
MAX_LASSO_COORD_ABS = 1e150  # Prevents DuckDB SIMD `DOUBLE` multiplication overflow (`< 1.79e308`)
_STATIC_WIDGET_JS = Path(__file__).parent / "_static" / "widget.js"
_SAFE_FALLBACK_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_./:-]{0,127}$")


def _rdp_simplify_py(
    points: list[tuple[float, float]] | list[list[float]],
    max_vertices: int = 64,
) -> list[tuple[float, float]]:
    n_in = len(points)
    iter_pts = (
        (points[(k * n_in) // 16384] for k in range(16384))
        if n_in > 16384
        else points
    )
    cleaned: list[tuple[float, float]] = []
    for pt in iter_pts:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        x = float(pt[0])
        y = float(pt[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if cleaned and cleaned[-1] == (x, y):
            continue
        cleaned.append((x, y))
    if len(cleaned) > 3 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    if len(cleaned) <= max_vertices:
        return cleaned
    capped = (
        [cleaned[(k * len(cleaned)) // 4096] for k in range(4096)]
        if len(cleaned) > 4096
        else cleaned
    )

    def sq_seg_dist(
        px: float, py: float, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        try:
            dx = x2 - x1
            dy = y2 - y1
            x, y = x1, y1
            denom = dx * dx + dy * dy
            if math.isfinite(denom) and denom > 0.0:
                t = ((px - x1) * dx + (py - y1) * dy) / denom
                if t > 1.0:
                    x, y = x2, y2
                elif t > 0.0:
                    x += dx * t
                    y += dy * t
            return (px - x) ** 2 + (py - y) ** 2
        except OverflowError:
            return math.inf

    def rdp_pass(pts: list[tuple[float, float]], sq_tol: float) -> list[tuple[float, float]]:
        n = len(pts)
        if n <= 2:
            return list(pts)
        keep = bytearray(n)
        keep[0] = 1
        keep[n - 1] = 1
        stack = [(0, n - 1)]
        while stack:
            first, last = stack.pop()
            max_sq = 0.0
            idx = -1
            x1, y1 = pts[first]
            x2, y2 = pts[last]
            for i in range(first + 1, last):
                d_sq = sq_seg_dist(pts[i][0], pts[i][1], x1, y1, x2, y2)
                if d_sq > max_sq:
                    max_sq = d_sq
                    idx = i
            if max_sq > sq_tol and idx != -1:
                keep[idx] = 1
                stack.append((first, idx))
                stack.append((idx, last))
        return [pts[i] for i in range(n) if keep[i]]

    xs = [p[0] for p in capped]
    ys = [p[1] for p in capped]
    try:
        raw_diag_sq = (max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2
    except OverflowError:
        raw_diag_sq = math.inf

    if math.isfinite(raw_diag_sq):
        diag_sq = max(1e-24, raw_diag_sq)
        closed_in = [*capped, capped[0]]
        lo = 0.0
        hi = diag_sq * 0.25
        best: list[tuple[float, float]] | None = None
        for it in range(24):
            mid = diag_sq * 1e-8 if it == 0 else 0.5 * (lo + hi)
            cand = rdp_pass(closed_in, mid)
            open_cand = cand[:-1] if len(cand) > 1 else cand
            if len(open_cand) <= max_vertices:
                if len(open_cand) >= 3:
                    best = open_cand
                hi = mid
            else:
                lo = mid
        if best and 3 <= len(best) <= max_vertices:
            return best
    target = max(3, max_vertices)
    return [capped[min(len(capped) - 1, (k * len(capped)) // target)] for k in range(target)]


def build_lasso_sql(
    x_column: str,
    y_column: str,
    points: list[tuple[float, float]] | list[list[float]],
    max_vertices: int = 64,
) -> str:
    """Generates a division-free DuckDB `WHERE` predicate matching WebGPU RDP lasso evaluation."""
    verts = _rdp_simplify_py(points, max_vertices)
    if len(verts) < 3:
        return "FALSE"
    qx = quote_sql_ident(x_column)
    qy = quote_sql_ident(y_column)
    min_x = min(v[0] for v in verts)
    max_x = max(v[0] for v in verts)
    min_y = min(v[1] for v in verts)
    max_y = max(v[1] for v in verts)
    if not all(
        math.isfinite(v) and abs(v) <= MAX_LASSO_COORD_ABS
        for v in (min_x, max_x, min_y, max_y)
    ):
        return "FALSE"

    terms: list[str] = []
    n = len(verts)
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        dy = yj - yi
        if math.isfinite(dy) and abs(dy) >= 1e-15:
            try:
                slope = (xj - xi) / dy
            except OverflowError:
                return "FALSE"
            if not math.isfinite(slope) or abs(slope) > MAX_LASSO_COORD_ABS:
                return "FALSE"
            terms.append(
                f"CASE WHEN (({yi} > {qy}) <> ({yj} > {qy})) "
                f"AND ({qx} < (({qy} - {yi}) * {slope} + {xi})) THEN 1 ELSE 0 END"
            )
        j = i
    if not terms:
        return "FALSE"
    aabb = f"({qx} BETWEEN {min_x} AND {max_x}) AND ({qy} BETWEEN {min_y} AND {max_y})"
    parity = f"(({' + '.join(terms)}) % 2) = 1"
    return f"({aabb} AND {parity})"


class DeepgraphWidget(anywidget.AnyWidget):
    """
    Pure-WebGPU `.arrowtiles` scatterplot & ego-network `anywidget` for Marimo and Jupyter.

    - Streams `.arrowtiles` pyramids via zero-copy `mmap` over either a localhost HTTP 206 fast
      path (`transport="auto"`) or `<= 512 KiB`-chunked `CommRangeSource` binary WebSocket frames.
    - Updates reactive traitlets (`selected_row_ids`, `clicked_row`, `selection_clause`,
      `selection_sql`) **strictly on committed selections** (`click`, `brush-end`, `lasso-end`,
      `clear`) so Marimo notebooks never experience `pointermove` cell re-execution storms.
    - Reconstructs and column-validates `selection_sql` **server-side in Python** so untrusted
      client WebSocket payloads can never inject arbitrary SQL into `query_selected()`.
    """

    _esm = _STATIC_WIDGET_JS

    # Server-owned configuration traitlets (Python -> JS; client tampering reverted in Python)
    source_mode = traitlets.Unicode("auto").tag(sync=True)
    http_url = traitlets.Unicode("").tag(sync=True)
    archive_key = traitlets.Unicode("default").tag(sync=True)
    width = traitlets.Unicode("100%").tag(sync=True)
    height = traitlets.Int(560).tag(sync=True)
    power_preference = traitlets.Unicode("high-performance").tag(sync=True)
    show_toolbar = traitlets.Bool(True).tag(sync=True)
    # Python-only SQL schema traitlets (NOT synced over WebSocket so clients cannot redirect columns!)
    x_column = traitlets.Unicode("x")
    y_column = traitlets.Unicode("y")
    row_id_column = traitlets.Unicode("row_id")
    max_selected_row_ids = traitlets.Int(10_000).tag(sync=True)

    encoding = traitlets.Dict(default_value={}).tag(sync=True)
    view_options = traitlets.Dict(default_value={}).tag(sync=True)
    interaction_mode = traitlets.Unicode("pan").tag(sync=True)
    filter_spec = traitlets.Dict(allow_none=True, default_value=None).tag(sync=True)
    static_edges = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)
    edge_style = traitlets.Dict(default_value={}).tag(sync=True)

    has_sidecar = traitlets.Bool(False).tag(sync=True)
    has_edge_resolver = traitlets.Bool(False).tag(sync=True)
    resolve_edges_on_hover = traitlets.Bool(True).tag(sync=True)

    # Committed reactive output traitlets
    selected_row_ids = traitlets.List(
        traitlets.Union([traitlets.Int(), traitlets.Unicode()]),
        default_value=[],
    ).tag(sync=True)
    selected_row_ids_count = traitlets.Int(0).tag(sync=True)
    selected_row_ids_truncated = traitlets.Bool(False).tag(sync=True)
    clicked_row = traitlets.Dict(allow_none=True, default_value=None).tag(sync=True)
    selection_clause = traitlets.Dict(allow_none=True, default_value=None).tag(sync=True)
    selection_sql = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self,
        data: Any = None,
        *,
        x: str | None = None,
        y: str | None = None,
        color: str | dict[str, Any] | None = None,
        size: str | dict[str, Any] | None = None,
        row_id: str = "row_id",
        columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
        sort_col: str | None = None,
        max_capacity: int = 25_000,
        encoding: dict[str, Any] | None = None,
        view_options: dict[str, Any] | None = None,
        interaction_mode: str = "pan",
        filter_spec: dict[str, Any] | None = None,
        sidecar: Any | None = None,
        edges: Any | None = None,
        edge_style: dict[str, Any] | None = None,
        resolve_edges: EdgeResolverCallable | None = None,
        resolve_edges_on_hover: bool = True,
        bidirectional_edges: bool = False,
        max_ego_neighbors: int = 48,
        max_selected_row_ids: int = 10_000,
        max_filtered_row_ids: int = DEFAULT_MAX_FILTERED_ROW_IDS,
        transport: str = "auto",  # "auto" | "http" | "comm"
        width: str = "100%",
        height: int = 560,
        power_preference: str = "high-performance",
        show_toolbar: bool = True,
        cache_dir: str | os.PathLike[str] | None = None,
        max_cache_bytes: int = DEFAULT_MAX_CACHE_BYTES,
        **kwargs: Any,
    ) -> None:
        if not _STATIC_WIDGET_JS.is_file():
            raise RuntimeError(
                f"Missing built widget asset {_STATIC_WIDGET_JS}. Run `npm run build:widget` first."
            )

        # Accept `x_column` / `y_column` / `row_id_column` keyword aliases (`I4`)
        x = x or kwargs.pop("x_column", None)
        y = y or kwargs.pop("y_column", None)
        if "row_id_column" in kwargs:
            row_id = str(kwargs.pop("row_id_column"))

        self._archive_path: Path | None = None
        self._archive_file: Any = None
        self._archive_mmap: mmap.mmap | None = None
        self._archive_size: int = 0
        self._http_token: str | None = None
        self._sidecar_store: SidecarStore | None = None
        self._duck_edge_resolver: DuckDbEdgeResolver | None = None
        self._custom_edge_callable: EdgeResolverCallable | None = resolve_edges
        self._aborted_req_ids: set[str] = set()
        self._filter_seq: int = 0
        self._latest_row_filter: tuple[int, str, bytes] | None = None
        self.max_filtered_row_ids: int = max(0, int(max_filtered_row_ids))
        self._validated_selection_sql: str = ""
        self._updating_selection_sql: bool = False
        self._updating_server_traits: bool = False
        self._server_sync_values: dict[str, Any] = {
            "source_mode": "auto",
            "http_url": "",
            "archive_key": "default",
            "has_sidecar": False,
            "has_edge_resolver": False,
        }

        merged_enc: dict[str, Any] = dict(encoding or {})
        if color is not None and "color" not in merged_enc:
            merged_enc["color"] = {"field": color} if isinstance(color, str) else color
        if size is not None and "size" not in merged_enc:
            merged_enc["size"] = {"field": size} if isinstance(size, str) else size

        super().__init__(
            width=width,
            height=height,
            power_preference=power_preference,
            show_toolbar=show_toolbar,
            interaction_mode=interaction_mode,
            encoding=merged_enc,
            view_options=dict(view_options or {}),
            filter_spec=filter_spec,
            edge_style=dict(edge_style or {}),
            resolve_edges_on_hover=resolve_edges_on_hover,
            max_selected_row_ids=max(0, int(max_selected_row_ids)),
            **kwargs,
        )

        self.on_msg(self._on_frontend_msg)

        if data is not None:
            try:
                self._bind_data(
                    data,
                    x=x,
                    y=y,
                    row_id=row_id,
                    columns=columns,
                    categorical_columns=categorical_columns,
                    sort_col=sort_col,
                    max_capacity=max_capacity,
                    sidecar=sidecar,
                    edges=edges,
                    bidirectional_edges=bidirectional_edges,
                    max_ego_neighbors=max_ego_neighbors,
                    transport=transport,
                    cache_dir=cache_dir,
                    max_cache_bytes=max_cache_bytes,
                )
            except Exception:
                self.close()
                raise

    def _is_valid_sql_column(self, col: str) -> bool:
        if not col or not isinstance(col, str):
            return False
        if self._sidecar_store is not None:
            return col in self._sidecar_store.columns
        allowed_defaults = {self.x_column, self.y_column, self.row_id_column}
        return col in allowed_defaults or bool(_SAFE_FALLBACK_IDENT_RE.match(col))

    def _compute_validated_selection_sql(self) -> str:
        clicked = self.clicked_row
        if isinstance(clicked, dict) and clicked.get("rowId") is not None:
            raw_rid = clicked["rowId"]
            if isinstance(raw_rid, bool):
                return ""
            rid_col = self.row_id_column or "row_id"
            if not self._is_valid_sql_column(rid_col):
                return ""
            try:
                rid_int = int(raw_rid)
            except (TypeError, ValueError):
                return ""
            if rid_int < -(1 << 63) or rid_int >= (1 << 64):
                return ""
            return f"{quote_sql_ident(rid_col)} = {rid_int}"

        clause = self.selection_clause
        if not isinstance(clause, dict):
            return ""
        kind = clause.get("kind")
        if not kind or kind == "none":
            return ""

        raw_x = clause.get("xField")
        raw_y = clause.get("yField")
        x_col = str(raw_x) if isinstance(raw_x, str) and raw_x else (self.x_column or "x")
        y_col = str(raw_y) if isinstance(raw_y, str) and raw_y else (self.y_column or "y")
        if not (self._is_valid_sql_column(x_col) and self._is_valid_sql_column(y_col)):
            return ""

        if kind == "rect":
            rect = clause.get("rect")
            if not isinstance(rect, dict):
                return ""
            try:
                x0 = float(rect["minX"])
                x1 = float(rect["maxX"])
                y0 = float(rect["minY"])
                y1 = float(rect["maxY"])
            except (KeyError, TypeError, ValueError):
                return ""
            if not all(math.isfinite(v) for v in (x0, x1, y0, y1)):
                return ""
            lo_x, hi_x = min(x0, x1), max(x0, x1)
            lo_y, hi_y = min(y0, y1), max(y0, y1)
            qx = quote_sql_ident(x_col)
            qy = quote_sql_ident(y_col)
            return f"({qx} BETWEEN {lo_x} AND {hi_x}) AND ({qy} BETWEEN {lo_y} AND {hi_y})"

        if kind == "lasso":
            lasso = clause.get("lasso")
            pts = lasso.get("points") if isinstance(lasso, dict) else None
            if not isinstance(pts, list) or len(pts) < 3:
                return ""
            try:
                sql = build_lasso_sql(x_col, y_col, pts, 64)
            except (TypeError, ValueError):
                return ""
            return "" if sql == "FALSE" else sql

        return ""

    def _sync_validated_selection_sql(self) -> None:
        sql = self._compute_validated_selection_sql()
        self._validated_selection_sql = sql
        if self.selection_sql != sql:
            self._updating_selection_sql = True
            try:
                self.selection_sql = sql
            finally:
                self._updating_selection_sql = False

    @traitlets.observe(
        "selection_clause",
        "clicked_row",
        "x_column",
        "y_column",
        "row_id_column",
    )
    def _on_selection_inputs_changed(self, _change: Any) -> None:
        self._sync_validated_selection_sql()

    @traitlets.observe("selection_sql")
    def _guard_untrusted_selection_sql(self, change: Any) -> None:
        if self._updating_selection_sql:
            return
        if change["new"] != self._validated_selection_sql:
            self._updating_selection_sql = True
            try:
                self.selection_sql = self._validated_selection_sql
            finally:
                self._updating_selection_sql = False

    def _set_server_trait(self, name: str, val: Any) -> None:
        self._server_sync_values[name] = val
        self._updating_server_traits = True
        try:
            setattr(self, name, val)
        finally:
            self._updating_server_traits = False

    @traitlets.observe(
        "source_mode",
        "http_url",
        "archive_key",
        "has_sidecar",
        "has_edge_resolver",
    )
    def _guard_server_owned_traits(self, change: Any) -> None:
        if self._updating_server_traits:
            return
        name = str(change["name"])
        expected = self._server_sync_values.get(name)
        if change["new"] != expected:
            self._updating_server_traits = True
            try:
                setattr(self, name, expected)
            finally:
                self._updating_server_traits = False

    def _ensure_archive_mmap(self) -> mmap.mmap | None:
        """Lazily maps the local `.arrowtiles` archive only if `CommRangeSource` requests bytes (`I6`)."""
        if self._archive_mmap is not None:
            return self._archive_mmap
        if self._archive_path is None or self._archive_size <= 0:
            return None
        self._archive_file = self._archive_path.open("rb")
        self._archive_mmap = mmap.mmap(self._archive_file.fileno(), 0, access=mmap.ACCESS_READ)
        return self._archive_mmap

    def _bind_data(
        self,
        data: Any,
        *,
        x: str | None,
        y: str | None,
        row_id: str,
        columns: list[str] | None,
        categorical_columns: list[str] | None,
        sort_col: str | None,
        max_capacity: int,
        sidecar: Any | None,
        edges: Any | None,
        bidirectional_edges: bool,
        max_ego_neighbors: int,
        transport: str,
        cache_dir: str | os.PathLike[str] | None,
        max_cache_bytes: int,
    ) -> None:
        self._release_resources()

        resolved = resolve_dataset_input(
            data,
            x=x,
            y=y,
            row_id=row_id,
            columns=columns,
            categorical_columns=categorical_columns,
            sort_col=sort_col,
            max_capacity=max_capacity,
            sidecar=sidecar,
            cache_dir=cache_dir,
            max_cache_bytes=max_cache_bytes,
        )
        self._sidecar_store = resolved.sidecar
        self._set_server_trait("has_sidecar", self._sidecar_store is not None)
        self.x_column = resolved.x_column
        self.y_column = resolved.y_column
        self.row_id_column = resolved.row_id_column

        if edges is not None:
            self._duck_edge_resolver = DuckDbEdgeResolver(
                edges,
                nodes_sidecar=self._sidecar_store,
                x_column=self.x_column,
                y_column=self.y_column,
                row_id_column=self.row_id_column,
                bidirectional=bidirectional_edges,
                max_neighbors=max_ego_neighbors,
            )
        self._set_server_trait(
            "has_edge_resolver",
            self._duck_edge_resolver is not None or self._custom_edge_callable is not None,
        )

        if resolved.source_mode == "remote_url":
            rem_url = resolved.remote_url or ""
            self._set_server_trait("http_url", rem_url)
            self._set_server_trait("source_mode", "remote_url")
            self._set_server_trait("archive_key", f"remote:{rem_url}")
            return

        assert resolved.archive_path is not None
        apath = resolved.archive_path
        st = apath.stat()
        self._archive_path = apath
        self._archive_size = st.st_size

        if transport in ("auto", "http"):
            daemon = get_shared_range_daemon()
            tok, url = daemon.register_file(apath)
            self._http_token = tok
            self._set_server_trait("http_url", url)
            self._set_server_trait("source_mode", transport)
        elif transport == "comm":
            self._ensure_archive_mmap()
            self._set_server_trait("http_url", "")
            self._set_server_trait("source_mode", "comm")
        else:
            raise ValueError(
                f"Unsupported transport='{transport}'; expected 'auto', 'http', or 'comm'."
            )

        self._set_server_trait(
            "archive_key", f"{apath.name}:{self._archive_size}:{int(st.st_mtime_ns)}"
        )

    def _on_frontend_msg(
        self,
        _widget: Any,
        content: dict[str, Any],
        _buffers: list[Any] | None = None,
    ) -> None:
        if not isinstance(content, dict):
            return
        mtype = content.get("type")

        if mtype == "frontend_ready":
            if self._latest_row_filter is not None:
                seq, dtype, raw_bytes = self._latest_row_filter
                self._send_binary_row_filter(seq, dtype, raw_bytes)
            return

        if mtype in ("range_abort", "rpc_abort"):
            req_id = content.get("reqId")
            if isinstance(req_id, str):
                self._aborted_req_ids.add(req_id)
                if len(self._aborted_req_ids) > 512:
                    self._aborted_req_ids.clear()
            return

        if mtype == "range_req":
            req_id = str(content.get("reqId", ""))
            if req_id in self._aborted_req_ids:
                self._aborted_req_ids.discard(req_id)
                return
            raw_off = content.get("offset", 0)
            raw_len = content.get("length", 0)
            if isinstance(raw_off, bool) or isinstance(raw_len, bool):
                self.send(
                    {"type": "range_res", "reqId": req_id, "error": "Invalid range offset/length"}
                )
                return
            try:
                offset = max(0, int(raw_off))
                length = max(0, int(raw_len))
            except (TypeError, ValueError, OverflowError):
                self.send(
                    {"type": "range_res", "reqId": req_id, "error": "Invalid range offset/length"}
                )
                return
            if length > EXPLICIT_RANGE_MAX_BYTES:
                self.send(
                    {
                        "type": "range_res",
                        "reqId": req_id,
                        "error": f"Range length {length} exceeds {EXPLICIT_RANGE_MAX_BYTES} B limit",
                    }
                )
                return

            mm = self._ensure_archive_mmap()
            if mm is None:
                if self._archive_path is not None and self._archive_size == 0:
                    self.send(
                        {
                            "type": "range_res",
                            "reqId": req_id,
                            "partIndex": 0,
                            "totalParts": 1,
                            "byteOffset": 0,
                            "totalLength": 0,
                        },
                        buffers=[memoryview(b"")],
                    )
                else:
                    self.send(
                        {"type": "range_res", "reqId": req_id, "error": "No local archive mapped"}
                    )
                return

            offset = min(offset, self._archive_size)
            end = min(self._archive_size, offset + length)
            actual_len = max(0, end - offset)

            if actual_len <= COMM_CHUNK_BYTES:
                mv = memoryview(mm)[offset:end]
                try:
                    self.send(
                        {
                            "type": "range_res",
                            "reqId": req_id,
                            "partIndex": 0,
                            "totalParts": 1,
                            "byteOffset": 0,
                            "totalLength": actual_len,
                        },
                        buffers=[mv],
                    )
                finally:
                    mv.release()
                return

            total_parts = math.ceil(actual_len / COMM_CHUNK_BYTES)
            for part_idx in range(total_parts):
                if req_id in self._aborted_req_ids:
                    self._aborted_req_ids.discard(req_id)
                    return
                rel_off = part_idx * COMM_CHUNK_BYTES
                c_start = offset + rel_off
                c_end = min(end, c_start + COMM_CHUNK_BYTES)
                mv = memoryview(mm)[c_start:c_end]
                try:
                    self.send(
                        {
                            "type": "range_res",
                            "reqId": req_id,
                            "partIndex": part_idx,
                            "totalParts": total_parts,
                            "byteOffset": rel_off,
                            "totalLength": actual_len,
                        },
                        buffers=[mv],
                    )
                finally:
                    mv.release()
            return

        if mtype == "resolve_row":
            req_id = str(content.get("reqId", ""))
            if req_id in self._aborted_req_ids:
                self._aborted_req_ids.discard(req_id)
                return
            try:
                attrs = (
                    self._sidecar_store.lookup_row(content["rowId"])
                    if self._sidecar_store is not None
                    else {}
                )
                self.send({"type": "resolve_row_res", "reqId": req_id, "attributes": attrs})
            except Exception as exc:
                self.send({"type": "resolve_row_res", "reqId": req_id, "error": str(exc)})
            return

        if mtype == "resolve_edges":
            req_id = str(content.get("reqId", ""))
            if req_id in self._aborted_req_ids:
                self._aborted_req_ids.discard(req_id)
                return
            try:
                row_id = content["rowId"]
                trigger = str(content.get("trigger", "click"))
                dx = float(content.get("dataX", 0.0))
                dy = float(content.get("dataY", 0.0))
                if self._custom_edge_callable is not None:
                    edges_out = self._custom_edge_callable(row_id, trigger, dx, dy)
                elif self._duck_edge_resolver is not None:
                    edges_out = self._duck_edge_resolver.resolve_ego_specs(
                        row_id, data_x=dx, data_y=dy
                    )
                else:
                    edges_out = []
                self.send({"type": "resolve_edges_res", "reqId": req_id, "edges": edges_out})
            except Exception as exc:
                self.send({"type": "resolve_edges_res", "reqId": req_id, "error": str(exc)})

    # ------------------------------------------------------------------
    # Public Python Convenience API
    # ------------------------------------------------------------------

    def set_filter(
        self,
        *,
        rect: dict[str, float] | None = None,
        lasso: list[tuple[float, float]] | list[list[float]] | None = None,
        ranges: dict[str, tuple[float, float] | list[float]] | None = None,
        categories: dict[str, list[str | int]] | None = None,
        unselected_opacity: float = 0.0,
    ) -> None:
        """Applies a zero-SQL GPU uniform cross-filter (`Path A`). Pass all `None` to clear."""
        if rect is None and lasso is None and not ranges and not categories:
            self.filter_spec = None
            return
        spec: dict[str, Any] = {}
        if rect is not None:
            spec["rect"] = dict(rect)
        if lasso is not None:
            spec["lasso"] = {"points": [[float(p[0]), float(p[1])] for p in lasso]}
        if ranges:
            spec["ranges"] = {k: [float(v[0]), float(v[1])] for k, v in ranges.items()}
        if categories:
            spec["categories"] = {k: list(v) for k, v in categories.items()}
        if unselected_opacity > 0.0:
            spec["unselectedOpacity"] = float(unselected_opacity)
        self.filter_spec = spec

    def set_filtered_row_ids(self, ids: Any | None) -> None:
        """Streams a binary `u32`/`u64` `row_id` visibility filter (`<= 512 KiB` chunks) to WebGPU."""
        self._filter_seq += 1
        seq = self._filter_seq
        if ids is None:
            self._latest_row_filter = None
            self.send({"type": "rowid_filter_bin", "filterSeq": seq, "dtype": "clear"})
            return

        if isinstance(ids, np.ndarray):
            if np.issubdtype(ids.dtype, np.signedinteger):
                arr_u64 = np.asarray(ids, dtype=np.int64).view(np.uint64)
            else:
                arr_u64 = np.asarray(ids, dtype=np.uint64)
        elif isinstance(ids, (pa.Array, pa.ChunkedArray)):
            if pa.types.is_signed_integer(ids.type):
                arr_u64 = np.asarray(
                    ids.to_numpy(zero_copy_only=False), dtype=np.int64
                ).view(np.uint64)
            else:
                arr_u64 = np.asarray(
                    pa.compute.cast(ids, pa.uint64()).to_numpy(zero_copy_only=False),
                    dtype=np.uint64,
                )
        else:
            items = [int(v) & 0xFFFFFFFFFFFFFFFF for v in ids]
            arr_u64 = np.array(items, dtype=np.uint64)

        if self.max_filtered_row_ids > 0 and arr_u64.size > self.max_filtered_row_ids:
            raise ValueError(
                f"set_filtered_row_ids count ({arr_u64.size:,}) exceeds max_filtered_row_ids "
                f"({self.max_filtered_row_ids:,}); narrow the filter or use set_filter() GPU Path A."
            )

        if arr_u64.size == 0:
            self._latest_row_filter = (seq, "u32", b"")
            self._send_binary_row_filter(seq, "u32", b"")
            return
        if int(arr_u64.max()) <= 0xFFFFFFFF:
            raw_u32 = arr_u64.astype("<u4", copy=False).tobytes()
            self._latest_row_filter = (seq, "u32", raw_u32)
            self._send_binary_row_filter(seq, "u32", raw_u32)
        else:
            raw_u64 = arr_u64.astype("<u8", copy=False).tobytes()
            self._latest_row_filter = (seq, "u64", raw_u64)
            self._send_binary_row_filter(seq, "u64", raw_u64)

    def filter_by_sql(self, where_sql: str | None, params: Any = None) -> int:
        """Executes `SELECT row_id FROM sidecar WHERE <where_sql>` in DuckDB and pushes binary masks to WebGPU."""
        self._filter_seq += 1
        seq = self._filter_seq
        if not where_sql or not where_sql.strip():
            self._latest_row_filter = None
            self.send({"type": "rowid_filter_bin", "filterSeq": seq, "dtype": "clear"})
            return -1
        if self._sidecar_store is None:
            raise RuntimeError(
                "filter_by_sql requires a sidecar DataFrame, Arrow Table, or Parquet source."
            )
        dtype, raw_bytes = self._sidecar_store.query_row_ids_binary(
            where_sql, params=params, max_rows=self.max_filtered_row_ids
        )
        if dtype == "clear":
            self._latest_row_filter = None
            self.send({"type": "rowid_filter_bin", "filterSeq": seq, "dtype": "clear"})
            return -1
        elem_size = 8 if dtype == "u64" else 4
        count = len(raw_bytes) // elem_size
        self._latest_row_filter = (seq, dtype, raw_bytes)
        self._send_binary_row_filter(seq, dtype, raw_bytes)
        return count

    def _send_binary_row_filter(self, seq: int, dtype: str, raw_bytes: bytes) -> None:
        total_bytes = len(raw_bytes)
        if total_bytes <= COMM_CHUNK_BYTES:
            self.send(
                {
                    "type": "rowid_filter_bin",
                    "filterSeq": seq,
                    "dtype": dtype,
                    "partIndex": 0,
                    "totalParts": 1,
                    "byteOffset": 0,
                    "totalBytes": total_bytes,
                },
                buffers=[memoryview(raw_bytes)],
            )
            return
        total_parts = math.ceil(total_bytes / COMM_CHUNK_BYTES)
        mv_all = memoryview(raw_bytes)
        try:
            for part_idx in range(total_parts):
                off = part_idx * COMM_CHUNK_BYTES
                end = min(total_bytes, off + COMM_CHUNK_BYTES)
                self.send(
                    {
                        "type": "rowid_filter_bin",
                        "filterSeq": seq,
                        "dtype": dtype,
                        "partIndex": part_idx,
                        "totalParts": total_parts,
                        "byteOffset": off,
                        "totalBytes": total_bytes,
                    },
                    buffers=[mv_all[off:end]],
                )
        finally:
            mv_all.release()

    def query_selected(
        self,
        *,
        columns: str | list[str] = "*",
        limit: int | None = 1000,
    ) -> pa.Table:
        """Queries the backing DuckDB/Parquet sidecar using the server-validated `_validated_selection_sql`."""
        if self._sidecar_store is None:
            raise RuntimeError(
                "query_selected requires a sidecar DataFrame, Arrow Table, or Parquet source."
            )
        sql = self._validated_selection_sql
        if not sql:
            return self._sidecar_store.query_where("FALSE", columns=columns, limit=0)
        return self._sidecar_store.query_where(sql, columns=columns, limit=limit)

    def query_clicked_neighbors(self) -> pa.Table | None:
        """Returns the exact 1-hop DuckDB neighbor rows + attributes for the currently clicked node."""
        if self._duck_edge_resolver is None or not isinstance(self.clicked_row, dict):
            return None
        rid = self.clicked_row.get("rowId")
        if rid is None or isinstance(rid, bool):
            return None
        dx = self.clicked_row.get("dataX")
        dy = self.clicked_row.get("dataY")
        try:
            fx = float(dx) if dx is not None and not isinstance(dx, bool) else None
            fy = float(dy) if dy is not None and not isinstance(dy, bool) else None
            return self._duck_edge_resolver.query_neighbors_arrow(rid, data_x=fx, data_y=fy)
        except (TypeError, ValueError, OverflowError):
            return None

    def set_edges(
        self,
        edges: list[dict[str, Any]],
        style: dict[str, Any] | None = None,
    ) -> None:
        if style is not None:
            self.edge_style = dict(style)
        self.static_edges = list(edges)

    def fit_view(self) -> None:
        self.send({"type": "fit_view"})

    def set_viewport(
        self,
        *,
        x: float | None = None,
        y: float | None = None,
        zoom: float | None = None,
    ) -> None:
        vp: dict[str, float] = {}
        if x is not None:
            vp["x"] = float(x)
        if y is not None:
            vp["y"] = float(y)
        if zoom is not None:
            vp["zoom"] = float(zoom)
        if vp:
            self.send({"type": "set_viewport", "viewport": vp})

    def clear_selection(self) -> None:
        self.selected_row_ids = []
        self.selected_row_ids_count = 0
        self.selected_row_ids_truncated = False
        self.clicked_row = None
        self.selection_clause = {"kind": "none", "xField": None, "yField": None}
        self._sync_validated_selection_sql()
        self.send({"type": "clear_selection"})

    def _release_resources(self) -> None:
        if self._http_token is not None:
            try:
                get_shared_range_daemon().unregister_token(self._http_token)
            except Exception:
                pass
            self._http_token = None
        if self._archive_mmap is not None:
            try:
                self._archive_mmap.close()
            except BufferError:
                pass
            self._archive_mmap = None
        if self._archive_file is not None:
            try:
                self._archive_file.close()
            except Exception:
                pass
            self._archive_file = None
        self._archive_path = None
        self._latest_row_filter = None
        if self._duck_edge_resolver is not None:
            self._duck_edge_resolver.close()
            self._duck_edge_resolver = None
        if self._sidecar_store is not None:
            self._sidecar_store.close()
            self._sidecar_store = None

    def close(self) -> None:
        self._release_resources()
        super().close()

    def __enter__(self) -> DeepgraphWidget:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._release_resources()
        except Exception:
            pass


def plot(data: Any = None, **kwargs: Any) -> DeepgraphWidget:
    """Create and configure a WebGPU `DeepgraphWidget` from an `.arrowtiles` path, URL, Parquet file, or DataFrame."""
    return DeepgraphWidget(data, **kwargs)
