from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import narwhals as nw
import pyarrow as pa

from ._ingest import (
    MAX_JS_SAFE_INT,
    SidecarStore,
    _json_safe_value,
    _to_arrow_table,
    quote_sql_ident,
)

SPATIAL_KNN_MODES = frozenset({"spatial_knn", "knn"})


def _format_js_row_id(val: int) -> int | str:
    return val if abs(val) <= MAX_JS_SAFE_INT else str(val)


class DuckDbEdgeResolver:
    """
    Indexed DuckDB 1-hop ego-network / exact 2D spatial kNN resolver (`< 3 ms` per hover/click).

    Supports three backing modes:
    1. `edges="spatial_knn"` (or `"knn"`): Provably exact inscribed-ball 2D Euclidean kNN over
       `(x, y)`-sorted `_dg_nodes` (`0` precomputed edge storage, `0` misses across millions of rows).
    2. Pre-joined coordinate columns (`source_x`, `source_y`, `target_x`, `target_y`) on an edge table.
    3. Joining `(source_col, target_col)` against a node `SidecarStore` carrying `(row_id, x, y, ...)`.

    Supports the full `.arrowtiles v0.1` `UInt64` (`0 .. 2^64 - 1`) `row_id` range via `UBIGINT`
    as well as signed negative `BIGINT` IDs when present.
    """

    def __init__(
        self,
        edges: Any,
        *,
        nodes_sidecar: SidecarStore | None = None,
        x_column: str = "x",
        y_column: str = "y",
        row_id_column: str = "row_id",
        source_col: str = "source_row_id",
        target_col: str = "target_row_id",
        weight_col: str | None = "weight",
        bidirectional: bool = False,
        max_neighbors: int = 64,
    ) -> None:
        self.source_col = source_col
        self.target_col = target_col
        self.weight_col = weight_col
        self.bidirectional = bidirectional
        self.max_neighbors = max(1, int(max_neighbors))
        self.x_column = x_column
        self.y_column = y_column
        self.row_id_column = row_id_column

        self._lock = threading.Lock()
        self._con = duckdb.connect(":memory:")
        self._is_spatial_knn = isinstance(edges, str) and edges.strip().lower() in SPATIAL_KNN_MODES
        self._has_inline_xy = False
        self._has_node_table = False
        self._id_sql_type = "UBIGINT"
        self._node_attr_cols: list[str] = []
        self._hot_node_attr_cols: list[str] = []
        self._nodes_sidecar: SidecarStore | None = nodes_sidecar
        self._spatial_diag = 100.0
        try:
            self._setup_tables(edges, nodes_sidecar)
        except Exception:
            self.close()
            raise

    def _bind_id_param(self, rid: int) -> Any:
        if isinstance(rid, bool):
            return None
        if self._id_sql_type == "UBIGINT":
            if rid < 0 or rid > 0xFFFFFFFFFFFFFFFF:
                return None
            return str(rid)
        if rid < -(1 << 63) or rid >= (1 << 63):
            return None
        return rid

    def _id_placeholder(self) -> str:
        return f"CAST(? AS {self._id_sql_type})"

    def _materialize_nodes_table(
        self,
        nodes_sidecar: SidecarStore,
        *,
        sort_by_xy: bool,
    ) -> None:
        # Stream directly from Parquet in DuckDB C++ when available (saves duplicating 2.84M rows in PyArrow RAM!)
        registered_arrow = False
        if nodes_sidecar.parquet_path is not None and nodes_sidecar.parquet_path.is_file():
            pq_esc = str(nodes_sidecar.parquet_path).replace("'", "''")
            nodes_src_expr = f"read_parquet('{pq_esc}')"
            node_cols = nodes_sidecar.columns
        else:
            nodes_tbl = nodes_sidecar.query_where(None, limit=None)
            self._con.register("_dg_nodes_arrow", nodes_tbl)
            registered_arrow = True
            nodes_src_expr = "_dg_nodes_arrow"
            node_cols = list(nodes_tbl.column_names)

        try:
            if self.x_column not in node_cols or self.y_column not in node_cols:
                raise ValueError(
                    f"Node sidecar missing coordinates ({self.x_column}, {self.y_column}); found {node_cols}"
                )
            self._node_attr_cols = [
                c for c in node_cols if c not in {self.row_id_column, self.x_column, self.y_column}
            ]
            q_rid = quote_sql_ident(self.row_id_column)
            q_x = quote_sql_ident(self.x_column)
            q_y = quote_sql_ident(self.y_column)

            rid_type_row = self._con.sql(f"SELECT {q_rid} FROM {nodes_src_expr} LIMIT 0").types[0]
            if str(rid_type_row).upper().startswith("U"):
                self._id_sql_type = "UBIGINT"
            else:
                min_row = self._con.execute(f"SELECT min({q_rid}) FROM {nodes_src_expr}").fetchone()
                has_neg = min_row is not None and min_row[0] is not None and int(min_row[0]) < 0
                self._id_sql_type = "BIGINT" if has_neg else "UBIGINT"

            extra_sel = (
                ", " + ", ".join(quote_sql_ident(c) for c in self._node_attr_cols)
                if self._node_attr_cols
                else ""
            )
            order_sql = " ORDER BY nx, ny" if sort_by_xy else ""
            self._con.execute(
                f"CREATE TABLE _dg_nodes AS "
                f"SELECT CAST({q_rid} AS {self._id_sql_type}) AS node_id, "
                f"CAST({q_x} AS DOUBLE) AS nx, "
                f"CAST({q_y} AS DOUBLE) AS ny"
                f"{extra_sel} FROM {nodes_src_expr}{order_sql}"
            )
        finally:
            if registered_arrow:
                self._con.unregister("_dg_nodes_arrow")

        self._con.execute("CREATE INDEX idx_dg_nodes_id ON _dg_nodes (node_id)")
        if sort_by_xy:
            self._con.execute("CREATE INDEX idx_dg_nodes_nx ON _dg_nodes (nx)")
        self._has_node_table = True

        if sort_by_xy:
            b_row = self._con.execute(
                "SELECT min(nx), max(nx), min(ny), max(ny) FROM _dg_nodes WHERE isfinite(nx) AND isfinite(ny)"
            ).fetchone()
            if b_row and all(v is not None for v in b_row):
                span_x = max(1e-6, float(b_row[1]) - float(b_row[0]))
                span_y = max(1e-6, float(b_row[3]) - float(b_row[2]))
                self._spatial_diag = math.hypot(span_x, span_y)

    def _setup_tables(self, edges: Any, nodes_sidecar: SidecarStore | None) -> None:
        if self._is_spatial_knn:
            if nodes_sidecar is None:
                raise ValueError(
                    "edges='spatial_knn' requires sidecar=<nodes_table_or_parquet> with (row_id, x, y)."
                )
            self._materialize_nodes_table(nodes_sidecar, sort_by_xy=True)
            return

        if isinstance(edges, (str, os.PathLike)):
            ep = str(Path(edges).resolve()).replace("'", "''")
            raw_edges_expr = f"read_parquet('{ep}')"
        elif isinstance(edges, duckdb.DuckDBPyRelation):
            self._con.register("_dg_edges_in", _to_arrow_table(edges))
            raw_edges_expr = "_dg_edges_in"
        elif isinstance(edges, pa.Table):
            self._con.register("_dg_edges_in", edges)
            raw_edges_expr = "_dg_edges_in"
        elif isinstance(edges, list):
            self._con.register("_dg_edges_in", pa.Table.from_pylist(edges))
            raw_edges_expr = "_dg_edges_in"
        else:
            nw_df = nw.from_native(edges, pass_through=False)
            if isinstance(nw_df, nw.LazyFrame):
                nw_df = nw_df.collect()
            self._con.register("_dg_edges_in", nw_df.to_arrow())
            raw_edges_expr = "_dg_edges_in"

        edge_rel = self._con.sql(f"SELECT * FROM {raw_edges_expr} LIMIT 0")
        edge_cols = set(edge_rel.columns)
        if self.source_col not in edge_cols or self.target_col not in edge_cols:
            for sc, tc in [("source", "target"), ("src", "dst"), ("u", "v")]:
                if sc in edge_cols and tc in edge_cols:
                    self.source_col = sc
                    self.target_col = tc
                    break
            else:
                raise ValueError(
                    f"Edge table missing ({self.source_col}, {self.target_col}); found {sorted(edge_cols)}"
                )

        q_src = quote_sql_ident(self.source_col)
        q_dst = quote_sql_ident(self.target_col)
        w_expr = (
            f"CAST({quote_sql_ident(self.weight_col)} AS DOUBLE)"
            if self.weight_col and self.weight_col in edge_cols
            else "1.0"
        )
        has_inline_xy = {"source_x", "source_y", "target_x", "target_y"}.issubset(edge_cols)

        if has_inline_xy:
            src_t = str(self._con.sql(f"SELECT {q_src} FROM {raw_edges_expr} LIMIT 0").types[0]).upper()
            dst_t = str(self._con.sql(f"SELECT {q_dst} FROM {raw_edges_expr} LIMIT 0").types[0]).upper()
            if src_t.startswith("U") and dst_t.startswith("U"):
                self._id_sql_type = "UBIGINT"
            else:
                min_e = self._con.execute(
                    f"SELECT least(min({q_src}), min({q_dst})) FROM {raw_edges_expr}"
                ).fetchone()
                has_neg = min_e is not None and min_e[0] is not None and int(min_e[0]) < 0
                self._id_sql_type = "BIGINT" if has_neg else "UBIGINT"

            self._con.execute(
                f"CREATE TABLE _dg_edges AS "
                f"SELECT CAST({q_src} AS {self._id_sql_type}) AS src_id, "
                f"CAST({q_dst} AS {self._id_sql_type}) AS dst_id, "
                f"{w_expr} AS weight, "
                f"CAST(source_x AS DOUBLE) AS sx, CAST(source_y AS DOUBLE) AS sy, "
                f"CAST(target_x AS DOUBLE) AS tx, CAST(target_y AS DOUBLE) AS ty "
                f"FROM {raw_edges_expr}"
            )
            self._has_inline_xy = True
        else:
            if nodes_sidecar is None:
                raise ValueError(
                    "When edge table omits (source_x, source_y, target_x, target_y), "
                    "pass sidecar=<nodes_table_or_parquet> with x_column and y_column."
                )
            self._materialize_nodes_table(nodes_sidecar, sort_by_xy=False)

            self._con.execute(
                f"CREATE TABLE _dg_edges AS "
                f"SELECT src_id, dst_id, weight FROM ("
                f"  SELECT TRY_CAST({q_src} AS {self._id_sql_type}) AS src_id, "
                f"  TRY_CAST({q_dst} AS {self._id_sql_type}) AS dst_id, "
                f"  {w_expr} AS weight "
                f"  FROM {raw_edges_expr}"
                f") sub WHERE src_id IS NOT NULL AND dst_id IS NOT NULL"
            )

        self._con.execute("CREATE INDEX idx_dg_edges_src ON _dg_edges (src_id)")
        if self.bidirectional:
            self._con.execute("CREATE INDEX idx_dg_edges_dst ON _dg_edges (dst_id)")

    def _query_spatial_knn_locked(
        self,
        rid: int,
        *,
        data_x: float | None = None,
        data_y: float | None = None,
    ) -> tuple[float, float, dict[str, Any], list[tuple[Any, ...]], float] | None:
        r"""
        Executes provably exact inscribed-ball 2D Euclidean kNN over `(nx, ny)`-sorted `_dg_nodes`.

        If the $k$-th candidate inside `[qx - r, qx + r] x [qy - r, qy + r]` has $d_k \le r$, no point
        outside the box can have Euclidean distance $< r$, proving global optimality; if $d_k > r$
        (box-corner case $r < d_k \le \sqrt{2}r$), a single refinement at $r_{\text{exact}} = d_k \cdot (1 + 10^{-6})$
        captures 100% of points within the closed Euclidean ball of radius $d_k$.
        """
        bound_id = self._bind_id_param(rid)
        if bound_id is None:
            return None
        ph = self._id_placeholder()
        attr_sel = (
            ", " + ", ".join(quote_sql_ident(c) for c in self._node_attr_cols)
            if self._node_attr_cols
            else ""
        )

        hub_rows = self._con.execute(
            f"SELECT nx, ny{attr_sel} FROM _dg_nodes WHERE node_id = {ph} LIMIT 1",
            [bound_id],
        ).fetchall()
        src_attrs: dict[str, Any] = {}
        if hub_rows:
            hr = hub_rows[0]
            qx = float(hr[0]) if hr[0] is not None else float(data_x or 0.0)
            qy = float(hr[1]) if hr[1] is not None else float(data_y or 0.0)
            for idx, cname in enumerate(self._node_attr_cols):
                src_attrs[cname] = _json_safe_value(hr[2 + idx])
        else:
            if data_x is None or data_y is None:
                return None
            qx = float(data_x)
            qy = float(data_y)

        if not (math.isfinite(qx) and math.isfinite(qy)):
            return None

        k = self.max_neighbors
        box_sql = (
            f"SELECT node_id, nx, ny, "
            f"(nx - ?) * (nx - ?) + (ny - ?) * (ny - ?) AS d2{attr_sel} "
            f"FROM _dg_nodes "
            f"WHERE nx BETWEEN ? AND ? AND ny BETWEEN ? AND ? "
            f"AND node_id <> {ph} AND isfinite(nx) AND isfinite(ny) "
            f"ORDER BY d2 ASC, node_id ASC LIMIT {k}"
        )

        diag = max(1e-6, self._spatial_diag)
        radii = (
            diag * 0.00015,
            diag * 0.0015,
            diag * 0.006,
            diag * 0.025,
            diag * 0.12,
            diag * 2.5,
        )
        cand: list[tuple[Any, ...]] = []
        used_r = radii[-1]
        for r in radii:
            used_r = r
            cand = self._con.execute(
                box_sql,
                [qx, qx, qy, qy, qx - r, qx + r, qy - r, qy + r, bound_id],
            ).fetchall()
            if len(cand) >= k:
                break

        if cand:
            max_d2 = float(cand[-1][3])
            if max_d2 > used_r * used_r and used_r < diag * 2.0:
                r_exact = math.sqrt(max_d2) * 1.000001
                used_r = r_exact
                cand = self._con.execute(
                    box_sql,
                    [
                        qx,
                        qx,
                        qy,
                        qy,
                        qx - r_exact,
                        qx + r_exact,
                        qy - r_exact,
                        qy + r_exact,
                        bound_id,
                    ],
                ).fetchall()

        return (qx, qy, src_attrs, cand, used_r)

    def resolve_ego_specs(
        self,
        row_id: int | str,
        *,
        data_x: float | None = None,
        data_y: float | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(row_id, bool):
            return []
        rid = int(row_id)

        def _compact_ego_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
            # Omit multi-KB cold text strings on 45ms hover JSON (`EdgeOverlay` only reads chart-axis values)
            return {
                k: v for k, v in attrs.items() if not (isinstance(v, str) and len(v) > 128)
            }

        if self._is_spatial_knn:
            with self._lock:
                res = self._query_spatial_knn_locked(rid, data_x=data_x, data_y=data_y)
            if res is None:
                return []
            qx, qy, src_attrs, cand, _ = res
            compact_src = _compact_ego_attrs(src_attrs)
            src_id_js = _format_js_row_id(rid)
            out_specs: list[dict[str, Any]] = []
            for r in cand:
                tid = int(r[0])
                tx = float(r[1])
                ty = float(r[2])
                if not (math.isfinite(tx) and math.isfinite(ty)):
                    continue
                dist = math.sqrt(max(0.0, float(r[3])))
                weight = 1.0 / (1.0 + dist) if math.isfinite(dist) else 1.0
                t_attrs = _compact_ego_attrs(
                    {
                        cname: _json_safe_value(r[4 + idx])
                        for idx, cname in enumerate(self._node_attr_cols)
                    }
                )
                out_specs.append(
                    {
                        "sourceRowId": src_id_js,
                        "targetRowId": _format_js_row_id(tid),
                        "weight": weight,
                        "source": {"x": qx, "y": qy},
                        "target": {"x": tx, "y": ty},
                        "sourceAttributes": dict(compact_src),
                        "targetAttributes": t_attrs,
                        "directed": True,
                    }
                )
            return out_specs

        bound_id = self._bind_id_param(rid)
        if bound_id is None:
            return []
        ph = self._id_placeholder()

        if self._has_inline_xy:
            if not self.bidirectional:
                sql = (
                    f"SELECT src_id, dst_id, weight, sx, sy, tx, ty "
                    f"FROM _dg_edges WHERE src_id = {ph} ORDER BY weight DESC LIMIT {self.max_neighbors}"
                )
                params = [bound_id]
            else:
                sql = (
                    f"SELECT src_id, dst_id, weight, sx, sy, tx, ty FROM ("
                    f"  SELECT src_id, dst_id, weight, sx, sy, tx, ty FROM _dg_edges WHERE src_id = {ph} "
                    f"  UNION ALL "
                    f"  SELECT dst_id AS src_id, src_id AS dst_id, weight, tx AS sx, ty AS sy, sx AS tx, sy AS ty "
                    f"  FROM _dg_edges WHERE dst_id = {ph} AND src_id <> {ph}"
                    f") ORDER BY weight DESC LIMIT {self.max_neighbors}"
                )
                params = [bound_id, bound_id, bound_id]
            with self._lock:
                rows = self._con.execute(sql, params).fetchall()
            inline_specs: list[dict[str, Any]] = []
            for r in rows:
                if r[3] is None or r[4] is None or r[5] is None or r[6] is None:
                    continue
                sx, sy, tx, ty = float(r[3]), float(r[4]), float(r[5]), float(r[6])
                if not all(math.isfinite(v) for v in (sx, sy, tx, ty)):
                    continue
                w_raw = float(r[2]) if r[2] is not None else 1.0
                w_safe = w_raw if math.isfinite(w_raw) else 1.0
                inline_specs.append(
                    {
                        "sourceRowId": _format_js_row_id(int(r[0])),
                        "targetRowId": _format_js_row_id(int(r[1])),
                        "weight": w_safe,
                        "source": {"x": sx, "y": sy},
                        "target": {"x": tx, "y": ty},
                        "directed": not self.bidirectional,
                    }
                )
            return inline_specs

        extra_src = ", ".join(f"s.{quote_sql_ident(c)}" for c in self._node_attr_cols)
        extra_dst = ", ".join(f"t.{quote_sql_ident(c)}" for c in self._node_attr_cols)
        attr_select = f", {extra_src}, {extra_dst}" if self._node_attr_cols else ""

        if not self.bidirectional:
            ego_rel = f"SELECT src_id, dst_id, weight FROM _dg_edges WHERE src_id = {ph}"
            params = [bound_id]
        else:
            ego_rel = (
                f"SELECT src_id, dst_id, weight FROM _dg_edges WHERE src_id = {ph} "
                f"UNION ALL "
                f"SELECT dst_id AS src_id, src_id AS dst_id, weight FROM _dg_edges WHERE dst_id = {ph} AND src_id <> {ph}"
            )
            params = [bound_id, bound_id, bound_id]

        sql = (
            f"SELECT e.src_id, e.dst_id, e.weight, s.nx, s.ny, t.nx, t.ny{attr_select} "
            f"FROM ({ego_rel}) e "
            f"JOIN _dg_nodes s ON s.node_id = e.src_id "
            f"JOIN _dg_nodes t ON t.node_id = e.dst_id "
            f"ORDER BY e.weight DESC LIMIT {self.max_neighbors}"
        )

        with self._lock:
            rows = self._con.execute(sql, params).fetchall()
        k = len(self._node_attr_cols)
        specs: list[dict[str, Any]] = []
        for r in rows:
            if r[5] is None or r[6] is None:
                continue
            sx = float(r[3]) if r[3] is not None else float(data_x if data_x is not None else 0.0)
            sy = float(r[4]) if r[4] is not None else float(data_y if data_y is not None else 0.0)
            tx = float(r[5])
            ty = float(r[6])
            if not all(math.isfinite(v) for v in (sx, sy, tx, ty)):
                continue
            w_raw = float(r[2]) if r[2] is not None else 1.0
            w_safe = w_raw if math.isfinite(w_raw) else 1.0
            s_attrs: dict[str, Any] = {}
            t_attrs: dict[str, Any] = {}
            if k > 0:
                for idx, cname in enumerate(self._node_attr_cols):
                    s_val = _json_safe_value(r[7 + idx])
                    t_val = _json_safe_value(r[7 + k + idx])
                    if not (isinstance(s_val, str) and len(s_val) > 128):
                        s_attrs[cname] = s_val
                    if not (isinstance(t_val, str) and len(t_val) > 128):
                        t_attrs[cname] = t_val
            specs.append(
                {
                    "sourceRowId": _format_js_row_id(int(r[0])),
                    "targetRowId": _format_js_row_id(int(r[1])),
                    "weight": w_safe,
                    "source": {"x": sx, "y": sy},
                    "target": {"x": tx, "y": ty},
                    "sourceAttributes": s_attrs,
                    "targetAttributes": t_attrs,
                    "directed": not self.bidirectional,
                }
            )
        return specs

    def query_neighbors_arrow(
        self,
        row_id: int | str,
        *,
        data_x: float | None = None,
        data_y: float | None = None,
    ) -> pa.Table:
        """Returns a PyArrow Table of 1-hop neighbors + node attributes for live Marimo table verification."""
        id_pa_type = pa.uint64() if self._id_sql_type == "UBIGINT" else pa.int64()
        if isinstance(row_id, bool):
            bound_id = None
            rid = -1
        else:
            rid = int(row_id)
            bound_id = self._bind_id_param(rid)
        ph = self._id_placeholder()

        if self._is_spatial_knn:
            if bound_id is None:
                return pa.table(
                    {
                        "source_row_id": pa.array([], type=id_pa_type),
                        "target_row_id": pa.array([], type=id_pa_type),
                        "distance": pa.array([], type=pa.float64()),
                        "weight": pa.array([], type=pa.float64()),
                        "target_x": pa.array([], type=pa.float64()),
                        "target_y": pa.array([], type=pa.float64()),
                    }
                )
            extra_cols = (
                ", " + ", ".join(quote_sql_ident(c) for c in self._node_attr_cols)
                if self._node_attr_cols
                else ""
            )
            with self._lock:
                res = self._query_spatial_knn_locked(rid, data_x=data_x, data_y=data_y)
                if res is None or not res[3]:
                    return pa.table(
                        {
                            "source_row_id": pa.array([], type=id_pa_type),
                            "target_row_id": pa.array([], type=id_pa_type),
                            "distance": pa.array([], type=pa.float64()),
                            "weight": pa.array([], type=pa.float64()),
                            "target_x": pa.array([], type=pa.float64()),
                            "target_y": pa.array([], type=pa.float64()),
                        }
                    )
                qx, qy, _, _, final_r = res
                # Stream native Arrow columns directly from DuckDB (`0` `_json_safe_value` int/str mixing!)
                arrow_sql = (
                    f"SELECT "
                    f"{ph} AS source_row_id, "
                    f"node_id AS target_row_id, "
                    f"ROUND(sqrt(greatest(0.0, d2)), 5) AS distance, "
                    f"ROUND(1.0 / (1.0 + sqrt(greatest(0.0, d2))), 4) AS weight, "
                    f"nx AS target_x, "
                    f"ny AS target_y"
                    f"{extra_cols} "
                    f"FROM ("
                    f"  SELECT node_id, nx, ny, "
                    f"  (nx - ?) * (nx - ?) + (ny - ?) * (ny - ?) AS d2{extra_cols} "
                    f"  FROM _dg_nodes "
                    f"  WHERE nx BETWEEN ? AND ? AND ny BETWEEN ? AND ? "
                    f"  AND node_id <> {ph} AND isfinite(nx) AND isfinite(ny) "
                    f"  ORDER BY d2 ASC, node_id ASC LIMIT {self.max_neighbors}"
                    f") sub"
                )
                return _to_arrow_table(
                    self._con.sql(
                        arrow_sql,
                        params=[
                            bound_id,
                            qx,
                            qx,
                            qy,
                            qy,
                            qx - final_r,
                            qx + final_r,
                            qy - final_r,
                            qy + final_r,
                            bound_id,
                        ],
                    )
                )

        if bound_id is None:
            return pa.table(
                {
                    "source_row_id": pa.array([], type=id_pa_type),
                    "target_row_id": pa.array([], type=id_pa_type),
                    "weight": pa.array([], type=pa.float64()),
                    "target_x": pa.array([], type=pa.float64()),
                    "target_y": pa.array([], type=pa.float64()),
                }
            )

        if self._has_inline_xy or not self._has_node_table:
            if not self.bidirectional:
                sql = (
                    f"SELECT src_id AS source_row_id, dst_id AS target_row_id, weight, tx AS target_x, ty AS target_y "
                    f"FROM _dg_edges WHERE src_id = {ph} ORDER BY weight DESC LIMIT {self.max_neighbors}"
                )
                params = [bound_id]
            else:
                sql = (
                    f"SELECT src_id AS source_row_id, dst_id AS target_row_id, weight, tx AS target_x, ty AS target_y FROM ("
                    f"  SELECT src_id, dst_id, weight, tx, ty FROM _dg_edges WHERE src_id = {ph} "
                    f"  UNION ALL "
                    f"  SELECT dst_id AS src_id, src_id AS dst_id, weight, sx AS tx, sy AS ty "
                    f"  FROM _dg_edges WHERE dst_id = {ph} AND src_id <> {ph}"
                    f") ORDER BY weight DESC LIMIT {self.max_neighbors}"
                )
                params = [bound_id, bound_id, bound_id]
            with self._lock:
                return _to_arrow_table(self._con.sql(sql, params=params))

        extra_dst = (
            ", " + ", ".join(f"t.{quote_sql_ident(c)}" for c in self._node_attr_cols)
            if self._node_attr_cols
            else ""
        )
        if not self.bidirectional:
            ego_rel = f"SELECT src_id, dst_id, weight FROM _dg_edges WHERE src_id = {ph}"
            params = [bound_id]
        else:
            ego_rel = (
                f"SELECT src_id, dst_id, weight FROM _dg_edges WHERE src_id = {ph} "
                f"UNION ALL "
                f"SELECT dst_id AS src_id, src_id AS dst_id, weight FROM _dg_edges WHERE dst_id = {ph} AND src_id <> {ph}"
            )
            params = [bound_id, bound_id, bound_id]

        with self._lock:
            return _to_arrow_table(
                self._con.sql(
                    f"SELECT e.src_id AS source_row_id, e.dst_id AS target_row_id, "
                    f"ROUND(e.weight, 4) AS weight, t.nx AS target_x, t.ny AS target_y{extra_dst} "
                    f"FROM ({ego_rel}) e "
                    f"JOIN _dg_nodes t ON t.node_id = e.dst_id "
                    f"ORDER BY e.weight DESC LIMIT {self.max_neighbors}",
                    params=params,
                )
            )

    def close(self) -> None:
        with self._lock:
            try:
                self._con.close()
            except Exception:
                pass


EdgeResolverCallable = Callable[[int | str, str, float, float], list[dict[str, Any]]]
