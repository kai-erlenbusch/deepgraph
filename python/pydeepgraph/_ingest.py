from __future__ import annotations

import decimal
import hashlib
import json
import math
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import arrowtiles
import duckdb
import narwhals as nw
import numpy as np
import pyarrow as pa

from ._http_range import get_shared_range_daemon_if_running

DEFAULT_MAX_CACHE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB LRU cap
DEFAULT_MAX_FILTERED_ROW_IDS = 10_000_000  # 10M binary row_id safety ceiling
MAX_JS_SAFE_INT = 9007199254740991

_OPEN_SIDECAR_LOCK = threading.Lock()
_OPEN_SIDECAR_PATHS: dict[Path, int] = {}


def quote_sql_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def get_cache_dir(custom_dir: str | os.PathLike[str] | None = None) -> Path:
    raw = custom_dir or os.environ.get("PYDEEPGRAPH_CACHE_DIR")
    base = Path(raw) if raw else (Path(tempfile.gettempdir()) / "pydeepgraph_cache")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _to_arrow_table(obj: Any) -> pa.Table:
    if hasattr(obj, "to_arrow_table"):
        return obj.to_arrow_table()
    return obj.fetch_arrow_table()


def _prune_cache_dir(
    cache_dir: Path,
    max_bytes: int = DEFAULT_MAX_CACHE_BYTES,
    *,
    protect: Path | None = None,
) -> None:
    protected_stems: set[Path] = set()
    if protect is not None:
        protected_stems.add(protect.resolve().with_suffix(""))
    with _OPEN_SIDECAR_LOCK:
        for open_pq in _OPEN_SIDECAR_PATHS:
            protected_stems.add(open_pq.with_suffix(""))
    daemon = get_shared_range_daemon_if_running()
    if daemon is not None:
        for open_at in daemon.active_paths():
            protected_stems.add(open_at.with_suffix(""))

    files: list[tuple[float, int, Path]] = []
    total = 0
    for p in cache_dir.glob("*.arrowtiles"):
        try:
            st = p.stat()
            pq_path = p.with_suffix(".parquet")
            pq_sz = pq_path.stat().st_size if pq_path.is_file() else 0
            pair_sz = st.st_size + pq_sz
            files.append((st.st_mtime, pair_sz, p))
            total += pair_sz
        except OSError:
            continue
    if total <= max_bytes:
        return
    files.sort(key=lambda item: item[0])  # oldest mtime first
    for _, sz, p in files:
        if total <= max_bytes:
            break
        if p.resolve().with_suffix("") in protected_stems:
            continue
        try:
            p.unlink(missing_ok=True)
            p.with_suffix(".parquet").unlink(missing_ok=True)
            total -= sz
        except OSError:
            continue


def _json_safe_value(val: Any) -> Any:
    if val is None or isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val if abs(val) <= MAX_JS_SAFE_INT else str(val)
    if isinstance(val, str):
        return val
    if isinstance(val, float):
        return val if math.isfinite(val) else None
    if isinstance(val, decimal.Decimal):
        f = float(val)
        return f if math.isfinite(f) else str(val)
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    if isinstance(val, (bytes, bytearray, memoryview)):
        return bytes(val).decode("utf-8", errors="replace")
    return str(val)


class SidecarStore:
    """DuckDB-backed row/predicate sidecar for cold-column click enrichment & SQL row_id masks."""

    def __init__(
        self,
        source: Any,
        *,
        row_id_column: str = "row_id",
    ) -> None:
        self.row_id_column = row_id_column
        self._lock = threading.Lock()
        self._con = duckdb.connect(":memory:")
        self._table_expr: str
        self._columns: list[str] = []
        self._rid_is_unsigned: bool = False
        self.parquet_path: Path | None = None
        self._closed: bool = False
        try:
            self._register_source(source)
        except Exception:
            self.close()
            raise
        if self.parquet_path is not None:
            with _OPEN_SIDECAR_LOCK:
                _OPEN_SIDECAR_PATHS[self.parquet_path] = (
                    _OPEN_SIDECAR_PATHS.get(self.parquet_path, 0) + 1
                )

    def _register_source(self, source: Any) -> None:
        if isinstance(source, (str, os.PathLike)):
            resolved_pq = Path(source).resolve()
            self.parquet_path = resolved_pq
            p = str(resolved_pq).replace("'", "''")
            self._table_expr = f"read_parquet('{p}')"
        elif isinstance(source, duckdb.DuckDBPyRelation):
            arrow_tbl = _to_arrow_table(source)
            self._con.register("_dg_sidecar_arrow", arrow_tbl)
            self._table_expr = "_dg_sidecar_arrow"
        elif isinstance(source, pa.Table):
            self._con.register("_dg_sidecar_arrow", source)
            self._table_expr = "_dg_sidecar_arrow"
        else:
            nw_frame = nw.from_native(source, pass_through=False)
            if isinstance(nw_frame, nw.LazyFrame):
                nw_frame = nw_frame.collect()
            arrow_tbl = nw_frame.to_arrow()
            self._con.register("_dg_sidecar_arrow", arrow_tbl)
            self._table_expr = "_dg_sidecar_arrow"

        schema_rel = self._con.sql(f"SELECT * FROM {self._table_expr} LIMIT 0")
        self._columns = list(schema_rel.columns)
        col_types = {c: str(t).upper() for c, t in zip(schema_rel.columns, schema_rel.types, strict=False)}
        if self.row_id_column not in self._columns:
            q_rid = quote_sql_ident(self.row_id_column)
            self._con.execute(
                f"CREATE TABLE _dg_sidecar_mat AS "
                f"SELECT (row_number() OVER () - 1)::UBIGINT AS {q_rid}, * FROM {self._table_expr}"
            )
            self._table_expr = "_dg_sidecar_mat"
            self._columns = [self.row_id_column, *self._columns]
            self._rid_is_unsigned = True
        else:
            rid_t = col_types.get(self.row_id_column, "")
            self._rid_is_unsigned = rid_t.startswith("U")

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    def lookup_row(self, row_id: int | str) -> dict[str, Any]:
        if isinstance(row_id, bool):
            return {}
        q_rid = quote_sql_ident(self.row_id_column)
        rid_val = int(row_id)
        if self._rid_is_unsigned:
            if rid_val < 0 or rid_val > 0xFFFFFFFFFFFFFFFF:
                return {}
            sql = f"SELECT * FROM {self._table_expr} WHERE {q_rid} = CAST(? AS UBIGINT) LIMIT 1"
            bind_params: list[Any] = [str(rid_val)]
        else:
            if rid_val < -(1 << 63) or rid_val >= (1 << 63):
                return {}
            sql = f"SELECT * FROM {self._table_expr} WHERE {q_rid} = CAST(? AS BIGINT) LIMIT 1"
            bind_params = [rid_val]

        with self._lock:
            cur = self._con.execute(sql, bind_params)
            rows = cur.fetchall()
            desc = cur.description
        if not rows or desc is None:
            return {}
        cols = [d[0] for d in desc]
        return {col: _json_safe_value(val) for col, val in zip(cols, rows[0], strict=False)}

    def query_where(
        self,
        where_sql: str | None = None,
        *,
        params: Any = None,
        columns: str | list[str] = "*",
        limit: int | None = 1000,
    ) -> pa.Table:
        if columns == "*":
            col_sql = "*"
        elif isinstance(columns, str):
            col_sql = quote_sql_ident(columns)
        else:
            col_sql = ", ".join(quote_sql_ident(c) for c in columns)
        where_clause = f" WHERE ({where_sql})" if where_sql and where_sql.strip() else ""
        limit_clause = f" LIMIT {int(limit)}" if limit is not None else ""
        sql = f"SELECT {col_sql} FROM {self._table_expr}{where_clause}{limit_clause}"
        with self._lock:
            return _to_arrow_table(self._con.sql(sql, params=params))

    def query_row_ids_binary(
        self,
        where_sql: str,
        *,
        params: Any = None,
        max_rows: int = DEFAULT_MAX_FILTERED_ROW_IDS,
    ) -> tuple[str, bytes]:
        """Executes `SELECT row_id` in DuckDB and returns contiguous little-endian u32 or u64 bytes."""
        if not where_sql or not where_sql.strip():
            return ("clear", b"")
        q_rid = quote_sql_ident(self.row_id_column)
        rid_cast = "UBIGINT" if self._rid_is_unsigned else "BIGINT"
        limit_sql = f" LIMIT {int(max_rows) + 1}" if max_rows and max_rows > 0 else ""
        sql = (
            f"SELECT CAST({q_rid} AS {rid_cast}) AS rid "
            f"FROM {self._table_expr} WHERE ({where_sql}) AND {q_rid} IS NOT NULL{limit_sql}"
        )
        with self._lock:
            tbl = _to_arrow_table(self._con.sql(sql, params=params))
        if max_rows and max_rows > 0 and tbl.num_rows > max_rows:
            raise ValueError(
                f"SQL row_id filter matched > {max_rows:,} rows; narrow the predicate or use set_filter() GPU Path A."
            )
        if tbl.num_rows == 0:
            return ("u32", b"")
        raw_np = tbl.column(0).to_numpy(zero_copy_only=False)
        if self._rid_is_unsigned:
            arr_u64 = np.asarray(raw_np, dtype=np.uint64)
        else:
            arr_u64 = np.asarray(raw_np, dtype=np.int64).view(np.uint64)
        if int(arr_u64.max()) <= 0xFFFFFFFF:
            return ("u32", arr_u64.astype("<u4", copy=False).tobytes())
        return ("u64", arr_u64.astype("<u8", copy=False).tobytes())

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pq_path = self.parquet_path
            try:
                self._con.close()
            except Exception:
                pass
        if pq_path is not None:
            with _OPEN_SIDECAR_LOCK:
                rem = _OPEN_SIDECAR_PATHS.get(pq_path, 0) - 1
                if rem <= 0:
                    _OPEN_SIDECAR_PATHS.pop(pq_path, None)
                else:
                    _OPEN_SIDECAR_PATHS[pq_path] = rem


@dataclass
class ResolvedDataset:
    source_mode: str  # "local_archive" | "remote_url"
    archive_path: Path | None
    remote_url: str | None
    x_column: str
    y_column: str
    row_id_column: str
    sidecar: SidecarStore | None


def resolve_dataset_input(
    data: Any,
    *,
    x: str | None = None,
    y: str | None = None,
    row_id: str = "row_id",
    columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    sort_col: str | None = None,
    max_capacity: int = 25_000,
    sidecar: Any | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    max_cache_bytes: int = DEFAULT_MAX_CACHE_BYTES,
) -> ResolvedDataset:
    """Resolves a URL, `.arrowtiles` file, Parquet file, or any `narwhals`/Arrow/DuckDB table."""
    # 1. Remote HTTPS/HTTP `.arrowtiles` URL
    if isinstance(data, str) and (data.startswith("http://") or data.startswith("https://")):
        sc = SidecarStore(sidecar, row_id_column=row_id) if sidecar is not None else None
        return ResolvedDataset(
            source_mode="remote_url",
            archive_path=None,
            remote_url=data,
            x_column=x or "x",
            y_column=y or "y",
            row_id_column=row_id,
            sidecar=sc,
        )

    # 2. Existing local `.arrowtiles` path
    if isinstance(data, (str, os.PathLike)):
        p = Path(data)
        if p.suffix.lower() == ".arrowtiles":
            resolved = p.resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f".arrowtiles archive not found: {resolved}")
            sc = SidecarStore(sidecar, row_id_column=row_id) if sidecar is not None else None
            return ResolvedDataset(
                source_mode="local_archive",
                archive_path=resolved,
                remote_url=None,
                x_column=x or "x",
                y_column=y or "y",
                row_id_column=row_id,
                sidecar=sc,
            )

    # 3. Parquet path or in-memory DataFrame / Arrow Table / DuckDB relation -> content-hashed `.arrowtiles`
    if not x or not y:
        raise ValueError(
            "x='<col>' and y='<col>' are required when plotting a DataFrame, Arrow Table, or Parquet file."
        )

    cdir = get_cache_dir(cache_dir)
    con = duckdb.connect(":memory:")
    try:
        if isinstance(data, (str, os.PathLike)):
            pq_path = str(Path(data).resolve()).replace("'", "''")
            src_expr = f"read_parquet('{pq_path}')"
        elif isinstance(data, duckdb.DuckDBPyRelation):
            arr_tbl = _to_arrow_table(data)
            con.register("_dg_ingest_arrow", arr_tbl)
            src_expr = "_dg_ingest_arrow"
        elif isinstance(data, pa.Table):
            con.register("_dg_ingest_arrow", data)
            src_expr = "_dg_ingest_arrow"
        else:
            nw_df = nw.from_native(data, pass_through=False)
            if isinstance(nw_df, nw.LazyFrame):
                nw_df = nw_df.collect()
            arr_tbl = nw_df.to_arrow()
            con.register("_dg_ingest_arrow", arr_tbl)
            src_expr = "_dg_ingest_arrow"

        desc_rel = con.sql(f"SELECT * FROM {src_expr} LIMIT 0")
        src_cols = list(desc_rel.columns)
        src_types = [str(t) for t in desc_rel.types]
        if x not in src_cols or y not in src_cols:
            raise ValueError(
                f"Columns x='{x}', y='{y}' not found in input columns: {src_cols}"
            )

        # Fast content hash in DuckDB across ALL rows:
        # Format `sum(...)` (128-bit `HUGEINT`) directly as `::VARCHAR` so datasets with >= 131,072
        # rows never overflow `UINT64` (`2^64 / 2^47 = 131,072`), and include `row_number() OVER ()`
        # when `row_id` is absent so row permutations (`df` vs `df[::-1]`) never collide (`I1`).
        hash_args = ", ".join(quote_sql_ident(c) for c in src_cols)
        if row_id in src_cols:
            stats_row = con.execute(
                f"SELECT count(*)::BIGINT, "
                f"coalesce(bit_xor(hash({hash_args})), 0)::UBIGINT, "
                f"coalesce(sum(hash({hash_args})::UBIGINT & 281474976710655), 0)::VARCHAR "
                f"FROM {src_expr}"
            ).fetchone()
        else:
            stats_row = con.execute(
                f"SELECT count(*)::BIGINT, "
                f"coalesce(bit_xor(h), 0)::UBIGINT, "
                f"coalesce(sum(h & 281474976710655), 0)::VARCHAR "
                f"FROM (SELECT hash(row_number() OVER (), {hash_args}) AS h FROM {src_expr})"
            ).fetchone()
        assert stats_row is not None

        fingerprint_payload = json.dumps(
            {
                "spec": arrowtiles.SPEC_VERSION,
                "cols": list(zip(src_cols, src_types, strict=False)),
                "nrows": int(stats_row[0]),
                "hxor": str(stats_row[1]),
                "hsum": str(stats_row[2]),
                "x": x,
                "y": y,
                "row_id": row_id,
                "columns": columns,
                "categorical_columns": categorical_columns,
                "sort_col": sort_col,
                "max_capacity": max_capacity,
            },
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(fingerprint_payload).hexdigest()[:24]
        out_archive = cdir / f"dg_{digest}.arrowtiles"
        staged_parquet = cdir / f"dg_{digest}.parquet"

        if not out_archive.is_file() or not staged_parquet.is_file():
            q_rid = quote_sql_ident(row_id)
            if row_id in src_cols:
                stage_sql = f"SELECT * FROM {src_expr}"
            else:
                stage_sql = (
                    f"SELECT (row_number() OVER () - 1)::UBIGINT AS {q_rid}, * FROM {src_expr}"
                )
            tmp_pq = cdir / f"dg_{digest}_{uuid.uuid4().hex[:8]}.tmp.parquet"
            tmp_at = cdir / f"dg_{digest}_{uuid.uuid4().hex[:8]}.tmp.arrowtiles"
            try:
                pq_escaped = str(tmp_pq).replace("'", "''")
                con.execute(f"COPY ({stage_sql}) TO '{pq_escaped}' (FORMAT PARQUET)")
                tmp_pq.replace(staged_parquet)

                # Pass `columns=columns, keep_source_xy=False` directly so `arrowtiles.build_generic`
                # enforces `D7a` high-cardinality free-text auto-dropping while `staged_parquet`
                # retains every cold column for `SidecarStore` (`B3`)!
                arrowtiles.build_generic(
                    input_parquet=str(staged_parquet),
                    output_path=str(tmp_at),
                    x_col=x,
                    y_col=y,
                    id_col=row_id,
                    sort_col=sort_col,
                    columns=columns,
                    categorical_cols=categorical_columns,
                    keep_source_xy=False,
                    max_capacity=max_capacity,
                    show_progress=False,
                )
                tmp_at.replace(out_archive)
            finally:
                tmp_pq.unlink(missing_ok=True)
                tmp_at.unlink(missing_ok=True)
            _prune_cache_dir(cdir, max_cache_bytes, protect=out_archive)
        else:
            os.utime(out_archive, None)

        sc_source = sidecar if sidecar is not None else staged_parquet
        sc_store = SidecarStore(sc_source, row_id_column=row_id)
        return ResolvedDataset(
            source_mode="local_archive",
            archive_path=out_archive,
            remote_url=None,
            x_column=x,
            y_column=y,
            row_id_column=row_id,
            sidecar=sc_store,
        )
    finally:
        con.close()
