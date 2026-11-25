from __future__ import annotations

import datetime
import decimal
import json
import subprocess
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import arrowtiles
import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import pydeepgraph as dg
from pydeepgraph import (
    COMM_CHUNK_BYTES,
    DeepgraphWidget,
    LocalRangeDaemon,
    build_lasso_sql,
    plot,
    resolve_dataset_input,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_lmsys_sample(tmp_path: Path) -> Path:
    candidates = [
        REPO_ROOT.parent / "arrowtiles" / "samples" / "lmsys_sample.arrowtiles",
        REPO_ROOT.parent / "duckdb-arrowtiles" / "samples" / "lmsys_sample.arrowtiles",
    ]
    for c in candidates:
        if c.is_file():
            return c
    stub = tmp_path / "stub_sample.arrowtiles"
    stub.write_bytes(b"PMTiles\x03" + b"\x00" * 248)
    return stub


def _dispatch_custom_msg(widget: DeepgraphWidget, content: dict[str, Any]) -> None:
    """Exercise the full ipywidgets `_handle_msg` -> `_handle_custom_msg` -> `on_msg` chain."""
    widget._handle_msg(
        {
            "content": {"data": {"method": "custom", "content": content}},
            "buffers": [],
        }
    )


def _dispatch_state_update(widget: DeepgraphWidget, state_patch: dict[str, Any]) -> None:
    """Simulate a raw frontend `ModelUpdate` WebSocket state message."""
    widget._handle_msg(
        {
            "content": {
                "data": {
                    "method": "update",
                    "state": state_patch,
                    "buffer_paths": [],
                }
            },
            "buffers": [],
        }
    )


def _capture_sends(widget: DeepgraphWidget) -> list[tuple[dict[str, Any], list[bytes]]]:
    captured: list[tuple[dict[str, Any], list[bytes]]] = []

    def fake_send(content: dict[str, Any], buffers: list[Any] | None = None) -> None:
        raw_bufs = [bytes(b) for b in (buffers or [])]
        captured.append((content, raw_bufs))

    widget.send = fake_send  # type: ignore[method-assign]
    return captured


def test_static_widget_bundle_present_and_self_contained() -> None:
    bundle = Path(dg.__file__).parent / "_static" / "widget.js"
    assert bundle.is_file()
    size = bundle.stat().st_size
    assert 100_000 < size < 1_200_000
    text = bundle.read_text(encoding="utf-8")
    assert "render" in text
    for line in text.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("import "), f"Unexpected external import: {stripped[:120]}"


def test_content_hashed_ingest_row_order_sensitivity_and_lru(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    rng = np.random.default_rng(42)
    n = 400
    pl_df = pl.DataFrame(
        {
            "x": rng.normal(0.0, 10.0, size=n),
            "y": rng.normal(0.0, 10.0, size=n),
            "category": ["alpha" if i % 2 == 0 else "beta" for i in range(n)],
            "score": rng.uniform(0.0, 1.0, size=n),
        }
    )

    # 1. Polars DataFrame build
    res1 = resolve_dataset_input(
        pl_df,
        x="x",
        y="y",
        categorical_columns=["category"],
        cache_dir=cache_dir,
    )
    assert res1.source_mode == "local_archive"
    assert res1.archive_path is not None and res1.archive_path.is_file()
    assert res1.sidecar is not None
    assert "row_id" in res1.sidecar.columns
    row0_x = res1.sidecar.lookup_row(0)["x"]
    res1.sidecar.close()

    # 2. Equivalent Pandas DataFrame hits exact content-addressed cache!
    pd_df = pl_df.to_pandas()
    res2 = resolve_dataset_input(
        pd_df,
        x="x",
        y="y",
        categorical_columns=["category"],
        cache_dir=cache_dir,
    )
    assert res2.archive_path == res1.archive_path
    res2.sidecar.close()  # type: ignore[union-attr]

    # 3. Reversed DataFrame without `row_id` MUST NOT collide in cache (`I1`)!
    pl_rev = pl_df.reverse()
    res_rev = resolve_dataset_input(
        pl_rev,
        x="x",
        y="y",
        categorical_columns=["category"],
        cache_dir=cache_dir,
    )
    assert res_rev.archive_path != res1.archive_path
    assert res_rev.sidecar is not None
    assert res_rev.sidecar.lookup_row(0)["x"] != pytest.approx(row0_x)
    res_rev.sidecar.close()

    # 4. Mutating a SINGLE scalar value with identical shape/schema produces a NEW digest
    pd_mut = pd_df.copy()
    pd_mut.loc[17, "score"] = float(pd_mut.loc[17, "score"]) + 0.123456
    res3 = resolve_dataset_input(
        pd_mut,
        x="x",
        y="y",
        categorical_columns=["category"],
        cache_dir=cache_dir,
    )
    assert res3.archive_path != res1.archive_path
    res3.sidecar.close()  # type: ignore[union-attr]

    # 5. PyArrow Table & DuckDB relation & Parquet path match content hash
    pa_tbl = pa.Table.from_pandas(pd_df, preserve_index=False)
    pq_file = tmp_path / "pts.parquet"
    pq.write_table(pa_tbl, pq_file)

    con = duckdb.connect(":memory:")
    con.register("t_arrow", pa_tbl)
    rel = con.sql("SELECT * FROM t_arrow")
    res_duck = resolve_dataset_input(
        rel,
        x="x",
        y="y",
        categorical_columns=["category"],
        cache_dir=cache_dir,
    )
    assert res_duck.archive_path == res1.archive_path
    res_duck.sidecar.close()  # type: ignore[union-attr]
    con.close()

    res_pq = resolve_dataset_input(
        pq_file,
        x="x",
        y="y",
        categorical_columns=["category"],
        cache_dir=cache_dir,
    )
    assert res_pq.archive_path == res1.archive_path
    res_pq.sidecar.close()  # type: ignore[union-attr]

    # 6. LRU cache eviction counts BOTH `.arrowtiles` and paired `.parquet` sidecar (`I5`)
    pd_mut2 = pd_df.copy()
    pd_mut2.loc[0, "x"] = 9999.0
    res_evict = resolve_dataset_input(
        pd_mut2,
        x="x",
        y="y",
        categorical_columns=["category"],
        cache_dir=cache_dir,
        max_cache_bytes=1,
    )
    res_evict.sidecar.close()  # type: ignore[union-attr]
    assert len(list(cache_dir.glob("*.arrowtiles"))) <= 1
    assert len(list(cache_dir.glob("*.parquet"))) <= 1


def test_d7a_cold_text_autodrop_and_sidecar_retention(tmp_path: Path) -> None:
    """Verifies `B3`: high-cardinality free-text columns (> 4,096 distinct) are auto-dropped from
    `.arrowtiles` via `D7a` and `x`/`y` are not duplicated as Float64, while `SidecarStore` still
    resolves full cold text."""
    rng = np.random.default_rng(7)
    n = 6_000
    df = pl.DataFrame(
        {
            "umap_x": rng.uniform(-5.0, 5.0, size=n),
            "umap_y": rng.uniform(-5.0, 5.0, size=n),
            "cluster": [f"c_{i % 5}" for i in range(n)],
            "prompt_text": [f"Unique long user prompt #{i}: " + ("token " * 12) for i in range(n)],
        }
    )
    res = resolve_dataset_input(df, x="umap_x", y="umap_y", cache_dir=tmp_path / "d7a_cache")
    try:
        assert res.archive_path is not None
        with arrowtiles.ArrowTilesReader(res.archive_path) as reader:
            tile_cols = set(reader.schema().names)
            meta = reader.metadata()
        # Free-text column must NOT bloat GPU tiles, and duplicate Float64 x/y must not be stored
        assert "prompt_text" not in tile_cols
        assert "umap_x" not in tile_cols
        assert "umap_y" not in tile_cols
        assert "prompt_text" in (meta.get("dropped_columns") or [])
        # Yet cold SidecarStore retains full `prompt_text` for click inspection & SQL search!
        assert res.sidecar is not None
        row42 = res.sidecar.lookup_row(42)
        assert row42["prompt_text"].startswith("Unique long user prompt #42:")
    finally:
        if res.sidecar is not None:
            res.sidecar.close()


def test_local_http_range_daemon_206_pna_and_multi_widget_refcount(tmp_path: Path) -> None:
    payload = bytes((i * 31) & 0xFF for i in range(100_000))
    fpath = tmp_path / "mock.arrowtiles"
    fpath.write_bytes(payload)

    daemon = LocalRangeDaemon()
    try:
        tok1, url1 = daemon.register_file(fpath)
        tok2, url2 = daemon.register_file(fpath)
        # Distinct per-registration bearer tokens (`B5`) sharing one underlying ref-counted mmap!
        assert tok1 != tok2
        assert url1 != url2

        opt_req = urllib.request.Request(url1, method="OPTIONS")
        with urllib.request.urlopen(opt_req, timeout=3) as resp:
            assert resp.status == 204
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
            assert resp.headers.get("Access-Control-Allow-Private-Network") == "true"

        # Closing/unregistering tok1 MUST NOT revoke tok2 (`B5`)!
        daemon.unregister_token(tok1)
        with pytest.raises(urllib.error.HTTPError) as exc_tok1:
            urllib.request.urlopen(url1, timeout=3)
        assert exc_tok1.value.code == 403
        exc_tok1.value.close()

        rng_req2 = urllib.request.Request(url2, headers={"Range": "bytes=1024-5119"})
        with urllib.request.urlopen(rng_req2, timeout=3) as resp2:
            assert resp2.status == 206
            assert resp2.read() == payload[1024:5120]

        # Unregistering the final reference revokes tok2 and closes the mmap
        daemon.unregister_token(tok2)
        with pytest.raises(urllib.error.HTTPError) as exc_tok2:
            urllib.request.urlopen(rng_req2, timeout=3)
        assert exc_tok2.value.code == 403
        exc_tok2.value.close()
    finally:
        daemon.shutdown()

    # Also verify `DeepgraphWidget(..., transport="http")` keeps `_archive_mmap` lazy (`I6`)
    # and survives sibling widget close (`B5`)
    w1 = plot(fpath, transport="http")
    w2 = plot(fpath, transport="http")
    try:
        assert w1._archive_mmap is None
        assert w1.http_url != w2.http_url
        w1.close()
        req = urllib.request.Request(w2.http_url, headers={"Range": "bytes=0-63"})
        with urllib.request.urlopen(req, timeout=3) as r_ok:
            assert r_ok.status == 206
            assert r_ok.read() == payload[:64]
    finally:
        w2.close()


def test_comm_range_chunking_and_abort(tmp_path: Path) -> None:
    raw = np.arange(350_000, dtype="<u4").tobytes()  # 1,400,000 B
    archive_path = tmp_path / "big_test.arrowtiles"
    archive_path.write_bytes(raw)

    w = plot(archive_path, transport="comm")
    try:
        sends = _capture_sends(w)

        _dispatch_custom_msg(w, {"type": "range_req", "reqId": "r1", "offset": 128, "length": 4096})
        assert len(sends) == 1
        meta1, bufs1 = sends[0]
        assert meta1["totalParts"] == 1
        assert bufs1[0] == raw[128 : 128 + 4096]

        sends.clear()
        _dispatch_custom_msg(
            w, {"type": "range_req", "reqId": "r_big", "offset": 10_000, "length": 1_350_000}
        )
        assert len(sends) == 3
        assembled = bytearray(1_350_000)
        for idx, (meta, bufs) in enumerate(sends):
            assert meta["partIndex"] == idx
            assert meta["totalParts"] == 3
            assert len(bufs[0]) <= COMM_CHUNK_BYTES
            off = meta["byteOffset"]
            assembled[off : off + len(bufs[0])] = bufs[0]
        assert bytes(assembled) == raw[10_000 : 10_000 + 1_350_000]

        sends.clear()
        _dispatch_custom_msg(w, {"type": "range_abort", "reqId": "r_cancel"})
        _dispatch_custom_msg(
            w, {"type": "range_req", "reqId": "r_cancel", "offset": 0, "length": 65536}
        )
        assert len(sends) == 0
    finally:
        w.close()


def test_sidecar_enrichment_ubigint_max_range_aliases_and_param_filters(tmp_path: Path) -> None:
    """Covers `B4` (`UInt64 >= 2^63`), `I2` (stringified JSON-safe `int > 2^53-1`),
    `I4` (`x_column`/`y_column` aliases), `I5` (`filterSeq`), and parameterized `filter_by_sql`."""
    sample = _find_lmsys_sample(tmp_path)
    gaia_u64_top = (1 << 63) + 42  # >= 2^63 (`UBIGINT` high bit set!)
    nodes_full = pa.table(
        {
            "row_id": pa.array([10, 40064, gaia_u64_top, 38479], type=pa.uint64()),
            "umap_x": [1.5, 88.7607, -12.0, 87.7994],
            "umap_y": [2.5, 69.1274, 44.0, 29.4974],
            "prompt": [
                "hello world",
                "Explain WebGPU atomic splatting",
                "Gaia DR3 high-bit star",
                "Compare Llama-3 vs Mixtral",
            ],
            "cost_usd": [
                decimal.Decimal("0.0125"),
                decimal.Decimal("0.4500"),
                decimal.Decimal("1.0000"),
                decimal.Decimal("0.0500"),
            ],
            "created_at": [
                datetime.datetime(2026, 5, 1, 12, 30, 0),
                datetime.datetime(2026, 5, 2, 8, 15, 0),
                datetime.datetime(2026, 5, 3, 0, 0, 0),
                datetime.datetime(2026, 5, 2, 9, 0, 0),
            ],
            "weird_nan": [float("nan"), 3.25, float("inf"), 1.0],
        }
    )

    edges_tbl = pa.table(
        {
            "source_row_id": pa.array([38479, 40064, gaia_u64_top], type=pa.uint64()),
            "target_row_id": pa.array([40064, 10, 40064], type=pa.uint64()),
            "weight": [0.94, 0.81, 0.99],
        }
    )

    # Test `x_column` / `y_column` aliases (`I4`)
    w = plot(
        sample,
        x_column="umap_x",
        y_column="umap_y",
        sidecar=nodes_full,
        edges=edges_tbl,
        bidirectional_edges=True,
        transport="comm",
    )
    try:
        assert w.x_column == "umap_x"
        assert w.y_column == "umap_y"
        sends = _capture_sends(w)

        # 1. Lookup `UInt64 >= 2^63` (`B4` + `I2`: `row_id` stringified so JS JSON.parse won't round!)
        _dispatch_custom_msg(
            w, {"type": "resolve_row", "reqId": "row_hi", "rowId": str(gaia_u64_top)}
        )
        assert len(sends) == 1
        hi_attrs = sends[0][0]["attributes"]
        assert hi_attrs["row_id"] == str(gaia_u64_top)
        assert hi_attrs["prompt"] == "Gaia DR3 high-bit star"
        assert hi_attrs["weird_nan"] is None
        json.dumps(sends[0][0], allow_nan=False)

        # 2. Bidirectional 1-hop DuckDB ego-edge resolution with `UInt64 >= 2^63` neighbor (`B4`)
        sends.clear()
        _dispatch_custom_msg(
            w,
            {
                "type": "resolve_edges",
                "reqId": "ego_1",
                "rowId": 40064,
                "trigger": "click",
                "dataX": 88.7607,
                "dataY": 69.1274,
            },
        )
        assert len(sends) == 1
        edges_out = sends[0][0]["edges"]
        assert len(edges_out) == 3
        assert edges_out[0]["targetRowId"] == str(gaia_u64_top)
        assert [edges_out[1]["targetRowId"], edges_out[2]["targetRowId"]] == [38479, 10]

        # 3. Parameterized `filter_by_sql(where_sql, params)` + monotonic `filterSeq` (`I5`)
        sends.clear()
        cnt_u32 = w.filter_by_sql("prompt ILIKE ?", ["%WebGPU%"])
        assert cnt_u32 == 1
        seq1 = sends[0][0]["filterSeq"]
        assert seq1 >= 1
        assert sends[0][0]["dtype"] == "u32"
        assert np.frombuffer(sends[0][1][0], dtype="<u4").tolist() == [40064]

        sends.clear()
        cnt_u64 = w.filter_by_sql("prompt ILIKE ?", ["%high-bit%"])
        assert cnt_u64 == 1
        assert sends[0][0]["filterSeq"] > seq1
        assert sends[0][0]["dtype"] == "u64"
        assert int(np.frombuffer(sends[0][1][0], dtype="<u8")[0]) == gaia_u64_top
    finally:
        w.close()


def test_exact_inscribed_ball_spatial_knn_matches_bruteforce_100_percent(tmp_path: Path) -> None:
    """Verifies `B2`: `edges='spatial_knn'` returns 100% exact Euclidean top-10 neighbors
    across dense cores, extreme outliers, and box-corner ($r < d_k \\le \\sqrt{2}r$) configurations."""
    sample = _find_lmsys_sample(tmp_path)
    rng = np.random.default_rng(1234)
    # Mix a tight dense cluster, a sparse ring, and extreme margin outliers (like lmsys_full_v01)
    xs = np.concatenate(
        [
            rng.normal(5.0, 0.03, size=1_500),
            rng.normal(-12.0, 1.8, size=400),
            np.array([-450.0, 480.0, 5.0 + 0.151, 5.0 - 0.151]),
        ]
    )
    ys = np.concatenate(
        [
            rng.normal(8.0, 0.03, size=1_500),
            rng.normal(20.0, 1.8, size=400),
            np.array([-420.0, 510.0, 8.0 + 0.151, 8.0 - 0.151]),
        ]
    )
    n = len(xs)
    # Include UInt64 IDs above 2^63 as well
    rids = np.array([(1 << 63) + i if i % 7 == 0 else i + 1000 for i in range(n)], dtype=np.uint64)
    nodes_tbl = pa.table(
        {
            "row_id": pa.array(rids, type=pa.uint64()),
            "x": xs,
            "y": ys,
            "model": [f"m_{i % 4}" for i in range(n)],
        }
    )

    w = plot(
        sample,
        x="x",
        y="y",
        sidecar=nodes_tbl,
        edges="spatial_knn",
        max_ego_neighbors=10,
        transport="comm",
    )
    try:
        probe_indices = [0, 1, 15, 777, 1499, 1500, 1899, n - 4, n - 3, n - 2, n - 1]
        for idx in probe_indices:
            q_rid = int(rids[idx])
            qx, qy = float(xs[idx]), float(ys[idx])
            d2 = (xs - qx) ** 2 + (ys - qy) ** 2
            d2[idx] = np.inf
            # Exact brute-force top-10 (tie-broken by row_id ASC)
            bf_order = np.lexsort((rids, d2))[:10]
            expected_ids = [int(rids[j]) for j in bf_order]

            w.clicked_row = {"rowId": str(q_rid), "dataX": qx, "dataY": qy}
            nbr_tbl = w.query_clicked_neighbors()
            assert nbr_tbl is not None
            got_ids = [int(v) for v in nbr_tbl["target_row_id"].to_pylist()]
            assert got_ids == expected_ids
    finally:
        w.close()


def test_server_side_selection_sql_blocks_client_sql_injection(tmp_path: Path) -> None:
    """Verifies Reviewer 2 Blocking #3: untrusted client `selection_sql` updates or injected
    column names in `selection_clause` are rejected and never reach DuckDB."""
    sample = _find_lmsys_sample(tmp_path)
    sidecar = pa.table(
        {
            "row_id": pa.array([1, 2, 3], type=pa.uint64()),
            "x": [1.0, 5.0, 9.0],
            "y": [1.0, 5.0, 9.0],
            "secret": ["a", "b", "c"],
        }
    )
    w = plot(sample, x="x", y="y", sidecar=sidecar, transport="comm")
    try:
        # 1. Attempt raw client injection into `selection_sql`
        _dispatch_state_update(
            w,
            {"selection_sql": "1=1) UNION ALL SELECT 999, 0, 0, content FROM read_text('README.md') --"},
        )
        assert w.selection_sql == ""
        assert w.query_selected().num_rows == 0

        # 2. Attempt SQL injection through `selection_clause.xField`
        _dispatch_state_update(
            w,
            {
                "selection_clause": {
                    "kind": "rect",
                    "xField": 'x" BETWEEN 0 AND 10 OR 1=1 --',
                    "yField": "y",
                    "rect": {"minX": 0.0, "maxX": 2.0, "minY": 0.0, "maxY": 2.0},
                }
            },
        )
        assert w.selection_sql == ""
        assert w.query_selected().num_rows == 0

        # 3. Legitimate structured `rect` selection builds clean server-validated SQL and returns exact row
        _dispatch_state_update(
            w,
            {
                "selection_clause": {
                    "kind": "rect",
                    "xField": "x",
                    "yField": "y",
                    "rect": {"minX": 4.0, "maxX": 6.0, "minY": 4.0, "maxY": 6.0},
                }
            },
        )
        assert w.selection_sql == '("x" BETWEEN 4.0 AND 6.0) AND ("y" BETWEEN 4.0 AND 6.0)'
        sel = w.query_selected()
        assert sel["row_id"].to_pylist() == [2]
    finally:
        w.close()


def test_python_and_js_rdp_lasso_sql_exact_parity() -> None:
    if not (REPO_ROOT / "dist" / "mosaic.js").is_file():
        pytest.skip("dist/mosaic.js absent in standalone Python sdist environment")
    pts: list[list[float]] = []
    spikes = 5
    total = 500
    for k in range(total):
        t = (k / total) * (2.0 * np.pi)
        spoke = (k * spikes * 2) / total
        frac = spoke - int(spoke)
        r_a = 10.0 if int(spoke) % 2 == 0 else 3.8
        r_b = 3.8 if int(spoke) % 2 == 0 else 10.0
        r = r_a + (r_b - r_a) * frac
        pts.append([round(float(r * np.cos(t)), 6), round(float(r * np.sin(t)), 6)])

    py_sql = build_lasso_sql("x", "y", pts, 64)

    node_script = (
        "import { buildLassoDuckDbSql } from './dist/mosaic.js';"
        f"const pts = {json.dumps(pts)};"
        "process.stdout.write(buildLassoDuckDbSql('x', 'y', pts, 64));"
    )
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", node_script],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    js_sql = proc.stdout.strip()

    con = duckdb.connect(":memory:")
    xs = np.linspace(-11.0, 11.0, 50)
    gx, gy = np.meshgrid(xs, xs)
    grid_tbl = pa.table({"id": np.arange(2500), "x": gx.ravel(), "y": gy.ravel()})
    con.register("grid", grid_tbl)
    py_hits = [r[0] for r in con.execute(f"SELECT id FROM grid WHERE {py_sql} ORDER BY id").fetchall()]
    js_hits = [r[0] for r in con.execute(f"SELECT id FROM grid WHERE {js_sql} ORDER BY id").fetchall()]
    con.close()

    assert len(py_hits) > 200
    assert py_hits == js_hits


def test_mixed_uint64_neighbors_arrow_frontend_ready_replay_and_edge_guards(tmp_path: Path) -> None:
    """Covers Goldfish findings: native Arrow types on mixed `<= 2^53-1` / `> 2^53-1` `spatial_knn`
    sidecar columns, `frontend_ready` filter replay, context manager cleanup, `1e308`/`1e200` lasso
    guards, `bool` `rowId` rejection, and `query_where(columns='single_col')`."""
    sample = _find_lmsys_sample(tmp_path)
    nodes_tbl = pa.table(
        {
            "row_id": pa.array([10, 20, (1 << 63) + 5], type=pa.uint64()),
            "x": [0.0, 0.1, 0.2],
            "y": [0.0, 0.1, 0.2],
            # Mixed-magnitude uint64 column straddling 2^53 - 1 (previously raised ArrowTypeError!)
            "ext_id": pa.array([42, (1 << 60) + 9, (1 << 63) + 77], type=pa.uint64()),
        }
    )

    with plot(
        sample,
        x="x",
        y="y",
        sidecar=nodes_tbl,
        edges="spatial_knn",
        max_ego_neighbors=5,
        transport="comm",
    ) as w:
        sends = _capture_sends(w)

        # 1. Mixed-magnitude `uint64` sidecar column in `query_clicked_neighbors()` preserves Arrow uint64
        w.clicked_row = {"rowId": 10, "dataX": 0.0, "dataY": 0.0}
        nbr_tbl = w.query_clicked_neighbors()
        assert nbr_tbl is not None
        assert nbr_tbl.num_rows == 2
        assert pa.types.is_uint64(nbr_tbl.schema.field("target_row_id").type)
        assert pa.types.is_uint64(nbr_tbl.schema.field("ext_id").type)
        assert nbr_tbl["ext_id"].to_pylist() == [(1 << 60) + 9, (1 << 63) + 77]

        # Missing hub row_id without fallback dataX/dataY returns typed empty table with `uint64` IDs
        w.clicked_row = {"rowId": 999999}
        empty_nbr = w.query_clicked_neighbors()
        assert empty_nbr is not None and empty_nbr.num_rows == 0
        assert pa.types.is_uint64(empty_nbr.schema.field("target_row_id").type)

        # 2. Single-string `columns="ext_id"` in `query_selected` / `SidecarStore.query_where` does not split characters
        _dispatch_state_update(
            w,
            {
                "clicked_row": {"rowId": 10},
                "selection_clause": {"kind": "point", "xField": "x", "yField": "y"},
            },
        )
        single_col_tbl = w.query_selected(columns="ext_id")
        assert single_col_tbl.column_names == ["ext_id"]
        assert single_col_tbl["ext_id"].to_pylist() == [42]

        # 3. Pre-mount `filter_by_sql` is replayed upon `frontend_ready` and cleared on `filter_by_sql(None)`
        matched = w.filter_by_sql("ext_id > 1000")
        assert matched == 2
        sends.clear()
        _dispatch_custom_msg(w, {"type": "frontend_ready"})
        assert len(sends) == 1
        assert sends[0][0]["type"] == "rowid_filter_bin"
        assert sends[0][0]["dtype"] == "u64"
        assert np.frombuffer(sends[0][1][0], dtype="<u8").tolist() == [20, (1 << 63) + 5]

        w.filter_by_sql(None)
        sends.clear()
        _dispatch_custom_msg(w, {"type": "frontend_ready"})
        assert sends == []

        # 4. Boolean `clicked_row.rowId` is rejected (`isinstance(True, int)` in Python!)
        _dispatch_state_update(
            w,
            {
                "clicked_row": {"rowId": True},
                "selection_clause": {"kind": "point", "xField": "x", "yField": "y"},
            },
        )
        assert w.selection_sql == ""

    # `__exit__` closed sidecar cleanly
    assert w._sidecar_store is None

    # 5. `build_lasso_sql` `1e150` DuckDB `DOUBLE` overflow & 20,000-vertex guards
    assert build_lasso_sql("x", "y", [[-1e250, -1e200], [1e250, 0.0], [0.0, 1e200]]) == "FALSE"
    huge_pts = [
        [float(np.cos(k * 0.0005) * 1e120), float(np.sin(k * 0.0005) * 1e120)]
        for k in range(20_000)
    ]
    huge_sql = build_lasso_sql("x", "y", huge_pts, 64)
    assert "BETWEEN" in huge_sql
    assert "inf" not in huge_sql.lower()
    assert "nan" not in huge_sql.lower()


def test_goldfish_hugeint_hash_untrusted_bounds_and_signed_rowids(tmp_path: Path) -> None:
    """Covers second-pass Goldfish findings: >= 131,072-row `HUGEINT` hash sum (`> 2^64-1`),
    `range_req` bounds/malformed input, client `row_id_column`/`http_url` tamper protection,
    negative `int64` `row_id` filtering, and live-sidecar LRU cache protection."""
    # 1. 150,000-row Arrow table (`> 131,072` `UINT64` sum threshold!) hashes & builds without ConversionException
    n_big = 150_000
    rng = np.random.default_rng(99)
    big_tbl = pa.table(
        {
            "x": rng.uniform(0.0, 100.0, size=n_big).astype(np.float32),
            "y": rng.uniform(0.0, 100.0, size=n_big).astype(np.float32),
        }
    )
    cdir = tmp_path / "big_cache"
    res_big = resolve_dataset_input(big_tbl, x="x", y="y", max_capacity=50_000, cache_dir=cdir)
    try:
        assert res_big.archive_path is not None and res_big.archive_path.is_file()
        assert res_big.sidecar is not None
        # Live `res_big.sidecar` MUST NOT have its `.parquet` deleted when a second build runs with `max_cache_bytes=1`!
        tiny_tbl = pa.table({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        res_tiny = resolve_dataset_input(
            tiny_tbl, x="x", y="y", cache_dir=cdir, max_cache_bytes=1
        )
        try:
            assert res_big.sidecar.parquet_path is not None
            assert res_big.sidecar.parquet_path.is_file()
            assert res_big.sidecar.lookup_row(100) != {}
        finally:
            if res_tiny.sidecar is not None:
                res_tiny.sidecar.close()
    finally:
        if res_big.sidecar is not None:
            res_big.sidecar.close()

    # 2. Negative `int64` `row_id`s + untrusted client `row_id_column` / `http_url` / `range_req` bounds
    sample = _find_lmsys_sample(tmp_path)
    neg_nodes = pa.table(
        {
            "row_id": pa.array([-42, -1, 7], type=pa.int64()),
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 1.0, 2.0],
            "tag": ["neg42", "neg1", "pos7"],
        }
    )
    with plot(sample, x="x", y="y", sidecar=neg_nodes, edges="spatial_knn", transport="comm") as w:
        sends = _capture_sends(w)

        # Negative `int64` `filter_by_sql` and `set_filtered_row_ids(pa.array(...))` succeed via two's-complement u64
        cnt_neg = w.filter_by_sql("row_id < 0")
        assert cnt_neg == 2
        assert sends[-1][0]["dtype"] == "u64"
        sends.clear()
        w.set_filtered_row_ids(pa.array([-42, 7], type=pa.int64()))
        assert sends[-1][0]["dtype"] == "u64"

        # Untrusted client `ModelUpdate` cannot redirect `row_id_column` or tamper with `source_mode` / `http_url`
        _dispatch_state_update(
            w,
            {
                "row_id_column": "tag",
                "source_mode": "http",
                "http_url": "http://evil.example/payload",
                "clicked_row": {"rowId": -42},
                "selection_clause": {"kind": "point"},
            },
        )
        assert w.row_id_column == "row_id"
        assert w.source_mode == "comm"
        assert w.http_url == ""
        assert w.selection_sql == '"row_id" = -42'

        # Malformed `clicked_row.rowId` returns `None` from `query_clicked_neighbors()` instead of raising ValueError
        _dispatch_state_update(w, {"clicked_row": {"rowId": "not_an_int"}})
        assert w.query_clicked_neighbors() is None

        # Malformed & oversize `range_req` return clean `range_res` error packets instead of throwing or OOMing
        sends.clear()
        _dispatch_custom_msg(w, {"type": "range_req", "reqId": "bad1", "offset": None, "length": "nan"})
        assert sends[-1][0].get("error") is not None
        sends.clear()
        _dispatch_custom_msg(
            w, {"type": "range_req", "reqId": "huge1", "offset": 0, "length": 500_000_000}
        )
        assert "exceeds" in str(sends[-1][0].get("error", ""))


def test_built_wheel_and_sdist_contain_widget_js_license_and_notice() -> None:
    wheels = sorted((REPO_ROOT / "python" / "dist").glob("pydeepgraph-*.whl"))
    sdists = sorted((REPO_ROOT / "python" / "dist").glob("pydeepgraph-*.tar.gz"))
    if not wheels or not sdists:
        pytest.skip("Wheel/sdist not built yet")
    with zipfile.ZipFile(wheels[-1]) as zf:
        names = set(zf.namelist())
        assert any(n.endswith("pydeepgraph/_static/widget.js") for n in names)
        assert any(n.endswith("pydeepgraph/py.typed") for n in names)
        assert any("LICENSE" in n for n in names)
        assert any("NOTICE" in n for n in names)
    with tarfile.open(sdists[-1], "r:gz") as tf:
        tnames = set(tf.getnames())
        assert any(n.endswith("pydeepgraph/_static/widget.js") for n in tnames)
        assert any(n.endswith("pydeepgraph/py.typed") for n in tnames)
        assert any(n.endswith("/LICENSE") for n in tnames)
        assert any(n.endswith("/NOTICE") for n in tnames)
