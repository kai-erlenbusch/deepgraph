import marimo

__generated_with = "0.26.0"
app = marimo.App(width="full")


@app.cell
def _():
    import hashlib
    import os
    import uuid
    from pathlib import Path

    import arrowtiles
    import duckdb
    import marimo as mo
    import pyarrow.parquet as pq

    import pydeepgraph as dg
    from pydeepgraph import get_cache_dir

    return (
        Path,
        arrowtiles,
        dg,
        duckdb,
        get_cache_dir,
        hashlib,
        mo,
        os,
        pq,
        uuid,
    )


@app.cell
def _(Path, arrowtiles, duckdb, get_cache_dir, hashlib, os, pq, uuid):
    def _discover_lmsys_archive() -> Path:
        env_override = os.environ.get("PYDEEPGRAPH_LMSYS_ARCHIVE")
        if env_override:
            p = Path(env_override).expanduser().resolve()
            if p.is_file():
                return p
            raise FileNotFoundError(f"PYDEEPGRAPH_LMSYS_ARCHIVE not found: {p}")

        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / "public" / "lmsys_full_v01.arrowtiles",
            repo_root.parent / "arrowtiles" / "samples" / "lmsys_full_v01.arrowtiles",
            repo_root.parent / "duckdb-arrowtiles" / "samples" / "lmsys_full_v01.arrowtiles",
            repo_root / "public" / "lmsys_sample.arrowtiles",
            repo_root.parent / "arrowtiles" / "samples" / "lmsys_sample.arrowtiles",
            repo_root.parent / "duckdb-arrowtiles" / "samples" / "lmsys_sample.arrowtiles",
        ]
        for cand in candidates:
            if cand.is_file():
                return cand.resolve()
        raise FileNotFoundError(
            "No LMSys .arrowtiles archive found. Generate samples in ../arrowtiles or set PYDEEPGRAPH_LMSYS_ARCHIVE."
        )

    def _load_or_build_full_lmsys_sidecar(archive_path: Path) -> tuple[Path, int, float]:
        st = archive_path.stat()
        cache_key = hashlib.sha256(
            f"{archive_path.name}:{st.st_size}:{st.st_mtime_ns}:full_xy_v2".encode()
        ).hexdigest()[:20]
        cdir = get_cache_dir()
        nodes_pq = cdir / f"lmsys_nodes_{cache_key}.parquet"

        with arrowtiles.ArrowTilesReader(archive_path) as reader:
            meta = reader.metadata()
            bounds = meta.get("bounds") or {}
            span_x = float(bounds.get("max_x", 100.0)) - float(bounds.get("min_x", 0.0))
            span_y = float(bounds.get("max_y", 100.0)) - float(bounds.get("min_y", 0.0))
            archive_span = max(1e-6, span_x, span_y)

            if nodes_pq.is_file():
                n_nodes = pq.read_metadata(nodes_pq).num_rows
                return nodes_pq, n_nodes, archive_span

            # Stream decoded pyramid tiles (`Z=0..maxZ`) directly into a temporary Parquet file
            # via `pq.ParquetWriter` (`O(tile)` RAM instead of buffering 2.84M rows in Python lists),
            # then let DuckDB sort by `(x, y)` into the atomic cached `.parquet` sidecar.
            want_cols = [
                "row_id",
                "x_decoded",
                "y_decoded",
                "model",
                "num_of_tokens",
                "language",
                "role",
                "clusters_28",
            ]
            raw_tmp_pq = cdir / f"lmsys_raw_{cache_key}_{uuid.uuid4().hex[:8]}.tmp.parquet"
            tmp_pq = cdir / f"lmsys_nodes_{cache_key}_{uuid.uuid4().hex[:8]}.tmp.parquet"
            writer: pq.ParquetWriter | None = None
            avail: set[str] = set()
            try:
                for (_z, _tx, _ty), tbl in reader.iter_tiles(
                    decode_coords=True, decode_categories=True
                ):
                    if tbl.num_rows > 0:
                        present = [c for c in want_cols if c in tbl.schema.names]
                        sub = tbl.select(present)
                        if writer is None:
                            avail = set(sub.schema.names)
                            writer = pq.ParquetWriter(raw_tmp_pq, sub.schema)
                        writer.write_table(sub)
                if writer is not None:
                    writer.close()
                    writer = None

                lang_expr = "CAST(language AS VARCHAR)" if "language" in avail else "'english'"
                role_expr = "CAST(role AS VARCHAR)" if "role" in avail else "'user'"
                cluster_expr = (
                    "CAST(clusters_28 AS VARCHAR)"
                    if "clusters_28" in avail
                    else "'umap'"
                )

                raw_escaped = str(raw_tmp_pq).replace("'", "''")
                tmp_escaped = str(tmp_pq).replace("'", "''")

                con = duckdb.connect(":memory:")
                try:
                    con.execute(
                        f"""
                        COPY (
                            SELECT
                                CAST(row_id AS UBIGINT) AS row_id,
                                CAST(x_decoded AS DOUBLE) AS x,
                                CAST(y_decoded AS DOUBLE) AS y,
                                CAST(model AS VARCHAR) AS model,
                                CAST(num_of_tokens AS DOUBLE) AS num_of_tokens,
                                {lang_expr} AS language,
                                {role_expr} AS role,
                                {cluster_expr} AS topic_cluster,
                                concat(
                                    'model=', CAST(model AS VARCHAR),
                                    ' | lang=', {lang_expr},
                                    ' | tokens=', CAST(CAST(round(num_of_tokens) AS BIGINT) AS VARCHAR),
                                    ' | umap=(', CAST(round(x_decoded, 2) AS VARCHAR), ', ', CAST(round(y_decoded, 2) AS VARCHAR), ')'
                                ) AS record_summary
                            FROM read_parquet('{raw_escaped}')
                            ORDER BY x, y
                        ) TO '{tmp_escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)
                        """
                    )
                finally:
                    con.close()

                tmp_pq.replace(nodes_pq)
            finally:
                if writer is not None:
                    writer.close()
                raw_tmp_pq.unlink(missing_ok=True)
                tmp_pq.unlink(missing_ok=True)

        n_nodes = pq.read_metadata(nodes_pq).num_rows
        return nodes_pq, n_nodes, archive_span

    lmsys_archive = _discover_lmsys_archive()
    lmsys_nodes_pq, indexed_nodes, archive_span = _load_or_build_full_lmsys_sidecar(
        lmsys_archive
    )
    transport_pref = os.environ.get("PYDEEPGRAPH_TRANSPORT", "auto")
    power_pref = os.environ.get("PYDEEPGRAPH_POWER_PREFERENCE", "high-performance")
    return (
        archive_span,
        indexed_nodes,
        lmsys_archive,
        lmsys_nodes_pq,
        power_pref,
        transport_pref,
    )


@app.cell
def _(
    archive_span,
    dg,
    lmsys_archive,
    lmsys_nodes_pq,
    mo,
    power_pref,
    transport_pref,
):
    scatter = mo.ui.anywidget(
        dg.plot(
            lmsys_archive,
            x="x",
            y="y",
            color={"field": "model", "scale": "viridis"},
            size={"field": "num_of_tokens"},
            sidecar=lmsys_nodes_pq,
            edges="spatial_knn",
            edge_style={
                "width": [1.2, 3.0],
                "curvature": 0.22,
                "opacity": 0.88,
                "animatePulse": True,
            },
            resolve_edges_on_hover=True,
            max_ego_neighbors=10,
            transport=transport_pref,
            power_preference=power_pref,
            height=560,
        )
    )

    def _on_zoom_ego(_: object) -> None:
        clicked = scatter.widget.clicked_row
        if not clicked:
            return
        cx = float(clicked.get("dataX", 0.0))
        cy = float(clicked.get("dataY", 0.0))
        nbrs = scatter.widget.query_clicked_neighbors()
        max_delta = 0.0
        if nbrs is not None and nbrs.num_rows > 0:
            for tx, ty in zip(
                nbrs["target_x"].to_pylist(),
                nbrs["target_y"].to_pylist(),
                strict=False,
            ):
                max_delta = max(max_delta, abs(float(tx) - cx), abs(float(ty) - cy))
        if max_delta > 1e-6:
            target_zoom = min(2000.0, max(8.0, 0.32 * archive_span / max_delta))
        else:
            target_zoom = 24.0
        scatter.widget.set_viewport(x=cx, y=cy, zoom=target_zoom)

    def _on_reset_view(_: object) -> None:
        scatter.widget.fit_view()

    sql_search = mo.ui.text(
        value="",
        placeholder="e.g. mixtral, llama, english, tokens=4...",
        label="DuckDB Sidecar Filter (`record_summary ILIKE`):",
        full_width=True,
    )
    zoom_ego_btn = mo.ui.button(
        label="Zoom to Clicked Ego-Neighborhood",
        kind="neutral",
        on_click=_on_zoom_ego,
    )
    reset_view_btn = mo.ui.button(
        label="Reset Full View",
        kind="neutral",
        on_click=_on_reset_view,
    )
    return reset_view_btn, scatter, sql_search, zoom_ego_btn


@app.cell
def _(mo, scatter, sql_search):
    _q = sql_search.value.strip()
    if _q:
        _matched = scatter.widget.filter_by_sql(
            "record_summary ILIKE ?",
            [f"%{_q}%"],
        )
        sql_filter_badge = mo.md(
            f"**Active DuckDB `row_id` Bitmask Filter:** `{_matched:,}` matching nodes for `{_q}`"
        )
    else:
        scatter.widget.filter_by_sql(None)
        sql_filter_badge = mo.md(
            "*Type in the search box above to push a binary `u32` DuckDB `row_id` bitmask filter across all pyramid levels.*"
        )
    return (sql_filter_badge,)


@app.cell
def _(
    indexed_nodes,
    lmsys_archive,
    mo,
    reset_view_btn,
    scatter,
    sql_filter_badge,
    sql_search,
    zoom_ego_btn,
):
    _state = scatter.value
    _sel_sql = _state.get("selection_sql") or ""
    _clicked = _state.get("clicked_row")

    if _clicked:
        _nbr_arrow = scatter.widget.query_clicked_neighbors()
        _nbr_rows = _nbr_arrow.to_pylist() if _nbr_arrow is not None else []
        ego_panel = mo.vstack(
            [
                mo.md(
                    f"### Ground-Truth 1-Hop kNN Neighbors for `row_id={_clicked.get('rowId')}`\n"
                    f"**Hub Coordinates:** `({float(_clicked.get('dataX', 0.0)):.3f}, {float(_clicked.get('dataY', 0.0)):.3f})` · "
                    f"**Summary:** `{_clicked.get('attributes', {}).get('record_summary', 'n/a')}`"
                ),
                mo.ui.table(_nbr_rows, pagination=True, page_size=10),
            ]
        )
    else:
        ego_panel = mo.md(
            "### Ground-Truth 1-Hop kNN Verification\n"
            "*Click any dot on the WebGPU canvas (at any zoom level `Z = 0..7`) to pin its 10 exact Euclidean nearest-neighbor Bézier arcs, then click **Zoom to Clicked Ego-Neighborhood** to frame all 10 connected nodes.*"
        )

    if _sel_sql:
        _sel_arrow = scatter.widget.query_selected(
            columns=["row_id", "model", "num_of_tokens", "language", "x", "y", "record_summary"],
            limit=250,
        )
        selection_panel = mo.vstack(
            [
                mo.md(
                    f"### Committed Brush / Lasso / Click Selection (`{_sel_arrow.num_rows}` preview rows)\n"
                    f"**Server-Validated DuckDB SQL:** `{_sel_sql}`"
                ),
                mo.ui.table(_sel_arrow.to_pylist(), pagination=True, page_size=8),
            ]
        )
    else:
        selection_panel = mo.md(
            "### Committed Selection SQL (`Brush` / `Lasso`)\n"
            "*Switch toolbar to **Brush** or **Lasso** and drag across any cluster — reactive DuckDB queries run once on `pointerup`.*"
        )

    header = mo.md(
        f"## Deepgraph + Exact DuckDB Spatial 10-NN Ego-Graph (`{lmsys_archive.name}`)\n"
        f"Indexed **`{indexed_nodes:,}`** UMAP nodes (`100%` of pyramid levels) with `< 2 ms` exact inscribed-ball DuckDB 10-NN lookup."
    )
    controls_row = mo.hstack(
        [sql_search, zoom_ego_btn, reset_view_btn],
        widths=[3, 1.15, 0.85],
        align="end",
    )
    main_layout = mo.vstack(
        [
            header,
            controls_row,
            sql_filter_badge,
            mo.hstack([scatter, ego_panel], widths=[1.35, 1.0], align="start"),
            selection_panel,
        ]
    )
    main_layout
    return ego_panel, header, main_layout, selection_panel


if __name__ == "__main__":
    app.run()
