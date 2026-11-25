import marimo

__generated_with = "0.26.0"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo

    import pydeepgraph as dg

    return Path, dg, mo, os


@app.cell
def _(Path, os):
    def _discover_gaia_archive() -> Path:
        env_override = os.environ.get("PYDEEPGRAPH_GAIA_ARCHIVE")
        if env_override:
            p = Path(env_override).expanduser().resolve()
            if p.is_file():
                return p
            raise FileNotFoundError(f"PYDEEPGRAPH_GAIA_ARCHIVE not found: {p}")

        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / "public" / "gaia_full.arrowtiles",
            repo_root.parent / "arrowtiles" / "samples" / "gaia_sample.arrowtiles",
            repo_root.parent / "duckdb-arrowtiles" / "samples" / "gaia_sample.arrowtiles",
        ]
        for cand in candidates:
            if cand.is_file():
                return cand.resolve()
        raise FileNotFoundError(
            "No Gaia .arrowtiles archive found. Place gaia_full.arrowtiles in public/ or build ../arrowtiles/samples."
        )

    gaia_archive = _discover_gaia_archive()
    archive_gb = gaia_archive.stat().st_size / (1024**3)
    power_pref = os.environ.get("PYDEEPGRAPH_POWER_PREFERENCE", "high-performance")
    transport_pref = os.environ.get("PYDEEPGRAPH_TRANSPORT", "auto")
    return archive_gb, gaia_archive, power_pref, transport_pref


@app.cell
def _(dg, gaia_archive, mo, power_pref, transport_pref):
    projection_toggle = mo.ui.radio(
        options={
            "Galactic Sky Map (Spatial Optical Splatting)": "sky",
            "Hertzsprung–Russell Diagram (bp_rp vs phot_g_mean_mag)": "hr",
        },
        value="Galactic Sky Map (Spatial Optical Splatting)",
        label="Projection Mode (2-Second GPU Transition):",
    )

    # `max_selected_row_ids=0` keeps `enableRowIdIndex=False` so Milky Way brushes across
    # multi-million loaded stars transmit only the compact `< 500 B` geometric `selection_clause`.
    gaia_scatter = mo.ui.anywidget(
        dg.plot(
            gaia_archive,
            x="x_hammer",
            y="y_hammer",
            row_id="row_id",
            encoding={
                "color": {"field": "bp_rp", "domain": [-0.5, 3.2]},
            },
            view_options={"pipeline": "optical", "flipY": False},
            max_selected_row_ids=0,
            transport=transport_pref,
            power_preference=power_pref,
            height=620,
        )
    )
    return gaia_scatter, projection_toggle


@app.cell
def _(gaia_scatter, projection_toggle):
    if projection_toggle.value == "hr":
        gaia_scatter.widget.encoding = {
            "x": {"field": "bp_rp", "domain": [-0.6, 4.5]},
            "y": {"field": "phot_g_mean_mag", "domain": [3.0, 21.5]},
            "color": {"field": "bp_rp", "scale": "coolwarm", "domain": [-0.5, 3.5]},
            "size": {"field": "phot_g_mean_mag", "domain": [3.0, 21.0], "range": [2.4, 0.55]},
            "flipY": True,
            "aggregation": "points",
        }
        gaia_scatter.widget.view_options = {
            "pipeline": "cartesian",
            "aggregation": "points",
            "flipY": True,
        }
        gaia_scatter.widget.fit_view()
    else:
        # Set `encoding` (including `"flipY": False` so the 2s HR -> Sky morph transitions atomically!)
        # BEFORE `view_options={"pipeline": "optical"}` so optical RGB mode stays active!
        gaia_scatter.widget.encoding = {
            "x": None,
            "y": None,
            "color": {"field": "bp_rp", "domain": [-0.5, 3.2]},
            "flipY": False,
        }
        gaia_scatter.widget.view_options = {"pipeline": "optical", "flipY": False}
        gaia_scatter.widget.fit_view()
    return


@app.cell
def _(archive_gb, gaia_archive, gaia_scatter, mo, projection_toggle):
    _state = gaia_scatter.value
    _sel_sql = _state.get("selection_sql") or "*No active brush/lasso selection*"
    _clicked = _state.get("clicked_row")

    view = mo.vstack(
        [
            mo.md(
                f"## Gaia DR3 Zero-Copy WebGPU Cruiser (`{gaia_archive.name}` — `{archive_gb:.2f} GiB`)\n"
                "Streams Zstd-compressed `.arrowtiles` leaves via read-only OS `mmap` (`0 B` heap copy) with full `u64` `source_id` precision."
            ),
            projection_toggle,
            gaia_scatter,
            mo.md(
                f"**Server-Validated DuckDB Selection Predicate:** `{_sel_sql}`  \n"
                f"**Selected Star (`u64` safe):** `{_clicked}`"
            ),
        ]
    )
    view
    return (view,)


if __name__ == "__main__":
    app.run()
