# `pydeepgraph` — WebGPU Billion-Point & Graph Widget for Python, DuckDB & Marimo

`pydeepgraph` brings [`deepgraph`](https://github.com/kai-erlenbusch/deepgraph)'s pure-WebGPU `.arrowtiles` pyramid engine into **Marimo**, **JupyterLab**, **VS Code Notebooks**, and **Google Colab** via [`anywidget`](https://anywidget.dev/).

## Quick Start

```python
import pydeepgraph as dg

# 1. Stream an existing .arrowtiles archive directly from NVMe or HTTPS (0 B copied into RAM)
w = dg.plot("lmsys_full_v01.arrowtiles", color="model")

# 2. Or pass any Polars / Pandas / PyArrow / DuckDB relation (auto-built & cached via arrowtiles)
w = dg.plot(df, x="umap_x", y="umap_y", color="cluster", edges="spatial_knn")
```

### Reactive Marimo Integration (`mo.ui.anywidget`)

```python
import marimo as mo
import pydeepgraph as dg

view = mo.ui.anywidget(
    dg.plot(
        "lmsys_full_v01.arrowtiles",
        x="x",
        y="y",
        sidecar=prompts_parquet,
        edges="spatial_knn",
        interaction_mode="brush",
    )
)
view
```

Downstream cells react strictly on committed selections (`click`, `brush-end`, `lasso-end`, `clear`) via server-validated `view.value["selection_sql"]`, `view.widget.query_selected()`, `view.widget.query_clicked_neighbors()`, `view.value["selected_row_ids"]`, and `view.value["clicked_row"]`.

## License

Copyright (c) 2026 Kai Erlenbusch. Licensed under **CC BY-NC 4.0** (`LICENSE`) for non-commercial use; commercial licensing inquiries: `erlenbusch@uchicago.edu`.
