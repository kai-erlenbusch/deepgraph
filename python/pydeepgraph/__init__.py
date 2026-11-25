from __future__ import annotations

from ._edges import DuckDbEdgeResolver
from ._http_range import LocalRangeDaemon, get_shared_range_daemon
from ._ingest import SidecarStore, get_cache_dir, resolve_dataset_input
from .widget import COMM_CHUNK_BYTES, DeepgraphWidget, build_lasso_sql, plot

__version__ = "0.1.0"

__all__ = [
    "COMM_CHUNK_BYTES",
    "DeepgraphWidget",
    "DuckDbEdgeResolver",
    "LocalRangeDaemon",
    "SidecarStore",
    "__version__",
    "build_lasso_sql",
    "get_cache_dir",
    "get_shared_range_daemon",
    "plot",
    "resolve_dataset_input",
]
