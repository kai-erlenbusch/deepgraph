---
log_id: 2026-11-24_03
date: 2026-11-24
type: WALKTHROUGH
status: executed
scope: [all]
source_artifact: step3b_pydeepgraph_marimo_walkthrough.md
conversation_id: 08f564f2-95c1-4b5f-b446-5fb13d4d48ab
---

# Step 3B Execution Walkthrough: `pydeepgraph==0.1.0` AnyWidget, Reactive Marimo DuckDB 10-NN & Pre-Release Hardening

## 1. Summary of Completed Work

Implemented, reviewed, and pre-release hardened **Step 3B (`pydeepgraph==0.1.0`)** inside `python/` and `src/widget/` alongside **Step 3A/3B TypeScript updates (`deepgraph@0.1.2`)**, with zero `.arrowtiles` `v0.1` wire-format changes:

1. **Self-Contained `anywidget` ESM Bundle (`src/widget/widget_entry.ts`, `vite.widget.config.ts`, `python/scripts/sync_assets.mjs`):**
   - Bundles `deepgraph`, `@uwdata/flechette`, `pmtiles`, `p-queue`, `fflate`, and the cached `Blob` worker + base64 `zstd.wasm` into `python/pydeepgraph/_static/widget.js` (`758.98 kB` raw / `228.60 kB` gzipped, `0` external module specifiers).
   - Syncs committed reactive traits (`selected_row_ids`, `selected_row_ids_count`, `selected_row_ids_truncated`, `clicked_row`, `selection_clause`) only on `pointerup` / `click` / `clear` (`0` `model.save_changes()` calls during `pointermove` hover), while `selection_sql` is reconstructed and column-validated server-side in Python (`x_column`, `y_column`, and `row_id_column` kept Python-only and server-owned traits protected against client writes).
   - Preserves cold sidecar summary/prompt priority ahead of retained GPU tile columns in `ViewerLayer.ts` and `.dg-anywidget-inspect`, replays pre-mount/reload binary `row_id` filters on `frontend_ready`, and guards `Deepgraph.create()` / `onArchiveChange` against adapter or abort failures.
2. **Reader-Refcounted HTTP `206` & Bounded `<= 512 KiB` `CommRangeSource` Transports (`python/pydeepgraph/_http_range.py`, `python/pydeepgraph/widget.py`):**
   - **Localhost HTTP `206` Fast Path (`transport="auto"` / `"http"`, strict `206` probe):** Process-shared `127.0.0.1:0` `ThreadingHTTPServer` (`timeout = 30`) minting unique per-widget 128-bit bearer tokens backed by reader-refcounted `(path, size, mtime_ns)` `mmap` entries (`acquire_view` / `release_view`), with lazy `self._archive_mmap` in `DeepgraphWidget`.
   - **`<= 512 KiB` Binary WebSocket Fallback (`transport="comm"`):** Multi-part `range_req` / `range_res` chunking (`COMM_CHUNK_BYTES = 524_288`, `maxLeafPrefetchBytes = 262_144`, `<= 64 MiB` per-request cap, and `bool`/parse validation).
3. **Universal DataFrame Ingestion, `UBIGINT`/Signed `BIGINT` Sidecar & Exact Inscribed-Ball 2D Spatial kNN (`python/pydeepgraph/_ingest.py`, `python/pydeepgraph/_edges.py`):**
   - `resolve_dataset_input`: Preserves `arrowtiles` `D7a` high-cardinality free-text auto-dropping (`columns=columns, keep_source_xy=False`), formats DuckDB's 128-bit `HUGEINT` content-hash sum as `::VARCHAR` (preventing `UINT64` overflow past `131,072` rows), hashes `row_number() OVER ()` when `row_id` is absent, protects open `.parquet` / `.arrowtiles` files during `2 GiB` LRU cache pruning, and unlinks `.tmp.*` files in `finally:`.
   - `SidecarStore` & `DuckDbEdgeResolver`: Full `UInt64` (`0 .. 2^64 - 1`) and negative two's-complement `BIGINT` support, native Arrow `.to_arrow_table()` output in `query_neighbors_arrow()`, parameterized `filter_by_sql(where_sql, params=None)` (`10M` default binary row ceiling), and exact inscribed-ball 2D Euclidean spatial kNN (`edges="spatial_knn"`).
   - Worker & Mosaic Hardening (`src/PMTilesClient.ts`, `src/mosaic.ts`): Drops `cachedTileRowIndex` on aborted worker decodes and deduplicates across workers, bounds `ensureResidentTilesIndexed` concurrency (`<= 6`), adds `evalSeq` stale-query protection + `IS NOT NULL` + multi-chunk `RecordBatch` typed-array extraction in `deepgraph/mosaic`, and bounds lasso coordinates at `1e150` against DuckDB SIMD `DOUBLE` overflow.
4. **Reactive Marimo Showcases & OIDC-Isolated Release Workflow (`python/examples/*`, `.github/workflows/release.yml`, `AGENTS.md`):**
   - `marimo_lmsys_mosaic.py`: Streams all pyramid levels (`2,835,490` rows on `lmsys_full_v01.arrowtiles`) via `pq.ParquetWriter` into an `(x, y)`-sorted Parquet sidecar (`0.65s` warm load) with `edges="spatial_knn"`, adaptive ego-zoom, and 1-Hop Neighbor Verification Table.
   - `marimo_gaia_1_35b.py`: Uses `x="x_hammer", y="y_hammer"`, `pipeline="optical"`, `max_selected_row_ids=0`, and atomic `"flipY": False` `HR → Sky` transitions.
   - `.github/workflows/release.yml`: Splits unprivileged `build-npm` / `build-pypi` (`contents: read`) from minimal `publish-npm` / `publish-pypi` jobs (`id-token: write`, `contents: write`, SHA-pinned `gh-action-pypi-publish`, tag + `__version__` assertions).

---

## 2. Verification Results

- `npm run typecheck`: **PASS** (`0` errors)
- `npm run build:lib && npm run build:widget && npm run verify:types && npm run build:demo`: **PASS** (`index.js` `73.56 kB` gz, `standalone.js` `223.81 kB` gz, `mosaic.js` `4.86 kB` gz, `widget.js` `228.60 kB` gz)
- `npm run test:unit`: **20 / 20 passed**
- `python -m build --no-isolation ./python`: **PASS** (`pydeepgraph-0.1.0-py3-none-any.whl` + `pydeepgraph-0.1.0.tar.gz`, `0` warnings)
- `pytest python/tests -v -W error::UserWarning -W "error::DeprecationWarning:pydeepgraph.*" -W error::SyntaxWarning`: **12 / 12 passed** (`0` warnings)
- `python -m marimo check python/examples/marimo_lmsys_mosaic.py python/examples/marimo_gaia_1_35b.py`: **PASS**
- `npm run test:e2e`: **7 / 7 passed** across all `6` Playwright specs (`1.4m` on Intel `xe-lpg` iGPU, `0 MiB` touched on RTX 5090, `liveBytes === 0`)
- `npm pack --dry-run`: **PASS** (`deepgraph-0.1.2.tgz`, `460.5 kB`, `12` files; `arrowtiles==0.1.1` verified live on PyPI)
