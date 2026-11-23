---
log_id: 2026-11-22_03
date: 2026-11-22
type: WALKTHROUGH
status: executed
scope: [all]
source_artifact: walkthrough.md
conversation_id: e91adbce-084e-445b-98b1-a434c428c86d
---

# Walkthrough: `D12c` Stage 2a GitHub Renames, Multi-OS Wheel Matrix & DuckDB 2.0 Determinism Fix

## 1. Summary of Changes

### Step 1 — Path & Markdown Link Normalization (`ac55f0a` in `arrowtiles`)
- Normalized all local `file:///` and developer-machine paths across all 34 Markdown files in both repositories (`6` in `deepgraph`, `28` in `arrowtiles`) into valid repo-relative or `https://github.com/kai-erlenbusch/.../blob/main/...` links (`0` broken links, `0` local path leaks).

### Step 2 — `D12c` Stage 2a GitHub Renames & Packaging/CI Hardening
- **GitHub Repository Alignment:**
  - Renamed and archived the June 2026 prototype repository as `kai-erlenbusch/deepgraph-webgpu-legacy`.
  - Renamed `kai-erlenbusch/duckdb-arrowtiles` → [`kai-erlenbusch/arrowtiles`](https://github.com/kai-erlenbusch/arrowtiles) and `kai-erlenbusch/deepgraph-arrowtiles-sandbox` → [`kai-erlenbusch/deepgraph`](https://github.com/kai-erlenbusch/deepgraph) (preserving `"private": true` on `deepgraph` until explicit Stage 2b registry sign-off).
- **`deepgraph` CI & Tarball Verification:**
  - Pruned stale Three.js lockfile remnants and updated `.github/workflows/ci.yml` to Node 24 (`npm install --no-audit --no-fund`) to avoid `npm/cli#4828` optional cross-OS Rollup/Rolldown WASM peer resolution failures on Linux runners.
  - Verified `npm pack --dry-run`: `deepgraph-0.1.0.tgz` (`320.5 kB` packed, `1.2 MB` unpacked) ships strictly the 10 expected distribution files (`dist/*`, `docs/API.md`, `README.md`, `LICENSE`, `NOTICE`, `package.json`).
- **`arrowtiles` Multi-OS `abi3` Wheel Matrix & `pmtiles` Crate Pruning (`b3fa783`, `9bb9e83`):**
  - Added `.github/workflows/release.yml` building 5 native `abi3-py39` wheels (`linux-x86_64`, native `ubuntu-24.04-arm` `linux-aarch64`, `macos-aarch64`, `macos-x86_64`, `windows-x64`) + `sdist`, with PyPI OIDC Trusted Publishing strictly gated behind `if: startsWith(github.ref, 'refs/tags/v')`.
  - Set `default-features = false` on `pmtiles = { version = "0.23.0", default-features = false, features = ["write", "zstd"] }`, stripping `3,024` lines of unused `aws-sdk-s3` / `aws-lc-sys` BoringSSL crates from `Cargo.lock`.
  - Added explicit `"cffi>=1.15.0"` runtime dependency in `pyproject.toml` (`import pyarrow.cffi as pc` in `arrowtiles/builder.py`).

### Step 3 — DuckDB `2.0.1` + PyArrow `26.0.1` Compatibility & Canonical `NULL` Slot Sanitizer
- **Root Cause Caught by Fresh-Venv CI (`duckdb==2.0.1`, `pyarrow==26.0.1`):**
  1. **Arrow IPC `NULL`-Slot Scratch Bytes:** DuckDB 2.0's multi-threaded vectorized execution leaves uninitialized allocator scratch bytes in flat-vector value buffers at masked-out `NULL` positions (permitted by the Arrow in-memory specification), which `StreamWriter` previously wrote verbatim into per-tile Arrow IPC `RecordBatch` buffers before Zstd compression—causing logically identical archives (`pa.Table.equals == True`) with nullable columns (`grp` `UInt8`) to have different compressed tile SHA-256 digests across consecutive runs.
  2. **DB-API Cursor Exhaustion & Relation Arrow Exports:** DuckDB 2.0 raises `_duckdb.ResourceInUseException` if `.execute(...).fetchone()` is followed by `con.register(...)` before exhausting the cursor, and emits `UserWarning` when calling `.to_arrow_reader()` / `.to_arrow_table()` on `.execute(...)` instead of `con.sql(...)`.
- **Fixes Applied:**
  - Added `canonicalize_tile_column` using `arrow::array::MutableArrayData` in `arrowtiles-engine/src/lib.rs` (`flush_chunk_buffer`) so every serialized tile column has `offset == 0`, zeroed validity/boolean tail padding bits, and deterministic zero bytes / empty slices at all `NULL` positions regardless of upstream DuckDB/Arrow allocator scratch memory.
  - Updated `arrowtiles/builder.py`, `scripts/build_samples.py`, and `tests/` to use `con.sql(...).to_arrow_reader(batch_size=100_000)` / `con.sql(...).to_arrow_table()` for zero-extra-copy Arrow streaming and `.fetchall()[0]` for scalar/single-row SQL queries across both DuckDB `1.x` and `2.0+`.
  - Expanded `test_byte_identical_sha256_unique_and_tied_priorities` in `tests/test_determinism_and_completeness.py` to cover nullable `UInt8`, nullable `Float64`, nullable `Boolean`, and nullable `Utf8` columns simultaneously.

---

## 2. Verification Results

- **Local Pytest (`duckdb==1.5.3` + `pyarrow==24.0.0`, `-W error::UserWarning`):** **`53 passed in 18.03s`** (`0` failures, `0` warnings).
- **Local Pytest (`duckdb==2.0.1` + `pyarrow==26.0.1`, `-W error::UserWarning`):** **`53 passed in 26.14s`** (`0` failures, `0` warnings).
- **GitHub Actions (`kai-erlenbusch/deepgraph` `CI` + `kai-erlenbusch/arrowtiles` `CI` & `Release Wheels & PyPI` dry-run):** Verified green on `main` with PyPI/npm registry publishing held gated behind Stage 2b owner sign-off.
