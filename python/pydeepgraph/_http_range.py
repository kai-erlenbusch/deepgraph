from __future__ import annotations

import atexit
import mmap
import os
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

OPEN_RANGE_CAP_BYTES = 16 * 1024 * 1024  # 16 MiB cap on open-ended `bytes=start-`
EXPLICIT_RANGE_MAX_BYTES = 64 * 1024 * 1024  # 64 MiB max explicit range
_RANGE_RE = re.compile(r"^bytes=(\d+)-(\d*)$")
_FileIdentityKey = tuple[Path, int, int]


class _MappedArchiveEntry:
    __slots__ = (
        "path",
        "file_key",
        "size",
        "refcount",
        "_file",
        "_mmap",
        "_lock",
        "_active_readers",
        "_closed",
    )

    def __init__(self, path: Path, file_key: _FileIdentityKey) -> None:
        self.path = path
        self.file_key = file_key
        self.size = file_key[1]
        self.refcount = 0
        self._file = self.path.open("rb")
        self._mmap: mmap.mmap | None = (
            mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            if self.size > 0
            else None
        )
        self._lock = threading.Lock()
        self._active_readers = 0
        self._closed = False

    def acquire_view(self, offset: int, length: int) -> memoryview | None:
        with self._lock:
            if self._closed or length <= 0 or self._mmap is None:
                return None
            mv = memoryview(self._mmap)[offset : offset + length]
            self._active_readers += 1
            return mv

    def release_view(self, mv: memoryview | None) -> None:
        if mv is not None:
            mv.release()
        with self._lock:
            if self._active_readers > 0:
                self._active_readers -= 1
            if self._closed and self._active_readers == 0:
                self._finalize_close_locked()

    def _finalize_close_locked(self) -> None:
        if self._mmap is not None:
            try:
                self._mmap.close()
            except BufferError:
                pass
            self._mmap = None
        try:
            self._file.close()
        except Exception:
            pass

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._active_readers == 0:
                self._finalize_close_locked()


class _RangeRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 30

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        # Silence per-tile HTTP stdout noise inside Jupyter / Marimo notebooks
        return

    def _add_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Range, If-None-Match, Content-Type, Access-Control-Request-Private-Network",
        )
        self.send_header(
            "Access-Control-Expose-Headers",
            "Content-Range, Content-Length, Accept-Ranges, ETag",
        )
        # Chrome Private Network Access (HTTPS Colab / JupyterHub -> 127.0.0.1)
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._add_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self) -> None:  # noqa: N802
        self._serve_range(send_body=False)

    def do_GET(self) -> None:  # noqa: N802
        self._serve_range(send_body=True)

    def _serve_range(self, *, send_body: bool) -> None:
        daemon: LocalRangeDaemon = self.server.daemon  # type: ignore[attr-defined]
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) != 2 or parts[0] != "tiles" or not parts[1].endswith(".arrowtiles"):
            self.send_response(403)
            self._add_cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        token = parts[1][: -len(".arrowtiles")]
        entry = daemon.get_entry(token)
        if entry is None:
            self.send_response(403)
            self._add_cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        range_hdr = self.headers.get("Range")
        if not range_hdr:
            chunk_len = min(entry.size, OPEN_RANGE_CAP_BYTES) if send_body else entry.size
            is_partial = send_body and chunk_len < entry.size and chunk_len > 0
            self.send_response(206 if is_partial else 200)
            self._add_cors_headers()
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            if is_partial:
                self.send_header("Content-Range", f"bytes 0-{chunk_len - 1}/{entry.size}")
            self.send_header("Content-Length", str(chunk_len))
            self.end_headers()
            if send_body and chunk_len > 0:
                mv = entry.acquire_view(0, chunk_len)
                try:
                    if mv is not None:
                        self.wfile.write(mv)
                except ConnectionError:
                    pass
                finally:
                    entry.release_view(mv)
            return

        m = _RANGE_RE.match(range_hdr.strip())
        if not m:
            self.send_response(416)
            self._add_cors_headers()
            self.send_header("Content-Range", f"bytes */{entry.size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        start = int(m.group(1))
        has_end = m.group(2) != ""
        end = (
            min(int(m.group(2)), entry.size - 1)
            if has_end
            else min(start + OPEN_RANGE_CAP_BYTES - 1, entry.size - 1)
        )
        if start < 0 or start >= entry.size or start > end:
            self.send_response(416)
            self._add_cors_headers()
            self.send_header("Content-Range", f"bytes */{entry.size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        chunk_len = end - start + 1
        if has_end and chunk_len > EXPLICIT_RANGE_MAX_BYTES:
            self.send_response(413)
            self._add_cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        self.send_response(206)
        self._add_cors_headers()
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{entry.size}")
        self.send_header("Content-Length", str(chunk_len))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()

        if send_body and chunk_len > 0:
            mv = entry.acquire_view(start, chunk_len)
            try:
                if mv is not None:
                    self.wfile.write(mv)
            except ConnectionError:
                pass
            finally:
                entry.release_view(mv)


class LocalRangeDaemon:
    """Process-shared `127.0.0.1` HTTP 206 byte-range server backed by ref-counted read-only `mmap`."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._lock = threading.Lock()
        self._entries_by_file: dict[_FileIdentityKey, _MappedArchiveEntry] = {}
        self._token_to_entry: dict[str, _MappedArchiveEntry] = {}
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._host = host
        self._requested_port = port
        self.port: int = 0

    def start(self) -> int:
        with self._lock:
            if self._server is not None:
                return self.port
            srv = ThreadingHTTPServer((self._host, self._requested_port), _RangeRequestHandler)
            srv.daemon_threads = True
            srv.daemon = self  # type: ignore[attr-defined]
            self.port = int(srv.server_address[1])
            t = threading.Thread(
                target=srv.serve_forever,
                name="pydeepgraph-http-range",
                daemon=True,
            )
            t.start()
            self._server = srv
            self._thread = t
            return self.port

    def register_file(self, path: str | os.PathLike[str]) -> tuple[str, str]:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Archive not found: {resolved}")
        st = resolved.stat()
        fkey: _FileIdentityKey = (resolved, int(st.st_size), int(st.st_mtime_ns))
        port = self.start()
        with self._lock:
            entry = self._entries_by_file.get(fkey)
            if entry is None:
                entry = _MappedArchiveEntry(resolved, fkey)
                self._entries_by_file[fkey] = entry
            entry.refcount += 1
            # Mint a unique per-registration bearer token so closing/GC'ing one widget
            # never revokes sibling widgets or active Marimo re-run instances!
            token = secrets.token_urlsafe(18)
            while token in self._token_to_entry:
                token = secrets.token_urlsafe(18)
            self._token_to_entry[token] = entry
            return token, f"http://127.0.0.1:{port}/tiles/{token}.arrowtiles"

    def get_entry(self, token: str) -> _MappedArchiveEntry | None:
        with self._lock:
            return self._token_to_entry.get(token)

    def unregister_token(self, token: str) -> None:
        to_close: _MappedArchiveEntry | None = None
        with self._lock:
            entry = self._token_to_entry.pop(token, None)
            if entry is not None:
                entry.refcount -= 1
                if entry.refcount <= 0:
                    self._entries_by_file.pop(entry.file_key, None)
                    to_close = entry
        if to_close is not None:
            to_close.close()

    def active_paths(self) -> set[Path]:
        with self._lock:
            return {e.path.resolve() for e in self._entries_by_file.values()}

    def shutdown(self) -> None:
        with self._lock:
            srv = self._server
            self._server = None
            entries = list(self._entries_by_file.values())
            self._entries_by_file.clear()
            self._token_to_entry.clear()
        for e in entries:
            e.close()
        if srv is not None:
            srv.shutdown()
            srv.server_close()


_GLOBAL_DAEMON_LOCK = threading.Lock()
_GLOBAL_DAEMON: LocalRangeDaemon | None = None


def get_shared_range_daemon() -> LocalRangeDaemon:
    global _GLOBAL_DAEMON
    with _GLOBAL_DAEMON_LOCK:
        if _GLOBAL_DAEMON is None:
            _GLOBAL_DAEMON = LocalRangeDaemon()
            atexit.register(_GLOBAL_DAEMON.shutdown)
        return _GLOBAL_DAEMON


def get_shared_range_daemon_if_running() -> LocalRangeDaemon | None:
    with _GLOBAL_DAEMON_LOCK:
        return _GLOBAL_DAEMON
