import { defineConfig, type Plugin } from 'vite';
import fs from 'node:fs';
import path from 'node:path';

interface CachedFd {
  fd: number;
  size: number;
  mtimeMs: number;
}

const OPEN_RANGE_CAP_BYTES = 16 * 1024 * 1024; // Cap open-ended `bytes=start-` slices at 16 MiB
const EXPLICIT_RANGE_MAX_BYTES = 64 * 1024 * 1024; // Allow explicit `bytes=start-end` up to 64 MiB; reject larger with 413

function isContainedIn(rootDir: string, candidatePath: string): boolean {
  const rel = path.relative(rootDir, candidatePath);
  return rel !== '' && !rel.startsWith('..') && !path.isAbsolute(rel);
}

/**
 * Serves HTTP Range requests on massive `.arrowtiles` files (e.g. 31 GB `gaia_full.arrowtiles`)
 * using a persistent OS file descriptor (`fs.read`) instead of `sirv`'s per-request
 * `fs.open` + `fs.createReadStream` + stream pipe + `fs.close` lifecycle.
 * Cuts local NVMe byte-range response latency from ~25–35ms down to ~0.5–2ms.
 */
function arrowtilesFastRangePlugin(): Plugin {
  const fdCache = new Map<string, CachedFd>();

  const closeCachedFd = (absPath: string) => {
    const existing = fdCache.get(absPath);
    if (existing) {
      try {
        fs.closeSync(existing.fd);
      } catch {
        // Ignore close errors on stale handle cleanup
      }
      fdCache.delete(absPath);
    }
  };

  const getOrOpenFd = (absPath: string): CachedFd | null => {
    let stat: fs.Stats;
    try {
      stat = fs.statSync(absPath);
    } catch {
      closeCachedFd(absPath);
      return null;
    }
    if (!stat.isFile()) {
      closeCachedFd(absPath);
      return null;
    }

    const cached = fdCache.get(absPath);
    if (cached && cached.size === stat.size && cached.mtimeMs === stat.mtimeMs) {
      return cached;
    }
    if (cached) {
      closeCachedFd(absPath);
    }

    const fd = fs.openSync(absPath, 'r');
    const entry: CachedFd = { fd, size: stat.size, mtimeMs: stat.mtimeMs };
    fdCache.set(absPath, entry);
    return entry;
  };

  return {
    name: 'arrowtiles-fast-range-server',
    configureServer(server) {
      const workspaceRoot = path.resolve(process.cwd());
      const publicRoot = path.resolve(workspaceRoot, 'public');
      const preferredSamplesRoot = path.resolve(workspaceRoot, '..', 'arrowtiles', 'samples');
      const siblingSamplesRoot = fs.existsSync(preferredSamplesRoot)
        ? preferredSamplesRoot
        : path.resolve(workspaceRoot, '..', 'duckdb-arrowtiles', 'samples');

      server.httpServer?.on('close', () => {
        for (const entry of fdCache.values()) {
          try {
            fs.closeSync(entry.fd);
          } catch {
            // Ignore close errors on shutdown
          }
        }
        fdCache.clear();
      });

      server.middlewares.use((req, res, next) => {
        if (!req.url || (req.method !== 'GET' && req.method !== 'HEAD')) {
          return next();
        }

        const rawPath = req.url.split('?')[0];
        let cleanUrl: string;
        try {
          cleanUrl = decodeURIComponent(rawPath);
        } catch {
          res.statusCode = 400;
          res.end('Bad Request: Malformed URL encoding');
          return;
        }

        if (!cleanUrl.endsWith('.arrowtiles')) {
          return next();
        }

        if (cleanUrl.includes('\0')) {
          res.statusCode = 400;
          res.end('Bad Request');
          return;
        }

        const relPath = cleanUrl.replace(/^\/+/, '');
        if (path.win32.isAbsolute(relPath) || path.posix.isAbsolute(relPath)) {
          res.statusCode = 403;
          res.end('Forbidden');
          return;
        }

        const publicCandidate = path.resolve(publicRoot, relPath);
        const rootCandidate = path.resolve(workspaceRoot, relPath);

        if (
          !isContainedIn(publicRoot, publicCandidate) ||
          !isContainedIn(workspaceRoot, rootCandidate)
        ) {
          res.statusCode = 403;
          res.end('Forbidden');
          return;
        }

        const rangeHeader = req.headers.range;
        if (!rangeHeader || !rangeHeader.startsWith('bytes=')) {
          return next();
        }

        let samplesCandidate: string | null = null;
        if (relPath.startsWith('samples/') && !relPath.slice('samples/'.length).includes('..')) {
          const subRel = relPath.slice('samples/'.length);
          const cand = path.resolve(siblingSamplesRoot, subRel);
          if (isContainedIn(siblingSamplesRoot, cand)) {
            samplesCandidate = cand;
          }
        }

        const entry =
          getOrOpenFd(publicCandidate) ||
          getOrOpenFd(rootCandidate) ||
          (samplesCandidate ? getOrOpenFd(samplesCandidate) : null);
        if (!entry) {
          return next();
        }

        const match = /^bytes=(\d+)-(\d*)$/.exec(rangeHeader);
        if (!match) {
          return next();
        }

        const start = Number(match[1]);
        const hasExplicitEnd = match[2] !== '';
        const targetEnd = hasExplicitEnd
          ? Math.min(Number(match[2]), entry.size - 1)
          : Math.min(start + OPEN_RANGE_CAP_BYTES - 1, entry.size - 1);

        if (
          Number.isNaN(start) ||
          Number.isNaN(targetEnd) ||
          start > targetEnd ||
          start >= entry.size
        ) {
          res.statusCode = 416;
          res.setHeader('Content-Range', `bytes */${entry.size}`);
          res.end();
          return;
        }

        const chunkLen = targetEnd - start + 1;
        if (hasExplicitEnd && chunkLen > EXPLICIT_RANGE_MAX_BYTES) {
          res.statusCode = 413;
          res.end('Payload Too Large: Explicit byte range exceeds 64 MiB limit');
          return;
        }

        res.statusCode = 206;
        res.setHeader('Content-Type', 'application/octet-stream');
        res.setHeader('Accept-Ranges', 'bytes');
        res.setHeader('Content-Range', `bytes ${start}-${targetEnd}/${entry.size}`);
        res.setHeader('Content-Length', chunkLen);
        res.setHeader('Cache-Control', 'public, max-age=3600');

        if (req.method === 'HEAD') {
          res.end();
          return;
        }

        const buf = Buffer.allocUnsafe(chunkLen);
        fs.read(entry.fd, buf, 0, chunkLen, start, (err, bytesRead) => {
          if (err) {
            if (!res.headersSent) res.statusCode = 500;
            res.end();
            return;
          }
          res.end(bytesRead === chunkLen ? buf : buf.subarray(0, bytesRead));
        });
      });
    }
  };
}

export default defineConfig({
  plugins: [arrowtilesFastRangePlugin()],
  assetsInclude: ['**/*.arrowtiles', '**/*.parquet', '**/*.duckdb'],
  build: {
    // public/ holds multi-GB .arrowtiles datasets served via HTTP Range; never copy 59 GB into dist/
    copyPublicDir: false,
  },
  server: {
    port: 5173,
    host: '127.0.0.1',
    strictPort: true,
    fs: {
      strict: true,
      allow: ['.']
    },
    watch: {
      ignored: ['.logs/**', 'research/**', '**/*.parquet', '**/s3_cache/**', '**/data/**', '**/dist/**', '**/*.arrowtiles', '.gemini/**', '.system_generated/**', '**/.gemini/**', '**/.system_generated/**']
    }
  },
  optimizeDeps: {
    exclude: ['apache-arrow', 'pmtiles', 'lil-gui']
  }
});
