---
log_id: 2026-11-22_05
date: 2026-11-22
type: WALKTHROUGH
status: executed
scope: [all]
source_artifact: walkthrough.md
conversation_id: e91adbce-084e-445b-98b1-a434c428c86d
---

# Walkthrough: `D12c` Stage 2b — `v0.1.0` Public Release (`deepgraph` & `arrowtiles`)

## 1. Release Summary

| Ecosystem Target | Version & License | Git Tag / Commit | Public Registry / Release URL |
| :--- | :--- | :---: | :--- |
| **`arrowtiles` (PyPI)** | `0.1.0` (`Apache-2.0`) | `v0.1.0` (`5c10adb`) | `https://pypi.org/project/arrowtiles/0.1.0/` (`5` `cp39-abi3` wheels + `sdist`) |
| **`deepgraph` (npm)** | `0.1.0` (`CC-BY-NC-4.0` + Commercial via `erlenbusch@uchicago.edu`) | `v0.1.0` (`cc24d00`) | `https://www.npmjs.com/package/deepgraph` (`10` files, `323.2 kB` packed) |
| **`arrowtiles` (GitHub)** | `Apache-2.0` + `SPEC.md` `v0.1` | `v0.1.0` / `ab73901` | `https://github.com/kai-erlenbusch/arrowtiles/releases/tag/v0.1.0` (`6` assets) |
| **`deepgraph` (GitHub)** | `CC-BY-NC-4.0` + Commercial | `v0.1.0` (`cc24d00`) | `https://github.com/kai-erlenbusch/deepgraph/releases/tag/v0.1.0` |

---

## 2. Execution Details

1. **`deepgraph` (`v0.1.0` — npm & GitHub):**
   - Removed `"private": true` in `package.json` (`cc24d00`), updated `docs/API.md` and `AGENTS.md`, and verified `npm run typecheck`, `npm run build:lib` (`dist/index.js` `46.61 kB` gzip, `dist/standalone.js` `99.96 kB` gzip, self-contained `pmtiles.worker` with inlined `zstd.wasm`), and `npm run verify:types` (`bundler` + `nodenext` + real Vite consumer app).
   - Tagged `v0.1.0`, verified GitHub Actions `CI` run `45099691291` (`success`), created GitHub Release `v0.1.0`, and published `deepgraph@0.1.0` to `registry.npmjs.org` (`shasum: e554f341fcc153038e7db2cc94fd499dd39dcfed`, `10` files, `1.20 MB` unpacked).
2. **`arrowtiles` (`v0.1.0` — PyPI & GitHub):**
   - Configured PyPI OIDC Trusted Publisher (`kai-erlenbusch/arrowtiles`, `release.yml`, environment `pypi`) and pushed annotated tag `v0.1.0` (`5c10adb`).
   - GitHub Actions `Release Wheels & PyPI` run `45099688720` built and uploaded all 6 distributions to PyPI (`manylinux2014_x86_64`, `manylinux2014_aarch64`, `macosx_11_0_arm64`, `macosx_10_12_x86_64`, `win_amd64` `cp39-abi3` wheels + `arrowtiles-0.1.0.tar.gz`).
   - Attached all 6 distributions to GitHub Release `v0.1.0` and patched `.github/workflows/release.yml` (`ab73901`) to pass `--repo "${GITHUB_REPOSITORY}"` explicitly on `gh release` commands in jobs without a git checkout.

---

## 3. Live Registry Verification

- **npm (`npm view deepgraph@0.1.0 --json`):**
  - `version = "0.1.0"`, `license = "CC-BY-NC-4.0"`, `author = "Kai Erlenbusch <erlenbusch@uchicago.edu>"`, `exports["."] = "./dist/index.js"`, `exports["./standalone"] = "./dist/standalone.js"`.
- **PyPI (`https://pypi.org/pypi/arrowtiles/0.1.0/json`):**
  - `version = "0.1.0"`, `license = "Apache-2.0"`, `requires_python = ">=3.10"`, `6` distribution files verified.
