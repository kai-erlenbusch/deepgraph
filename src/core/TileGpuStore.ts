import type { TileData } from '../PMTilesClient.ts';
import type {
    BoundingBox,
    ColorScaleName,
    ExperimentalColorScaleName,
} from '../lib/types.ts';

export type BuiltinColorScaleName = ColorScaleName;
export type { ExperimentalColorScaleName };

export type AnyColorScaleName = BuiltinColorScaleName | ExperimentalColorScaleName;

export interface InternalAxisConfig {
    field: string;
    transform: 'literal' | 'linear';
    domain: [number, number];
    range: [number, number];
}

export interface InternalEncodingConfig {
    x: InternalAxisConfig;
    y: InternalAxisConfig;
    color: { field: string; range: AnyColorScaleName; domain: [number, number] };
    size: { field: string; domain: [number, number]; range: [number, number] };
}

export const DEFAULT_CATEGORICAL_PALETTE: readonly number[] = [
    // 0..9: Tableau 10
    0x4e79a7, 0xf28e2c, 0xe15759, 0x76b7b2, 0x59a14f,
    0xedc949, 0xaf7aa1, 0xff9da7, 0x9c755f, 0xbab0ab,
    // 10..19: Tableau 20 light companions
    0xa0cbe8, 0xffbe7d, 0xff9d9a, 0x86bcb6, 0x8cd17d,
    0xf1ce63, 0xd4a6c8, 0xfabfd2, 0xd7b5a6, 0x79706e,
    // 20..31: Glasbey / Kelly high-contrast accents
    0x1f77b4, 0xd62728, 0x2ca02c, 0x9467bd, 0x8c564b,
    0xe377c2, 0xbcbd22, 0x17becf, 0x393b79, 0x637939,
    0x8c6d31, 0x843c39,
    // 32..63: 32 distinct perceptually spaced hues/tones
    0x7b4173, 0x3182bd, 0xe6550d, 0x31a354, 0x756bb1, 0x6baed6, 0xfd8d3c, 0x74c476,
    0x9e9ac8, 0x9ecae1, 0xfdae6b, 0xa1d99b, 0xbcbddc, 0xad494a, 0xd6616b, 0xe7969c,
    0xde9ed6, 0xce6dbd, 0xa55194, 0xbd9e39, 0xe7ba52, 0xe7cb94, 0xb5cf6b, 0xcedb9c,
    0x5254a3, 0x6b6ecf, 0x9c9ede, 0x00a087, 0x3c5488, 0xf39b7f, 0x8491b4, 0x91d1c2,
];

export interface GpuTileSlot {
    key: string;
    totalRows: number;
    activeRows: number;
    visible: boolean;
    tileOffsetX: number; // Float64 exact [0..1)
    tileOffsetY: number; // Float64 exact [0..1)
    tileScale: number;   // Float64 exact 2^-z
    spawnTime: number;
    buffers: Map<string, GPUBuffer>;
}

/**
 * Direct WebGPU Tile Buffer Store (`D9` Compute-Only).
 *
 * Manages deterministic, exact-sized `GPUBuffer` lifecycles per tile and column
 * directly on `GPUDevice` without Three.js geometry or node-material wrappers.
 */
export class TileGpuStore {
    public readonly maxSlots: number;
    private device: GPUDevice;
    public slots: (GpuTileSlot | null)[];
    public tileKeyToSlot = new Map<string, number>();
    private slotMagBins: Int32Array;
    private lastMagBin = -1;
    private lastIsOpticalMode = false;

    public currentConfig: InternalEncodingConfig = {
        x: { field: 'x_u16', transform: 'linear', domain: [0, 1], range: [-1, 1] },
        y: { field: 'y_u16', transform: 'linear', domain: [0, 1], range: [1, -1] },
        color: { field: 'x_u16', range: 'viridis', domain: [0, 1] },
        size: { field: 'x_u16', domain: [0, 1], range: [0.5, 3.0] },
    };
    public previousConfig: InternalEncodingConfig = this.currentConfig;

    // 2-second WGSL axis transition state (`D9`)
    public isTransitioning = false;
    public transitionT = 1.0;
    private transitionStartMs = 0;
    private readonly transitionDurationMs = 2000;

    // Shader mode & uniform state
    public computeMode = 1; // 0 = Optical Sky, 1 = Log-Density Points, 2 = Mean Field, 3 = Star Trails, 4 = 3D Streamlines
    public isOpticalPipeline = false;
    public magnitudeField = 'x_u16';
    public opticalColorField = 'bp_rp';
    public metricType = 0;
    public auxField1 = '';
    public auxField2 = '';
    public auxField3 = '';
    /** Optional unmapped filter columns bound into idle `@binding(4)` / `@binding(5)` on spatial maps (`Step 3A`). */
    public filterExtraField1 = '';
    public filterExtraField2 = '';
    public trailMode = 0;
    public trailTimeKyr = 400.0;
    public minParallax = 8.0;
    public kernelRadius = 1.0;
    public lightBackground = false;
    public invertColorScale = false;
    public animateFlow = false;

    // Phase 8b multiscale sizing & exposure state
    public pointSize = 1.0;
    public zoomBalance = 0.22;
    public pointOpacity: 'auto' | number = 'auto';
    public exposure: 'auto' | number = 'auto';

    public globalTimeSec = 0.0;
    public maxMagCutoff = 14.0;
    public rootBounds: BoundingBox;

    public categoricalDictionaries: Record<string, string[]> = {};
    public colOrigins: Record<string, number> = {};
    public categoricalPalette: readonly number[] = DEFAULT_CATEGORICAL_PALETTE;
    public paletteDirty = true;

    private onReleaseCpuTileBuffers?: (tile: TileData) => void;
    public onSlotBindingsInvalidated?: (slot: number) => void;
    public onSlotDisposed?: (slot: number) => void;
    public onSlotMaskUploaded?: (slot: number, maskSlices: ReadonlyArray<ArrayBuffer>) => void;

    /** @internal Diagnostic flag used strictly by the Phase 6 negative-control VRAM leak test. */
    public debugSkipBufferDestroy = false;
    private debugSkippedBuffers: GPUBuffer[] = [];

    constructor(
        device: GPUDevice,
        rootBounds: BoundingBox,
        maxSlots = 264,
        onReleaseCpuTileBuffers?: (tile: TileData) => void
    ) {
        this.device = device;
        this.maxSlots = Math.max(64, maxSlots);
        this.slots = new Array(this.maxSlots).fill(null);
        this.slotMagBins = new Int32Array(this.maxSlots * 16);
        this.rootBounds = { ...rootBounds };
        this.onReleaseCpuTileBuffers = onReleaseCpuTileBuffers;
    }

    public setRootBounds(bounds: BoundingBox): void {
        this.rootBounds = { ...bounds };
    }

    public getCategoryColorHex(index: number): number {
        if (index < this.categoricalPalette.length) {
            return this.categoricalPalette[index];
        }
        // Deterministic golden-ratio HSV fallback for categories >= 64 (up to 65,535)
        const h = (((index * 0.61803398875) % 1.0) + 1.0) % 1.0;
        const s = 0.65;
        const v = 0.88;
        const i = Math.floor(h * 6.0);
        const f = h * 6.0 - i;
        const p = v * (1.0 - s);
        const q = v * (1.0 - f * s);
        const t = v * (1.0 - (1.0 - f) * s);
        let r = 0,
            g = 0,
            b = 0;
        switch (i % 6) {
            case 0: r = v; g = t; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = t; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = t; g = p; b = v; break;
            default: r = v; g = p; b = q; break;
        }
        return (
            ((Math.round(r * 255) & 255) << 16) |
            ((Math.round(g * 255) & 255) << 8) |
            (Math.round(b * 255) & 255)
        );
    }

    public clearDictionaries(): void {
        this.categoricalDictionaries = {};
        this.colOrigins = {};
    }

    public setDictionaries(dictionaries: Record<string, string[]>): void {
        for (const [colName, dict] of Object.entries(dictionaries)) {
            if (dict && dict.length > 0) {
                this.categoricalDictionaries[colName] = dict;
            }
        }
    }

    public getActiveColumns(config: InternalEncodingConfig = this.currentConfig): string[] {
        const cols = new Set<string>(['xy_u16']);
        const addIfColumn = (f: string | undefined) => {
            if (f && f !== 'x_u16' && f !== 'y_u16' && f !== 'xy_u16') {
                cols.add(f);
            }
        };

        addIfColumn(config.x.field);
        addIfColumn(config.y.field);
        addIfColumn(config.color.field);
        addIfColumn(config.size.field);
        addIfColumn(this.auxField1);
        addIfColumn(this.auxField2);
        addIfColumn(this.auxField3);
        addIfColumn(this.filterExtraField1);
        addIfColumn(this.filterExtraField2);

        if (this.isTransitioning) {
            addIfColumn(this.previousConfig.x.field);
            addIfColumn(this.previousConfig.y.field);
        }

        if (this.isOpticalPipeline || this.computeMode === 0 || this.computeMode === 3) {
            addIfColumn(this.opticalColorField);
            addIfColumn(this.magnitudeField);
        }

        return Array.from(cols);
    }

    public updateEncoding(nextConfig: InternalEncodingConfig, animateAxes = true): string[] {
        const prev = this.currentConfig;
        const axesChanged =
            animateAxes &&
            (prev.x.field !== nextConfig.x.field ||
                prev.y.field !== nextConfig.y.field ||
                Math.sign(prev.y.range[1] - prev.y.range[0]) !==
                    Math.sign(nextConfig.y.range[1] - nextConfig.y.range[0]));

        if (axesChanged) {
            this.previousConfig = {
                x: { ...prev.x, domain: [...prev.x.domain], range: [...prev.x.range] },
                y: { ...prev.y, domain: [...prev.y.domain], range: [...prev.y.range] },
                color: { ...prev.color, domain: [...prev.color.domain] },
                size: { ...prev.size, domain: [...prev.size.domain], range: [...prev.size.range] },
            };
            this.isTransitioning = true;
            this.transitionStartMs = performance.now();
            this.transitionT = 0.0;
        }

        this.currentConfig = {
            x: { ...nextConfig.x, domain: [...nextConfig.x.domain], range: [...nextConfig.x.range] },
            y: { ...nextConfig.y, domain: [...nextConfig.y.domain], range: [...nextConfig.y.range] },
            color: { ...nextConfig.color, domain: [...nextConfig.color.domain] },
            size: {
                ...nextConfig.size,
                domain: [...nextConfig.size.domain],
                range: [...nextConfig.size.range],
            },
        };

        this.refreshAllSlotActiveRows();
        for (let i = 0; i < this.maxSlots; i++) {
            const s = this.slots[i];
            if (s && s.visible && !this.isSlotReadyForCurrentAxes(s)) {
                s.visible = false;
            }
        }
        return this.getActiveColumns(this.currentConfig);
    }

    private isSlotReadyForCurrentAxes(slot: GpuTileSlot): boolean {
        if (slot.activeRows <= 0 || !slot.buffers.has('xy_u16')) return false;
        const isOptical =
            this.isOpticalPipeline || this.computeMode === 0 || this.computeMode === 3;
        if (isOptical) return true;
        const xf = this.currentConfig.x.field;
        const yf = this.currentConfig.y.field;
        if (xf && xf !== 'x_u16' && xf !== 'y_u16' && !slot.buffers.has(xf)) return false;
        if (yf && yf !== 'y_u16' && yf !== 'x_u16' && !slot.buffers.has(yf)) return false;
        if (this.filterExtraField1 && !slot.buffers.has(this.filterExtraField1)) return false;
        if (this.filterExtraField2 && !slot.buffers.has(this.filterExtraField2)) return false;
        return true;
    }

    private getMagBinForCutoff(maxMag: number): number {
        return Math.min(15, Math.max(0, Math.ceil((maxMag - 13.5) * 2.0)));
    }

    /**
     * Builds a 16-bin lossless prefix count table for a magnitude-sorted tile buffer.
     * Uses `Number.isFinite(m)` (never `m > 0.0`) so bright negative-magnitude stars (e.g. Sirius G = -1.46)
     * map cleanly into bin 0 and are never culled.
     */
    private computeSlotMagBins(
        slotIdx: number,
        numItems: number,
        magBuffer: ArrayBuffer | undefined
    ): void {
        const baseIdx = slotIdx * 16;
        if (!magBuffer || numItems <= 0) {
            for (let b = 0; b < 16; b++) this.slotMagBins[baseIdx + b] = numItems;
            return;
        }

        const maxFloats = Math.min(numItems, Math.floor(magBuffer.byteLength / 4));
        const magArr = new Float32Array(magBuffer, 0, maxFloats);
        for (let b = 0; b < 16; b++) this.slotMagBins[baseIdx + b] = 0;

        let lowestUnfilled = 0;
        for (let i = maxFloats - 1; i >= 0 && lowestUnfilled < 16; i--) {
            const m = magArr[i];
            if (Number.isFinite(m)) {
                const bin = m > 30.0 ? 0 : Math.min(15, Math.max(0, Math.ceil((m - 13.5) * 2.0)));
                const count = i + 1;
                for (let b = Math.max(lowestUnfilled, bin); b < 16; b++) {
                    if (this.slotMagBins[baseIdx + b] === 0) {
                        this.slotMagBins[baseIdx + b] = count;
                    }
                }
                while (lowestUnfilled < 16 && this.slotMagBins[baseIdx + lowestUnfilled] > 0) {
                    lowestUnfilled++;
                }
            }
        }
    }

    private applySlotActiveRows(slotIdx: number): void {
        const slot = this.slots[slotIdx];
        if (!slot) return;
        if (slot.totalRows <= 0) {
            slot.activeRows = 0;
            return;
        }
        const isOptical = this.isOpticalPipeline || this.computeMode === 0 || this.computeMode === 3;
        if (isOptical) {
            const bin = this.getMagBinForCutoff(this.maxMagCutoff);
            slot.activeRows = this.slotMagBins[slotIdx * 16 + bin];
        } else {
            slot.activeRows = slot.totalRows;
        }
    }

    private refreshAllSlotActiveRows(): void {
        for (let i = 0; i < this.maxSlots; i++) {
            if (this.slots[i]) {
                this.applySlotActiveRows(i);
            }
        }
    }

    public updateFrameState(
        cameraZoom: number,
        nowMs = performance.now(),
        baseZoom = 1.0
    ): void {
        this.globalTimeSec = nowMs * 0.001;

        if (this.isTransitioning) {
            const rawProgress = Math.min(
                1.0,
                Math.max(0.0, (nowMs - this.transitionStartMs) / this.transitionDurationMs)
            );
            // Smoothstep cubic easing
            this.transitionT = rawProgress * rawProgress * (3.0 - 2.0 * rawProgress);
            if (rawProgress >= 1.0) {
                this.isTransitioning = false;
                this.transitionT = 1.0;
                this.previousConfig = this.currentConfig;
            }
        } else {
            this.transitionT = 1.0;
        }

        const relZoom = (cameraZoom || 1.0) / Math.max(1e-30, baseZoom);
        const currentZoom = Math.log2(Math.max(0.1, relZoom));
        let magOffset = 0;
        if (currentZoom <= 1.2) {
            magOffset = 0;
        } else if (currentZoom <= 2.5) {
            magOffset = ((currentZoom - 1.2) / 1.3) * 1.5;
        } else if (currentZoom <= 4.5) {
            const t = (currentZoom - 2.5) / 2.0;
            magOffset = 1.5 + t * (2.0 - t) * 6.5;
        } else {
            magOffset = 8.0;
        }
        this.maxMagCutoff = Math.min(22.0, 14.0 + magOffset);

        const isOptical = this.isOpticalPipeline || this.computeMode === 0 || this.computeMode === 3;
        const magBin = this.getMagBinForCutoff(this.maxMagCutoff);
        if (magBin !== this.lastMagBin || isOptical !== this.lastIsOpticalMode) {
            this.lastMagBin = magBin;
            this.lastIsOpticalMode = isOptical;
            this.refreshAllSlotActiveRows();
        }
    }

    public getActivePointsCount(): number {
        let sum = 0;
        for (let i = 0; i < this.maxSlots; i++) {
            const s = this.slots[i];
            if (s && s.visible) {
                sum += s.activeRows;
            }
        }
        return sum;
    }

    public getLiveTileGpuBytes(): number {
        let bytes = 0;
        for (let i = 0; i < this.maxSlots; i++) {
            const s = this.slots[i];
            if (!s) continue;
            for (const buf of s.buffers.values()) {
                bytes += buf.size;
            }
        }
        return bytes;
    }

    private findFreeSlot(): number {
        for (let i = 0; i < this.maxSlots; i++) {
            if (!this.slots[i]) return i;
        }
        return -1;
    }

    private uploadColumnBuffer(
        slot: GpuTileSlot,
        colName: string,
        cpuBuf: ArrayBuffer,
        numRows: number
    ): void {
        const byteLength = Math.min(cpuBuf.byteLength, numRows * 4);
        if (byteLength <= 0) return;
        const alignedSize = Math.max(16, (byteLength + 3) & ~3);

        let gpuBuf = slot.buffers.get(colName);
        if (gpuBuf && gpuBuf.size !== alignedSize) {
            gpuBuf.destroy();
            gpuBuf = undefined;
        }
        if (!gpuBuf) {
            gpuBuf = this.device.createBuffer({
                label: `TileSlot_${slot.key}_${colName}`,
                size: alignedSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            });
            slot.buffers.set(colName, gpuBuf);
        }

        const alignedWriteBytes = byteLength & ~3;
        if (alignedWriteBytes > 0) {
            this.device.queue.writeBuffer(gpuBuf, 0, cpuBuf, 0, alignedWriteBytes);
        }
    }

    /**
     * Synchronizes visible tiles with GPU slots.
     * Note: Off-screen tiles remain cached in VRAM until `PMTilesClient` evicts them via `unloadTile(key)`,
     * allowing CPU `tile.columns` `ArrayBuffer`s to be released immediately after GPU upload (`0 MB` duplicate CPU retention).
     */
    public updateVisibleTiles(visibleTiles: TileData[]): boolean {
        const frameStart = performance.now();
        let hasPendingUpdates = false;

        // Mark all slots invisible first; only currently visible tiles are dispatched this frame
        for (let i = 0; i < this.maxSlots; i++) {
            const s = this.slots[i];
            if (s) s.visible = false;
        }

        let uploadedThisFrame = 0;
        const activeCols = this.getActiveColumns(this.currentConfig);

        for (const tile of visibleTiles) {
            let slotIdx = this.tileKeyToSlot.get(tile.key);
            const isNewSlot = slotIdx === undefined;
            const needsUpload = isNewSlot || Boolean(tile.needsUpdate);

            if (needsUpload && uploadedThisFrame >= 2 && performance.now() - frameStart > 8.0) {
                if (slotIdx !== undefined && this.slots[slotIdx]) {
                    this.slots[slotIdx]!.visible = this.isSlotReadyForCurrentAxes(
                        this.slots[slotIdx]!
                    );
                }
                hasPendingUpdates = true;
                continue;
            }

            if (slotIdx === undefined) {
                slotIdx = this.findFreeSlot();
                if (slotIdx === -1) continue;

                const [zStr, txStr, tyStr] = tile.key.split('/');
                const z = parseInt(zStr, 10);
                const scale = Math.pow(0.5, z); // Exact IEEE-754 Float64 power-of-two!
                const tileOffsetX = parseInt(txStr, 10) * scale;
                const tileOffsetY = parseInt(tyStr, 10) * scale;

                const newSlot: GpuTileSlot = {
                    key: tile.key,
                    totalRows: tile.numRows,
                    activeRows: 0,
                    visible: false,
                    tileOffsetX,
                    tileOffsetY,
                    tileScale: scale,
                    spawnTime: performance.now() * 0.001,
                    buffers: new Map(),
                };
                this.slots[slotIdx] = newSlot;
                this.tileKeyToSlot.set(tile.key, slotIdx);
                tile.needsUpdate = true;
            }

            const slot = this.slots[slotIdx]!;
            if (tile.needsUpdate) {
                const [zStr, txStr, tyStr] = tile.key.split('/');
                const z = parseInt(zStr, 10);
                const scale = Math.pow(0.5, z);
                slot.tileOffsetX = parseInt(txStr, 10) * scale;
                slot.tileOffsetY = parseInt(tyStr, 10) * scale;
                slot.tileScale = scale;
                slot.totalRows = tile.numRows;
                slot.spawnTime = performance.now() * 0.001;

                const magBuf =
                    tile.columns?.[this.magnitudeField] ??
                    tile.columns?.['phot_g_mean_mag'] ??
                    tile.columns?.['abs_m'];
                this.computeSlotMagBins(slotIdx, tile.numRows, magBuf);
                this.applySlotActiveRows(slotIdx);

                for (const col of activeCols) {
                    const cpuBuf = tile.columns?.[col];
                    if (cpuBuf) {
                        this.uploadColumnBuffer(slot, col, cpuBuf, tile.numRows);
                    }
                }
                // Also upload any extra decoded columns present on `tile.columns` before releasing CPU RAM
                if (tile.columns) {
                    for (const [col, cpuBuf] of Object.entries(tile.columns)) {
                        if (!slot.buffers.has(col) && cpuBuf) {
                            this.uploadColumnBuffer(slot, col, cpuBuf, tile.numRows);
                        }
                    }
                }

                if (tile.rowMaskSlices && this.onSlotMaskUploaded) {
                    this.onSlotMaskUploaded(slotIdx, tile.rowMaskSlices);
                } else if (isNewSlot && this.onSlotMaskUploaded) {
                    this.onSlotMaskUploaded(slotIdx, []);
                }

                tile.needsUpdate = false;
                if (this.onReleaseCpuTileBuffers) {
                    this.onReleaseCpuTileBuffers(tile);
                }
                if (this.onSlotBindingsInvalidated) {
                    this.onSlotBindingsInvalidated(slotIdx);
                }
                uploadedThisFrame++;
            }

            slot.visible = this.isSlotReadyForCurrentAxes(slot);
        }

        return hasPendingUpdates;
    }

    public unloadTile(key: string): void {
        const slotIdx = this.tileKeyToSlot.get(key);
        if (slotIdx === undefined) return;
        const slot = this.slots[slotIdx];
        if (slot) {
            for (const buf of slot.buffers.values()) {
                if (this.debugSkipBufferDestroy) {
                    this.debugSkippedBuffers.push(buf);
                } else {
                    buf.destroy();
                }
            }
            slot.buffers.clear();
        }
        this.slots[slotIdx] = null;
        this.tileKeyToSlot.delete(key);
        if (this.onSlotDisposed) {
            this.onSlotDisposed(slotIdx);
        }
    }

    /** @internal Reclaims buffers intentionally held during the negative-control VRAM leak test. */
    public flushDebugSkippedBuffers(): number {
        const count = this.debugSkippedBuffers.length;
        for (const buf of this.debugSkippedBuffers) {
            buf.destroy();
        }
        this.debugSkippedBuffers.length = 0;
        return count;
    }

    public clearAllTiles(): void {
        this.flushDebugSkippedBuffers();
        for (let i = 0; i < this.maxSlots; i++) {
            const slot = this.slots[i];
            if (slot) {
                for (const buf of slot.buffers.values()) {
                    buf.destroy();
                }
                slot.buffers.clear();
                this.slots[i] = null;
                if (this.onSlotDisposed) {
                    this.onSlotDisposed(i);
                }
            }
        }
        this.tileKeyToSlot.clear();
    }

    public dispose(): void {
        this.clearAllTiles();
        this.clearDictionaries();
    }
}
