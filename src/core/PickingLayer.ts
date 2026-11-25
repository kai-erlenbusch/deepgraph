import {
    buildSharedFilterWGSL,
    FilterGpuResources,
    GLOBAL_FILTER_UNIFORM_BYTES,
    SLOT_MASK_ROWS_PER_SLICE,
    SLOT_MASK_SLICE_BYTES,
    type AxisProjectionContext,
} from './FilterGpuResources.ts';
import type { Camera2D, Renderer } from './Renderer.ts';
import type { InternalAxisConfig, TileGpuStore } from './TileGpuStore.ts';

const SLOT_UNIFORM_STRIDE = 256;
const UNIFORM_BINDING_SIZE = 240; // 60 x 4-byte scalars (16-byte aligned, <= 256)
const MAX_ROWS_PER_CHUNK = SLOT_MASK_ROWS_PER_SLICE; // 262,144 (exact 32 KiB mask slice & <= 20-bit localRowIdx)
const NO_HIT_SENTINEL = 0xffffffff;
const WINNER_COORD_READBACK_BYTES = 32; // [packedXY_u32, x_f32, y_f32, prevX_f32, prevY_f32, pad...]

export interface GpuPickWinner {
    slotIndex: number;
    tileKey: string;
    z: number;
    x: number;
    y: number;
    rawRowIdx: number;
    distCssPx: number;
    xU16: number;
    yU16: number;
    gpuAxisAttributes: Record<string, number>;
}

interface CachedPickBindGroup {
    bindGroup: GPUBindGroup;
    xyBuf: GPUBuffer;
    colorBuf: GPUBuffer;
    sizeBuf: GPUBuffer;
    xBuf: GPUBuffer;
    yBuf: GPUBuffer;
    prevXBuf: GPUBuffer;
    prevYBuf: GPUBuffer;
    globalFilterBuf: GPUBuffer;
    maskBuf: GPUBuffer;
}

/**
 * Race-Free Per-Slot WebGPU Compute `PickingLayer` (`Phase 7 — 05_PLAN §9`).
 *
 * - Zero per-frame overhead during pan/zoom; dispatches only on click (`< 4 CSS px` travel) or throttled hover.
 * - Uses its own `@compute` pipeline (`1` dynamic uniform + `8` storage buffers <= universal WebGPU baseline).
 * - Writes `(dist16th_u12 << 20u) | localRowIdx_u20` via `atomicMin` into a ~1 KB (`264 * 4 = 1,056 B`)
 *   `slotBestHit` buffer over a `9 x 9 CSS px` cursor window (supporting > 1,048,575-row oversize leaves
 *   via automatic multi-chunk passes).
 * - Breaks distance ties across slots by **coarser zoom `z` first** (higher global pyramid priority)
 *   and `rawRowIdx` second.
 */
export class PickingLayer {
    private rendererWrapper: Renderer;
    private maxSlots: number;
    private device!: GPUDevice;

    private pipeline!: GPUComputePipeline;
    private bindGroupLayout!: GPUBindGroupLayout;
    private filterResources: FilterGpuResources | null = null;
    private ownsFilterResources = false;

    private uniformBuffer: GPUBuffer | null = null;
    private uniformArrayBuffer: ArrayBuffer;
    private uniformF32: Float32Array;
    private uniformU32: Uint32Array;

    private dummyStorageBuffer: GPUBuffer | null = null;
    private slotBestHitBuffer: GPUBuffer | null = null;
    private readbackBuffer: GPUBuffer | null = null;
    private winnerCoordBuffer: GPUBuffer | null = null;
    private resetArray: Uint32Array;
    private zeroCoordReset = new Uint32Array(WINNER_COORD_READBACK_BYTES / 4);
    private slotBindGroups: (CachedPickBindGroup | null)[];

    public lastDispatchedSlotCount = 0;
    public lastCulledSlotCount = 0;

    private initialized = false;
    private disposed = false;
    private activePickPromise: Promise<GpuPickWinner | null> | null = null;

    constructor(
        rendererWrapper: Renderer,
        maxSlots = 264,
        filterResources?: FilterGpuResources
    ) {
        this.rendererWrapper = rendererWrapper;
        this.maxSlots = Math.max(64, maxSlots);
        this.filterResources = filterResources ?? null;
        this.uniformArrayBuffer = new ArrayBuffer(this.maxSlots * SLOT_UNIFORM_STRIDE);
        this.uniformF32 = new Float32Array(this.uniformArrayBuffer);
        this.uniformU32 = new Uint32Array(this.uniformArrayBuffer);
        this.resetArray = new Uint32Array(this.maxSlots).fill(NO_HIT_SENTINEL);
        this.slotBindGroups = new Array(this.maxSlots).fill(null);
    }

    public invalidateSlot(slot: number): void {
        if (slot >= 0 && slot < this.slotBindGroups.length) {
            this.slotBindGroups[slot] = null;
        }
    }

    public get isBusy(): boolean {
        return this.activePickPromise !== null;
    }

    public init(): boolean {
        if (this.disposed) return false;
        if (this.initialized) return true;

        const device = this.rendererWrapper.device;
        if (!device) return false;
        this.device = device;

        if (!this.filterResources) {
            this.filterResources = new FilterGpuResources(device, this.maxSlots);
            this.ownsFilterResources = true;
        }
        this.filterResources.init();

        this.uniformBuffer = this.device.createBuffer({
            label: 'PickingLayer_SlotUniforms',
            size: this.maxSlots * SLOT_UNIFORM_STRIDE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.dummyStorageBuffer = this.device.createBuffer({
            label: 'PickingLayer_DummyStorage',
            size: 64,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const hitBufBytes = this.maxSlots * 4;
        this.slotBestHitBuffer = this.device.createBuffer({
            label: 'PickingLayer_SlotBestHit_1KB',
            size: hitBufBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        this.readbackBuffer = this.device.createBuffer({
            label: 'PickingLayer_Readback_1KB',
            size: hitBufBytes,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        this.winnerCoordBuffer = this.device.createBuffer({
            label: 'PickingLayer_WinnerCoord_32B',
            size: WINNER_COORD_READBACK_BYTES,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        this.createPipeline();
        this.initialized = true;
        return true;
    }

    private createPipeline(): void {
        const shaderCode = /* wgsl */ `
struct PickSlotUniforms {
    camLeft: f32,
    camRight: f32,
    camBottom: f32,
    camTop: f32,
    screenWidth: u32,
    screenHeight: u32,
    chunkRows: u32,
    mode: u32,

    tileOffsetX: f32,
    tileOffsetY: f32,
    tileScale: f32,
    rowBaseOffset: u32,

    cursorBackingX: f32,
    cursorBackingY: f32,
    maxMag: f32,
    dpr: f32,

    xDomMin: f32,
    xDomMax: f32,
    xRangeMin: f32,
    xRangeMax: f32,

    yDomMin: f32,
    yDomMax: f32,
    yRangeMin: f32,
    yRangeMax: f32,

    colorMin: f32,
    colorMax: f32,
    sizeMin: f32,
    sizeMax: f32,

    sizeRangeMin: f32,
    sizeRangeMax: f32,
    flags: u32,
    slotIndex: u32,

    zoomT: f32,
    halfWindowCssPx: f32,
    transitionT: f32,
    extraFlags: u32,

    rtcBaseX: f32,
    rtcStepX: f32,
    rtcBaseY: f32,
    rtcStepY: f32,

    invCamSpanX: f32,
    invCamSpanY: f32,
    pad0: f32,
    pad1: f32,

    prevXDomMin: f32,
    prevXDomMax: f32,
    prevXRangeMin: f32,
    prevXRangeMax: f32,

    prevYDomMin: f32,
    prevYDomMax: f32,
    prevYRangeMin: f32,
    prevYRangeMax: f32,

    prevRtcBaseX: f32,
    prevRtcStepX: f32,
    prevRtcBaseY: f32,
    prevRtcStepY: f32,
};

@group(0) @binding(0) var<uniform> u: PickSlotUniforms;
@group(0) @binding(1) var<storage, read> xyBuf: array<u32>;
@group(0) @binding(2) var<storage, read> colorBuf: array<f32>;
@group(0) @binding(3) var<storage, read> sizeBuf: array<f32>;
@group(0) @binding(4) var<storage, read> xBuf: array<f32>;
@group(0) @binding(5) var<storage, read> yBuf: array<f32>;
@group(0) @binding(6) var<storage, read> prevXBuf: array<f32>;
@group(0) @binding(7) var<storage, read> prevYBuf: array<f32>;
@group(0) @binding(8) var<storage, read_write> slotBestHit: array<atomic<u32>>;
${buildSharedFilterWGSL(9, 10)}

fn pcg_hash(v_in: u32) -> f32 {
    let state = v_in * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    let res = (word >> 22u) ^ word;
    return f32(res) / 4294967295.0;
}

fn map_linear(val: f32, domMin: f32, domMax: f32, rangeMin: f32, rangeMax: f32) -> f32 {
    let span = domMax - domMin;
    let safeSpan = select(span, 1.0, abs(span) < 1e-12);
    let t = clamp((val - domMin) / safeSpan, 0.0, 1.0);
    return mix(rangeMin, rangeMax, t);
}

fn project_non_spatial_rtc(
    val: f32,
    domMin: f32,
    domMax: f32,
    rtcAnchor: f32,
    rtcScale: f32,
    isLiteral: bool
) -> f32 {
    let effVal = select(
        clamp(val, min(domMin, domMax), max(domMin, domMax)),
        val,
        isLiteral
    );
    if (rtcScale == 0.0) {
        return rtcAnchor;
    }
    return (effVal - rtcAnchor) * rtcScale;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let localIdx = gid.x;
    if (localIdx >= u.chunkRows) {
        return;
    }
    let i = u.rowBaseOffset + localIdx;

    let isXSpatial = (u.flags & 1u) != 0u;
    let isYSpatial = (u.flags & 2u) != 0u;
    let xIsY16 = (u.flags & 4096u) != 0u;
    let yIsX16 = (u.flags & 8192u) != 0u;
    let isTransitioning = (u.extraFlags & 65536u) != 0u;
    let isXLiteral = (u.extraFlags & (1u << 21u)) != 0u;
    let isYLiteral = (u.extraFlags & (1u << 22u)) != 0u;
    let prevIsXLiteral = (u.extraFlags & (1u << 23u)) != 0u;
    let prevIsYLiteral = (u.extraFlags & (1u << 24u)) != 0u;
    let hasColorBuf = (u.extraFlags & (1u << 25u)) != 0u;
    let hasSizeBuf = (u.extraFlags & (1u << 26u)) != 0u;
    let hasXBuf = (u.extraFlags & (1u << 27u)) != 0u;
    let hasYBuf = (u.extraFlags & (1u << 28u)) != 0u;
    let hasPrevXBuf = (u.extraFlags & (1u << 29u)) != 0u;
    let hasPrevYBuf = (u.extraFlags & (1u << 30u)) != 0u;

    let xyPacked = xyBuf[i];
    let rawX = f32(xyPacked & 0xFFFFu) / 65535.0;
    let rawY = f32((xyPacked >> 16u) & 0xFFFFu) / 65535.0;

    let camSpanX = max(1e-12, u.camRight - u.camLeft);
    let camSpanY = max(1e-12, u.camTop - u.camBottom);

    if (isXSpatial && isYSpatial && !isTransitioning) {
        let coarseRelX = u.rtcBaseX + select(rawX, rawY, xIsY16) * u.rtcStepX;
        let coarseFromTopY = u.rtcBaseY + select(rawY, rawX, yIsX16) * u.rtcStepY;
        let padX = max(abs(u.rtcStepX) * 0.002, camSpanX * 0.02);
        let padY = max(abs(u.rtcStepY) * 0.002, camSpanY * 0.02);
        if (coarseRelX < -padX || coarseRelX > camSpanX + padX ||
            coarseFromTopY < -padY || coarseFromTopY > camSpanY + padY) {
            return;
        }
    }

    // Visibility parity with ComputeRasterizer: never pick culled/invisible rows
    var rawSize = 0.0;
    if (hasSizeBuf) {
        rawSize = sizeBuf[i];
    }
    if (u.mode == 0u || u.mode == 3u) {
        if (!(rawSize == rawSize)) {
            return;
        }
        let isTokens = rawSize > 30.0;
        if (!isTokens && rawSize > u.maxMag) {
            return;
        }
    }

    let colorSrc = (u.flags >> 14u) & 3u;
    let isCatMode = (u.mode == 1u) && ((u.flags & 4u) != 0u);
    if (u.mode == 1u && !isCatMode && colorSrc == 0u && hasColorBuf) {
        let cv = colorBuf[i];
        if (!(cv == cv)) {
            return;
        }
    }
    if (u.mode == 2u) {
        let metricType = (u.extraFlags >> 8u) & 15u;
        if (metricType == 0u || metricType == 5u || metricType == 6u) {
            if (!hasColorBuf) {
                return;
            }
            let cv = colorBuf[i];
            if (!(cv == cv) || (metricType == 6u && cv <= 0.0)) {
                return;
            }
        }
    }

    // Exact pcg_hash jitter + Float64 RTC / custom chart axis parity
    let jitterX = (pcg_hash(i) - 0.5) / 65535.0;
    let jitterY = (pcg_hash(i + 12345u) - 0.5) / 65535.0;
    let subX = rawX + jitterX;
    let subY = rawY + jitterY;
    let globalX = u.tileOffsetX + subX * u.tileScale;
    let globalY = u.tileOffsetY + subY * u.tileScale;

    var relWorldX = 0.0;
    if (isXSpatial) {
        relWorldX = u.rtcBaseX + select(subX, subY, xIsY16) * u.rtcStepX;
    } else {
        if (!hasXBuf) {
            return;
        }
        let xv = xBuf[i];
        if (!(xv == xv)) {
            return;
        }
        relWorldX = project_non_spatial_rtc(
            xv,
            u.xDomMin,
            u.xDomMax,
            u.rtcBaseX,
            u.rtcStepX,
            isXLiteral
        );
    }

    var fromTopY = 0.0;
    if (isYSpatial) {
        fromTopY = u.rtcBaseY + select(subY, subX, yIsX16) * u.rtcStepY;
    } else {
        if (!hasYBuf) {
            return;
        }
        let yv = yBuf[i];
        if (!(yv == yv)) {
            return;
        }
        fromTopY = project_non_spatial_rtc(
            yv,
            u.yDomMin,
            u.yDomMax,
            u.rtcBaseY,
            u.rtcStepY,
            isYLiteral
        );
    }

    // Never pick filtered-out or dimmed context points (Step 3A parity with ComputeRasterizer)
    if (filterU.header.x != 0u) {
        var filtCVal = 0.0;
        if (hasColorBuf) { filtCVal = colorBuf[i]; }
        var filtSVal = 0.0;
        if (hasSizeBuf) { filtSVal = sizeBuf[i]; }
        var filtXVal = 0.0;
        if (hasXBuf) { filtXVal = xBuf[i]; }
        var filtYVal = 0.0;
        if (hasYBuf) { filtYVal = yBuf[i]; }
        if (!eval_filter_selected(
            localIdx,
            relWorldX,
            fromTopY,
            globalX,
            globalY,
            filtCVal, hasColorBuf,
            filtSVal, hasSizeBuf,
            filtXVal, hasXBuf,
            filtYVal, hasYBuf
        )) {
            return;
        }
    }

    if (isTransitioning) {
        let prevIsXSpatial = (u.extraFlags & 131072u) != 0u;
        let prevIsYSpatial = (u.extraFlags & 262144u) != 0u;
        let prevXIsY16 = (u.extraFlags & 524288u) != 0u;
        let prevYIsX16 = (u.extraFlags & 1048576u) != 0u;

        var prevRelX = 0.0;
        if (prevIsXSpatial) {
            prevRelX = u.prevRtcBaseX + select(subX, subY, prevXIsY16) * u.prevRtcStepX;
        } else {
            if (!hasPrevXBuf) {
                return;
            }
            let pxv = prevXBuf[i];
            if (!(pxv == pxv)) {
                return;
            }
            prevRelX = project_non_spatial_rtc(
                pxv,
                u.prevXDomMin,
                u.prevXDomMax,
                u.prevRtcBaseX,
                u.prevRtcStepX,
                prevIsXLiteral
            );
        }

        var prevFromTopY = 0.0;
        if (prevIsYSpatial) {
            prevFromTopY = u.prevRtcBaseY + select(subY, subX, prevYIsX16) * u.prevRtcStepY;
        } else {
            if (!hasPrevYBuf) {
                return;
            }
            let pyv = prevYBuf[i];
            if (!(pyv == pyv)) {
                return;
            }
            prevFromTopY = project_non_spatial_rtc(
                pyv,
                u.prevYDomMin,
                u.prevYDomMax,
                u.prevRtcBaseY,
                u.prevRtcStepY,
                prevIsYLiteral
            );
        }

        relWorldX = mix(prevRelX, relWorldX, u.transitionT);
        fromTopY = mix(prevFromTopY, fromTopY, u.transitionT);
    }

    let ndcX = relWorldX * u.invCamSpanX;
    let ndcY = fromTopY * u.invCamSpanY;
    if (ndcX < 0.0 || ndcX >= 1.0 || ndcY < 0.0 || ndcY >= 1.0) {
        return;
    }

    let fx = ndcX * f32(u.screenWidth);
    let fy = ndcY * f32(u.screenHeight);
    let safeDpr = max(1.0, u.dpr);
    let dxCss = (fx - u.cursorBackingX) / safeDpr;
    let dyCss = (fy - u.cursorBackingY) / safeDpr;

    if (abs(dxCss) > u.halfWindowCssPx || abs(dyCss) > u.halfWindowCssPx) {
        return;
    }
    let distCss = sqrt(dxCss * dxCss + dyCss * dyCss);
    if (distCss > u.halfWindowCssPx) {
        return;
    }

    // Pack 12-bit distance in sixteenths of a CSS pixel (0..4094) + 20-bit localRowIdx
    let dist16th = min(u32(round(distCss * 16.0)), 4094u);
    let candidate = (dist16th << 20u) | (localIdx & 0xFFFFFu);
    atomicMin(&slotBestHit[u.slotIndex], candidate);
}
`;

        // 1 dynamic uniform buffer (0) + 8 storage buffers (1..8) <= universal WebGPU baseline!
        this.bindGroupLayout = this.device.createBindGroupLayout({
            label: 'PickingLayer_ComputeBGL',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'uniform',
                        hasDynamicOffset: true,
                        minBindingSize: UNIFORM_BINDING_SIZE,
                    },
                },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                {
                    binding: 9,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform', minBindingSize: GLOBAL_FILTER_UNIFORM_BYTES },
                },
                {
                    binding: 10,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'uniform',
                        hasDynamicOffset: true,
                        minBindingSize: SLOT_MASK_SLICE_BYTES,
                    },
                },
            ],
        });

        const module = this.device.createShaderModule({
            label: 'PickingLayer_ComputeShader',
            code: shaderCode,
        });

        this.pipeline = this.device.createComputePipeline({
            label: 'PickingLayer_ComputePipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.bindGroupLayout],
            }),
            compute: {
                module,
                entryPoint: 'main',
            },
        });
    }

    private computeRtcAxis(
        axis: InternalAxisConfig,
        tileOffsetX: number,
        tileOffsetY: number,
        tileScale: number,
        isYAxis: boolean,
        camLeftOrTop: number
    ): { base: number; step: number } {
        const isSpatial = axis.field === 'x_u16' || axis.field === 'y_u16';
        const domSpan = axis.domain[1] - axis.domain[0];
        const safeDomSpan = Math.abs(domSpan) < 1e-12 ? 1.0 : domSpan;
        if (!isSpatial) {
            const worldSpan = axis.range[1] - axis.range[0];
            if (Math.abs(worldSpan) < 1e-12) {
                return {
                    base: isYAxis ? camLeftOrTop - axis.range[0] : axis.range[0] - camLeftOrTop,
                    step: 0.0,
                };
            }
            const worldScale = worldSpan / safeDomSpan;
            const camAnchorData = axis.domain[0] + (camLeftOrTop - axis.range[0]) / worldScale;
            return {
                base: camAnchorData,
                step: isYAxis ? -worldScale : worldScale,
            };
        }

        const isY16 = axis.field === 'y_u16';
        const tileOff = isY16 ? tileOffsetY : tileOffsetX;
        const worldScale = (axis.range[1] - axis.range[0]) / safeDomSpan;
        const worldTile0 = axis.range[0] + (tileOff - axis.domain[0]) * worldScale;

        if (isYAxis) {
            return {
                base: camLeftOrTop - worldTile0,
                step: -tileScale * worldScale,
            };
        }
        return {
            base: worldTile0 - camLeftOrTop,
            step: tileScale * worldScale,
        };
    }

    private getOrCreateSlotBindGroup(
        slot: number,
        xyBuf: GPUBuffer,
        colorBuf: GPUBuffer,
        sizeBuf: GPUBuffer,
        xBuf: GPUBuffer,
        yBuf: GPUBuffer,
        prevXBuf: GPUBuffer,
        prevYBuf: GPUBuffer,
        globalFilterBuf: GPUBuffer,
        maskBuf: GPUBuffer
    ): GPUBindGroup {
        const cached = this.slotBindGroups[slot];
        if (
            cached &&
            cached.xyBuf === xyBuf &&
            cached.colorBuf === colorBuf &&
            cached.sizeBuf === sizeBuf &&
            cached.xBuf === xBuf &&
            cached.yBuf === yBuf &&
            cached.prevXBuf === prevXBuf &&
            cached.prevYBuf === prevYBuf &&
            cached.globalFilterBuf === globalFilterBuf &&
            cached.maskBuf === maskBuf
        ) {
            return cached.bindGroup;
        }

        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.uniformBuffer!,
                        offset: 0,
                        size: UNIFORM_BINDING_SIZE,
                    },
                },
                { binding: 1, resource: { buffer: xyBuf } },
                { binding: 2, resource: { buffer: colorBuf } },
                { binding: 3, resource: { buffer: sizeBuf } },
                { binding: 4, resource: { buffer: xBuf } },
                { binding: 5, resource: { buffer: yBuf } },
                { binding: 6, resource: { buffer: prevXBuf } },
                { binding: 7, resource: { buffer: prevYBuf } },
                { binding: 8, resource: { buffer: this.slotBestHitBuffer! } },
                {
                    binding: 9,
                    resource: {
                        buffer: globalFilterBuf,
                        offset: 0,
                        size: GLOBAL_FILTER_UNIFORM_BYTES,
                    },
                },
                {
                    binding: 10,
                    resource: {
                        buffer: maskBuf,
                        offset: 0,
                        size: SLOT_MASK_SLICE_BYTES,
                    },
                },
            ],
        });

        this.slotBindGroups[slot] = {
            bindGroup,
            xyBuf,
            colorBuf,
            sizeBuf,
            xBuf,
            yBuf,
            prevXBuf,
            prevYBuf,
            globalFilterBuf,
            maskBuf,
        };
        return bindGroup;
    }

    public async pick(
        cssX: number,
        cssY: number,
        camera: Camera2D,
        tileStore: TileGpuStore,
        halfWindowCssPx = 4.5,
        projCtx?: AxisProjectionContext
    ): Promise<GpuPickWinner | null> {
        if (this.disposed) return null;
        if (this.activePickPromise) {
            await this.activePickPromise.catch(() => null);
            if (this.disposed) return null;
        }

        const runPromise = this.executePickPass(
            cssX,
            cssY,
            camera,
            tileStore,
            halfWindowCssPx,
            projCtx
        );
        this.activePickPromise = runPromise;
        try {
            return await runPromise;
        } finally {
            if (this.activePickPromise === runPromise) {
                this.activePickPromise = null;
            }
        }
    }

    private async executePickPass(
        cssX: number,
        cssY: number,
        camera: Camera2D,
        tileStore: TileGpuStore,
        halfWindowCssPx: number,
        projCtx?: AxisProjectionContext
    ): Promise<GpuPickWinner | null> {
        if (!this.init()) return null;
        if (
            !this.uniformBuffer ||
            !this.dummyStorageBuffer ||
            !this.slotBestHitBuffer ||
            !this.readbackBuffer ||
            !this.winnerCoordBuffer ||
            !this.filterResources
        ) {
            return null;
        }

        const canvas = this.rendererWrapper.canvas;
        const screenWidth = Math.max(1, canvas.width);
        const screenHeight = Math.max(1, canvas.height);
        const dpr = Math.max(1, this.rendererWrapper.dpr);
        const cursorBackingX = cssX * dpr;
        const cursorBackingY = cssY * dpr;

        const camLeft = camera.position.x + camera.left / camera.zoom;
        const camRight = camera.position.x + camera.right / camera.zoom;
        const camBottom = camera.position.y + camera.bottom / camera.zoom;
        const camTop = camera.position.y + camera.top / camera.zoom;
        const invCamSpanX = 1.0 / Math.max(1e-12, camRight - camLeft);
        const invCamSpanY = 1.0 / Math.max(1e-12, camTop - camBottom);

        const computeMode = tileStore.computeMode;
        const isOptical = computeMode === 0 || computeMode === 3;
        const cfg = tileStore.currentConfig;
        const prevCfg = tileStore.previousConfig;

        const magField = tileStore.magnitudeField || 'x_u16';
        const optColorField = tileStore.opticalColorField || 'bp_rp';

        const xAxis: InternalAxisConfig = isOptical
            ? { field: 'x_u16', transform: 'linear', domain: cfg.x.domain, range: cfg.x.range }
            : cfg.x;
        const yAxis: InternalAxisConfig = isOptical
            ? { field: 'y_u16', transform: 'linear', domain: cfg.y.domain, range: cfg.y.range }
            : cfg.y;

        const colorField = isOptical ? optColorField : cfg.color.field;
        const sizeField = isOptical ? magField : cfg.size.field;

        const isXSpatial = xAxis.field === 'x_u16' || xAxis.field === 'y_u16';
        const isYSpatial = yAxis.field === 'y_u16' || yAxis.field === 'x_u16';
        const isXLiteral = !isXSpatial && xAxis.transform === 'literal';
        const isYLiteral = !isYSpatial && yAxis.transform === 'literal';
        const xIsY16 = xAxis.field === 'y_u16';
        const yIsX16 = yAxis.field === 'x_u16';

        const isTransitioning =
            tileStore.isTransitioning && !isOptical && computeMode !== 3 && computeMode !== 4;
        const prevIsXSpatial = prevCfg.x.field === 'x_u16' || prevCfg.x.field === 'y_u16';
        const prevIsYSpatial = prevCfg.y.field === 'y_u16' || prevCfg.y.field === 'x_u16';
        const prevIsXLiteral = !prevIsXSpatial && prevCfg.x.transform === 'literal';
        const prevIsYLiteral = !prevIsYSpatial && prevCfg.y.transform === 'literal';
        const prevXIsY16 = prevCfg.x.field === 'y_u16';
        const prevYIsX16 = prevCfg.y.field === 'x_u16';

        const dict =
            !isOptical && colorField ? tileStore.categoricalDictionaries[colorField] : undefined;
        const isCategorical = Boolean(dict && dict.length > 0);
        const colorSrc = colorField === 'x_u16' ? 1 : colorField === 'y_u16' ? 2 : 0;
        const sizeSrc = sizeField === 'x_u16' ? 1 : sizeField === 'y_u16' ? 2 : 0;

        const flags =
            (isXSpatial ? 1 : 0) |
            (isYSpatial ? 2 : 0) |
            (isCategorical ? 4 : 0) |
            (xIsY16 ? 4096 : 0) |
            (yIsX16 ? 8192 : 0) |
            (colorSrc << 14) |
            (sizeSrc << 16);

        const extraFlags =
            (((tileStore.metricType & 15) << 8) |
                (isTransitioning ? 65536 : 0) |
                (prevIsXSpatial ? 131072 : 0) |
                (prevIsYSpatial ? 262144 : 0) |
                (prevXIsY16 ? 524288 : 0) |
                (prevYIsX16 ? 1048576 : 0) |
                (isXLiteral ? 1 << 21 : 0) |
                (isYLiteral ? 1 << 22 : 0) |
                (prevIsXLiteral ? 1 << 23 : 0) |
                (prevIsYLiteral ? 1 << 24 : 0)) >>>
            0;

        const zoomT = Math.max(0.0, Math.min(1.0, this.rendererWrapper.zoomT));
        const logZoom = Math.max(0.0, this.rendererWrapper.logZoom);
        const smoothT = isOptical
            ? Math.max(0.0, Math.min(1.0, (logZoom - 5.20) / 3.80))
            : Math.max(0.0, Math.min(1.0, (logZoom - 2.5) / 6.0));
        const smoothReveal = smoothT * smoothT * (3.0 - 2.0 * smoothT);
        const minBaseCss = isOptical ? 2.5 : 2.25;
        const maxBaseCss = isOptical ? 6.0 : 6.5;
        const maxDiamCss =
            (minBaseCss + (maxBaseCss - minBaseCss) * smoothReveal) *
            Math.max(1.0, tileStore.pointSize);
        const effectiveHalfWindowCssPx = Math.max(halfWindowCssPx, 0.5 * maxDiamCss + 1.5);
        const maxMag = tileStore.maxMagCutoff;
        const dummy = this.dummyStorageBuffer;
        const limitSlots = Math.min(this.maxSlots, tileStore.maxSlots);
        const xOrExtra1Field = isXSpatial
            ? tileStore.auxField1 || tileStore.filterExtraField1
            : xAxis.field;
        const yOrExtra2Field = isYSpatial
            ? tileStore.auxField2 || tileStore.filterExtraField2
            : yAxis.field;

        const resolvedProjCtx: AxisProjectionContext = projCtx ?? {
            xAxis,
            yAxis,
            rootBounds: tileStore.rootBounds,
            effectiveFlipY: false,
            colOrigins: tileStore.colOrigins,
        };
        this.filterResources.updateFrameUniforms(camera, tileStore, resolvedProjCtx, [
            colorSrc === 0 ? colorField : '',
            sizeSrc === 0 ? sizeField : '',
            xOrExtra1Field,
            yOrExtra2Field,
        ]);
        const globalFilterBuf = this.filterResources.globalFilterBuffer!;
        const activeMaskBuf = this.filterResources.getActiveMaskBuffer()!;

        // Determine maximum chunk count across visible slots (1 for all standard <= 262,144-row tiles)
        let maxChunks = 1;
        for (let s = 0; s < limitSlots; s++) {
            const slotObj = tileStore.slots[s];
            if (slotObj && slotObj.visible && slotObj.activeRows > 0) {
                const chunks = Math.ceil(slotObj.activeRows / MAX_ROWS_PER_CHUNK);
                if (chunks > maxChunks) maxChunks = chunks;
            }
        }

        let bestOverall: {
            slotIndex: number;
            tileKey: string;
            expectedXyBuf: GPUBuffer;
            z: number;
            x: number;
            y: number;
            dist16th: number;
            rawRowIdx: number;
        } | null = null;

        this.lastDispatchedSlotCount = 0;
        this.lastCulledSlotCount = 0;
        const windowBackingPx = effectiveHalfWindowCssPx * dpr + 2.0;

        for (let chunkIdx = 0; chunkIdx < maxChunks; chunkIdx++) {
            const rowBaseOffset = chunkIdx * MAX_ROWS_PER_CHUNK;
            const dispatches: {
                slot: number;
                maskByteOffset: number;
                bindGroup: GPUBindGroup;
                workgroups: number;
            }[] = [];
            const dispatchedSnapshots = new Map<
                number,
                { key: string; xyBuf: GPUBuffer; activeRows: number }
            >();
            let maxSlotWritten = 0;

            for (let slotIdx = 0; slotIdx < limitSlots; slotIdx++) {
                const slotObj = tileStore.slots[slotIdx];
                if (!slotObj || !slotObj.visible || slotObj.activeRows <= rowBaseOffset) continue;

                const xyBuf = slotObj.buffers.get('xy_u16');
                if (!xyBuf) continue;

                const rtcX = this.computeRtcAxis(
                    xAxis,
                    slotObj.tileOffsetX,
                    slotObj.tileOffsetY,
                    slotObj.tileScale,
                    false,
                    camLeft
                );
                const rtcY = this.computeRtcAxis(
                    yAxis,
                    slotObj.tileOffsetX,
                    slotObj.tileOffsetY,
                    slotObj.tileScale,
                    true,
                    camTop
                );

                // Spatial slot cursor-window culling: skip slots whose screen-space bounds
                // cannot intersect cursorBacking +- (halfWindowCssPx * dpr + 2)
                if (!isTransitioning) {
                    if (isXSpatial) {
                        const px0 = rtcX.base * invCamSpanX * screenWidth;
                        const px1 = (rtcX.base + rtcX.step) * invCamSpanX * screenWidth;
                        const jitterPadX =
                            Math.abs(rtcX.step * invCamSpanX * screenWidth) * (2.0 / 65535.0);
                        const minPx = Math.min(px0, px1) - jitterPadX;
                        const maxPx = Math.max(px0, px1) + jitterPadX;
                        if (
                            maxPx < cursorBackingX - windowBackingPx ||
                            minPx > cursorBackingX + windowBackingPx
                        ) {
                            if (chunkIdx === 0) this.lastCulledSlotCount++;
                            continue;
                        }
                    }
                    if (isYSpatial) {
                        const py0 = rtcY.base * invCamSpanY * screenHeight;
                        const py1 = (rtcY.base + rtcY.step) * invCamSpanY * screenHeight;
                        const jitterPadY =
                            Math.abs(rtcY.step * invCamSpanY * screenHeight) * (2.0 / 65535.0);
                        const minPy = Math.min(py0, py1) - jitterPadY;
                        const maxPy = Math.max(py0, py1) + jitterPadY;
                        if (
                            maxPy < cursorBackingY - windowBackingPx ||
                            minPy > cursorBackingY + windowBackingPx
                        ) {
                            if (chunkIdx === 0) this.lastCulledSlotCount++;
                            continue;
                        }
                    }
                }
                if (chunkIdx === 0) this.lastDispatchedSlotCount++;

                const chunkRows = Math.min(
                    MAX_ROWS_PER_CHUNK,
                    slotObj.activeRows - rowBaseOffset
                );
                const colorBuf =
                    (colorSrc === 0 ? slotObj.buffers.get(colorField) : undefined) || dummy;
                const sizeBuf =
                    (sizeSrc === 0 ? slotObj.buffers.get(sizeField) : undefined) || dummy;
                const xBuf =
                    (xOrExtra1Field ? slotObj.buffers.get(xOrExtra1Field) : undefined) || dummy;
                const yBuf =
                    (yOrExtra2Field ? slotObj.buffers.get(yOrExtra2Field) : undefined) || dummy;
                const prevXBuf =
                    (isTransitioning && !prevIsXSpatial && prevCfg.x.field
                        ? slotObj.buffers.get(prevCfg.x.field)
                        : undefined) || dummy;
                const prevYBuf =
                    (isTransitioning && !prevIsYSpatial && prevCfg.y.field
                        ? slotObj.buffers.get(prevCfg.y.field)
                        : undefined) || dummy;

                const base = slotIdx * (SLOT_UNIFORM_STRIDE / 4);
                this.uniformF32[base + 0] = camLeft;
                this.uniformF32[base + 1] = camRight;
                this.uniformF32[base + 2] = camBottom;
                this.uniformF32[base + 3] = camTop;
                this.uniformU32[base + 4] = screenWidth;
                this.uniformU32[base + 5] = screenHeight;
                this.uniformU32[base + 6] = chunkRows;
                this.uniformU32[base + 7] = computeMode;

                this.uniformF32[base + 8] = slotObj.tileOffsetX;
                this.uniformF32[base + 9] = slotObj.tileOffsetY;
                this.uniformF32[base + 10] = slotObj.tileScale;
                this.uniformU32[base + 11] = rowBaseOffset;

                this.uniformF32[base + 12] = cursorBackingX;
                this.uniformF32[base + 13] = cursorBackingY;
                this.uniformF32[base + 14] = maxMag;
                this.uniformF32[base + 15] = dpr;

                this.uniformF32[base + 16] = xAxis.domain[0];
                this.uniformF32[base + 17] = xAxis.domain[1];
                this.uniformF32[base + 18] = xAxis.range[0];
                this.uniformF32[base + 19] = xAxis.range[1];

                this.uniformF32[base + 20] = yAxis.domain[0];
                this.uniformF32[base + 21] = yAxis.domain[1];
                this.uniformF32[base + 22] = yAxis.range[0];
                this.uniformF32[base + 23] = yAxis.range[1];

                this.uniformF32[base + 24] = cfg.color.domain[0];
                this.uniformF32[base + 25] = cfg.color.domain[1];
                this.uniformF32[base + 26] = cfg.size.domain[0];
                this.uniformF32[base + 27] = cfg.size.domain[1];

                this.uniformF32[base + 28] = cfg.size.range[0];
                this.uniformF32[base + 29] = cfg.size.range[1];
                this.uniformU32[base + 30] = flags;
                this.uniformU32[base + 31] = slotIdx;

                const slotBufFlags =
                    (colorBuf !== dummy ? 1 << 25 : 0) |
                    (sizeBuf !== dummy ? 1 << 26 : 0) |
                    (xBuf !== dummy ? 1 << 27 : 0) |
                    (yBuf !== dummy ? 1 << 28 : 0) |
                    (prevXBuf !== dummy ? 1 << 29 : 0) |
                    (prevYBuf !== dummy ? 1 << 30 : 0);

                this.uniformF32[base + 32] = zoomT;
                this.uniformF32[base + 33] = effectiveHalfWindowCssPx;
                this.uniformF32[base + 34] = tileStore.transitionT;
                this.uniformU32[base + 35] = (extraFlags | slotBufFlags) >>> 0;

                this.uniformF32[base + 36] = rtcX.base;
                this.uniformF32[base + 37] = rtcX.step;
                this.uniformF32[base + 38] = rtcY.base;
                this.uniformF32[base + 39] = rtcY.step;

                this.uniformF32[base + 40] = invCamSpanX;
                this.uniformF32[base + 41] = invCamSpanY;
                this.uniformF32[base + 42] = 0.0;
                this.uniformF32[base + 43] = 0.0;

                this.uniformF32[base + 44] = prevCfg.x.domain[0];
                this.uniformF32[base + 45] = prevCfg.x.domain[1];
                this.uniformF32[base + 46] = prevCfg.x.range[0];
                this.uniformF32[base + 47] = prevCfg.x.range[1];

                this.uniformF32[base + 48] = prevCfg.y.domain[0];
                this.uniformF32[base + 49] = prevCfg.y.domain[1];
                this.uniformF32[base + 50] = prevCfg.y.range[0];
                this.uniformF32[base + 51] = prevCfg.y.range[1];

                if (isTransitioning) {
                    const prevRtcX = this.computeRtcAxis(
                        prevCfg.x,
                        slotObj.tileOffsetX,
                        slotObj.tileOffsetY,
                        slotObj.tileScale,
                        false,
                        camLeft
                    );
                    const prevRtcY = this.computeRtcAxis(
                        prevCfg.y,
                        slotObj.tileOffsetX,
                        slotObj.tileOffsetY,
                        slotObj.tileScale,
                        true,
                        camTop
                    );
                    this.uniformF32[base + 52] = prevRtcX.base;
                    this.uniformF32[base + 53] = prevRtcX.step;
                    this.uniformF32[base + 54] = prevRtcY.base;
                    this.uniformF32[base + 55] = prevRtcY.step;
                }

                const bindGroup = this.getOrCreateSlotBindGroup(
                    slotIdx,
                    xyBuf,
                    colorBuf,
                    sizeBuf,
                    xBuf,
                    yBuf,
                    prevXBuf,
                    prevYBuf,
                    globalFilterBuf,
                    activeMaskBuf
                );
                dispatches.push({
                    slot: slotIdx,
                    maskByteOffset: this.filterResources.getMaskDynamicOffset(slotIdx, chunkIdx),
                    bindGroup,
                    workgroups: Math.ceil(chunkRows / 256),
                });
                dispatchedSnapshots.set(slotIdx, {
                    key: slotObj.key,
                    xyBuf,
                    activeRows: slotObj.activeRows,
                });
                maxSlotWritten = slotIdx;
            }

            if (dispatches.length === 0) continue;

            const uploadBytes = (maxSlotWritten + 1) * SLOT_UNIFORM_STRIDE;
            this.device.queue.writeBuffer(
                this.uniformBuffer,
                0,
                this.uniformArrayBuffer,
                0,
                uploadBytes
            );
            this.device.queue.writeBuffer(this.slotBestHitBuffer, 0, this.resetArray);

            const hitBufBytes = this.maxSlots * 4;
            const encoder = this.device.createCommandEncoder({
                label: 'PickingLayer_CommandEncoder',
            });
            const pass = encoder.beginComputePass({ label: 'PickingLayer_ComputePass' });
            pass.setPipeline(this.pipeline);
            for (let i = 0; i < dispatches.length; i++) {
                const item = dispatches[i];
                pass.setBindGroup(0, item.bindGroup, [
                    item.slot * SLOT_UNIFORM_STRIDE,
                    item.maskByteOffset,
                ]);
                pass.dispatchWorkgroups(item.workgroups);
            }
            pass.end();
            encoder.copyBufferToBuffer(
                this.slotBestHitBuffer,
                0,
                this.readbackBuffer,
                0,
                hitBufBytes
            );
            this.device.queue.submit([encoder.finish()]);

            await this.readbackBuffer.mapAsync(GPUMapMode.READ, 0, hitBufBytes);
            if (this.disposed) {
                try {
                    this.readbackBuffer.unmap();
                } catch {
                    // Ignore
                }
                return null;
            }

            const words = new Uint32Array(
                this.readbackBuffer.getMappedRange(0, hitBufBytes)
            ).slice();
            this.readbackBuffer.unmap();

            for (let slotIdx = 0; slotIdx < limitSlots; slotIdx++) {
                const w = words[slotIdx];
                if (w === NO_HIT_SENTINEL || w === undefined) continue;
                const snap = dispatchedSnapshots.get(slotIdx);
                if (!snap) continue;
                const slotObj = tileStore.slots[slotIdx];
                if (
                    !slotObj ||
                    slotObj.key !== snap.key ||
                    slotObj.buffers.get('xy_u16') !== snap.xyBuf
                ) {
                    continue;
                }

                const dist16th = w >>> 20;
                const localIdx = w & 0xfffff;
                const rawRowIdx = rowBaseOffset + localIdx;
                if (rawRowIdx >= slotObj.activeRows || rawRowIdx >= snap.activeRows) continue;

                const [zStr, xStr, yStr] = snap.key.split('/');
                const z = parseInt(zStr, 10);
                const x = parseInt(xStr, 10);
                const y = parseInt(yStr, 10);

                // Priority-aware tie-breaking (05_PLAN §9):
                // 1) Smaller distance (`dist16th`)
                // 2) Coarser zoom level `z` (higher global pyramid priority)
                // 3) Smaller `rawRowIdx` (higher intra-tile priority)
                const isBetter =
                    bestOverall === null ||
                    dist16th < bestOverall.dist16th ||
                    (dist16th === bestOverall.dist16th &&
                        (z < bestOverall.z ||
                            (z === bestOverall.z && rawRowIdx < bestOverall.rawRowIdx)));

                if (isBetter) {
                    bestOverall = {
                        slotIndex: slotIdx,
                        tileKey: snap.key,
                        expectedXyBuf: snap.xyBuf,
                        z,
                        x,
                        y,
                        dist16th,
                        rawRowIdx,
                    };
                }
            }
        }

        if (!bestOverall) return null;

        // Instant GPU Winner Coordinate Readback (`32 B` MAP_READ):
        // Reads the winning row's `xy_u16` word (plus active chart-axis floats) directly from VRAM
        // so `HighlightOverlay.hoverMarker` renders immediately at 60 FPS without waiting on network!
        let xU16 = 0;
        let yU16 = 0;
        const gpuAxisAttributes: Record<string, number> = {};
        const winnerSlot = tileStore.slots[bestOverall.slotIndex];
        const winnerXyBuf = winnerSlot?.buffers.get('xy_u16');
        const byteOff = bestOverall.rawRowIdx * 4;

        if (
            !winnerSlot ||
            winnerSlot.key !== bestOverall.tileKey ||
            winnerXyBuf !== bestOverall.expectedXyBuf ||
            !winnerXyBuf ||
            byteOff + 4 > winnerXyBuf.size
        ) {
            return null;
        }

        {
            this.device.queue.writeBuffer(this.winnerCoordBuffer, 0, this.zeroCoordReset);
            const copyEnc = this.device.createCommandEncoder({
                label: 'PickingLayer_WinnerCoordCopy',
            });
            copyEnc.copyBufferToBuffer(winnerXyBuf, byteOff, this.winnerCoordBuffer, 0, 4);

            const copyXField = !isXSpatial && xAxis.field ? xAxis.field : null;
            const copyYField = !isYSpatial && yAxis.field ? yAxis.field : null;
            const copyPrevXField =
                isTransitioning && !prevIsXSpatial && prevCfg.x.field ? prevCfg.x.field : null;
            const copyPrevYField =
                isTransitioning && !prevIsYSpatial && prevCfg.y.field ? prevCfg.y.field : null;

            const wXBuf = copyXField ? winnerSlot.buffers.get(copyXField) : undefined;
            if (wXBuf && byteOff + 4 <= wXBuf.size) {
                copyEnc.copyBufferToBuffer(wXBuf, byteOff, this.winnerCoordBuffer, 4, 4);
            }
            const wYBuf = copyYField ? winnerSlot.buffers.get(copyYField) : undefined;
            if (wYBuf && byteOff + 4 <= wYBuf.size) {
                copyEnc.copyBufferToBuffer(wYBuf, byteOff, this.winnerCoordBuffer, 8, 4);
            }
            const wPrevXBuf = copyPrevXField ? winnerSlot.buffers.get(copyPrevXField) : undefined;
            if (wPrevXBuf && byteOff + 4 <= wPrevXBuf.size) {
                copyEnc.copyBufferToBuffer(wPrevXBuf, byteOff, this.winnerCoordBuffer, 12, 4);
            }
            const wPrevYBuf = copyPrevYField ? winnerSlot.buffers.get(copyPrevYField) : undefined;
            if (wPrevYBuf && byteOff + 4 <= wPrevYBuf.size) {
                copyEnc.copyBufferToBuffer(wPrevYBuf, byteOff, this.winnerCoordBuffer, 16, 4);
            }

            this.device.queue.submit([copyEnc.finish()]);
            await this.winnerCoordBuffer.mapAsync(GPUMapMode.READ, 0, WINNER_COORD_READBACK_BYTES);
            if (this.disposed) {
                try {
                    this.winnerCoordBuffer.unmap();
                } catch {
                    // Ignore
                }
                return null;
            }
            const postSlot = tileStore.slots[bestOverall.slotIndex];
            if (
                !postSlot ||
                postSlot.key !== bestOverall.tileKey ||
                postSlot.buffers.get('xy_u16') !== bestOverall.expectedXyBuf
            ) {
                this.winnerCoordBuffer.unmap();
                return null;
            }
            const mapped = this.winnerCoordBuffer.getMappedRange(0, WINNER_COORD_READBACK_BYTES);
            const u32 = new Uint32Array(mapped);
            const f32 = new Float32Array(mapped);
            const packedXY = u32[0] ?? 0;
            xU16 = packedXY & 0xffff;
            yU16 = (packedXY >>> 16) & 0xffff;
            if (copyXField && wXBuf) {
                gpuAxisAttributes[copyXField] = f32[1] + (tileStore.colOrigins[copyXField] ?? 0);
            }
            if (copyYField && wYBuf) {
                gpuAxisAttributes[copyYField] = f32[2] + (tileStore.colOrigins[copyYField] ?? 0);
            }
            if (copyPrevXField && wPrevXBuf) {
                gpuAxisAttributes[copyPrevXField] =
                    f32[3] + (tileStore.colOrigins[copyPrevXField] ?? 0);
            }
            if (copyPrevYField && wPrevYBuf) {
                gpuAxisAttributes[copyPrevYField] =
                    f32[4] + (tileStore.colOrigins[copyPrevYField] ?? 0);
            }
            this.winnerCoordBuffer.unmap();
        }

        return {
            slotIndex: bestOverall.slotIndex,
            tileKey: bestOverall.tileKey,
            z: bestOverall.z,
            x: bestOverall.x,
            y: bestOverall.y,
            rawRowIdx: bestOverall.rawRowIdx,
            distCssPx: bestOverall.dist16th / 16.0,
            xU16,
            yU16,
            gpuAxisAttributes,
        };
    }

    public getLiveGpuBytes(): number {
        return (
            (this.uniformBuffer?.size ?? 0) +
            (this.dummyStorageBuffer?.size ?? 0) +
            (this.slotBestHitBuffer?.size ?? 0) +
            (this.readbackBuffer?.size ?? 0) +
            (this.winnerCoordBuffer?.size ?? 0) +
            (this.ownsFilterResources ? (this.filterResources?.getLiveGpuBytes() ?? 0) : 0)
        );
    }

    public dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        if (this.uniformBuffer) {
            this.uniformBuffer.destroy();
            this.uniformBuffer = null;
        }
        if (this.dummyStorageBuffer) {
            this.dummyStorageBuffer.destroy();
            this.dummyStorageBuffer = null;
        }
        if (this.slotBestHitBuffer) {
            this.slotBestHitBuffer.destroy();
            this.slotBestHitBuffer = null;
        }
        if (this.readbackBuffer) {
            this.readbackBuffer.destroy();
            this.readbackBuffer = null;
        }
        if (this.winnerCoordBuffer) {
            this.winnerCoordBuffer.destroy();
            this.winnerCoordBuffer = null;
        }
        if (this.ownsFilterResources && this.filterResources) {
            this.filterResources.dispose();
        }
        this.filterResources = null;
        for (let i = 0; i < this.slotBindGroups.length; i++) {
            this.slotBindGroups[i] = null;
        }
        this.initialized = false;
    }
}
