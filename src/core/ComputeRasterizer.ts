import {
    buildSharedFilterWGSL,
    FilterGpuResources,
    GLOBAL_FILTER_UNIFORM_BYTES,
    SLOT_MASK_ROWS_PER_SLICE,
    SLOT_MASK_SLICE_BYTES,
    type AxisProjectionContext,
} from './FilterGpuResources.ts';
import type { Camera2D, Renderer } from './Renderer.ts';
import type { AnyColorScaleName, InternalAxisConfig, TileGpuStore } from './TileGpuStore.ts';

const SLOT_UNIFORM_STRIDE = 256; // WebGPU minUniformBufferOffsetAlignment is 256 bytes
const UNIFORM_BINDING_SIZE = 256; // 64 x 4-byte scalars (16-byte aligned, exact 256-byte slot stride)

function smoothstepCpu(edge0: number, edge1: number, x: number): number {
    const t = Math.max(0.0, Math.min(1.0, (x - edge0) / Math.max(1e-6, edge1 - edge0)));
    return t * t * (3.0 - 2.0 * t);
}

interface CachedSlotBindGroup {
    bindGroup: GPUBindGroup;
    uniformBuf: GPUBuffer;
    xyBuf: GPUBuffer;
    colorBuf: GPUBuffer;
    sizeBuf: GPUBuffer;
    xBuf: GPUBuffer;
    yBuf: GPUBuffer;
    auxOrPrevXBuf: GPUBuffer;
    prevYBuf: GPUBuffer;
    globalFilterBuf: GPUBuffer;
    maskBuf: GPUBuffer;
}

/**
 * Native WebGPU Compute Shader Atomic Point Rasterizer (`D9` Compute-Only + `D10` Split-by-Mode).
 *
 * - Unified `2 x u32` (`8 B/px` = `112.5 MiB` at 5K Retina) atomic accumulation buffer across all modes:
 *   - Modes 0 & 3 (Optical Sky & Star Trails): `16-bit` fixed-point RGB channels (`word0 = R16 | G16<<16`, `word1 = B16`)
 *     with per-point ACES + sRGB (`D10`).
 *   - Mode 1 (Categorical & Generic Numeric Log-Density): `word0 = exact u32 hitWeightSum` via
 *     `prevW = atomicAdd(&accumBuf[base2], w)`, `word1 = R10 | G10<<10 | B10<<20` strictly interval-gated
 *     on `prevW < 64u` (`64 * 15 = 960 < 1024`, zero TOCTOU race and zero channel carry), resolved via
 *     `log2(1 + count)` brightness + mean RGB (`D10`).
 *   - Mode 2 (Mean Field): `word0 = valSum_u32`, `word1 = weightSum_u32` (zero carry).
 *   - Mode 4 (3D Motion + LIC Streamlines): `word0 = valSum_u32`, `word1 = weightSum_u24 | (lineHits_u8 << 24)`.
 * - CPU `Float64` Relative-to-Center (`RTC`) deep-zoom projection (`F-07`) preserving full 24-bit sub-pixel
 *   precision past `z = 20`.
 * - Smooth 2-second WGSL axis transitions (`mix(prevRelPos, currRelPos, transitionT)`).
 */
export class ComputeRasterizer {
    private rendererWrapper: Renderer;
    private maxSlots: number;
    private device!: GPUDevice;
    private context!: GPUCanvasContext;
    private canvasFormat!: GPUTextureFormat;

    private computePipeline!: GPUComputePipeline;
    private computeUnselectedPipeline!: GPUComputePipeline;
    private computeSelectedPipeline!: GPUComputePipeline;
    private computeBindGroupLayout!: GPUBindGroupLayout;
    private resolvePipeline!: GPURenderPipeline;
    private resolveDimBgPipeline!: GPURenderPipeline;
    private resolveFgOverlayPipeline!: GPURenderPipeline;
    private resolveBindGroupLayout!: GPUBindGroupLayout;

    private filterResources: FilterGpuResources | null = null;
    private ownsFilterResources = false;
    private allocatedUniformSlots: number;

    private uniformBuffer: GPUBuffer | null = null;
    private uniformArrayBuffer: ArrayBuffer;
    private uniformF32: Float32Array;
    private uniformU32: Uint32Array;

    private dummyStorageBuffer: GPUBuffer | null = null;
    private paletteBuffer: GPUBuffer | null = null;
    private accumBuffer: GPUBuffer | null = null;
    private resolveBindGroup: GPUBindGroup | null = null;
    private slotBindGroups: (CachedSlotBindGroup | null)[];

    private width = 0;
    private height = 0;
    private maxAccumPixels = 0;
    private initialized = false;
    private disposed = false;

    constructor(
        rendererWrapper: Renderer,
        maxSlots = 264,
        filterResources?: FilterGpuResources
    ) {
        this.rendererWrapper = rendererWrapper;
        this.maxSlots = Math.max(64, maxSlots);
        this.allocatedUniformSlots = this.maxSlots;
        this.filterResources = filterResources ?? null;
        this.uniformArrayBuffer = new ArrayBuffer(this.allocatedUniformSlots * SLOT_UNIFORM_STRIDE);
        this.uniformF32 = new Float32Array(this.uniformArrayBuffer);
        this.uniformU32 = new Uint32Array(this.uniformArrayBuffer);
        this.slotBindGroups = new Array(this.maxSlots).fill(null);
    }

    public invalidateSlot(slot: number): void {
        if (slot >= 0 && slot < this.slotBindGroups.length) {
            this.slotBindGroups[slot] = null;
        }
    }

    public init(): boolean {
        if (this.disposed) return false;
        if (this.initialized) return true;

        const device = this.rendererWrapper.device;
        const context = this.rendererWrapper.context;
        if (!device || !context) {
            return false;
        }

        this.device = device;
        this.context = context;
        this.canvasFormat = this.rendererWrapper.canvasFormat;

        if (!this.filterResources) {
            this.filterResources = new FilterGpuResources(device, this.maxSlots);
            this.ownsFilterResources = true;
        }
        this.filterResources.init();

        this.uniformBuffer = this.device.createBuffer({
            label: 'ComputeRasterizer_SlotUniforms',
            size: this.allocatedUniformSlots * SLOT_UNIFORM_STRIDE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.dummyStorageBuffer = this.device.createBuffer({
            label: 'ComputeRasterizer_DummyStorage',
            size: 64,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // 64 x u32 = 256-byte uniform palette buffer (array<vec4<u32>, 16>) so storage-buffer count stays <= 8
        this.paletteBuffer = this.device.createBuffer({
            label: 'ComputeRasterizer_PaletteUniform',
            size: 64 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.createPipelines();
        this.ensureScreenBuffers();
        this.initialized = true;
        return true;
    }

    private ensureScreenBuffers(): void {
        if (
            !this.uniformBuffer ||
            !this.filterResources ||
            !this.filterResources.globalFilterBuffer
        ) {
            return;
        }
        const canvas = this.rendererWrapper.canvas;
        const w = Math.max(1, canvas.width);
        const h = Math.max(1, canvas.height);

        if (w === this.width && h === this.height && this.accumBuffer && this.resolveBindGroup) {
            return;
        }

        this.width = w;
        this.height = h;

        if (this.accumBuffer) {
            this.accumBuffer.destroy();
            this.accumBuffer = null;
        }

        // Unified 2 x u32 (8 bytes/pixel = 112.5 MiB at 5120x2880 5K Retina < 128 MiB WebGPU baseline limit)
        const maxBind = Math.min(
            this.device.limits?.maxStorageBufferBindingSize ?? 134_217_728,
            this.device.limits?.maxBufferSize ?? 268_435_456
        );
        const requestedBytes = this.width * this.height * 8;
        const alignedMaxBind = Math.floor(Number(maxBind) / 8) * 8;
        const safeBytes = Math.max(16, Math.min(alignedMaxBind, requestedBytes));
        this.maxAccumPixels = Math.floor(safeBytes / 8);

        this.accumBuffer = this.device.createBuffer({
            label: 'ComputeRasterizer_AccumBuffer_2xU32',
            size: safeBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        for (let i = 0; i < this.maxSlots; i++) {
            this.slotBindGroups[i] = null;
        }

        this.resolveBindGroup = this.device.createBindGroup({
            layout: this.resolveBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.uniformBuffer,
                        offset: 0,
                        size: UNIFORM_BINDING_SIZE,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.accumBuffer,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.filterResources.globalFilterBuffer,
                    },
                },
            ],
        });
    }

    private ensureUniformSlotCapacity(requiredSlots: number): void {
        if (!this.device || requiredSlots <= this.allocatedUniformSlots) return;
        const nextSlots = Math.max(requiredSlots, Math.ceil(this.allocatedUniformSlots * 1.5));
        this.allocatedUniformSlots = nextSlots;
        this.uniformArrayBuffer = new ArrayBuffer(nextSlots * SLOT_UNIFORM_STRIDE);
        this.uniformF32 = new Float32Array(this.uniformArrayBuffer);
        this.uniformU32 = new Uint32Array(this.uniformArrayBuffer);
        if (this.uniformBuffer) {
            this.uniformBuffer.destroy();
        }
        this.uniformBuffer = this.device.createBuffer({
            label: 'ComputeRasterizer_SlotUniforms',
            size: nextSlots * SLOT_UNIFORM_STRIDE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        for (let i = 0; i < this.slotBindGroups.length; i++) {
            this.slotBindGroups[i] = null;
        }
        if (this.accumBuffer && this.filterResources?.globalFilterBuffer) {
            this.resolveBindGroup = this.device.createBindGroup({
                layout: this.resolveBindGroupLayout,
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: this.uniformBuffer,
                            offset: 0,
                            size: UNIFORM_BINDING_SIZE,
                        },
                    },
                    { binding: 1, resource: { buffer: this.accumBuffer } },
                    { binding: 2, resource: { buffer: this.filterResources.globalFilterBuffer } },
                ],
            });
        }
    }

    private createPipelines(): void {
        const sharedUniformStructWGSL = /* wgsl */ `
struct SlotUniforms {
    camLeft: f32,
    camRight: f32,
    camBottom: f32,
    camTop: f32,
    screenWidth: u32,
    screenHeight: u32,
    numPoints: u32,
    mode: u32,

    tileOffsetX: f32,
    tileOffsetY: f32,
    tileScale: f32,
    spawnTime: f32,

    globalTime: f32,
    zoomT: f32,
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
    dictLen: u32,

    trailTimeKyr: f32,
    minParallax: f32,
    kernelRadius: f32,
    extraFlags: u32,

    rtcBaseX: f32,
    rtcStepX: f32,
    rtcBaseY: f32,
    rtcStepY: f32,

    invCamSpanX: f32,
    invCamSpanY: f32,
    transitionT: f32,
    maxAccumPixels: u32,

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

    logZoom: f32,
    pointSizeScale: f32,
    zoomBalance: f32,
    maxPointDiameterPx: f32,

    resolvedPointOpacity: f32,
    resolvedExposure: f32,
    mode1RefCount: f32,
    rowBaseOffset: u32,
};
`;

        const sharedColormapWGSL = /* wgsl */ `
fn inverse_hammer(gx: f32, gy: f32, out_lb: ptr<function, vec2<f32>>) -> bool {
    let xH = (gx * 2.0 - 1.0) * 2.82842712;
    let yH = (1.0 - gy * 2.0) * 1.41421356;
    let z2 = 1.0 - (xH * xH) * 0.0625 - (yH * yH) * 0.25;
    if (z2 <= 0.0001) {
        return false;
    }
    let z = sqrt(z2);
    let l = 2.0 * atan2(z * xH, 4.0 * z2 - 2.0);
    let sinB = clamp(z * yH, -1.0, 1.0);
    let b = asin(sinB);
    (*out_lb) = vec2<f32>(-l, b);
    return true;
}

fn forward_hammer_vec(r: vec3<f32>) -> vec2<f32> {
    let l = atan2(r.y, r.x);
    let cosB = sqrt(max(0.0, 1.0 - r.z * r.z));
    let denom = sqrt(max(1e-8, 1.0 + cosB * cos(l * 0.5)));
    let xH = (-2.0 * 1.41421356 * cosB * sin(l * 0.5)) / denom;
    let yH = (1.41421356 * r.z) / denom;
    let gx = xH / 5.65685425 + 0.5;
    let gy = 0.5 - yH / 2.82842712;
    return vec2<f32>(gx, gy);
}

fn ramp5(t: f32, c1: vec3<f32>, c2: vec3<f32>, c3: vec3<f32>, c4: vec3<f32>, c5: vec3<f32>) -> vec3<f32> {
    var c = mix(c1, c2, smoothstep(0.0, 0.25, t));
    c = mix(c, c3, smoothstep(0.25, 0.50, t));
    c = mix(c, c4, smoothstep(0.50, 0.75, t));
    return mix(c, c5, smoothstep(0.75, 1.0, t));
}

fn eval_ramp(t: f32, scaleId: u32) -> vec3<f32> {
    if (scaleId == 1u) {
        return ramp5(t,
            vec3<f32>(0.050, 0.030, 0.528),
            vec3<f32>(0.494, 0.012, 0.658),
            vec3<f32>(0.798, 0.280, 0.470),
            vec3<f32>(0.973, 0.580, 0.254),
            vec3<f32>(0.940, 0.975, 0.131));
    }
    if (scaleId == 2u) {
        return ramp5(t,
            vec3<f32>(0.001, 0.000, 0.014),
            vec3<f32>(0.316, 0.071, 0.485),
            vec3<f32>(0.716, 0.215, 0.475),
            vec3<f32>(0.987, 0.535, 0.382),
            vec3<f32>(0.987, 0.991, 0.750));
    }
    if (scaleId == 3u) {
        return ramp5(t,
            vec3<f32>(0.001, 0.000, 0.014),
            vec3<f32>(0.341, 0.062, 0.429),
            vec3<f32>(0.735, 0.216, 0.330),
            vec3<f32>(0.978, 0.557, 0.035),
            vec3<f32>(0.988, 0.998, 0.645));
    }
    if (scaleId == 4u) {
        return ramp5(t,
            vec3<f32>(0.035, 0.188, 0.420),
            vec3<f32>(0.259, 0.573, 0.776),
            vec3<f32>(0.965, 0.965, 0.965),
            vec3<f32>(0.937, 0.396, 0.282),
            vec3<f32>(0.502, 0.000, 0.149));
    }
    if (scaleId == 5u) {
        return ramp5(t,
            vec3<f32>(0.230, 0.299, 0.754),
            vec3<f32>(0.553, 0.690, 0.996),
            vec3<f32>(0.865, 0.865, 0.865),
            vec3<f32>(0.957, 0.604, 0.482),
            vec3<f32>(0.706, 0.016, 0.150));
    }
    if (scaleId == 6u) {
        return ramp5(t,
            vec3<f32>(0.050, 0.050, 0.650),
            vec3<f32>(0.000, 0.650, 0.950),
            vec3<f32>(0.350, 0.900, 0.350),
            vec3<f32>(0.980, 0.750, 0.050),
            vec3<f32>(0.800, 0.050, 0.050));
    }
    if (scaleId == 7u) {
        return ramp5(t,
            vec3<f32>(0.110, 0.240, 0.660),
            vec3<f32>(0.420, 0.470, 0.780),
            vec3<f32>(0.830, 0.760, 0.860),
            vec3<f32>(0.880, 0.560, 0.620),
            vec3<f32>(0.680, 0.140, 0.220));
    }
    if (scaleId == 8u) {
        return ramp5(t,
            vec3<f32>(0.985, 0.975, 0.960),
            vec3<f32>(0.965, 0.840, 0.600),
            vec3<f32>(0.910, 0.530, 0.220),
            vec3<f32>(0.680, 0.190, 0.110),
            vec3<f32>(0.220, 0.030, 0.040));
    }
    if (scaleId == 9u) {
        return ramp5(t,
            vec3<f32>(0.012, 0.008, 0.018),
            vec3<f32>(0.240, 0.050, 0.040),
            vec3<f32>(0.680, 0.210, 0.050),
            vec3<f32>(0.960, 0.620, 0.160),
            vec3<f32>(0.995, 0.975, 0.840));
    }
    return ramp5(t,
        vec3<f32>(0.267, 0.005, 0.329),
        vec3<f32>(0.230, 0.322, 0.546),
        vec3<f32>(0.128, 0.567, 0.551),
        vec3<f32>(0.369, 0.789, 0.383),
        vec3<f32>(0.993, 0.906, 0.144));
}
`;

        const computeShaderCode = /* wgsl */ `
${sharedUniformStructWGSL}

struct PaletteUniforms {
    colors: array<vec4<u32>, 16>,
};

@group(0) @binding(0) var<uniform> u: SlotUniforms;
@group(0) @binding(1) var<storage, read> xyBuf: array<u32>;
@group(0) @binding(2) var<storage, read> colorBuf: array<f32>;
@group(0) @binding(3) var<storage, read> sizeBuf: array<f32>;
@group(0) @binding(4) var<storage, read> xBuf: array<f32>;
@group(0) @binding(5) var<storage, read> yBuf: array<f32>;
@group(0) @binding(6) var<uniform> palette: PaletteUniforms;
@group(0) @binding(7) var<storage, read_write> accumBuf: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> auxBuf: array<f32>;
@group(0) @binding(9) var<storage, read> prevYBuf: array<f32>;
${buildSharedFilterWGSL(10, 11)}

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

fn icrs_pm_to_gal_vel(l: f32, b: f32, pmra: f32, pmdec: f32, out_uGal: ptr<function, vec3<f32>>) -> vec3<f32> {
    let cosB = cos(b);
    let uGal = vec3<f32>(cosB * cos(l), cosB * sin(l), sin(b));
    (*out_uGal) = uGal;

    let r0 = vec3<f32>(-0.05487556, -0.87343709, -0.48383502);
    let r1 = vec3<f32>( 0.49410943, -0.44482963,  0.74698224);
    let r2 = vec3<f32>(-0.86766615, -0.19807637,  0.45598378);

    let uEq = vec3<f32>(
        r0.x * uGal.x + r1.x * uGal.y + r2.x * uGal.z,
        r0.y * uGal.x + r1.y * uGal.y + r2.y * uGal.z,
        r0.z * uGal.x + r1.z * uGal.y + r2.z * uGal.z
    );
    let cosD = sqrt(max(1e-8, uEq.x * uEq.x + uEq.y * uEq.y));
    let sinD = uEq.z;

    let pEq = vec3<f32>(-uEq.y / cosD, uEq.x / cosD, 0.0);
    let qEq = vec3<f32>(-sinD * uEq.x / cosD, -sinD * uEq.y / cosD, cosD);

    let radPerKyr = 4.8481368e-6;
    let duEq = (pmra * pEq + pmdec * qEq) * radPerKyr;

    return vec3<f32>(
        dot(r0, duEq),
        dot(r1, duEq),
        dot(r2, duEq)
    );
}

${sharedColormapWGSL}

fn eval_color(rawVal: f32, globalX: f32, globalY: f32) -> vec3<f32> {
    let isOptical = (u.mode == 0u) || (u.mode == 3u);
    let isCat = (u.flags & 4u) != 0u;
    let colorSrc = (u.flags >> 14u) & 3u;
    var val = rawVal;
    if (colorSrc == 1u) {
        val = globalX;
    } else if (colorSrc == 2u) {
        val = globalY;
    }
    if (!isOptical && isCat && u.dictLen > 0u) {
        if (!(val == val) || val < 0.0) {
            return vec3<f32>(0.42, 0.42, 0.46);
        }
        let code = u32(clamp(val, 0.0, f32(u.dictLen - 1u)));
        if (code < 64u) {
            let hex = palette.colors[code >> 2u][code & 3u];
            return vec3<f32>(
                f32((hex >> 16u) & 255u) / 255.0,
                f32((hex >> 8u) & 255u) / 255.0,
                f32(hex & 255u) / 255.0
            );
        }
        let h6 = fract(f32(code) * 0.61803398875) * 6.0;
        let hi = u32(floor(h6)) % 6u;
        let f = h6 - floor(h6);
        let v = 0.88;
        let s = 0.65;
        let p = v * (1.0 - s);
        let q = v * (1.0 - f * s);
        let t = v * (1.0 - (1.0 - f) * s);
        if (hi == 0u) { return vec3<f32>(v, t, p); }
        if (hi == 1u) { return vec3<f32>(q, v, p); }
        if (hi == 2u) { return vec3<f32>(p, v, t); }
        if (hi == 3u) { return vec3<f32>(p, q, v); }
        if (hi == 4u) { return vec3<f32>(t, p, v); }
        return vec3<f32>(v, p, q);
    }

    if (!(val == val)) {
        val = 0.0;
    }

    if (isOptical) {
        var c = mix(vec3<f32>(0.0667, 0.1333, 0.6667), vec3<f32>(0.3333, 0.6667, 0.8667), smoothstep(-5.0, -2.5, val));
        c = mix(c, vec3<f32>(1.0, 1.0, 1.0), smoothstep(-2.5, 0.0, val));
        c = mix(c, vec3<f32>(1.0, 0.6, 0.0), smoothstep(0.0, 2.5, val));
        return mix(c, vec3<f32>(0.8, 0.1333, 0.0), smoothstep(2.5, 5.0, val));
    }

    let cSpan = max(u.colorMax - u.colorMin, 1e-6);
    var t = clamp((val - u.colorMin) / cSpan, 0.0, 1.0);
    if ((u.extraFlags & 2u) != 0u) {
        t = 1.0 - t;
    }
    let scaleId = (u.flags >> 8u) & 15u;
    return eval_ramp(t, scaleId);
}

fn eval_numeric_t(rawVal: f32, globalX: f32, globalY: f32) -> f32 {
    let colorSrc = (u.flags >> 14u) & 3u;
    var val = rawVal;
    if (colorSrc == 1u) {
        val = globalX;
    } else if (colorSrc == 2u) {
        val = globalY;
    }
    if (!(val == val)) {
        val = 0.0;
    }
    let cSpan = max(u.colorMax - u.colorMin, 1e-6);
    var t = clamp((val - u.colorMin) / cSpan, 0.0, 1.0);
    if ((u.extraFlags & 2u) != 0u) {
        t = 1.0 - t;
    }
    return t;
}

fn three_aces_filmic(color: vec3<f32>, exposure: f32) -> vec3<f32> {
    let c = (color * exposure) / 0.6;
    let aIn = vec3<f32>(
        0.59719 * c.r + 0.35458 * c.g + 0.04823 * c.b,
        0.07600 * c.r + 0.90834 * c.g + 0.01566 * c.b,
        0.02840 * c.r + 0.13383 * c.g + 0.83777 * c.b
    );
    let num = aIn * (aIn + 0.0245786) - 0.000090537;
    let den = aIn * (0.983729 * aIn + 0.4329510) + 0.238081;
    let odt = num / den;
    let aOut = vec3<f32>(
         1.60475 * odt.r - 0.53108 * odt.g - 0.07367 * odt.b,
        -0.10208 * odt.r + 1.10813 * odt.g - 0.00605 * odt.b,
        -0.00327 * odt.r - 0.07276 * odt.g + 1.07602 * odt.b
    );
    return clamp(aOut, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_to_srgb_ch(v: f32) -> f32 {
    if (v <= 0.0031308) {
        return max(0.0, v * 12.92);
    }
    let s1 = sqrt(v);
    let s2 = sqrt(s1);
    let s3 = sqrt(s2);
    return clamp(0.662002687 * s1 + 0.684122060 * s2 - 0.323583601 * s3 - 0.022541147 * v, 0.0, 1.0);
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        linear_to_srgb_ch(c.r),
        linear_to_srgb_ch(c.g),
        linear_to_srgb_ch(c.b)
    );
}

// Modes 0 & 3 (Optical Sky & Star Trails — D10):
// 16-bit channels in 2 x u32 (word0 = R16 | G16<<16, word1 = B16 | fluxTickets16<<16), scale 1.0 = 1024.0.
// Reserves ticket = ceil(max(rU, gU, bU) / 16) (1..32) gated at <= 3840u (3840 * 16 = 61,440 < 65,535),
// allowing up to 3,840 stars/pixel (40x higher than 96-hit cap) with zero R16->G16/B16 carry and 0.0000% shimmer
// even on 80+ SM discrete GPUs where hundreds of warps hit Galactic-core pixels concurrently.
fn splat_optical_pixel(px: i32, py: i32, baseColor: vec3<f32>, safeAlpha: f32, exposure: f32) {
    if (px < 0 || py < 0 || px >= i32(u.screenWidth) || py >= i32(u.screenHeight)) {
        return;
    }
    let idx = u32(py) * u.screenWidth + u32(px);
    if (idx >= u.maxAccumPixels) {
        return;
    }
    let base2 = idx * 2u;
    let preW1 = atomicLoad(&accumBuf[base2 + 1u]);
    if ((preW1 >> 16u) >= 3840u) {
        return;
    }

    let srgb = linear_to_srgb(three_aces_filmic(baseColor * safeAlpha, exposure));
    let scale = 1024.0;
    let rU = min(u32(max(0.0, srgb.r * scale + 0.5)), 512u);
    let gU = min(u32(max(0.0, srgb.g * scale + 0.5)), 512u);
    let bU = min(u32(max(0.0, srgb.b * scale + 0.5)), 512u);
    let cMax = max(rU, max(gU, bU));
    if (cMax == 0u) {
        return;
    }

    let ticket = (cMax + 15u) >> 4u;
    let prevW1 = atomicAdd(&accumBuf[base2 + 1u], ticket << 16u);
    let prevTickets = prevW1 >> 16u;
    if (prevTickets + ticket > 3840u) {
        atomicSub(&accumBuf[base2 + 1u], ticket << 16u);
        return;
    }

    let rgPack = rU | (gU << 16u);
    if (rgPack > 0u) {
        atomicAdd(&accumBuf[base2], rgPack);
    }
    if (bU > 0u) {
        atomicAdd(&accumBuf[base2 + 1u], bU);
    }
}

// Mode 1 (BOTH Categorical & Generic Numeric Log-Density — D10 + v0.1.1 Pushback #11):
// - Categorical (isCat=true):
//   - word0 = rSum_u16 | (wRgbSum_u16 << 16), word1 = gSum_u16 | (bSum_u16 << 16)
//   - Q = 12.0 (64/64 unique palette colors), wRgb = max(1u, (w + 1u) >> 1u), interval-gated at
//     wRgbSum < 5120u (5120 * 12 = 61,440 < 65,535 -> 2,560 center hits, zero carry, 0.0000% shimmer)
// - Numeric (isCat=false):
//   - word0 = exact 32-bit weighted hit count, word1 = sum(tU * wVal) gated at prevW < 1048576u
fn splat_log_density_pixel(px: i32, py: i32, baseColor: vec3<f32>, tNorm: f32, w: u32, isCat: bool) {
    if (px < 0 || py < 0 || px >= i32(u.screenWidth) || py >= i32(u.screenHeight) || w == 0u) {
        return;
    }
    let idx = u32(py) * u.screenWidth + u32(px);
    if (idx >= u.maxAccumPixels) {
        return;
    }
    let base2 = idx * 2u;
    if (isCat) {
        let wRgb = max(1u, (w + 1u) >> 1u);
        let preW0 = atomicLoad(&accumBuf[base2]);
        if ((preW0 >> 16u) >= 5120u) {
            return;
        }
        let prevW0 = atomicAdd(&accumBuf[base2], wRgb << 16u);
        let prevWRgb = prevW0 >> 16u;
        if (prevWRgb >= 5120u) {
            atomicSub(&accumBuf[base2], wRgb << 16u);
            return;
        }
        let effWRgb = min(wRgb, 5120u - prevWRgb);
        if (effWRgb < wRgb) {
            atomicSub(&accumBuf[base2], (wRgb - effWRgb) << 16u);
        }
        let rQ = u32(clamp(round(baseColor.r * 12.0), 0.0, 12.0)) * effWRgb;
        let gQ = u32(clamp(round(baseColor.g * 12.0), 0.0, 12.0)) * effWRgb;
        let bQ = u32(clamp(round(baseColor.b * 12.0), 0.0, 12.0)) * effWRgb;
        if (rQ > 0u) {
            atomicAdd(&accumBuf[base2], rQ);
        }
        let gbPack = gQ | (bQ << 16u);
        if (gbPack > 0u) {
            atomicAdd(&accumBuf[base2 + 1u], gbPack);
        }
    } else {
        let prevW = atomicAdd(&accumBuf[base2], w);
        if (prevW < 1048576u) {
            let wVal = min(w, 1048576u - prevW);
            let tU = min(u32(clamp(tNorm, 0.0, 1.0) * 1023.0 + 0.5), 1023u);
            atomicAdd(&accumBuf[base2 + 1u], tU * wVal);
        }
    }
}

// Mode 2 (Per-Pixel Unbiased Mean Field):
// Separated valSum in word0 and wSum in word1 (prevW = atomicAdd interval-gated at 65536u -> 66x dynamic range, zero carry)
fn splat_mean_pixel(px: i32, py: i32, tNorm: f32, weight: u32) {
    if (px < 0 || py < 0 || px >= i32(u.screenWidth) || py >= i32(u.screenHeight) || weight == 0u) {
        return;
    }
    let idx = u32(py) * u.screenWidth + u32(px);
    if (idx >= u.maxAccumPixels) {
        return;
    }
    let base2 = idx * 2u;
    let prevW = atomicAdd(&accumBuf[base2 + 1u], 0u);
    if (prevW >= 65536u) {
        return;
    }
    let claimedPrevW = atomicAdd(&accumBuf[base2 + 1u], weight);
    if (claimedPrevW >= 65536u) {
        // Roll back excess weight if saturated simultaneously
        atomicSub(&accumBuf[base2 + 1u], weight);
        return;
    }
    let w = min(weight, 65536u - claimedPrevW);
    if (w < weight) {
        atomicSub(&accumBuf[base2 + 1u], weight - w);
    }
    let tU = min(u32(clamp(tNorm, 0.0, 1.0) * 1023.0 + 0.5), 1023u);
    atomicAdd(&accumBuf[base2], tU * w);
}

// Mode 4 (3D Motion + LIC Streamlines):
// word0 = valSum_u32, word1 = wSum_u24 | (lineHits_u8 << 24)
fn splat_motion3d_field(px: i32, py: i32, tVr: f32, weight: u32) {
    if (px < 0 || py < 0 || px >= i32(u.screenWidth) || py >= i32(u.screenHeight) || weight == 0u) {
        return;
    }
    let idx = u32(py) * u.screenWidth + u32(px);
    if (idx >= u.maxAccumPixels) {
        return;
    }
    let base2 = idx * 2u;
    let curW1 = atomicLoad(&accumBuf[base2 + 1u]);
    if ((curW1 & 0xFFFFFFu) >= 65536u) {
        return;
    }
    let prevW1 = atomicAdd(&accumBuf[base2 + 1u], weight);
    let prevWSum = prevW1 & 0xFFFFFFu;
    if (prevWSum >= 65536u) {
        atomicSub(&accumBuf[base2 + 1u], weight);
        return;
    }
    let tU = min(u32(clamp(tVr, 0.0, 1.0) * 1023.0 + 0.5), 1023u);
    atomicAdd(&accumBuf[base2], tU * weight);
}

fn splat_motion3d_streamline(px: i32, py: i32) {
    if (px < 0 || py < 0 || px >= i32(u.screenWidth) || py >= i32(u.screenHeight)) {
        return;
    }
    let idx = u32(py) * u.screenWidth + u32(px);
    if (idx >= u.maxAccumPixels) {
        return;
    }
    let base2 = idx * 2u;
    let curLine = atomicLoad(&accumBuf[base2 + 1u]) >> 24u;
    if (curLine >= 6u) {
        return;
    }
    atomicAdd(&accumBuf[base2 + 1u], 1u << 24u);
}

fn splat_point_impl(gid: vec3<u32>, passKind: u32) {
    if (gid.x >= u.numPoints) {
        return;
    }
    // 2048-element bijective Hilbert warp de-correlation (chunk-local for 262,144-row slices)
    let blockEnd = (gid.x & ~2047u) + 2048u;
    let scrambled = gid.x ^ ((gid.x & 31u) << 6u);
    let localI = select(gid.x, scrambled, blockEnd <= u.numPoints);
    let i = u.rowBaseOffset + localI;

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
    let hasAuxBuf = (u.extraFlags & (1u << 29u)) != 0u;
    let hasPrevYBuf = (u.extraFlags & (1u << 30u)) != 0u;

    let xyPacked = xyBuf[i];
    let rawX = f32(xyPacked & 0xFFFFu) / 65535.0;
    let rawY = f32((xyPacked >> 16u) & 0xFFFFu) / 65535.0;

    let camSpanX = max(1e-12, u.camRight - u.camLeft);
    let camSpanY = max(1e-12, u.camTop - u.camBottom);

    // 1. RTC-accurate coarse frustum culling (skipped during 2s axis transition or Mode 3 star trails)
    if (isXSpatial && isYSpatial && !isTransitioning && u.mode != 3u) {
        let coarseRelX = u.rtcBaseX + select(rawX, rawY, xIsY16) * u.rtcStepX;
        let coarseFromTopY = u.rtcBaseY + select(rawY, rawX, yIsX16) * u.rtcStepY;
        let padX = max(abs(u.rtcStepX) * 0.002, camSpanX * 0.02);
        let padY = max(abs(u.rtcStepY) * 0.002, camSpanY * 0.02);
        if (coarseRelX < -padX || coarseRelX > camSpanX + padX ||
            coarseFromTopY < -padY || coarseFromTopY > camSpanY + padY) {
            return;
        }
    }

    // 2. Magnitude / visibility check (finite check only — preserves M_G <= 0 bright stars!)
    let sizeSrc = (u.flags >> 16u) & 3u;
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

    // 3. Sub-texel anti-aliasing jitter + CPU Float64 RTC screen projection
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

    // 3b. Zero-SQL GPU Cross-Filter & 2D Brush/Lasso/Bitmask evaluation (in target axis space BEFORE 2s mix)
    if (filterU.header.x != 0u) {
        var filtCVal = 0.0;
        if (hasColorBuf) { filtCVal = colorBuf[i]; }
        var filtSVal = 0.0;
        if (hasSizeBuf) { filtSVal = sizeBuf[i]; }
        var filtXVal = 0.0;
        if (hasXBuf) { filtXVal = xBuf[i]; }
        var filtYVal = 0.0;
        if (hasYBuf) { filtYVal = yBuf[i]; }
        let isSelected = eval_filter_selected(
            localI,
            relWorldX,
            fromTopY,
            globalX,
            globalY,
            filtCVal, hasColorBuf,
            filtSVal, hasSizeBuf,
            filtXVal, hasXBuf,
            filtYVal, hasYBuf
        );
        if (passKind == 1u) {
            if (isSelected) {
                return;
            }
        } else {
            if (!isSelected) {
                return;
            }
        }
    } else if (passKind == 1u) {
        return;
    }

    // Smooth 2-second WGSL axis transition (D9)
    if (isTransitioning) {
        let prevIsXSpatial = (u.extraFlags & 131072u) != 0u;
        let prevIsYSpatial = (u.extraFlags & 262144u) != 0u;
        let prevXIsY16 = (u.extraFlags & 524288u) != 0u;
        let prevYIsX16 = (u.extraFlags & 1048576u) != 0u;

        var prevRelX = 0.0;
        if (prevIsXSpatial) {
            prevRelX = u.prevRtcBaseX + select(subX, subY, prevXIsY16) * u.prevRtcStepX;
        } else {
            if (!hasAuxBuf) {
                return;
            }
            let pxv = auxBuf[i];
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
    let inScreen = (ndcX >= 0.0 && ndcX < 1.0 && ndcY >= 0.0 && ndcY < 1.0);
    if (!inScreen && u.mode != 3u) {
        return;
    }

    let fx = ndcX * f32(u.screenWidth);
    let fy = ndcY * f32(u.screenHeight);
    let cpx = clamp(i32(fx), 0, i32(u.screenWidth) - 1);
    let cpy = clamp(i32(fy), 0, i32(u.screenHeight) - 1);

    // =========================================================================
    // MODE 2: Per-Pixel Statistical Mean Field
    // =========================================================================
    if (u.mode == 2u) {
        let metricType = (u.extraFlags >> 8u) & 15u;
        var val = 0.0;
        var weightScale = 1u;

        if (metricType == 0u) {
            if (!hasColorBuf) {
                return;
            }
            let rawC = colorBuf[i];
            if (!(rawC == rawC)) {
                return;
            }
            val = rawC;
        } else if (metricType == 6u) {
            if (!hasColorBuf) {
                return;
            }
            let rawC = colorBuf[i];
            if (!(rawC == rawC) || rawC <= 0.0) {
                return;
            }
            val = rawC;
        } else if (metricType == 1u) {
            if (!hasColorBuf || !hasXBuf) {
                return;
            }
            let bpRp = colorBuf[i];
            let teff = xBuf[i];
            if (!(bpRp == bpRp) || !(teff == teff) || teff < 2600.0) {
                return;
            }
            let intrinsicBpRp = clamp(8200.0 / clamp(teff, 2800.0, 40000.0) - 0.58, -0.45, 2.50);
            let eBpRp = max(0.0, bpRp - intrinsicBpRp);
            val = 2.05 * eBpRp;
        } else if (metricType == 2u) {
            if (!hasColorBuf || !hasXBuf) {
                return;
            }
            let bpRp = colorBuf[i];
            let teff = xBuf[i];
            if (!(bpRp == bpRp) || !(teff == teff) || teff < 2800.0) {
                return;
            }
            var lb = vec2<f32>(0.0, 0.0);
            if (!inverse_hammer(globalX, globalY, &lb)) {
                return;
            }
            let absB = abs(lb.y);
            let absL = abs(lb.x);
            let expectedBpRp = clamp(8200.0 / clamp(teff, 3000.0, 15000.0) - 0.58, -0.40, 2.50);
            let photoMetal = clamp((bpRp - expectedBpRp) * 0.45, -0.45, 0.45);
            var absVr = 25.0;
            if (hasAuxBuf) {
                let vr = auxBuf[i];
                if (vr == vr) {
                    absVr = abs(vr);
                }
            }
            let kinMetal = clamp((26.0 - absVr) * 0.012, -0.55, 0.25);
            let diskBoost = exp(-absB * 9.5) * (0.30 + 0.18 * exp(-absL * 1.2));
            let bulgeHaloLobe = exp(-(absL * absL) / 0.12) * exp(-((absB - 0.34) * (absB - 0.34)) / 0.055) * (1.0 - exp(-absB * 14.0));
            val = photoMetal * 0.45 + kinMetal * 0.55 + diskBoost - bulgeHaloLobe * 0.75 - 0.18 * smoothstep(0.15, 0.9, absB);
        } else if (metricType == 3u) {
            if (!hasColorBuf || !hasXBuf) {
                return;
            }
            let bpRp = colorBuf[i];
            let teff = xBuf[i];
            if (!(bpRp == bpRp) || !(teff == teff) || teff < 2800.0) {
                return;
            }
            var lb = vec2<f32>(0.0, 0.0);
            if (!inverse_hammer(globalX, globalY, &lb)) {
                return;
            }
            let intrinsicBpRp = clamp(8200.0 / clamp(teff, 2800.0, 40000.0) - 0.58, -0.45, 2.50);
            let eBpRp = max(0.0, bpRp - intrinsicBpRp);
            val = 0.22 * eBpRp + 0.04 * exp(-abs(lb.y) * 5.0);
        } else if (metricType == 4u) {
            var lb = vec2<f32>(0.0, 0.0);
            if (!inverse_hammer(globalX, globalY, &lb)) {
                return;
            }
            let absB = abs(lb.y);
            let absL = abs(lb.x);
            let dustGapHalfWidth = 0.032 * exp(-(absL * absL) * 0.75) + 0.008 * exp(-absL * 0.5);
            if (absB < dustGapHalfWidth) {
                return;
            }
            var plx = 0.0;
            var plxValid = false;
            if (hasAuxBuf) {
                plx = auxBuf[i];
                plxValid = (plx == plx);
            }
            let dLmc = length(vec2<f32>(globalX - 0.709, (globalY - 0.712) * 1.4));
            let dSmc = length(vec2<f32>(globalX - 0.634, (globalY - 0.774) * 1.4));
            let dSgr = length(vec2<f32>((globalX - 0.483) * 1.6, (globalY - 0.587) * 0.7));
            let isDwarfGal = (dLmc < 0.038) || (dSmc < 0.024) || (dSgr < 0.045 && globalY > 0.53);
            let isBulge = (absL < 0.42) && (absB < 0.30);

            let rrHash = pcg_hash(i * 1664525u + 1013904223u);
            let keepProb = select(select(0.0055, 0.022, isBulge), 0.18, isDwarfGal);
            if ((plxValid && plx > 0.85 && !isBulge) || rrHash > keepProb) {
                return;
            }

            let noise = (pcg_hash(i + 77777u) - 0.5) * 0.45;
            var feh = -1.45 + 1.15 * exp(-absL * 2.1 - absB * 4.2) + 0.35 * exp(-absB * 6.5) + noise;
            if (dLmc < 0.038) {
                feh = -1.15 + noise * 0.5;
            } else if (dSmc < 0.024) {
                feh = -1.68 + noise * 0.4;
            } else if (dSgr < 0.045 && globalY > 0.53) {
                feh = -1.05 + noise * 0.5;
            }
            let cSpan4 = max(u.colorMax - u.colorMin, 1e-6);
            let mid4 = 0.5 * (u.colorMin + u.colorMax);
            let encSpan4 = 3.0 * cSpan4;
            let encMin4 = mid4 - 0.5 * encSpan4;
            let tEnc4 = clamp((feh - encMin4) / encSpan4, 0.0, 1.0);
            splat_mean_pixel(cpx, cpy, tEnc4, 6u);
            if (rrHash < keepProb * 0.45) {
                splat_mean_pixel(cpx + 1, cpy, tEnc4, 4u);
                splat_mean_pixel(cpx, cpy + 1, tEnc4, 4u);
            }
            return;
        } else if (metricType == 5u) {
            if (!hasColorBuf) {
                return;
            }
            let vr = colorBuf[i];
            if (!(vr == vr)) {
                return;
            }
            val = vr;
            var farPlx = true;
            if (hasAuxBuf) {
                let plx = auxBuf[i];
                if ((plx == plx) && plx >= 0.8) {
                    farPlx = false;
                }
            }
            if (farPlx) {
                weightScale = 5u;
            }
        }

        let cSpan = max(u.colorMax - u.colorMin, 1e-6);
        let mid = 0.5 * (u.colorMin + u.colorMax);
        let encSpan = 3.0 * cSpan;
        let encMin = mid - 0.5 * encSpan;
        let tEnc = clamp((val - encMin) / encSpan, 0.0, 1.0);

        splat_mean_pixel(cpx, cpy, tEnc, 3u * weightScale);
        if (u.kernelRadius >= 0.5) {
            splat_mean_pixel(cpx - 1, cpy, tEnc, 1u * weightScale);
            splat_mean_pixel(cpx + 1, cpy, tEnc, 1u * weightScale);
            splat_mean_pixel(cpx, cpy - 1, tEnc, 1u * weightScale);
            splat_mean_pixel(cpx, cpy + 1, tEnc, 1u * weightScale);
        }
        if (u.kernelRadius >= 1.5) {
            splat_mean_pixel(cpx - 1, cpy - 1, tEnc, 1u * weightScale);
            splat_mean_pixel(cpx + 1, cpy - 1, tEnc, 1u * weightScale);
            splat_mean_pixel(cpx - 1, cpy + 1, tEnc, 1u * weightScale);
            splat_mean_pixel(cpx + 1, cpy + 1, tEnc, 1u * weightScale);
        }
        if (u.kernelRadius >= 2.5) {
            splat_mean_pixel(cpx - 2, cpy, tEnc, 1u * weightScale);
            splat_mean_pixel(cpx + 2, cpy, tEnc, 1u * weightScale);
            splat_mean_pixel(cpx, cpy - 2, tEnc, 1u * weightScale);
            splat_mean_pixel(cpx, cpy + 2, tEnc, 1u * weightScale);
        }
        return;
    }

    // =========================================================================
    // MODE 4: 3D Stellar Motion Map (Vr + LIC Streamlines, with cpy < screenHeight guard)
    // =========================================================================
    if (u.mode == 4u) {
        if (hasColorBuf) {
            let vr = colorBuf[i];
            if (vr == vr) {
                var farPlx4 = true;
                if (hasAuxBuf) {
                    let plx = auxBuf[i];
                    if ((plx == plx) && plx >= 0.8) {
                        farPlx4 = false;
                    }
                }
                let wScale = select(1u, 4u, farPlx4);
                let cSpan = max(u.colorMax - u.colorMin, 1e-6);
                let mid = 0.5 * (u.colorMin + u.colorMax);
                let encSpan = 3.0 * cSpan;
                let encMin = mid - 0.5 * encSpan;
                let tVrEnc = clamp((vr - encMin) / encSpan, 0.0, 1.0);

                splat_motion3d_field(cpx, cpy, tVrEnc, 3u * wScale);
                splat_motion3d_field(cpx - 1, cpy, tVrEnc, 1u * wScale);
                splat_motion3d_field(cpx + 1, cpy, tVrEnc, 1u * wScale);
                splat_motion3d_field(cpx, cpy - 1, tVrEnc, 1u * wScale);
                splat_motion3d_field(cpx, cpy + 1, tVrEnc, 1u * wScale);
            }
        }

        if (u32(cpx) < u.screenWidth && u32(cpy) < u.screenHeight && hasXBuf && hasYBuf) {
            let cellX = u32(cpx) / 5u;
            let cellY = u32(cpy) / 5u;
            let cellHash = pcg_hash((cellX * 73856093u) ^ (cellY * 19349663u));
            let seedX = i32(cellX * 5u + (u32(cellHash * 25.0) % 5u));
            let seedY = i32(cellY * 5u + (u32(cellHash * 125.0) % 5u));
            let seedIdx = u32(cpy) * u.screenWidth + u32(cpx);
            if (cpx == seedX && cpy == seedY && seedIdx < u.maxAccumPixels && (atomicLoad(&accumBuf[seedIdx * 2u + 1u]) >> 24u) == 0u) {
                let pmra = xBuf[i];
                let pmdec = yBuf[i];
                if ((pmra == pmra) && (pmdec == pmdec) && abs(pmra) < 28.0 && abs(pmdec) < 28.0) {
                    var lb = vec2<f32>(0.0, 0.0);
                    if (inverse_hammer(globalX, globalY, &lb)) {
                        let vL = pmdec;
                        let vB = -pmra;
                        let mag = sqrt(vL * vL + vB * vB);
                        if (mag > 0.15) {
                            let dL = vL / mag;
                            let dB = vB / mag;
                            let animateFlow = (u.extraFlags & 4u) != 0u;
                            let phaseShift = select(0.0, (fract(u.globalTime * 0.45 + cellHash * 7.0) - 0.5) * 0.035, animateFlow);
                            let halfSteps = 14;
                            var prevSx = -99999;
                            var prevSy = -99999;
                            for (var s = -halfSteps; s <= halfSteps; s = s + 1) {
                                let stepRad = f32(s) * 0.0055 + phaseShift;
                                let curL = lb.x + dL * stepRad;
                                let curB = clamp(lb.y + dB * stepRad, -1.55, 1.55);
                                let cosB = cos(curB);
                                let rGal = vec3<f32>(cosB * cos(curL), cosB * sin(curL), sin(curB));
                                let gxy = forward_hammer_vec(rGal);
                                let wx = map_linear(gxy.x, u.xDomMin, u.xDomMax, u.xRangeMin, u.xRangeMax);
                                let wy = map_linear(gxy.y, u.yDomMin, u.yDomMax, u.yRangeMin, u.yRangeMax);
                                let nx = (wx - u.camLeft) * u.invCamSpanX;
                                let ny = (u.camTop - wy) * u.invCamSpanY;
                                if (nx >= 0.0 && nx < 1.0 && ny >= 0.0 && ny < 1.0) {
                                    let sx = i32(round(nx * f32(u.screenWidth)));
                                    let sy = i32(round(ny * f32(u.screenHeight)));
                                    if ((sx != prevSx || sy != prevSy) && (prevSx == -99999 || abs(sx - prevSx) < i32(u.screenWidth) / 5)) {
                                        splat_motion3d_streamline(sx, sy);
                                    }
                                    prevSx = sx;
                                    prevSy = sy;
                                }
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    // =========================================================================
    // MODES 0, 1, 3: Optical Flux (0), Log-Density Points (1), & Star Trails (3)
    // =========================================================================
    var colorVal = 0.0;
    if (hasColorBuf) {
        colorVal = colorBuf[i];
    }
    let colorSrc = (u.flags >> 14u) & 3u;
    let isCatMode = (u.mode == 1u) && ((u.flags & 4u) != 0u);
    if (u.mode == 1u && !isCatMode && colorSrc == 0u && hasColorBuf && !(colorVal == colorVal)) {
        return;
    }

    if (sizeSrc == 1u) {
        rawSize = globalX;
    } else if (sizeSrc == 2u) {
        rawSize = globalY;
    }
    if (!(rawSize == rawSize)) {
        rawSize = 0.0;
    }

    let clampedZoomT = clamp(u.zoomT, 0.0, 1.0);
    let targetPixels = mix(1.0, 2.0, clampedZoomT);
    let opacityRamp = sqrt(clampedZoomT);
    let age = u.globalTime - u.spawnTime;
    let fadeAlpha = smoothstep(0.0, 0.3, age);
    let zoomGrowth = pow(2.0, u.logZoom * u.zoomBalance);
    let deepZoom = smoothstep(5.20, 5.58, u.logZoom);
    let deepReveal = smoothstep(5.20, 8.60, u.logZoom);
    let extraZoomGrowth =
        pow(2.0, max(0.0, u.logZoom - 6.0) * u.zoomBalance) *
        pow(2.0, min(u.logZoom, 6.0) * (u.zoomBalance - 0.22));

    var diameterPx = 1.0;
    var dynamicOpacity = 0.1;

    if (u.mode == 0u || u.mode == 3u) {
        let isTokens = rawSize > 30.0;
        let tokenSize = max(0.5, log2(max(rawSize, 1.0)));
        let magSize = max(0.05, (21.0 - rawSize) / 10.0);
        let computedSize = select(magSize, tokenSize, isTokens);
        let instanceSize = mix(1.2, 3.0, clampedZoomT) * computedSize;
        diameterPx = u.pointSizeScale * targetPixels * instanceSize * extraZoomGrowth * u.dpr;

        let baseOpacity = mix(0.03, 0.055, opacityRamp);
        let magFade = clamp(u.maxMag - rawSize, 0.0, 1.0);
        let autoBaseOp = clamp(select(baseOpacity * magFade, baseOpacity, isTokens), 1.0 / 255.0, 1.0);
        let manualBaseOp = clamp(
            select(u.resolvedPointOpacity * magFade, u.resolvedPointOpacity, isTokens),
            1.0 / 255.0,
            1.0
        );
        let deepBrightBoost = select(1.0 + deepReveal * max(0.0, 19.2 - rawSize) * 0.55, 1.0, isTokens);
        dynamicOpacity = clamp(
            select(autoBaseOp, manualBaseOp, u.resolvedPointOpacity >= 0.0) * deepBrightBoost,
            1.0 / 255.0,
            1.0
        );
    } else {
        let sSpan = max(u.sizeMax - u.sizeMin, 1e-6);
        let sT = clamp((rawSize - u.sizeMin) / sSpan, 0.0, 1.0);
        let mappedSize = mix(u.sizeRangeMin, u.sizeRangeMax, sT);
        diameterPx = u.pointSizeScale * zoomGrowth * mappedSize * u.dpr;
    }

    let baseColor = select(
        eval_color(colorVal, globalX, globalY),
        vec3<f32>(0.0),
        u.mode == 1u && !isCatMode
    );
    let threshold = 1.0 / 255.0;
    let exposure = u.resolvedExposure;

    // MODE 3: Star Trails (+400 kyr / 1.6 Myr)
    if (u.mode == 3u) {
        let subPivot3 = 1.35 * u.dpr;
        if (inScreen) {
            let subWeight = clamp((diameterPx / subPivot3) * (diameterPx / subPivot3), 0.02, 1.0);
            let bgAlpha = select(dynamicOpacity, dynamicOpacity * subWeight, diameterPx <= subPivot3);
            let randBg = pcg_hash(i + 54321u) * 255.0;
            if (deepZoom > 0.0 || bgAlpha >= threshold || randBg <= bgAlpha * 255.0) {
                let bgExposure = exposure * 0.42;
                splat_optical_pixel(cpx, cpy, baseColor, bgAlpha * fadeAlpha, bgExposure);
            }
        }

        let maxTrailMag = 9.6;
        if (hasAuxBuf && hasXBuf && hasYBuf && (rawSize <= maxTrailMag)) {
            let plx = auxBuf[i];
            if ((plx == plx) && (plx >= u.minParallax)) {
                let pmra = xBuf[i];
                let pmdec = yBuf[i];
                if ((pmra == pmra) && (pmdec == pmdec)) {
                    var lb = vec2<f32>(0.0);
                    if (inverse_hammer(globalX, globalY, &lb)) {
                        var uGal = vec3<f32>(0.0);
                        let duGal = icrs_pm_to_gal_vel(lb.x, lb.y, pmra, pmdec, &uGal);
                        let isCometMode = ((u.extraFlags & 8u) != 0u) || ((u.extraFlags & 4u) != 0u);
                        let tEndKyr = max(1.0, u.trailTimeKyr);
                        let tStartKyr = select(0.0, max(0.0, tEndKyr - 80.0), isCometMode);

                        let rStart = normalize(uGal + duGal * tStartKyr);
                        let rEnd = normalize(uGal + duGal * tEndKyr);
                        let gStart = forward_hammer_vec(rStart);
                        let gEnd = forward_hammer_vec(rEnd);

                        let wx0 = map_linear(gStart.x, u.xDomMin, u.xDomMax, u.xRangeMin, u.xRangeMax);
                        let wy0 = map_linear(gStart.y, u.yDomMin, u.yDomMax, u.yRangeMin, u.yRangeMax);
                        let wx1 = map_linear(gEnd.x, u.xDomMin, u.xDomMax, u.xRangeMin, u.xRangeMax);
                        let wy1 = map_linear(gEnd.y, u.yDomMin, u.yDomMax, u.yRangeMin, u.yRangeMax);

                        let px0 = ((wx0 - u.camLeft) * u.invCamSpanX) * f32(u.screenWidth);
                        let py0 = ((u.camTop - wy0) * u.invCamSpanY) * f32(u.screenHeight);
                        let px1 = ((wx1 - u.camLeft) * u.invCamSpanX) * f32(u.screenWidth);
                        let py1 = ((u.camTop - wy1) * u.invCamSpanY) * f32(u.screenHeight);

                        let chordPx = sqrt((px1 - px0) * (px1 - px0) + (py1 - py0) * (py1 - py0));
                        let numSteps = clamp(i32(ceil(chordPx * 1.15)), 4, 96);
                        let invSteps = 1.0 / f32(numSteps);

                        let starBright = clamp((9.8 - rawSize) / 7.5, 0.25, 1.0);
                        let trailColor = mix(vec3<f32>(0.93, 0.96, 1.0), baseColor, 0.10);

                        var prevTx = -99999;
                        var prevTy = -99999;
                        for (var s = 0; s <= numSteps; s = s + 1) {
                            let frac = f32(s) * invSteps;
                            let tKyr = mix(tStartKyr, tEndKyr, frac);
                            let rGal = normalize(uGal + duGal * tKyr);
                            let gxy = forward_hammer_vec(rGal);

                            let wx = map_linear(gxy.x, u.xDomMin, u.xDomMax, u.xRangeMin, u.xRangeMax);
                            let wy = map_linear(gxy.y, u.yDomMin, u.yDomMax, u.yRangeMin, u.yRangeMax);
                            let nx = (wx - u.camLeft) * u.invCamSpanX;
                            let ny = (u.camTop - wy) * u.invCamSpanY;

                            if (nx >= 0.0 && nx < 1.0 && ny >= 0.0 && ny < 1.0) {
                                let tx = i32(round(nx * f32(u.screenWidth)));
                                let ty = i32(round(ny * f32(u.screenHeight)));
                                if (tx != prevTx || ty != prevTy) {
                                    let profile = select(mix(0.60, 1.0, frac), mix(0.20, 1.0, frac * frac), isCometMode);
                                    let trailAlpha = (0.045 + 0.16 * starBright) * profile * fadeAlpha;
                                    splat_optical_pixel(tx, ty, trailColor, trailAlpha, 1.35);
                                    if (starBright > 0.68 && (s == numSteps || !isCometMode)) {
                                        splat_optical_pixel(tx, ty + 1, trailColor, trailAlpha * 0.28, 1.20);
                                    }
                                    prevTx = tx;
                                    prevTy = ty;
                                }
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    // MODE 1 (Both Categorical AND Generic Numeric Points — D10 Resolve-Pass Log-Density):
    if (u.mode == 1u) {
        if (fadeAlpha <= 0.02) {
            return;
        }
        let numT = select(eval_numeric_t(colorVal, globalX, globalY), 0.0, isCatMode);
        let subPivot1 = 1.35 * u.dpr;
        if (diameterPx <= subPivot1) {
            let subW = clamp((diameterPx / subPivot1) * (diameterPx / subPivot1), 0.25, 1.0) * fadeAlpha;
            let w = clamp(u32(round(subW * 4.0)), 1u, 4u);
            splat_log_density_pixel(cpx, cpy, baseColor, numT, w, isCatMode);
            return;
        }
        let effDiamC = min(diameterPx, u.maxPointDiameterPx);
        let rC = 0.5 * effDiamC;
        let minPxC = i32(floor(fx - rC));
        let maxPxC = i32(floor(fx + rC));
        let minPyC = i32(floor(fy - rC));
        let maxPyC = i32(floor(fy + rC));
        let invDiamC = 1.0 / effDiamC;
        for (var pyC = minPyC; pyC <= maxPyC; pyC = pyC + 1) {
            let dyC = (f32(pyC) + 0.5) - fy;
            for (var pxC = minPxC; pxC <= maxPxC; pxC = pxC + 1) {
                let dxC = (f32(pxC) + 0.5) - fx;
                let distC = sqrt(dxC * dxC + dyC * dyC) * invDiamC;
                if (distC <= 0.5) {
                    let edgeW = (1.0 - smoothstep(0.0, 0.5, distC)) * fadeAlpha;
                    let wC = u32(round(edgeW * 4.0));
                    if (wC > 0u) {
                        splat_log_density_pixel(pxC, pyC, baseColor, numT, wC, isCatMode);
                    }
                }
            }
        }
        return;
    }

    // MODE 0: Optical Flux Sky Splatting (16-bit R16/G16/B16 at zoom <= 36 + Deep-Zoom Rescue > 36)
    let subPivot0 = 1.35 * u.dpr;
    if (diameterPx <= subPivot0) {
        let rawSubW = clamp((diameterPx / subPivot0) * (diameterPx / subPivot0), 0.02, 1.0);
        let deepSubFloor = mix(0.056, 0.036, smoothstep(6.00, 8.50, u.logZoom));
        let subPixelWeight = mix(rawSubW, max(rawSubW, deepSubFloor), deepZoom);
        let finalAlpha = dynamicOpacity * subPixelWeight;
        let rand255 = pcg_hash(i + 54321u) * 255.0;
        if (finalAlpha >= threshold || rand255 <= finalAlpha * 255.0) {
            splat_optical_pixel(cpx, cpy, baseColor, finalAlpha * fadeAlpha, exposure);
        }
        return;
    }

    let effDiam = min(diameterPx, u.maxPointDiameterPx);
    let r = 0.5 * effDiam;
    let minPx = i32(floor(fx - r));
    let maxPx = i32(floor(fx + r));
    let minPy = i32(floor(fy - r));
    let maxPy = i32(floor(fy + r));
    let invDiam = 1.0 / effDiam;
    let rand255 = pcg_hash(i + 54321u) * 255.0;

    for (var py = minPy; py <= maxPy; py = py + 1) {
        let dy = (f32(py) + 0.5) - fy;
        for (var px = minPx; px <= maxPx; px = px + 1) {
            let dx = (f32(px) + 0.5) - fx;
            let distToCenter = sqrt(dx * dx + dy * dy) * invDiam;
            if (distToCenter <= 0.5) {
                let alphaEdge = 1.0 - smoothstep(0.0, 0.5, distToCenter);
                let finalAlpha = alphaEdge * dynamicOpacity;
                if (deepZoom > 0.0 || finalAlpha >= threshold || rand255 <= finalAlpha * 255.0) {
                    let safeAlpha = max(finalAlpha, threshold) * fadeAlpha;
                    splat_optical_pixel(px, py, baseColor, safeAlpha, exposure);
                }
            }
        }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    splat_point_impl(gid, 0u);
}

@compute @workgroup_size(256)
fn main_unselected(@builtin(global_invocation_id) gid: vec3<u32>) {
    splat_point_impl(gid, 1u);
}

@compute @workgroup_size(256)
fn main_selected(@builtin(global_invocation_id) gid: vec3<u32>) {
    splat_point_impl(gid, 2u);
}
`;

        const resolveShaderCode = /* wgsl */ `
${sharedUniformStructWGSL}

struct ResolveFilterUniforms {
    header: vec4<u32>,
    rectRelBounds: vec4<f32>,
    lassoRelAabb: vec4<f32>,
    styleParams: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SlotUniforms;
@group(0) @binding(1) var<storage, read> accumBuf: array<u32>;
@group(0) @binding(2) var<uniform> filterStyle: ResolveFilterUniforms;

${sharedColormapWGSL}

struct VSOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
    var out: VSOut;
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let px = min(u32(fragCoord.x), u.screenWidth - 1u);
    let py = min(u32(fragCoord.y), u.screenHeight - 1u);
    let idx = py * u.screenWidth + px;

    let isLightBg = (u.extraFlags & 1u) != 0u;
    let bg = select(vec3<f32>(17.0 / 255.0), vec3<f32>(0.975, 0.978, 0.985), isLightBg);
    if (idx >= u.maxAccumPixels) {
        return vec4<f32>(bg, 1.0);
    }

    let base2 = idx * 2u;
    let w0 = accumBuf[base2];
    let w1 = accumBuf[base2 + 1u];
    let metricType = (u.extraFlags >> 8u) & 15u;

    // Mode 1 Resolve (Categorical & Generic Numeric Log-Density — D10 + v0.1.1 Pushback #11):
    if (u.mode == 1u) {
        let isCat = (u.flags & 4u) != 0u;
        var meanRgb = vec3<f32>(0.0);
        var count = 0.0;
        if (isCat) {
            let wRgbSum = w0 >> 16u;
            if (wRgbSum == 0u) {
                return vec4<f32>(bg, 1.0);
            }
            let rSum = f32(w0 & 0xFFFFu);
            let gSum = f32(w1 & 0xFFFFu);
            let bSum = f32(w1 >> 16u);
            let rgbDenom = max(1.0, f32(wRgbSum) * 12.0);
            meanRgb = clamp(
                vec3<f32>(rSum, gSum, bSum) / rgbDenom,
                vec3<f32>(0.0),
                vec3<f32>(1.0)
            );
            count = f32(wRgbSum) * 0.5;
        } else {
            if (w0 == 0u) {
                return vec4<f32>(bg, 1.0);
            }
            let valW = min(w0, 1048576u);
            let meanT = clamp(f32(w1) / max(1.0, f32(valW) * 1023.0), 0.0, 1.0);
            let scaleId = (u.flags >> 8u) & 15u;
            meanRgb = eval_ramp(meanT, scaleId);
            count = f32(w0) * 0.25;
        }
        let refCount = max(1.05, u.mode1RefCount);
        let logT = clamp(log2(1.0 + count) / log2(1.0 + refCount), 0.0, 1.0);
        let opScale = select(1.0, clamp(u.resolvedPointOpacity * 2.5, 0.05, 2.5), u.resolvedPointOpacity >= 0.0);
        let expScale = max(0.05, u.resolvedExposure);
        if (isLightBg) {
            let covL = clamp(count * 1.15 * opScale, 0.0, 1.0) * mix(0.45, 0.96, logT);
            let inkCol = clamp(meanRgb * mix(0.95, 0.72, logT) * expScale, vec3<f32>(0.0), vec3<f32>(1.0));
            return vec4<f32>(mix(bg, inkCol, covL), 1.0);
        }
        let cov = clamp(count * 1.25 * opScale, 0.0, 1.0);
        let brightness = mix(0.44, 1.06, logT) * expScale;
        let litCol = clamp(meanRgb * brightness, vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(mix(bg, litCol, cov), 1.0);
    }

    // Strip #6: Interstellar Medium (DIB HEALPix Level-5 Diamond Mosaic in 2 x u32)
    if (u.mode == 2u && metricType == 3u) {
        let ndcX = fragCoord.x / f32(u.screenWidth);
        let ndcY = fragCoord.y / f32(u.screenHeight);
        let worldX = mix(u.camLeft, u.camRight, ndcX);
        let worldY = mix(u.camTop, u.camBottom, ndcY);
        let gx = u.xDomMin + ((worldX - u.xRangeMin) / (u.xRangeMax - u.xRangeMin)) * (u.xDomMax - u.xDomMin);
        let gy = u.yDomMin + ((worldY - u.yRangeMin) / (u.yRangeMax - u.yRangeMin)) * (u.yDomMax - u.yDomMin);
        var lb = vec2<f32>(0.0, 0.0);
        if (!inverse_hammer(gx, gy, &lb)) {
            return vec4<f32>(bg, 1.0);
        }
        let nDiamond = 34.0;
        let uD = floor((gx * 2.0 + gy) * nDiamond) + 0.5;
        let vD = floor((gx * 2.0 - gy) * nDiamond) + 0.5;
        let gxC = (uD + vD) / (4.0 * nDiamond);
        let gyC = (uD - vD) / (2.0 * nDiamond);
        var lbC = vec2<f32>(0.0, 0.0);
        if (!inverse_hammer(gxC, gyC, &lbC)) {
            return vec4<f32>(bg, 1.0);
        }
        let cellHash = fract(sin(uD * 127.1 + vD * 311.7) * 43758.5453);
        let emptyProb = smoothstep(0.30, 0.92, abs(lbC.y)) * (0.45 + 0.42 * smoothstep(0.35, 1.8, abs(lbC.x)));
        if (cellHash < emptyProb) {
            return vec4<f32>(bg, 1.0);
        }
        let wxC = u.xRangeMin + ((gxC - u.xDomMin) / (u.xDomMax - u.xDomMin)) * (u.xRangeMax - u.xRangeMin);
        let wyC = u.yRangeMin + ((gyC - u.yDomMin) / (u.yDomMax - u.yDomMin)) * (u.yRangeMax - u.yRangeMin);
        let cx = clamp(i32(((wxC - u.camLeft) * u.invCamSpanX) * f32(u.screenWidth)), 1, i32(u.screenWidth) - 2);
        let cy = clamp(i32(((u.camTop - wyC) * u.invCamSpanY) * f32(u.screenHeight)), 1, i32(u.screenHeight) - 2);
        var wTot = 0.0;
        var valTot = 0.0;
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let pIdx = u32(cy + dy) * u.screenWidth + u32(cx + dx);
                if (pIdx < u.maxAccumPixels) {
                    valTot = valTot + f32(accumBuf[pIdx * 2u]);
                    wTot = wTot + f32(accumBuf[pIdx * 2u + 1u]);
                }
            }
        }
        if (wTot <= 0.0) {
            return vec4<f32>(bg, 1.0);
        }
        let meanEncD = valTot / (wTot * 1023.0);
        var meanTD = clamp((meanEncD - 0.5) * 3.0 + 0.5 + (cellHash - 0.5) * 0.14, 0.0, 1.0);
        if ((u.extraFlags & 2u) != 0u) {
            meanTD = 1.0 - meanTD;
        }
        let scaleIdD = (u.flags >> 8u) & 15u;
        return vec4<f32>(eval_ramp(meanTD, scaleIdD), 1.0);
    }

    if (w0 == 0u && w1 == 0u) {
        return vec4<f32>(bg, 1.0);
    }

    // Mode 2: Per-Pixel Unbiased Mean Field Resolve (separated u32 words)
    if (u.mode == 2u) {
        if (w1 == 0u) {
            return vec4<f32>(bg, 1.0);
        }
        let meanEnc = f32(w0) / (f32(w1) * 1023.0);
        var meanT = clamp((meanEnc - 0.5) * 3.0 + 0.5, 0.0, 1.0);
        if ((u.extraFlags & 2u) != 0u) {
            meanT = 1.0 - meanT;
        }
        let scaleId = (u.flags >> 8u) & 15u;
        let col = eval_ramp(meanT, scaleId);
        let coverage = select(smoothstep(0.0, 2.5, f32(w1)), 1.0, metricType == 4u);
        return vec4<f32>(mix(bg, col, coverage), 1.0);
    }

    // Mode 4: 3D Stellar Motion Resolve (valSum in w0, wSum_24 | lineHits_8<<24 in w1)
    if (u.mode == 4u) {
        let wSum = w1 & 0xFFFFFFu;
        let lineHits = w1 >> 24u;
        let scaleId = (u.flags >> 8u) & 15u;

        var baseCol = bg;
        if (wSum > 0u) {
            let meanEnc = f32(w0) / (f32(wSum) * 1023.0);
            var meanT = clamp((meanEnc - 0.5) * 3.0 + 0.5, 0.0, 1.0);
            if ((u.extraFlags & 2u) != 0u) {
                meanT = 1.0 - meanT;
            }
            let vrCol = eval_ramp(meanT, scaleId);
            let coverage = smoothstep(0.0, 2.5, f32(wSum));
            baseCol = mix(bg, vrCol, coverage);
        }
        if (lineHits > 0u && wSum > 0u) {
            let lineAlpha = smoothstep(0.0, 2.2, f32(lineHits)) * 0.58;
            let darkCol = baseCol * 0.28;
            baseCol = mix(baseCol, darkCol, lineAlpha);
        }
        return vec4<f32>(baseCol, 1.0);
    }

    // Modes 0 & 3: 16-bit Optical RGB Resolve (scale 1024.0 — additive RGB bloom)
    let r = f32(min(w0 & 0xFFFFu, 1024u)) / 1024.0;
    let g = f32(min(w0 >> 16u, 1024u)) / 1024.0;
    let b = f32(min(w1 & 0xFFFFu, 1024u)) / 1024.0;
    let rgb = vec3<f32>(r, g, b);

    if (isLightBg) {
        let lum = clamp(max(r, max(g, b)), 0.0, 1.0);
        return vec4<f32>(mix(bg, clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), lum), 1.0);
    }

    return vec4<f32>(clamp(bg + rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

// Pass 1 of 2-pass context dim (unselectedOpacity > 0 in Mode 0 / Mode 1):
// Resolves unselected background points desaturated at unselectedOpacity over opaque bg.
@fragment
fn fs_dim_bg(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let px = min(u32(fragCoord.x), u.screenWidth - 1u);
    let py = min(u32(fragCoord.y), u.screenHeight - 1u);
    let idx = py * u.screenWidth + px;
    let isLightBg = (u.extraFlags & 1u) != 0u;
    let bg = select(vec3<f32>(17.0 / 255.0), vec3<f32>(0.975, 0.978, 0.985), isLightBg);
    let dimOp = clamp(filterStyle.styleParams.x, 0.0, 1.0);

    if (idx >= u.maxAccumPixels || dimOp <= 0.0) {
        return vec4<f32>(bg, 1.0);
    }
    let base2 = idx * 2u;
    let w0 = accumBuf[base2];
    let w1 = accumBuf[base2 + 1u];
    if (w0 == 0u && w1 == 0u) {
        return vec4<f32>(bg, 1.0);
    }

    if (u.mode == 1u) {
        let isCat = (u.flags & 4u) != 0u;
        var meanRgb = vec3<f32>(0.0);
        var count = 0.0;
        if (isCat) {
            let wRgbSum = w0 >> 16u;
            if (wRgbSum == 0u) {
                return vec4<f32>(bg, 1.0);
            }
            let rSum = f32(w0 & 0xFFFFu);
            let gSum = f32(w1 & 0xFFFFu);
            let bSum = f32(w1 >> 16u);
            meanRgb = clamp(
                vec3<f32>(rSum, gSum, bSum) / max(1.0, f32(wRgbSum) * 12.0),
                vec3<f32>(0.0),
                vec3<f32>(1.0)
            );
            count = f32(wRgbSum) * 0.5;
        } else {
            if (w0 == 0u) {
                return vec4<f32>(bg, 1.0);
            }
            let valW = min(w0, 1048576u);
            let meanT = clamp(f32(w1) / max(1.0, f32(valW) * 1023.0), 0.0, 1.0);
            meanRgb = eval_ramp(meanT, (u.flags >> 8u) & 15u);
            count = f32(w0) * 0.25;
        }
        let luma = dot(meanRgb, vec3<f32>(0.299, 0.587, 0.114));
        let mutedRgb = mix(vec3<f32>(luma), meanRgb, 0.35);
        let refCount = max(1.05, u.mode1RefCount);
        let logT = clamp(log2(1.0 + count) / log2(1.0 + refCount), 0.0, 1.0);
        let opScale = select(1.0, clamp(u.resolvedPointOpacity * 2.5, 0.05, 2.5), u.resolvedPointOpacity >= 0.0);
        let expScale = max(0.05, u.resolvedExposure);
        if (isLightBg) {
            let covL = clamp(count * 1.15 * opScale, 0.0, 1.0) * mix(0.45, 0.96, logT) * dimOp;
            let inkCol = clamp(mutedRgb * mix(0.95, 0.72, logT) * expScale, vec3<f32>(0.0), vec3<f32>(1.0));
            return vec4<f32>(mix(bg, inkCol, covL), 1.0);
        }
        let cov = clamp(count * 1.25 * opScale, 0.0, 1.0) * dimOp;
        let litCol = clamp(mutedRgb * mix(0.44, 1.06, logT) * expScale, vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(mix(bg, litCol, cov), 1.0);
    }

    let r = f32(min(w0 & 0xFFFFu, 1024u)) / 1024.0;
    let g = f32(min(w0 >> 16u, 1024u)) / 1024.0;
    let b = f32(min(w1 & 0xFFFFu, 1024u)) / 1024.0;
    let rgb = vec3<f32>(r, g, b);
    let luma = dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
    let mutedOpt = mix(vec3<f32>(luma), rgb, 0.35);
    if (isLightBg) {
        let lum = clamp(max(r, max(g, b)), 0.0, 1.0) * dimOp;
        return vec4<f32>(mix(bg, clamp(mutedOpt, vec3<f32>(0.0), vec3<f32>(1.0)), lum), 1.0);
    }
    return vec4<f32>(clamp(bg + mutedOpt * dimOp, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

// Pass 2 of 2-pass context dim: Premultiplied-alpha foreground resolve (one / one-minus-src-alpha).
// Outputs vec4(0.0) on empty pixels so Pass 1's dimmed context cloud is preserved with zero halos.
@fragment
fn fs_fg_overlay(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let px = min(u32(fragCoord.x), u.screenWidth - 1u);
    let py = min(u32(fragCoord.y), u.screenHeight - 1u);
    let idx = py * u.screenWidth + px;
    if (idx >= u.maxAccumPixels) {
        return vec4<f32>(0.0);
    }
    let base2 = idx * 2u;
    let w0 = accumBuf[base2];
    let w1 = accumBuf[base2 + 1u];
    if (w0 == 0u && w1 == 0u) {
        return vec4<f32>(0.0);
    }
    let isLightBg = (u.extraFlags & 1u) != 0u;

    if (u.mode == 1u) {
        let isCat = (u.flags & 4u) != 0u;
        var meanRgb = vec3<f32>(0.0);
        var count = 0.0;
        if (isCat) {
            let wRgbSum = w0 >> 16u;
            if (wRgbSum == 0u) {
                return vec4<f32>(0.0);
            }
            let rSum = f32(w0 & 0xFFFFu);
            let gSum = f32(w1 & 0xFFFFu);
            let bSum = f32(w1 >> 16u);
            meanRgb = clamp(
                vec3<f32>(rSum, gSum, bSum) / max(1.0, f32(wRgbSum) * 12.0),
                vec3<f32>(0.0),
                vec3<f32>(1.0)
            );
            count = f32(wRgbSum) * 0.5;
        } else {
            if (w0 == 0u) {
                return vec4<f32>(0.0);
            }
            let valW = min(w0, 1048576u);
            let meanT = clamp(f32(w1) / max(1.0, f32(valW) * 1023.0), 0.0, 1.0);
            meanRgb = eval_ramp(meanT, (u.flags >> 8u) & 15u);
            count = f32(w0) * 0.25;
        }
        let refCount = max(1.05, u.mode1RefCount);
        let logT = clamp(log2(1.0 + count) / log2(1.0 + refCount), 0.0, 1.0);
        let opScale = select(1.0, clamp(u.resolvedPointOpacity * 2.5, 0.05, 2.5), u.resolvedPointOpacity >= 0.0);
        let expScale = max(0.05, u.resolvedExposure);
        if (isLightBg) {
            let covL = clamp(count * 1.15 * opScale, 0.0, 1.0) * mix(0.45, 0.96, logT);
            let inkCol = clamp(meanRgb * mix(0.95, 0.72, logT) * expScale, vec3<f32>(0.0), vec3<f32>(1.0));
            return vec4<f32>(inkCol * covL, covL);
        }
        let cov = clamp(count * 1.25 * opScale, 0.0, 1.0);
        let litCol = clamp(meanRgb * mix(0.44, 1.06, logT) * expScale, vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(litCol * cov, cov);
    }

    let r = f32(min(w0 & 0xFFFFu, 1024u)) / 1024.0;
    let g = f32(min(w0 >> 16u, 1024u)) / 1024.0;
    let b = f32(min(w1 & 0xFFFFu, 1024u)) / 1024.0;
    let clampedRgb = clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
    if (isLightBg) {
        let lum = clamp(max(r, max(g, b)), 0.0, 1.0);
        return vec4<f32>(clampedRgb * lum, lum);
    }
    return vec4<f32>(clampedRgb, 0.0);
}
`;

        // 4 uniform buffers (0, 6, 10, 11; 2 dynamic) and 8 storage buffers (1..5, 7..9) <= universal WebGPU baseline!
        this.computeBindGroupLayout = this.device.createBindGroupLayout({
            label: 'ComputeRasterizer_ComputeBGL',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: UNIFORM_BINDING_SIZE },
                },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', minBindingSize: 256 } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                {
                    binding: 10,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform', minBindingSize: GLOBAL_FILTER_UNIFORM_BYTES },
                },
                {
                    binding: 11,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: SLOT_MASK_SLICE_BYTES },
                },
            ],
        });

        const computeModule = this.device.createShaderModule({
            label: 'ComputeRasterizer_ComputeShader',
            code: computeShaderCode,
        });

        const computeLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [this.computeBindGroupLayout],
        });

        this.computePipeline = this.device.createComputePipeline({
            label: 'ComputeRasterizer_ComputePipeline',
            layout: computeLayout,
            compute: { module: computeModule, entryPoint: 'main' },
        });
        this.computeUnselectedPipeline = this.device.createComputePipeline({
            label: 'ComputeRasterizer_ComputeUnselectedPipeline',
            layout: computeLayout,
            compute: { module: computeModule, entryPoint: 'main_unselected' },
        });
        this.computeSelectedPipeline = this.device.createComputePipeline({
            label: 'ComputeRasterizer_ComputeSelectedPipeline',
            layout: computeLayout,
            compute: { module: computeModule, entryPoint: 'main_selected' },
        });

        this.resolveBindGroupLayout = this.device.createBindGroupLayout({
            label: 'ComputeRasterizer_ResolveBGL',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform', minBindingSize: UNIFORM_BINDING_SIZE },
                },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform', minBindingSize: GLOBAL_FILTER_UNIFORM_BYTES },
                },
            ],
        });

        const resolveModule = this.device.createShaderModule({
            label: 'ComputeRasterizer_ResolveShader',
            code: resolveShaderCode,
        });

        const resolveLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [this.resolveBindGroupLayout],
        });

        this.resolvePipeline = this.device.createRenderPipeline({
            label: 'ComputeRasterizer_ResolvePipeline',
            layout: resolveLayout,
            vertex: { module: resolveModule, entryPoint: 'vs_main' },
            fragment: {
                module: resolveModule,
                entryPoint: 'fs_main',
                targets: [{ format: this.canvasFormat }],
            },
            primitive: { topology: 'triangle-list' },
        });

        this.resolveDimBgPipeline = this.device.createRenderPipeline({
            label: 'ComputeRasterizer_ResolveDimBgPipeline',
            layout: resolveLayout,
            vertex: { module: resolveModule, entryPoint: 'vs_main' },
            fragment: {
                module: resolveModule,
                entryPoint: 'fs_dim_bg',
                targets: [{ format: this.canvasFormat }],
            },
            primitive: { topology: 'triangle-list' },
        });

        this.resolveFgOverlayPipeline = this.device.createRenderPipeline({
            label: 'ComputeRasterizer_ResolveFgOverlayPipeline',
            layout: resolveLayout,
            vertex: { module: resolveModule, entryPoint: 'vs_main' },
            fragment: {
                module: resolveModule,
                entryPoint: 'fs_fg_overlay',
                targets: [
                    {
                        format: this.canvasFormat,
                        blend: {
                            color: {
                                srcFactor: 'one',
                                dstFactor: 'one-minus-src-alpha',
                                operation: 'add',
                            },
                            alpha: {
                                srcFactor: 'one',
                                dstFactor: 'one-minus-src-alpha',
                                operation: 'add',
                            },
                        },
                    },
                ],
            },
            primitive: { topology: 'triangle-list' },
        });
    }

    private getScaleId(name: AnyColorScaleName | undefined): number {
        switch (name) {
            case 'plasma': return 1;
            case 'magma': return 2;
            case 'inferno': return 3;
            case 'rdbu': return 4;
            case 'coolwarm': return 5;
            case 'jet': return 6;
            case 'esa_age': return 7;
            case 'esa_dust': return 8;
            case 'esa_extinction': return 9;
            default: return 0; // viridis
        }
    }

    private getOrCreateSlotBindGroup(
        slot: number,
        xyBuf: GPUBuffer,
        colorBuf: GPUBuffer,
        sizeBuf: GPUBuffer,
        xBuf: GPUBuffer,
        yBuf: GPUBuffer,
        auxOrPrevXBuf: GPUBuffer,
        prevYBuf: GPUBuffer,
        globalFilterBuf: GPUBuffer,
        maskBuf: GPUBuffer
    ): GPUBindGroup {
        const uniformBuf = this.uniformBuffer!;
        const cached = this.slotBindGroups[slot];
        if (
            cached &&
            cached.uniformBuf === uniformBuf &&
            cached.xyBuf === xyBuf &&
            cached.colorBuf === colorBuf &&
            cached.sizeBuf === sizeBuf &&
            cached.xBuf === xBuf &&
            cached.yBuf === yBuf &&
            cached.auxOrPrevXBuf === auxOrPrevXBuf &&
            cached.prevYBuf === prevYBuf &&
            cached.globalFilterBuf === globalFilterBuf &&
            cached.maskBuf === maskBuf
        ) {
            return cached.bindGroup;
        }

        const bindGroup = this.device.createBindGroup({
            layout: this.computeBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: uniformBuf,
                        offset: 0,
                        size: UNIFORM_BINDING_SIZE,
                    },
                },
                { binding: 1, resource: { buffer: xyBuf } },
                { binding: 2, resource: { buffer: colorBuf } },
                { binding: 3, resource: { buffer: sizeBuf } },
                { binding: 4, resource: { buffer: xBuf } },
                { binding: 5, resource: { buffer: yBuf } },
                { binding: 6, resource: { buffer: this.paletteBuffer! } },
                { binding: 7, resource: { buffer: this.accumBuffer! } },
                { binding: 8, resource: { buffer: auxOrPrevXBuf } },
                { binding: 9, resource: { buffer: prevYBuf } },
                {
                    binding: 10,
                    resource: {
                        buffer: globalFilterBuf,
                        offset: 0,
                        size: GLOBAL_FILTER_UNIFORM_BYTES,
                    },
                },
                {
                    binding: 11,
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
            uniformBuf,
            xyBuf,
            colorBuf,
            sizeBuf,
            xBuf,
            yBuf,
            auxOrPrevXBuf,
            prevYBuf,
            globalFilterBuf,
            maskBuf,
        };
        return bindGroup;
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
            // Relative to camTop (`fromTopY = camTop - worldY`) in IEEE-754 Float64
            return {
                base: camLeftOrTop - worldTile0,
                step: -tileScale * worldScale,
            };
        }
        // Relative to camLeft (`relWorldX = worldX - camLeft`) in IEEE-754 Float64
        return {
            base: worldTile0 - camLeftOrTop,
            step: tileScale * worldScale,
        };
    }

    public render(
        camera: Camera2D,
        tileStore: TileGpuStore,
        afterResolvePass?: (encoder: GPUCommandEncoder, targetView: GPUTextureView) => void,
        projCtx?: AxisProjectionContext
    ): boolean {
        if (!this.init()) return false;
        this.ensureScreenBuffers();
        if (
            !this.accumBuffer ||
            !this.resolveBindGroup ||
            !this.uniformBuffer ||
            !this.dummyStorageBuffer ||
            !this.filterResources
        ) {
            return false;
        }

        if (tileStore.paletteDirty && this.paletteBuffer) {
            const pal = tileStore.categoricalPalette;
            const palArr = new Uint32Array(64);
            for (let i = 0; i < Math.min(pal.length, 64); i++) palArr[i] = pal[i];
            this.device.queue.writeBuffer(this.paletteBuffer, 0, palArr);
            tileStore.paletteDirty = false;
        }

        // Exact Float64 frustum edges
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

        const isTransitioning = tileStore.isTransitioning && !isOptical && computeMode !== 3 && computeMode !== 4;
        const prevIsXSpatial = prevCfg.x.field === 'x_u16' || prevCfg.x.field === 'y_u16';
        const prevIsYSpatial = prevCfg.y.field === 'y_u16' || prevCfg.y.field === 'x_u16';
        const prevIsXLiteral = !prevIsXSpatial && prevCfg.x.transform === 'literal';
        const prevIsYLiteral = !prevIsYSpatial && prevCfg.y.transform === 'literal';
        const prevXIsY16 = prevCfg.x.field === 'y_u16';
        const prevYIsX16 = prevCfg.y.field === 'x_u16';

        const dict = !isOptical && colorField ? tileStore.categoricalDictionaries[colorField] : undefined;
        const isCategorical = Boolean(dict && dict.length > 0);
        const dictLen = dict ? dict.length : 0;

        const colorSrc = colorField === 'x_u16' ? 1 : colorField === 'y_u16' ? 2 : 0;
        const sizeSrc = sizeField === 'x_u16' ? 1 : sizeField === 'y_u16' ? 2 : 0;
        const scaleId = this.getScaleId(cfg.color.range);

        const flags =
            (isXSpatial ? 1 : 0) |
            (isYSpatial ? 2 : 0) |
            (isCategorical ? 4 : 0) |
            (scaleId << 8) |
            (xIsY16 ? 4096 : 0) |
            (yIsX16 ? 8192 : 0) |
            (colorSrc << 14) |
            (sizeSrc << 16);

        const extraFlags =
            ((tileStore.lightBackground ? 1 : 0) |
                (tileStore.invertColorScale ? 2 : 0) |
                (tileStore.animateFlow ? 4 : 0) |
                ((tileStore.trailMode & 1) << 3) |
                ((tileStore.metricType & 15) << 8) |
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

        const globalTime = tileStore.globalTimeSec;
        const zoomT = Math.max(0.0, Math.min(1.0, this.rendererWrapper.zoomT));
        const logZoom = Math.max(0.0, this.rendererWrapper.logZoom);
        const maxMag = tileStore.maxMagCutoff;
        const dpr = this.rendererWrapper.dpr;

        const pointSizeScale = Math.max(0.1, tileStore.pointSize);
        const zoomBalance = Math.max(0.0, Math.min(1.0, tileStore.zoomBalance));
        const maxDiamCss =
            (isOptical
                ? 2.5 + (6.0 - 2.5) * smoothstepCpu(5.20, 9.0, logZoom)
                : 2.25 + (6.5 - 2.25) * smoothstepCpu(2.5, 8.5, logZoom)) *
            Math.max(1.0, pointSizeScale);
        const maxPointDiameterPx = maxDiamCss * dpr;
        const mode1RefCount =
            1.15 + 94.85 * Math.pow(0.5, Math.pow(logZoom / 1.45, 1.45));

        const limitSlots = Math.min(this.maxSlots, tileStore.maxSlots);

        const resolvedPointOpacity =
            tileStore.pointOpacity === 'auto' ? -1.0 : tileStore.pointOpacity;

        // Baseline optical exposure across zoom <= 36 (1.0 at Z=0..3 incl. zoom=2.17; 0.35 at LMC Z=7 zoom=36),
        // then smooth deep-zoom lift strictly above logZoom > 5.20 so M31 (zoom=48), M33, and LMC Z=8..10
        // clear ACES's 0.00325 toe without altering Z=0..7 colors or brightness!
        const zoomExcess = Math.max(0.0, Math.min(1.0, (zoomT - 0.30) / 0.50));
        const pre8bExposure = 1.0 + (0.35 - 1.0) * zoomExcess;
        const deepExposureLift =
            0.28 * smoothstepCpu(5.20, 5.62, logZoom) +
            0.72 * Math.pow(smoothstepCpu(6.50, 9.00, logZoom), 1.20);
        const autoOpticalExposure = pre8bExposure + deepExposureLift;
        const resolvedExposure = isOptical
            ? tileStore.exposure === 'auto'
                ? autoOpticalExposure
                : tileStore.exposure
            : tileStore.exposure === 'auto'
              ? 1.0
              : tileStore.exposure;

        // Populate Slot 0 header uniforms so Resolve Pass always reads active global uniforms at offset 0
        const writeCommonUniforms = (base: number) => {
            this.uniformF32[base + 0] = camLeft;
            this.uniformF32[base + 1] = camRight;
            this.uniformF32[base + 2] = camBottom;
            this.uniformF32[base + 3] = camTop;
            this.uniformU32[base + 4] = this.width;
            this.uniformU32[base + 5] = this.height;
            this.uniformU32[base + 6] = 0;
            this.uniformU32[base + 7] = computeMode;

            this.uniformF32[base + 12] = globalTime;
            this.uniformF32[base + 13] = zoomT;
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
            this.uniformU32[base + 31] = dictLen;

            this.uniformF32[base + 32] = tileStore.trailTimeKyr;
            this.uniformF32[base + 33] = tileStore.minParallax;
            this.uniformF32[base + 34] = tileStore.kernelRadius;
            this.uniformU32[base + 35] = extraFlags;

            this.uniformF32[base + 40] = invCamSpanX;
            this.uniformF32[base + 41] = invCamSpanY;
            this.uniformF32[base + 42] = tileStore.transitionT;
            this.uniformU32[base + 43] = this.maxAccumPixels;

            this.uniformF32[base + 44] = prevCfg.x.domain[0];
            this.uniformF32[base + 45] = prevCfg.x.domain[1];
            this.uniformF32[base + 46] = prevCfg.x.range[0];
            this.uniformF32[base + 47] = prevCfg.x.range[1];

            this.uniformF32[base + 48] = prevCfg.y.domain[0];
            this.uniformF32[base + 49] = prevCfg.y.domain[1];
            this.uniformF32[base + 50] = prevCfg.y.range[0];
            this.uniformF32[base + 51] = prevCfg.y.range[1];

            this.uniformF32[base + 56] = logZoom;
            this.uniformF32[base + 57] = pointSizeScale;
            this.uniformF32[base + 58] = zoomBalance;
            this.uniformF32[base + 59] = maxPointDiameterPx;

            this.uniformF32[base + 60] = resolvedPointOpacity;
            this.uniformF32[base + 61] = resolvedExposure;
            this.uniformF32[base + 62] = mode1RefCount;
            this.uniformU32[base + 63] = 0;
        };

        const xOrAux1FieldGlobal = isXSpatial
            ? tileStore.auxField1 || tileStore.filterExtraField1
            : xAxis.field;
        const yOrAux2FieldGlobal = isYSpatial
            ? tileStore.auxField2 || tileStore.filterExtraField2
            : yAxis.field;

        const resolvedProjCtx: AxisProjectionContext = projCtx ?? {
            xAxis,
            yAxis,
            rootBounds: tileStore.rootBounds,
            effectiveFlipY: false,
            colOrigins: tileStore.colOrigins,
        };

        const { useTwoPassDim } = this.filterResources.updateFrameUniforms(
            camera,
            tileStore,
            resolvedProjCtx,
            [
                colorSrc === 0 ? colorField : '',
                sizeSrc === 0 ? sizeField : '',
                xOrAux1FieldGlobal,
                yOrAux2FieldGlobal,
            ]
        );
        const globalFilterBuf = this.filterResources.globalFilterBuffer!;
        const activeMaskBuf = this.filterResources.getActiveMaskBuffer()!;
        const chunkOversizeLeaves = this.filterResources.hasSlotBitmask;

        if (chunkOversizeLeaves) {
            let extraChunks = 0;
            for (let s = 0; s < limitSlots; s++) {
                const slotObj = tileStore.slots[s];
                if (slotObj && slotObj.visible && slotObj.activeRows > SLOT_MASK_ROWS_PER_SLICE) {
                    extraChunks += Math.ceil(slotObj.activeRows / SLOT_MASK_ROWS_PER_SLICE) - 1;
                }
            }
            if (extraChunks > 0) {
                this.ensureUniformSlotCapacity(this.maxSlots + extraChunks);
            }
        }

        writeCommonUniforms(0);

        const activeSlots: {
            uniformSlot: number;
            maskByteOffset: number;
            bindGroup: GPUBindGroup;
            workgroups: number;
        }[] = [];
        let maxSlotWritten = 0;
        let nextExtraUniformSlot = this.maxSlots;
        const dummy = this.dummyStorageBuffer;

        for (let slotIdx = 0; slotIdx < limitSlots; slotIdx++) {
            const slotObj = tileStore.slots[slotIdx];
            if (!slotObj || !slotObj.visible || slotObj.activeRows <= 0) continue;

            const xyBuf = slotObj.buffers.get('xy_u16');
            if (!xyBuf) continue;

            const colorBuf = (colorSrc === 0 ? slotObj.buffers.get(colorField) : undefined) || dummy;
            const sizeBuf = (sizeSrc === 0 ? slotObj.buffers.get(sizeField) : undefined) || dummy;
            const xBuf =
                (xOrAux1FieldGlobal ? slotObj.buffers.get(xOrAux1FieldGlobal) : undefined) || dummy;
            const yBuf =
                (yOrAux2FieldGlobal ? slotObj.buffers.get(yOrAux2FieldGlobal) : undefined) || dummy;

            const auxOrPrevXField = isTransitioning
                ? prevIsXSpatial
                    ? ''
                    : prevCfg.x.field
                : tileStore.auxField3;
            const prevYField = isTransitioning && !prevIsYSpatial ? prevCfg.y.field : '';
            const auxOrPrevXBuf =
                (auxOrPrevXField ? slotObj.buffers.get(auxOrPrevXField) : undefined) || dummy;
            const prevYBuf = (prevYField ? slotObj.buffers.get(prevYField) : undefined) || dummy;

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
            const prevRtcX = isTransitioning
                ? this.computeRtcAxis(
                      prevCfg.x,
                      slotObj.tileOffsetX,
                      slotObj.tileOffsetY,
                      slotObj.tileScale,
                      false,
                      camLeft
                  )
                : null;
            const prevRtcY = isTransitioning
                ? this.computeRtcAxis(
                      prevCfg.y,
                      slotObj.tileOffsetX,
                      slotObj.tileOffsetY,
                      slotObj.tileScale,
                      true,
                      camTop
                  )
                : null;

            const slotBufFlags =
                (colorBuf !== dummy ? 1 << 25 : 0) |
                (sizeBuf !== dummy ? 1 << 26 : 0) |
                (xBuf !== dummy ? 1 << 27 : 0) |
                (yBuf !== dummy ? 1 << 28 : 0) |
                (auxOrPrevXBuf !== dummy ? 1 << 29 : 0) |
                (prevYBuf !== dummy ? 1 << 30 : 0);

            const bindGroup = this.getOrCreateSlotBindGroup(
                slotIdx,
                xyBuf,
                colorBuf,
                sizeBuf,
                xBuf,
                yBuf,
                auxOrPrevXBuf,
                prevYBuf,
                globalFilterBuf,
                activeMaskBuf
            );

            const numChunks =
                chunkOversizeLeaves && slotObj.activeRows > SLOT_MASK_ROWS_PER_SLICE
                    ? Math.ceil(slotObj.activeRows / SLOT_MASK_ROWS_PER_SLICE)
                    : 1;

            for (let chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
                const uniformSlot = chunkIdx === 0 ? slotIdx : nextExtraUniformSlot++;
                const rowBaseOffset = chunkIdx * SLOT_MASK_ROWS_PER_SLICE;
                const chunkCount =
                    numChunks === 1
                        ? slotObj.activeRows
                        : Math.min(SLOT_MASK_ROWS_PER_SLICE, slotObj.activeRows - rowBaseOffset);
                if (chunkCount <= 0) continue;

                const base = uniformSlot * (SLOT_UNIFORM_STRIDE / 4);
                writeCommonUniforms(base);
                this.uniformU32[base + 35] = (extraFlags | slotBufFlags) >>> 0;
                this.uniformU32[base + 6] = chunkCount;
                this.uniformU32[base + 63] = rowBaseOffset;

                this.uniformF32[base + 8] = slotObj.tileOffsetX;
                this.uniformF32[base + 9] = slotObj.tileOffsetY;
                this.uniformF32[base + 10] = slotObj.tileScale;
                this.uniformF32[base + 11] = slotObj.spawnTime;

                this.uniformF32[base + 36] = rtcX.base;
                this.uniformF32[base + 37] = rtcX.step;
                this.uniformF32[base + 38] = rtcY.base;
                this.uniformF32[base + 39] = rtcY.step;

                if (prevRtcX && prevRtcY) {
                    this.uniformF32[base + 52] = prevRtcX.base;
                    this.uniformF32[base + 53] = prevRtcX.step;
                    this.uniformF32[base + 54] = prevRtcY.base;
                    this.uniformF32[base + 55] = prevRtcY.step;
                }

                activeSlots.push({
                    uniformSlot,
                    maskByteOffset: this.filterResources.getMaskDynamicOffset(slotIdx, chunkIdx),
                    bindGroup,
                    workgroups: Math.ceil(chunkCount / 256),
                });
                if (uniformSlot > maxSlotWritten) {
                    maxSlotWritten = uniformSlot;
                }
            }
        }

        const uploadBytes = (maxSlotWritten + 1) * SLOT_UNIFORM_STRIDE;
        this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformArrayBuffer, 0, uploadBytes);

        const encoder = this.device.createCommandEncoder({
            label: 'ComputeRasterizer_CommandEncoder',
        });

        const currentTextureView = this.context.getCurrentTexture().createView();
        const clearVal = tileStore.lightBackground
            ? { r: 0.975, g: 0.978, b: 0.985, a: 1.0 }
            : { r: 17 / 255, g: 17 / 255, b: 17 / 255, a: 1.0 };

        if (useTwoPassDim) {
            // Subpass 1: Splat !selected -> Resolve desaturated context cloud at `unselectedOpacity`
            encoder.clearBuffer(this.accumBuffer);
            if (activeSlots.length > 0) {
                const bgCompute = encoder.beginComputePass({
                    label: 'ComputeRasterizer_UnselectedPass',
                });
                bgCompute.setPipeline(this.computeUnselectedPipeline);
                for (let i = 0; i < activeSlots.length; i++) {
                    const item = activeSlots[i];
                    bgCompute.setBindGroup(0, item.bindGroup, [
                        item.uniformSlot * SLOT_UNIFORM_STRIDE,
                        item.maskByteOffset,
                    ]);
                    bgCompute.dispatchWorkgroups(item.workgroups);
                }
                bgCompute.end();
            }

            const bgResolve = encoder.beginRenderPass({
                label: 'ComputeRasterizer_ResolveDimBgPass',
                colorAttachments: [
                    {
                        view: currentTextureView,
                        clearValue: clearVal,
                        loadOp: 'clear',
                        storeOp: 'store',
                    },
                ],
            });
            bgResolve.setPipeline(this.resolveDimBgPipeline);
            bgResolve.setBindGroup(0, this.resolveBindGroup);
            bgResolve.draw(3, 1, 0, 0);
            bgResolve.end();

            // Subpass 2: Clear accumBuffer -> Splat selected (full ticket budget!) -> Premultiplied overlay resolve
            encoder.clearBuffer(this.accumBuffer);
            if (activeSlots.length > 0) {
                const fgCompute = encoder.beginComputePass({
                    label: 'ComputeRasterizer_SelectedPass',
                });
                fgCompute.setPipeline(this.computeSelectedPipeline);
                for (let i = 0; i < activeSlots.length; i++) {
                    const item = activeSlots[i];
                    fgCompute.setBindGroup(0, item.bindGroup, [
                        item.uniformSlot * SLOT_UNIFORM_STRIDE,
                        item.maskByteOffset,
                    ]);
                    fgCompute.dispatchWorkgroups(item.workgroups);
                }
                fgCompute.end();
            }

            const fgResolve = encoder.beginRenderPass({
                label: 'ComputeRasterizer_ResolveFgOverlayPass',
                colorAttachments: [
                    {
                        view: currentTextureView,
                        loadOp: 'load',
                        storeOp: 'store',
                    },
                ],
            });
            fgResolve.setPipeline(this.resolveFgOverlayPipeline);
            fgResolve.setBindGroup(0, this.resolveBindGroup);
            fgResolve.draw(3, 1, 0, 0);
            fgResolve.end();
        } else {
            encoder.clearBuffer(this.accumBuffer);

            if (activeSlots.length > 0) {
                const computePass = encoder.beginComputePass({
                    label: 'ComputeRasterizer_SplatPass',
                });
                computePass.setPipeline(this.computePipeline);
                for (let i = 0; i < activeSlots.length; i++) {
                    const item = activeSlots[i];
                    computePass.setBindGroup(0, item.bindGroup, [
                        item.uniformSlot * SLOT_UNIFORM_STRIDE,
                        item.maskByteOffset,
                    ]);
                    computePass.dispatchWorkgroups(item.workgroups);
                }
                computePass.end();
            }

            const renderPass = encoder.beginRenderPass({
                label: 'ComputeRasterizer_ResolvePass',
                colorAttachments: [
                    {
                        view: currentTextureView,
                        clearValue: clearVal,
                        loadOp: 'clear',
                        storeOp: 'store',
                    },
                ],
            });
            renderPass.setPipeline(this.resolvePipeline);
            renderPass.setBindGroup(0, this.resolveBindGroup);
            renderPass.draw(3, 1, 0, 0);
            renderPass.end();
        }

        if (afterResolvePass) {
            afterResolvePass(encoder, currentTextureView);
        }

        this.device.queue.submit([encoder.finish()]);
        return true;
    }

    public getLiveGpuBytes(): number {
        return (
            (this.accumBuffer?.size ?? 0) +
            (this.uniformBuffer?.size ?? 0) +
            (this.paletteBuffer?.size ?? 0) +
            (this.dummyStorageBuffer?.size ?? 0) +
            (this.ownsFilterResources ? (this.filterResources?.getLiveGpuBytes() ?? 0) : 0)
        );
    }

    public dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        if (this.accumBuffer) {
            this.accumBuffer.destroy();
            this.accumBuffer = null;
        }
        if (this.uniformBuffer) {
            this.uniformBuffer.destroy();
            this.uniformBuffer = null;
        }
        if (this.dummyStorageBuffer) {
            this.dummyStorageBuffer.destroy();
            this.dummyStorageBuffer = null;
        }
        if (this.paletteBuffer) {
            this.paletteBuffer.destroy();
            this.paletteBuffer = null;
        }
        if (this.ownsFilterResources && this.filterResources) {
            this.filterResources.dispose();
        }
        this.filterResources = null;
        this.resolveBindGroup = null;
        for (let i = 0; i < this.slotBindGroups.length; i++) {
            this.slotBindGroups[i] = null;
        }
        this.initialized = false;
    }
}
