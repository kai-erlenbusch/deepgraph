import type { EdgeSpec, EdgeStyleOptions } from '../lib/types.ts';
import { projectDataToWorld64, type AxisProjectionContext } from './FilterGpuResources.ts';
import type { Camera2D, Renderer } from './Renderer.ts';
import type { TileGpuStore } from './TileGpuStore.ts';

export const MAX_EDGES_PER_LAYER = 100_000;
const BEZIER_SEGMENTS = 16;
const VERTS_PER_INSTANCE = BEZIER_SEGMENTS * 6; // 96 vertices per curved ribbon
// 12 x 4B = 48 B/instance:
// [0..3]: srcRelX, srcRelY, dstRelX, dstRelY (target axis world - anchor64)
// [4..7]: prevSrcRelX, prevSrcRelY, prevDstRelX, prevDstRelY (prev axis world - prevAnchor64)
// [8..11]: normWeight_f32, curvature_f32, rgbaPacked_u32, edgeFlags_u32
const BYTES_PER_EDGE_INSTANCE = 48;
const EDGE_UNIFORM_BINDING_SIZE = 128;
const EDGE_UNIFORM_SLOT_STRIDE = 256; // WebGPU minUniformBufferOffsetAlignment (slot 0 = static, slot 1 = ego)

interface PackedEdgeLayer {
    specs: EdgeSpec[];
    gpuBuffer: GPUBuffer | null;
    instanceCount: number;
    anchorWorldX: number;
    anchorWorldY: number;
    prevAnchorWorldX: number;
    prevAnchorWorldY: number;
    packedAxisKey: string;
}

function packColorRgbaU32(hex: number, alpha01: number): number {
    const r = (hex >>> 16) & 0xff;
    const g = (hex >>> 8) & 0xff;
    const b = hex & 0xff;
    const a = Math.max(0, Math.min(255, Math.round(alpha01 * 255)));
    return ((r | (g << 8) | (b << 16) | (a << 24)) >>> 0);
}

/**
 * Instanced Plain-WebGPU Quadratic Bézier Graph & Ego-Network Arc Pass (`Step 3A`).
 *
 * - Renders up to `100,000` curved arcs per layer (`staticLayer` + `egoLayer`) using
 *   `GPUBufferUsage.VERTEX` instance attributes (`0` storage buffers, `1` `128 B` uniform buffer).
 * - Projects endpoints into `Float64` dataset-anchor-relative world coordinates (`1D RTO`) once
 *   when edges or axes change, then uploads **only `128 B` of `Float64` camera-relative `RTC`
 *   uniforms per frame** (`O(1)` CPU per pan/zoom/transition frame at `180 FPS`).
 * - Right-handed perpendicular control-point offset automatically separates `A -> B` and `B -> A`
 *   bidirectional edges; `animatePulse` defaults to `false` (`0.0000%` stationary shimmer).
 */
export class EdgeOverlay {
    private rendererWrapper: Renderer;
    private device!: GPUDevice;
    private pipeline!: GPURenderPipeline;
    private bindGroupLayout!: GPUBindGroupLayout;
    private uniformBuffer: GPUBuffer | null = null;
    private bindGroup: GPUBindGroup | null = null;
    private uniformStaging = new ArrayBuffer(EDGE_UNIFORM_SLOT_STRIDE * 2);
    private uniformF32 = new Float32Array(this.uniformStaging);

    private staticLayer: PackedEdgeLayer = {
        specs: [],
        gpuBuffer: null,
        instanceCount: 0,
        anchorWorldX: 0,
        anchorWorldY: 0,
        prevAnchorWorldX: 0,
        prevAnchorWorldY: 0,
        packedAxisKey: '',
    };

    private egoLayer: PackedEdgeLayer = {
        specs: [],
        gpuBuffer: null,
        instanceCount: 0,
        anchorWorldX: 0,
        anchorWorldY: 0,
        prevAnchorWorldX: 0,
        prevAnchorWorldY: 0,
        packedAxisKey: '',
    };

    private style: Required<EdgeStyleOptions> = {
        width: 1.35,
        weightDomain: [0.0, 1.0],
        curvature: 0.18,
        opacity: 0.68,
        colorHex: 0x7dd3fc,
        directed: true,
        animatePulse: false,
    };

    private initialized = false;
    private disposed = false;

    constructor(rendererWrapper: Renderer, initialStyle?: EdgeStyleOptions) {
        this.rendererWrapper = rendererWrapper;
        if (initialStyle) {
            this.setStyle(initialStyle);
        }
    }

    public get isPulseAnimated(): boolean {
        return (
            this.style.animatePulse &&
            (this.staticLayer.specs.length > 0 || this.egoLayer.specs.length > 0)
        );
    }

    public setStyle(patch: EdgeStyleOptions): void {
        if (patch.width !== undefined) this.style.width = patch.width;
        if (patch.weightDomain !== undefined) {
            this.style.weightDomain = [patch.weightDomain[0], patch.weightDomain[1]];
        }
        if (patch.curvature !== undefined && Number.isFinite(patch.curvature)) {
            this.style.curvature = Math.max(-1.5, Math.min(1.5, patch.curvature));
        }
        if (patch.opacity !== undefined && Number.isFinite(patch.opacity)) {
            this.style.opacity = Math.max(0.0, Math.min(1.0, patch.opacity));
        }
        if (patch.colorHex !== undefined && Number.isFinite(patch.colorHex)) {
            this.style.colorHex = patch.colorHex >>> 0;
        }
        if (patch.directed !== undefined) this.style.directed = Boolean(patch.directed);
        if (patch.animatePulse !== undefined) {
            this.style.animatePulse = Boolean(patch.animatePulse);
        }
        // Force instance repack on next frame if per-edge defaults (color/curvature/weightDomain) changed
        this.staticLayer.packedAxisKey = '';
        this.egoLayer.packedAxisKey = '';
    }

    public setStaticEdges(edges: ReadonlyArray<EdgeSpec>, style?: EdgeStyleOptions): void {
        if (this.disposed) return;
        if (style) this.setStyle(style);
        this.staticLayer.specs = edges.slice(0, MAX_EDGES_PER_LAYER);
        this.staticLayer.packedAxisKey = '';
        if (this.staticLayer.specs.length === 0 && this.staticLayer.gpuBuffer) {
            this.staticLayer.gpuBuffer.destroy();
            this.staticLayer.gpuBuffer = null;
            this.staticLayer.instanceCount = 0;
        }
    }

    public setEgoEdges(edges: ReadonlyArray<EdgeSpec>): void {
        if (this.disposed) return;
        this.egoLayer.specs = edges.slice(0, MAX_EDGES_PER_LAYER);
        this.egoLayer.packedAxisKey = '';
        if (this.egoLayer.specs.length === 0 && this.egoLayer.gpuBuffer) {
            this.egoLayer.gpuBuffer.destroy();
            this.egoLayer.gpuBuffer = null;
            this.egoLayer.instanceCount = 0;
        }
    }

    public clearEgoEdges(): void {
        this.setEgoEdges([]);
    }

    public clearAll(): void {
        this.setStaticEdges([]);
        this.setEgoEdges([]);
    }

    public init(): boolean {
        if (this.disposed) return false;
        if (this.initialized) return true;

        const device = this.rendererWrapper.device;
        if (!device) return false;
        this.device = device;

        this.uniformBuffer = this.device.createBuffer({
            label: 'EdgeOverlay_GlobalUniforms_512B',
            size: EDGE_UNIFORM_SLOT_STRIDE * 2,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const shaderCode = /* wgsl */ `
struct EdgeUniforms {
    // x = rtcOffsetX (anchorWorldX - camLeft), y = rtcOffsetY (camTop - anchorWorldY),
    // z = prevRtcOffsetX, w = prevRtcOffsetY
    rtcOffsets: vec4<f32>,
    // x = invCamSpanX, y = invCamSpanY, z = screenW, w = screenH
    projAndScreen: vec4<f32>,
    // x = dpr, y = minWidthCssPx, z = maxWidthCssPx, w = transitionT
    widthAndTrans: vec4<f32>,
    // x = baseOpacity, y = globalTime, z = flags (bit0=isTransitioning, bit1=animatePulse, bit2=isLightBg), w = pad
    styleAndFlags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: EdgeUniforms;

struct VSIn {
    @builtin(vertex_index) vid: u32,
    @location(0) curEndpoints: vec4<f32>,  // srcRelX, srcRelY, dstRelX, dstRelY
    @location(1) prevEndpoints: vec4<f32>, // prevSrcRelX, prevSrcRelY, prevDstRelX, prevDstRelY
    @location(2) weightCurv: vec2<f32>,    // normWeight, curvature
    @location(3) colorAndFlags: vec2<u32>, // rgbaPacked, bit0=directed
};

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) vCrossPx: f32,
    @location(1) vHalfStrokePx: f32,
    @location(2) vT: f32,
    @location(3) vColor: vec4<f32>,
    @location(4) @interpolate(flat) vDirected: u32,
};

fn eval_qbezier(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, t: f32) -> vec2<f32> {
    let omt = 1.0 - t;
    return (omt * omt) * p0 + (2.0 * omt * t) * p1 + (t * t) * p2;
}

fn eval_qbezier_tan(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, t: f32) -> vec2<f32> {
    let v = 2.0 * (1.0 - t) * (p1 - p0) + 2.0 * t * (p2 - p1);
    let len = length(v);
    if (len < 1e-4) {
        let chord = p2 - p0;
        let cLen = max(1e-4, length(chord));
        return chord / cLen;
    }
    return v / len;
}

@vertex
fn vs_main(in: VSIn) -> VSOut {
    var out: VSOut;
    let flags = u32(u.styleAndFlags.z);
    let isTrans = (flags & 1u) != 0u;

    var srcRelCam = vec2<f32>(
        u.rtcOffsets.x + in.curEndpoints.x,
        u.rtcOffsets.y - in.curEndpoints.y
    );
    var dstRelCam = vec2<f32>(
        u.rtcOffsets.x + in.curEndpoints.z,
        u.rtcOffsets.y - in.curEndpoints.w
    );

    if (isTrans) {
        let prevSrcCam = vec2<f32>(
            u.rtcOffsets.z + in.prevEndpoints.x,
            u.rtcOffsets.w - in.prevEndpoints.y
        );
        let prevDstCam = vec2<f32>(
            u.rtcOffsets.z + in.prevEndpoints.z,
            u.rtcOffsets.w - in.prevEndpoints.w
        );
        srcRelCam = mix(prevSrcCam, srcRelCam, u.widthAndTrans.w);
        dstRelCam = mix(prevDstCam, dstRelCam, u.widthAndTrans.w);
    }

    let screenW = u.projAndScreen.z;
    let screenH = u.projAndScreen.w;
    let p0 = vec2<f32>(
        srcRelCam.x * u.projAndScreen.x * screenW,
        srcRelCam.y * u.projAndScreen.y * screenH
    );
    let p2 = vec2<f32>(
        dstRelCam.x * u.projAndScreen.x * screenW,
        dstRelCam.y * u.projAndScreen.y * screenH
    );

    let chord = p2 - p0;
    let chordLen = length(chord);
    if (chordLen < 0.35) {
        out.pos = vec4<f32>(2.0, 2.0, 0.0, 1.0);
        out.vCrossPx = 0.0;
        out.vHalfStrokePx = 0.0;
        out.vT = 0.0;
        out.vColor = vec4<f32>(0.0);
        out.vDirected = 0u;
        return out;
    }

    let dir = chord / chordLen;
    let perp = vec2<f32>(-dir.y, dir.x);
    let p1 = 0.5 * (p0 + p2) + perp * (in.weightCurv.y * chordLen);

    let dpr = max(1.0, u.widthAndTrans.x);
    let widthCss = mix(u.widthAndTrans.y, u.widthAndTrans.z, clamp(in.weightCurv.x, 0.0, 1.0));
    let halfStrokePx = max(0.45 * dpr, 0.5 * widthCss * dpr);
    let halfQuadPx = halfStrokePx + 1.25 * dpr;

    // Screen-space AABB cull across (p0, p1, p2)
    let minBox = min(min(p0, p1), p2) - vec2<f32>(halfQuadPx + 2.0);
    let maxBox = max(max(p0, p1), p2) + vec2<f32>(halfQuadPx + 2.0);
    if (maxBox.x < 0.0 || minBox.x > screenW || maxBox.y < 0.0 || minBox.y > screenH) {
        out.pos = vec4<f32>(2.0, 2.0, 0.0, 1.0);
        out.vCrossPx = 0.0;
        out.vHalfStrokePx = 0.0;
        out.vT = 0.0;
        out.vColor = vec4<f32>(0.0);
        out.vDirected = 0u;
        return out;
    }

    let segIdx = in.vid / 6u;
    let corner = in.vid % 6u;
    // 2 triangles per segment: (0,-1), (1,-1), (0,+1), (0,+1), (1,-1), (1,+1)
    let endSel = array<f32, 6>(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
    let sideSel = array<f32, 6>(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0);

    let t = (f32(segIdx) + endSel[corner]) / ${BEZIER_SEGMENTS}.0;
    let side = sideSel[corner];

    let centerPx = eval_qbezier(p0, p1, p2, t);
    let tanDir = eval_qbezier_tan(p0, p1, p2, t);
    let normDir = vec2<f32>(-tanDir.y, tanDir.x);
    let vertPx = centerPx + normDir * (side * halfQuadPx);

    let ndcX = (vertPx.x / screenW) * 2.0 - 1.0;
    let ndcY = 1.0 - (vertPx.y / screenH) * 2.0;

    let packedC = in.colorAndFlags.x;
    let r = f32(packedC & 0xFFu) / 255.0;
    let g = f32((packedC >> 8u) & 0xFFu) / 255.0;
    let b = f32((packedC >> 16u) & 0xFFu) / 255.0;
    let a = f32((packedC >> 24u) & 0xFFu) / 255.0;

    out.pos = vec4<f32>(ndcX, ndcY, 0.0, 1.0);
    out.vCrossPx = side * halfQuadPx;
    out.vHalfStrokePx = halfStrokePx;
    out.vT = t;
    out.vColor = vec4<f32>(r, g, b, a);
    out.vDirected = in.colorAndFlags.y & 1u;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let dpr = max(1.0, u.widthAndTrans.x);
    let flags = u32(u.styleAndFlags.z);
    let animatePulse = (flags & 2u) != 0u;
    let isLightBg = (flags & 4u) != 0u;

    let aa = 1.0 - smoothstep(
        max(0.20, in.vHalfStrokePx - 0.65 * dpr),
        in.vHalfStrokePx + 0.65 * dpr,
        abs(in.vCrossPx)
    );
    if (aa <= 0.005) {
        discard;
    }

    var dirAlpha = 1.0;
    var rgb = in.vColor.rgb;
    if (isLightBg) {
        rgb = rgb * 0.72;
    }

    if (in.vDirected != 0u) {
        dirAlpha = mix(0.36, 1.0, in.vT);
        rgb = mix(rgb * 0.78, min(vec3<f32>(1.0), rgb * 1.14), in.vT);
        if (animatePulse) {
            let wave = fract(in.vT - u.styleAndFlags.y * 0.55);
            let pulse = exp(-((wave - 0.5) * (wave - 0.5)) * 36.0);
            rgb = min(vec3<f32>(1.0), rgb + vec3<f32>(0.22) * pulse);
            dirAlpha = min(1.0, dirAlpha + 0.25 * pulse);
        }
    }

    let finalAlpha = clamp(aa * in.vColor.a * u.styleAndFlags.x * dirAlpha, 0.0, 1.0);
    if (finalAlpha <= 0.004) {
        discard;
    }
    return vec4<f32>(rgb * finalAlpha, finalAlpha);
}
`;

        this.bindGroupLayout = this.device.createBindGroupLayout({
            label: 'EdgeOverlay_BGL',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: 'uniform',
                        hasDynamicOffset: true,
                        minBindingSize: EDGE_UNIFORM_BINDING_SIZE,
                    },
                },
            ],
        });

        const module = this.device.createShaderModule({
            label: 'EdgeOverlay_Shader',
            code: shaderCode,
        });

        this.pipeline = this.device.createRenderPipeline({
            label: 'EdgeOverlay_Pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.bindGroupLayout],
            }),
            vertex: {
                module,
                entryPoint: 'vs_main',
                buffers: [
                    {
                        arrayStride: BYTES_PER_EDGE_INSTANCE,
                        stepMode: 'instance',
                        attributes: [
                            { shaderLocation: 0, offset: 0, format: 'float32x4' },
                            { shaderLocation: 1, offset: 16, format: 'float32x4' },
                            { shaderLocation: 2, offset: 32, format: 'float32x2' },
                            { shaderLocation: 3, offset: 40, format: 'uint32x2' },
                        ],
                    },
                ],
            },
            fragment: {
                module,
                entryPoint: 'fs_main',
                targets: [
                    {
                        format: this.rendererWrapper.canvasFormat,
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

        this.bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.uniformBuffer,
                        offset: 0,
                        size: EDGE_UNIFORM_BINDING_SIZE,
                    },
                },
            ],
        });

        this.initialized = true;
        return true;
    }

    private resolveEndpointCoord(
        axisField: string,
        pt: { x: number; y: number } | undefined,
        attrs: Record<string, unknown> | undefined,
        isY: boolean,
        categories: Record<string, string[]>
    ): number | null {
        if (axisField !== 'x_u16' && axisField !== 'y_u16') {
            if (attrs !== undefined) {
                const raw = attrs[axisField];
                if (raw === null || raw === undefined) return null;
                if (typeof raw === 'number') {
                    return Number.isFinite(raw) ? raw : null;
                }
                if (typeof raw === 'string') {
                    const dict = categories[axisField];
                    const idx = dict ? dict.indexOf(raw) : -1;
                    if (idx >= 0) return idx;
                    const trimmed = raw.trim();
                    if (!trimmed) return null;
                    const parsed = Number(trimmed);
                    return Number.isFinite(parsed) ? parsed : null;
                }
                return null;
            }
            if (pt) {
                const v = isY ? pt.y : pt.x;
                return Number.isFinite(v) ? v : null;
            }
            return null;
        }
        if (pt) {
            const v = isY ? pt.y : pt.x;
            if (Number.isFinite(v)) return v;
        }
        return null;
    }

    private ensureLayerPacked(
        layer: PackedEdgeLayer,
        projCtx: AxisProjectionContext,
        prevProjCtx: AxisProjectionContext | null,
        tileStore: TileGpuStore,
        axisKey: string
    ): void {
        if (layer.specs.length === 0) {
            layer.instanceCount = 0;
            return;
        }
        if (layer.packedAxisKey === axisKey && layer.gpuBuffer) {
            return;
        }

        const nMax = layer.specs.length;
        const cpuBuf = new ArrayBuffer(nMax * BYTES_PER_EDGE_INSTANCE);
        const f32 = new Float32Array(cpuBuf);
        const u32 = new Uint32Array(cpuBuf);

        let anchorX = 0;
        let anchorY = 0;
        let prevAnchorX = 0;
        let prevAnchorY = 0;
        let anchorSet = false;

        const wDom0 = this.style.weightDomain[0];
        const wSpan = Math.max(1e-9, this.style.weightDomain[1] - wDom0);
        const cats = tileStore.categoricalDictionaries;

        let validCount = 0;
        for (let i = 0; i < nMax; i++) {
            const e = layer.specs[i];
            const sx = this.resolveEndpointCoord(
                projCtx.xAxis.field,
                e.source,
                e.sourceAttributes,
                false,
                cats
            );
            const sy = this.resolveEndpointCoord(
                projCtx.yAxis.field,
                e.source,
                e.sourceAttributes,
                true,
                cats
            );
            const tx = this.resolveEndpointCoord(
                projCtx.xAxis.field,
                e.target,
                e.targetAttributes,
                false,
                cats
            );
            const ty = this.resolveEndpointCoord(
                projCtx.yAxis.field,
                e.target,
                e.targetAttributes,
                true,
                cats
            );
            if (sx === null || sy === null || tx === null || ty === null) continue;

            const curS = projectDataToWorld64(sx, sy, projCtx);
            const curT = projectDataToWorld64(tx, ty, projCtx);

            let prevS = curS;
            let prevT = curT;
            if (prevProjCtx) {
                const psx = this.resolveEndpointCoord(
                    prevProjCtx.xAxis.field,
                    e.source,
                    e.sourceAttributes,
                    false,
                    cats
                );
                const psy = this.resolveEndpointCoord(
                    prevProjCtx.yAxis.field,
                    e.source,
                    e.sourceAttributes,
                    true,
                    cats
                );
                const ptx = this.resolveEndpointCoord(
                    prevProjCtx.xAxis.field,
                    e.target,
                    e.targetAttributes,
                    false,
                    cats
                );
                const pty = this.resolveEndpointCoord(
                    prevProjCtx.yAxis.field,
                    e.target,
                    e.targetAttributes,
                    true,
                    cats
                );
                if (psx === null || psy === null || ptx === null || pty === null) {
                    continue;
                }
                prevS = projectDataToWorld64(psx, psy, prevProjCtx);
                prevT = projectDataToWorld64(ptx, pty, prevProjCtx);
            }

            if (!anchorSet) {
                anchorX = curS.worldX;
                anchorY = curS.worldY;
                prevAnchorX = prevS.worldX;
                prevAnchorY = prevS.worldY;
                anchorSet = true;
            }

            const base = validCount * (BYTES_PER_EDGE_INSTANCE / 4);
            f32[base + 0] = curS.worldX - anchorX;
            f32[base + 1] = curS.worldY - anchorY;
            f32[base + 2] = curT.worldX - anchorX;
            f32[base + 3] = curT.worldY - anchorY;

            f32[base + 4] = prevS.worldX - prevAnchorX;
            f32[base + 5] = prevS.worldY - prevAnchorY;
            f32[base + 6] = prevT.worldX - prevAnchorX;
            f32[base + 7] = prevT.worldY - prevAnchorY;

            const normW =
                e.weight !== undefined && Number.isFinite(e.weight)
                    ? Math.max(0.0, Math.min(1.0, (e.weight - wDom0) / wSpan))
                    : 0.5;
            const weightAlpha =
                e.weight !== undefined && Number.isFinite(e.weight)
                    ? 0.45 + 0.55 * normW
                    : 1.0;

            f32[base + 8] = normW;
            f32[base + 9] =
                e.curvature !== undefined && Number.isFinite(e.curvature)
                    ? Math.max(-1.5, Math.min(1.5, e.curvature))
                    : this.style.curvature;
            u32[base + 10] = packColorRgbaU32(
                e.colorHex !== undefined ? e.colorHex : this.style.colorHex,
                weightAlpha
            );
            const isDir = e.directed !== undefined ? e.directed : this.style.directed;
            u32[base + 11] = isDir ? 1 : 0;
            validCount++;
        }

        layer.anchorWorldX = anchorX;
        layer.anchorWorldY = anchorY;
        layer.prevAnchorWorldX = prevAnchorX;
        layer.prevAnchorWorldY = prevAnchorY;
        layer.instanceCount = validCount;
        layer.packedAxisKey = axisKey;

        if (validCount === 0) {
            return;
        }

        const requiredBytes = validCount * BYTES_PER_EDGE_INSTANCE;
        if (layer.gpuBuffer && layer.gpuBuffer.size < requiredBytes) {
            layer.gpuBuffer.destroy();
            layer.gpuBuffer = null;
        }
        if (!layer.gpuBuffer) {
            layer.gpuBuffer = this.device.createBuffer({
                label: 'EdgeOverlay_InstanceVertices',
                size: requiredBytes,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            });
        }
        this.device.queue.writeBuffer(layer.gpuBuffer, 0, cpuBuf, 0, requiredBytes);
    }

    private drawLayer(
        encoder: GPUCommandEncoder,
        targetView: GPUTextureView,
        layer: PackedEdgeLayer,
        slotIndex: 0 | 1,
        camLeft: number,
        camTop: number,
        invSpanX: number,
        invSpanY: number,
        screenW: number,
        screenH: number,
        dpr: number,
        isTransitioning: boolean,
        transitionT: number,
        lightBackground: boolean,
        globalTime: number
    ): void {
        if (!layer.gpuBuffer || layer.instanceCount <= 0 || !this.uniformBuffer || !this.bindGroup) {
            return;
        }

        const minW = Array.isArray(this.style.width) ? this.style.width[0] : this.style.width;
        const maxW = Array.isArray(this.style.width) ? this.style.width[1] : this.style.width;
        const flags =
            (isTransitioning ? 1 : 0) |
            (this.style.animatePulse ? 2 : 0) |
            (lightBackground ? 4 : 0);

        const base = slotIndex * (EDGE_UNIFORM_SLOT_STRIDE / 4);
        this.uniformF32[base + 0] = layer.anchorWorldX - camLeft;
        this.uniformF32[base + 1] = camTop - layer.anchorWorldY;
        this.uniformF32[base + 2] = layer.prevAnchorWorldX - camLeft;
        this.uniformF32[base + 3] = camTop - layer.prevAnchorWorldY;

        this.uniformF32[base + 4] = invSpanX;
        this.uniformF32[base + 5] = invSpanY;
        this.uniformF32[base + 6] = screenW;
        this.uniformF32[base + 7] = screenH;

        this.uniformF32[base + 8] = dpr;
        this.uniformF32[base + 9] = minW;
        this.uniformF32[base + 10] = maxW;
        this.uniformF32[base + 11] = transitionT;

        this.uniformF32[base + 12] = this.style.opacity;
        this.uniformF32[base + 13] = globalTime;
        this.uniformF32[base + 14] = flags;
        this.uniformF32[base + 15] = 0;

        const byteOff = slotIndex * EDGE_UNIFORM_SLOT_STRIDE;
        this.device.queue.writeBuffer(
            this.uniformBuffer,
            byteOff,
            this.uniformStaging,
            byteOff,
            EDGE_UNIFORM_BINDING_SIZE
        );

        const pass = encoder.beginRenderPass({
            label: slotIndex === 0 ? 'EdgeOverlay_StaticPass' : 'EdgeOverlay_EgoPass',
            colorAttachments: [
                {
                    view: targetView,
                    loadOp: 'load',
                    storeOp: 'store',
                },
            ],
        });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup, [byteOff]);
        pass.setVertexBuffer(0, layer.gpuBuffer);
        pass.draw(VERTS_PER_INSTANCE, layer.instanceCount, 0, 0);
        pass.end();
    }

    /**
     * Note: When BOTH `staticLayer` and `egoLayer` are non-empty in the same frame,
     * they may have distinct `anchorWorldX/Y`, so we give `egoLayer` its own `128 B`
     * dynamic offset or second uniform buffer! Let's check: `queue.writeBuffer` runs
     * before `encoder.finish()`, so giving `uniformBuffer` `2 * 256 = 512 B` with
     * `hasDynamicOffset: true` lets static (`offset 0`) and ego (`offset 256`) draw
     * in the exact same `GPUCommandEncoder` with zero uniform overwrite!
     */
    public encodePass(
        encoder: GPUCommandEncoder,
        targetView: GPUTextureView,
        camera: Camera2D,
        tileStore: TileGpuStore,
        projCtx: AxisProjectionContext,
        prevProjCtx: AxisProjectionContext | null
    ): void {
        if (this.disposed) return;
        const hasStatic = this.staticLayer.specs.length > 0;
        const hasEgo = this.egoLayer.specs.length > 0;
        if (!hasStatic && !hasEgo) return;
        if (!this.init() || !this.uniformBuffer || !this.bindGroup) return;

        const computeMode = tileStore.computeMode;
        const isOptical = computeMode === 0 || computeMode === 3;
        const isTransitioning =
            Boolean(prevProjCtx) &&
            tileStore.isTransitioning &&
            !isOptical &&
            computeMode !== 3 &&
            computeMode !== 4;
        const effPrevCtx = isTransitioning ? prevProjCtx : null;

        const axisKey = [
            projCtx.xAxis.field,
            projCtx.xAxis.domain[0],
            projCtx.xAxis.domain[1],
            projCtx.xAxis.range[0],
            projCtx.xAxis.range[1],
            projCtx.yAxis.field,
            projCtx.yAxis.domain[0],
            projCtx.yAxis.domain[1],
            projCtx.yAxis.range[0],
            projCtx.yAxis.range[1],
            projCtx.effectiveFlipY ? 1 : 0,
            effPrevCtx
                ? `${effPrevCtx.xAxis.field}:${effPrevCtx.xAxis.domain[0]}:${effPrevCtx.xAxis.domain[1]}:${effPrevCtx.yAxis.field}:${effPrevCtx.yAxis.domain[0]}:${effPrevCtx.yAxis.domain[1]}`
                : 'static',
        ].join('|');

        if (hasStatic) {
            this.ensureLayerPacked(this.staticLayer, projCtx, effPrevCtx, tileStore, axisKey);
        }
        if (hasEgo) {
            this.ensureLayerPacked(this.egoLayer, projCtx, effPrevCtx, tileStore, axisKey);
        }

        const canvas = this.rendererWrapper.canvas;
        const screenW = Math.max(1, canvas.width);
        const screenH = Math.max(1, canvas.height);
        const dpr = Math.max(1, this.rendererWrapper.dpr);

        const camLeft = camera.position.x + camera.left / camera.zoom;
        const camRight = camera.position.x + camera.right / camera.zoom;
        const camBottom = camera.position.y + camera.bottom / camera.zoom;
        const camTop = camera.position.y + camera.top / camera.zoom;
        const invSpanX = 1.0 / Math.max(1e-12, camRight - camLeft);
        const invSpanY = 1.0 / Math.max(1e-12, camTop - camBottom);

        if (hasStatic && this.staticLayer.instanceCount > 0) {
            this.drawLayer(
                encoder,
                targetView,
                this.staticLayer,
                0,
                camLeft,
                camTop,
                invSpanX,
                invSpanY,
                screenW,
                screenH,
                dpr,
                isTransitioning,
                tileStore.transitionT,
                tileStore.lightBackground,
                tileStore.globalTimeSec
            );
        }
        if (hasEgo && this.egoLayer.instanceCount > 0) {
            this.drawLayer(
                encoder,
                targetView,
                this.egoLayer,
                1,
                camLeft,
                camTop,
                invSpanX,
                invSpanY,
                screenW,
                screenH,
                dpr,
                isTransitioning,
                tileStore.transitionT,
                tileStore.lightBackground,
                tileStore.globalTimeSec
            );
        }
    }

    public getLiveGpuBytes(): number {
        return (
            (this.uniformBuffer?.size ?? 0) +
            (this.staticLayer.gpuBuffer?.size ?? 0) +
            (this.egoLayer.gpuBuffer?.size ?? 0)
        );
    }

    public dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        if (this.uniformBuffer) {
            this.uniformBuffer.destroy();
            this.uniformBuffer = null;
        }
        if (this.staticLayer.gpuBuffer) {
            this.staticLayer.gpuBuffer.destroy();
            this.staticLayer.gpuBuffer = null;
        }
        if (this.egoLayer.gpuBuffer) {
            this.egoLayer.gpuBuffer.destroy();
            this.egoLayer.gpuBuffer = null;
        }
        this.staticLayer.specs = [];
        this.egoLayer.specs = [];
        this.staticLayer.instanceCount = 0;
        this.egoLayer.instanceCount = 0;
        this.bindGroup = null;
        this.initialized = false;
    }
}
