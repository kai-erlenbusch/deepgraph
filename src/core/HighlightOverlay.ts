import type { Camera2D, Renderer } from './Renderer.ts';
import type { InternalAxisConfig, TileGpuStore } from './TileGpuStore.ts';

const MAX_OVERLAY_MARKERS = 64;
const FLOATS_PER_MARKER = 8; // centerNdc.xy, radiusNdc.xy, innerOuterPx.xy, colorKind.xy

export interface HighlightMarkerSpec {
    kind: 'hover' | 'selected';
    rowId: number;
    rawRowIdx: number;
    tileZ: number;
    tileX: number;
    tileY: number;
    xU16: number;
    yU16: number;
    attributes: Record<string, unknown>;
}

function pcgHashJS(vIn: number): number {
    const state = (Math.imul(vIn >>> 0, 747796405) + 2891336453) >>> 0;
    const shift = ((state >>> 28) + 4) >>> 0;
    const word = Math.imul((state >>> shift) ^ state, 277803737) >>> 0;
    const res = ((word >>> 22) ^ word) >>> 0;
    return res / 4294967295.0;
}

function mapLinearJS(
    val: number,
    domMin: number,
    domMax: number,
    rangeMin: number,
    rangeMax: number
): number {
    const span = domMax - domMin;
    const safeSpan = Math.abs(span) < 1e-12 ? 1.0 : span;
    const t = Math.max(0.0, Math.min(1.0, (val - domMin) / safeSpan));
    return rangeMin + (rangeMax - rangeMin) * t;
}

/**
 * Plain-WebGPU Hover & Selection Ring Overlay (`D9` + `Phase 7`).
 *
 * Renders crisp anti-aliased dual-contrast rings over picked/hovered points using
 * `loadOp: 'load'` in the same `GPUCommandEncoder` as `ComputeRasterizer`, tracking
 * pan, deep-zoom (`Float64` RTC), and 2-second smoothstep axis transitions with `0 B`
 * per-frame overhead when no markers are active.
 */
export class HighlightOverlay {
    private rendererWrapper: Renderer;
    private device!: GPUDevice;
    private pipeline!: GPURenderPipeline;
    private bindGroupLayout!: GPUBindGroupLayout;
    private markerBuffer: GPUBuffer | null = null;
    private bindGroup: GPUBindGroup | null = null;
    private stagingF32 = new Float32Array(MAX_OVERLAY_MARKERS * FLOATS_PER_MARKER);

    public hoverMarker: HighlightMarkerSpec | null = null;
    public selectedMarkers: HighlightMarkerSpec[] = [];

    private initialized = false;
    private disposed = false;

    constructor(rendererWrapper: Renderer) {
        this.rendererWrapper = rendererWrapper;
    }

    public init(): boolean {
        if (this.disposed) return false;
        if (this.initialized) return true;

        const device = this.rendererWrapper.device;
        if (!device) return false;
        this.device = device;

        this.markerBuffer = this.device.createBuffer({
            label: 'HighlightOverlay_MarkerUniforms',
            size: this.stagingF32.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const shaderCode = /* wgsl */ `
struct Marker {
    // x = centerBackingX, y = centerBackingY, z = screenWidth, w = screenHeight
    posAndScreen: vec4<f32>,
    // x = outerRadiusBackingPx, y = ringRadiusBackingPx, z = strokeBackingPx, w = styleKind (0 = hover, 1 = selected)
    ringParams: vec4<f32>,
};

struct MarkerBank {
    markers: array<Marker, 64>,
};

@group(0) @binding(0) var<uniform> bank: MarkerBank;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) localPx: vec2<f32>,
    @location(1) @interpolate(flat) instIdx: u32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32
) -> VSOut {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0)
    );
    let m = bank.markers[iid];
    let c = corners[vid];
    let localPx = c * m.ringParams.x;
    let screenPx = m.posAndScreen.xy + localPx;
    let ndcX = (screenPx.x / m.posAndScreen.z) * 2.0 - 1.0;
    let ndcY = 1.0 - (screenPx.y / m.posAndScreen.w) * 2.0;

    var out: VSOut;
    out.pos = vec4<f32>(ndcX, ndcY, 0.0, 1.0);
    out.localPx = localPx;
    out.instIdx = iid;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let m = bank.markers[in.instIdx];
    let d = length(in.localPx);
    let rRing = m.ringParams.y;
    let halfStroke = m.ringParams.z;
    let isSelected = m.ringParams.w >= 0.5;

    // Outer dark contrast halo so the ring is crisp over both bright cores and light backgrounds
    let haloDist = abs(d - rRing);
    let haloAlpha = (1.0 - smoothstep(halfStroke + 0.8, halfStroke + 2.4, haloDist)) * 0.72;
    let coreAlpha = 1.0 - smoothstep(max(0.4, halfStroke - 0.6), halfStroke + 0.7, haloDist);

    // Optional center crosshair dot on selected points
    let centerDot = select(0.0, 1.0 - smoothstep(1.2, 2.4, d), isSelected);

    let ringRgb = select(
        vec3<f32>(0.45, 0.90, 1.00),
        vec3<f32>(1.00, 0.82, 0.22),
        isSelected
    );

    let outAlpha = max(haloAlpha, max(coreAlpha, centerDot * 0.9));
    if (outAlpha <= 0.01) {
        discard;
    }
    let fgMix = clamp(max(coreAlpha, centerDot) / max(0.001, outAlpha), 0.0, 1.0);
    let rgb = mix(vec3<f32>(0.04, 0.05, 0.07), ringRgb, fgMix);
    return vec4<f32>(rgb, outAlpha);
}
`;

        this.bindGroupLayout = this.device.createBindGroupLayout({
            label: 'HighlightOverlay_BGL',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform', minBindingSize: this.stagingF32.byteLength },
                },
            ],
        });

        const module = this.device.createShaderModule({
            label: 'HighlightOverlay_Shader',
            code: shaderCode,
        });

        this.pipeline = this.device.createRenderPipeline({
            label: 'HighlightOverlay_Pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.bindGroupLayout],
            }),
            vertex: {
                module,
                entryPoint: 'vs_main',
            },
            fragment: {
                module,
                entryPoint: 'fs_main',
                targets: [
                    {
                        format: this.rendererWrapper.canvasFormat,
                        blend: {
                            color: {
                                srcFactor: 'src-alpha',
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
            primitive: {
                topology: 'triangle-list',
            },
        });

        this.bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: this.markerBuffer } }],
        });

        this.initialized = true;
        return true;
    }

    private projectAxisRelWorld(
        axis: InternalAxisConfig,
        marker: HighlightMarkerSpec,
        subX: number,
        subY: number,
        isYAxis: boolean,
        camLeftOrTop: number
    ): number | null {
        const isSpatial =
            axis.field === 'x_u16' || axis.field === 'y_u16' || axis.transform === 'literal';
        if (isSpatial) {
            const isY16 = axis.field === 'y_u16';
            const scale = Math.pow(2.0, -marker.tileZ);
            const tileOff = (isY16 ? marker.tileY : marker.tileX) * scale;
            const subCoord = isYAxis ? (axis.field === 'x_u16' ? subX : subY) : isY16 ? subY : subX;
            const domSpan = axis.domain[1] - axis.domain[0];
            const safeDomSpan = Math.abs(domSpan) < 1e-12 ? 1.0 : domSpan;
            const worldScale = (axis.range[1] - axis.range[0]) / safeDomSpan;
            const worldTile0 = axis.range[0] + (tileOff - axis.domain[0]) * worldScale;
            if (isYAxis) {
                return camLeftOrTop - worldTile0 - subCoord * scale * worldScale;
            }
            return worldTile0 - camLeftOrTop + subCoord * scale * worldScale;
        }

        const rawVal = marker.attributes[axis.field];
        const numVal = typeof rawVal === 'number' ? rawVal : Number(rawVal);
        if (!Number.isFinite(numVal)) return null;
        const mapped = mapLinearJS(
            numVal,
            axis.domain[0],
            axis.domain[1],
            axis.range[0],
            axis.range[1]
        );
        return isYAxis ? camLeftOrTop - mapped : mapped - camLeftOrTop;
    }

    public encodePass(
        encoder: GPUCommandEncoder,
        targetView: GPUTextureView,
        camera: Camera2D,
        tileStore: TileGpuStore
    ): void {
        if (this.disposed) return;
        const hasHover = Boolean(this.hoverMarker);
        const hasSelected = this.selectedMarkers.length > 0;
        if (!hasHover && !hasSelected) return;
        if (!this.init() || !this.markerBuffer || !this.bindGroup) return;

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

        const computeMode = tileStore.computeMode;
        const isOptical = computeMode === 0 || computeMode === 3;
        const cfg = tileStore.currentConfig;
        const prevCfg = tileStore.previousConfig;
        const xAxis: InternalAxisConfig = isOptical
            ? { field: 'x_u16', transform: 'linear', domain: cfg.x.domain, range: cfg.x.range }
            : cfg.x;
        const yAxis: InternalAxisConfig = isOptical
            ? { field: 'y_u16', transform: 'linear', domain: cfg.y.domain, range: cfg.y.range }
            : cfg.y;
        const isTransitioning =
            tileStore.isTransitioning && !isOptical && computeMode !== 3 && computeMode !== 4;

        const queueList: HighlightMarkerSpec[] = [];
        for (const sel of this.selectedMarkers) {
            if (queueList.length < MAX_OVERLAY_MARKERS) queueList.push(sel);
        }
        if (
            this.hoverMarker &&
            queueList.length < MAX_OVERLAY_MARKERS &&
            !queueList.some(
                (s) =>
                    s.rowId === this.hoverMarker!.rowId ||
                    (s.tileZ === this.hoverMarker!.tileZ &&
                        s.tileX === this.hoverMarker!.tileX &&
                        s.tileY === this.hoverMarker!.tileY &&
                        s.rawRowIdx === this.hoverMarker!.rawRowIdx)
            )
        ) {
            queueList.push(this.hoverMarker);
        }

        let count = 0;
        for (const m of queueList) {
            const subX = m.xU16 / 65535.0 + (pcgHashJS(m.rawRowIdx) - 0.5) / 65535.0;
            const subY = m.yU16 / 65535.0 + (pcgHashJS(m.rawRowIdx + 12345) - 0.5) / 65535.0;

            let relX = this.projectAxisRelWorld(xAxis, m, subX, subY, false, camLeft);
            let fromTopY = this.projectAxisRelWorld(yAxis, m, subX, subY, true, camTop);
            if (relX === null || fromTopY === null) continue;

            if (isTransitioning) {
                const prevRelX = this.projectAxisRelWorld(
                    prevCfg.x,
                    m,
                    subX,
                    subY,
                    false,
                    camLeft
                );
                const prevFromTopY = this.projectAxisRelWorld(
                    prevCfg.y,
                    m,
                    subX,
                    subY,
                    true,
                    camTop
                );
                if (prevRelX !== null && prevFromTopY !== null) {
                    const t = tileStore.transitionT;
                    relX = prevRelX + (relX - prevRelX) * t;
                    fromTopY = prevFromTopY + (fromTopY - prevFromTopY) * t;
                }
            }

            const bx = relX * invSpanX * screenW;
            const by = fromTopY * invSpanY * screenH;
            const margin = 24 * dpr;
            if (bx < -margin || bx > screenW + margin || by < -margin || by > screenH + margin) {
                continue;
            }

            const isSelected = m.kind === 'selected';
            const ringRadiusPx = (isSelected ? 8.5 : 6.5) * dpr;
            const halfStrokePx = (isSelected ? 1.35 : 1.05) * dpr;
            const outerQuadPx = ringRadiusPx + halfStrokePx + 4.0 * dpr;

            const base = count * FLOATS_PER_MARKER;
            this.stagingF32[base + 0] = bx;
            this.stagingF32[base + 1] = by;
            this.stagingF32[base + 2] = screenW;
            this.stagingF32[base + 3] = screenH;
            this.stagingF32[base + 4] = outerQuadPx;
            this.stagingF32[base + 5] = ringRadiusPx;
            this.stagingF32[base + 6] = halfStrokePx;
            this.stagingF32[base + 7] = isSelected ? 1.0 : 0.0;
            count++;
        }

        if (count === 0) return;

        this.device.queue.writeBuffer(
            this.markerBuffer,
            0,
            this.stagingF32.buffer,
            0,
            count * FLOATS_PER_MARKER * 4
        );

        const pass = encoder.beginRenderPass({
            label: 'HighlightOverlay_RenderPass',
            colorAttachments: [
                {
                    view: targetView,
                    loadOp: 'load',
                    storeOp: 'store',
                },
            ],
        });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.draw(6, count, 0, 0);
        pass.end();
    }

    public clear(): void {
        this.hoverMarker = null;
        this.selectedMarkers = [];
    }

    public getLiveGpuBytes(): number {
        return this.markerBuffer?.size ?? 0;
    }

    public dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.clear();
        if (this.markerBuffer) {
            this.markerBuffer.destroy();
            this.markerBuffer = null;
        }
        this.bindGroup = null;
        this.initialized = false;
    }
}
