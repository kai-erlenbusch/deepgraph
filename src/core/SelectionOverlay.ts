import type { LassoPolygonClause, RectBrushClause } from '../lib/types.ts';
import {
    MAX_LASSO_GPU_VERTICES,
    projectDataToWorld64,
    simplifyRdpPolygon,
    type AxisProjectionContext,
} from './FilterGpuResources.ts';
import type { Camera2D, Renderer } from './Renderer.ts';

// header(16B) + screenAndStyle(16B) + aabbBacking(16B) + 32 x vec4<f32> verts(512B) = 560B -> pad to 576B
const SELECTION_UNIFORM_BYTES = 576;

/**
 * Plain-WebGPU 2D Rectangle Brush & Concave Polygon Lasso Overlay (`Step 3A`).
 *
 * - Renders translucent selection fill + crisp anti-aliased dual-contrast border
 *   (`0` storage buffers, `1` uniform buffer `576 B`, `6`-vertex AABB quad) using
 *   exact fragment-stage even-odd polygon test + analytical segment SDF so concave
 *   64-vertex lassos never suffer triangle-fan self-overlap artifacts.
 * - Zero GPU work when no brush or lasso is active.
 */
export class SelectionOverlay {
    private rendererWrapper: Renderer;
    private device!: GPUDevice;
    private pipeline!: GPURenderPipeline;
    private bindGroupLayout!: GPUBindGroupLayout;
    private uniformBuffer: GPUBuffer | null = null;
    private bindGroup: GPUBindGroup | null = null;
    private stagingBuffer = new ArrayBuffer(SELECTION_UNIFORM_BYTES);
    private stagingU32 = new Uint32Array(this.stagingBuffer);
    private stagingF32 = new Float32Array(this.stagingBuffer);

    /** Optional live CSS polyline preview while drawing an open lasso (`< 3` vertices). */
    public liveLassoCssPoints: [number, number][] | null = null;

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

        this.uniformBuffer = this.device.createBuffer({
            label: 'SelectionOverlay_Uniforms_576B',
            size: SELECTION_UNIFORM_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const shaderCode = /* wgsl */ `
struct SelectionUniforms {
    // x = kind (1 = rect, 2 = closed lasso, 3 = open polyline), y = vertCount (<= 64), z = isLightBg, w = pad
    header: vec4<u32>,
    // x = screenW, y = screenH, z = dpr, w = fillAlpha
    screenAndStyle: vec4<f32>,
    // Backing-pixel AABB: (minX, maxX, minY, maxY)
    aabbPx: vec4<f32>,
    // Up to 64 backing-pixel vertices packed 2 per vec4<f32>
    verts: array<vec4<f32>, 32>,
    _pad: vec4<f32>,
};

@group(0) @binding(0) var<uniform> sel: SelectionUniforms;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) fragPx: vec2<f32>,
};

fn get_vert(i: u32) -> vec2<f32> {
    let p = sel.verts[i >> 1u];
    if ((i & 1u) == 0u) {
        return p.xy;
    }
    return p.zw;
}

fn seg_dist(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let denom = dot(ab, ab);
    var t = 0.0;
    if (denom > 1e-6) {
        t = clamp(dot(p - a, ab) / denom, 0.0, 1.0);
    }
    return length(p - (a + ab * t));
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0)
    );
    let c = corners[vid];
    let pad = 5.0 * max(1.0, sel.screenAndStyle.z);
    let minPx = vec2<f32>(sel.aabbPx.x - pad, sel.aabbPx.z - pad);
    let maxPx = vec2<f32>(sel.aabbPx.y + pad, sel.aabbPx.w + pad);
    let px = mix(minPx, maxPx, c);

    let ndcX = (px.x / sel.screenAndStyle.x) * 2.0 - 1.0;
    let ndcY = 1.0 - (px.y / sel.screenAndStyle.y) * 2.0;

    var out: VSOut;
    out.pos = vec4<f32>(ndcX, ndcY, 0.0, 1.0);
    out.fragPx = px;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let kind = sel.header.x;
    let n = sel.header.y;
    let isLight = sel.header.z != 0u;
    let dpr = max(1.0, sel.screenAndStyle.z);
    let p = in.fragPx;

    var inside = false;
    var minEdgeDist = 1e9;

    if (kind == 1u) {
        // Axis-aligned rectangle in aabbPx
        let minB = vec2<f32>(sel.aabbPx.x, sel.aabbPx.z);
        let maxB = vec2<f32>(sel.aabbPx.y, sel.aabbPx.w);
        inside = (p.x >= minB.x && p.x <= maxB.x && p.y >= minB.y && p.y <= maxB.y);
        let dLeft = seg_dist(p, vec2<f32>(minB.x, minB.y), vec2<f32>(minB.x, maxB.y));
        let dRight = seg_dist(p, vec2<f32>(maxB.x, minB.y), vec2<f32>(maxB.x, maxB.y));
        let dTop = seg_dist(p, vec2<f32>(minB.x, minB.y), vec2<f32>(maxB.x, minB.y));
        let dBot = seg_dist(p, vec2<f32>(minB.x, maxB.y), vec2<f32>(maxB.x, maxB.y));
        minEdgeDist = min(min(dLeft, dRight), min(dTop, dBot));
    } else if (n >= 2u) {
        let closed = (kind == 2u) && (n >= 3u);
        var j = select(0u, n - 1u, closed);
        let startIdx = select(1u, 0u, closed);
        for (var i = startIdx; i < n; i = i + 1u) {
            let vi = get_vert(i);
            let vj = get_vert(j);
            minEdgeDist = min(minEdgeDist, seg_dist(p, vj, vi));
            if (closed) {
                let condY = (vi.y > p.y) != (vj.y > p.y);
                if (condY) {
                    let xCross = ((vj.x - vi.x) * (p.y - vi.y)) / (vj.y - vi.y) + vi.x;
                    if (p.x < xCross) {
                        inside = !inside;
                    }
                }
            }
            j = i;
        }
    }

    let halfCore = 0.90 * dpr;
    let haloAlpha = (1.0 - smoothstep(halfCore + 0.6 * dpr, halfCore + 2.1 * dpr, minEdgeDist)) * 0.62;
    let coreAlpha = (1.0 - smoothstep(max(0.35, halfCore - 0.55 * dpr), halfCore + 0.55 * dpr, minEdgeDist)) * 0.95;
    let fillAlpha = select(0.0, sel.screenAndStyle.w, inside);

    let outAlpha = max(fillAlpha, max(haloAlpha, coreAlpha));
    if (outAlpha <= 0.005) {
        discard;
    }

    let accentRgb = select(
        vec3<f32>(0.22, 0.74, 0.97),
        vec3<f32>(0.01, 0.52, 0.78),
        isLight
    );
    let haloRgb = select(
        vec3<f32>(0.03, 0.05, 0.08),
        vec3<f32>(0.98, 0.99, 1.00),
        isLight
    );

    var rgb = accentRgb;
    if (haloAlpha > 0.01 || coreAlpha > 0.01) {
        let strokeMix = clamp(coreAlpha / max(0.001, max(haloAlpha, coreAlpha)), 0.0, 1.0);
        rgb = mix(haloRgb, accentRgb, strokeMix);
    }
    return vec4<f32>(rgb, outAlpha);
}
`;

        this.bindGroupLayout = this.device.createBindGroupLayout({
            label: 'SelectionOverlay_BGL',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform', minBindingSize: SELECTION_UNIFORM_BYTES },
                },
            ],
        });

        const module = this.device.createShaderModule({
            label: 'SelectionOverlay_Shader',
            code: shaderCode,
        });

        this.pipeline = this.device.createRenderPipeline({
            label: 'SelectionOverlay_Pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.bindGroupLayout],
            }),
            vertex: { module, entryPoint: 'vs_main' },
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
            primitive: { topology: 'triangle-list' },
        });

        this.bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
        });

        this.initialized = true;
        return true;
    }

    /** Tests whether `(cssX, cssY)` lies inside `rect` projected onto the current canvas (for spotlight-lens drag). */
    public isCssInsideRect(
        cssX: number,
        cssY: number,
        rect: RectBrushClause | null | undefined,
        camera: Camera2D,
        projCtx: AxisProjectionContext
    ): boolean {
        if (!rect) return false;
        const { width: cssW, height: cssH } = this.rendererWrapper.getContainerSize();
        const camLeft = camera.position.x + camera.left / camera.zoom;
        const camRight = camera.position.x + camera.right / camera.zoom;
        const camBottom = camera.position.y + camera.bottom / camera.zoom;
        const camTop = camera.position.y + camera.top / camera.zoom;
        const invSpanX = 1.0 / Math.max(1e-12, camRight - camLeft);
        const invSpanY = 1.0 / Math.max(1e-12, camTop - camBottom);

        const p0 = projectDataToWorld64(rect.minX, rect.minY, projCtx);
        const p1 = projectDataToWorld64(rect.maxX, rect.maxY, projCtx);
        const sx0 = (p0.worldX - camLeft) * invSpanX * cssW;
        const sx1 = (p1.worldX - camLeft) * invSpanX * cssW;
        const sy0 = (camTop - p0.worldY) * invSpanY * cssH;
        const sy1 = (camTop - p1.worldY) * invSpanY * cssH;

        const minX = Math.min(sx0, sx1);
        const maxX = Math.max(sx0, sx1);
        const minY = Math.min(sy0, sy1);
        const maxY = Math.max(sy0, sy1);
        return cssX >= minX && cssX <= maxX && cssY >= minY && cssY <= maxY;
    }

    public encodePass(
        encoder: GPUCommandEncoder,
        targetView: GPUTextureView,
        camera: Camera2D,
        projCtx: AxisProjectionContext,
        rect: RectBrushClause | null | undefined,
        lasso: LassoPolygonClause | null | undefined,
        lightBackground: boolean
    ): void {
        if (this.disposed) return;

        const hasLiveLine = Boolean(this.liveLassoCssPoints && this.liveLassoCssPoints.length >= 2);
        const hasRect = Boolean(
            rect &&
                Number.isFinite(rect.minX) &&
                Number.isFinite(rect.maxX) &&
                Number.isFinite(rect.minY) &&
                Number.isFinite(rect.maxY)
        );
        const hasLasso = Boolean(lasso && Array.isArray(lasso.points) && lasso.points.length >= 3);

        if (!hasLiveLine && !hasRect && !hasLasso) return;
        if (!this.init() || !this.uniformBuffer || !this.bindGroup) return;

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

        this.stagingU32.fill(0);
        this.stagingF32[4] = screenW;
        this.stagingF32[5] = screenH;
        this.stagingF32[6] = dpr;
        this.stagingF32[7] = lightBackground ? 0.10 : 0.085;
        this.stagingU32[2] = lightBackground ? 1 : 0;

        if (hasLiveLine && !hasLasso) {
            const simp = simplifyRdpPolygon(this.liveLassoCssPoints!, MAX_LASSO_GPU_VERTICES);
            if (simp.length < 2) return;
            let minX = Infinity;
            let maxX = -Infinity;
            let minY = Infinity;
            let maxY = -Infinity;
            const vertBase = 12;
            for (let i = 0; i < simp.length; i++) {
                const bx = simp[i][0] * dpr;
                const by = simp[i][1] * dpr;
                if (bx < minX) minX = bx;
                if (bx > maxX) maxX = bx;
                if (by < minY) minY = by;
                if (by > maxY) maxY = by;
                this.stagingF32[vertBase + i * 2] = bx;
                this.stagingF32[vertBase + i * 2 + 1] = by;
            }
            this.stagingU32[0] = 3;
            this.stagingU32[1] = simp.length;
            this.stagingF32[8] = minX;
            this.stagingF32[9] = maxX;
            this.stagingF32[10] = minY;
            this.stagingF32[11] = maxY;
        } else if (hasLasso && lasso) {
            const simp = simplifyRdpPolygon(lasso.points, MAX_LASSO_GPU_VERTICES);
            if (simp.length < 3) return;
            let minX = Infinity;
            let maxX = -Infinity;
            let minY = Infinity;
            let maxY = -Infinity;
            const vertBase = 12;
            for (let i = 0; i < simp.length; i++) {
                const pw = projectDataToWorld64(simp[i][0], simp[i][1], projCtx);
                const bx = (pw.worldX - camLeft) * invSpanX * screenW;
                const by = (camTop - pw.worldY) * invSpanY * screenH;
                if (bx < minX) minX = bx;
                if (bx > maxX) maxX = bx;
                if (by < minY) minY = by;
                if (by > maxY) maxY = by;
                this.stagingF32[vertBase + i * 2] = bx;
                this.stagingF32[vertBase + i * 2 + 1] = by;
            }
            if (maxX < -8 || minX > screenW + 8 || maxY < -8 || minY > screenH + 8) return;
            this.stagingU32[0] = 2;
            this.stagingU32[1] = simp.length;
            this.stagingF32[8] = minX;
            this.stagingF32[9] = maxX;
            this.stagingF32[10] = minY;
            this.stagingF32[11] = maxY;
        } else if (hasRect && rect) {
            const p0 = projectDataToWorld64(rect.minX, rect.minY, projCtx);
            const p1 = projectDataToWorld64(rect.maxX, rect.maxY, projCtx);
            const bx0 = (p0.worldX - camLeft) * invSpanX * screenW;
            const bx1 = (p1.worldX - camLeft) * invSpanX * screenW;
            const by0 = (camTop - p0.worldY) * invSpanY * screenH;
            const by1 = (camTop - p1.worldY) * invSpanY * screenH;

            const minX = Math.min(bx0, bx1);
            const maxX = Math.max(bx0, bx1);
            const minY = Math.min(by0, by1);
            const maxY = Math.max(by0, by1);
            if (maxX < -8 || minX > screenW + 8 || maxY < -8 || minY > screenH + 8) return;

            this.stagingU32[0] = 1;
            this.stagingU32[1] = 4;
            this.stagingF32[8] = minX;
            this.stagingF32[9] = maxX;
            this.stagingF32[10] = minY;
            this.stagingF32[11] = maxY;
        } else {
            return;
        }

        this.device.queue.writeBuffer(
            this.uniformBuffer,
            0,
            this.stagingBuffer,
            0,
            SELECTION_UNIFORM_BYTES
        );

        const pass = encoder.beginRenderPass({
            label: 'SelectionOverlay_Pass',
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
        pass.draw(6, 1, 0, 0);
        pass.end();
    }

    public getLiveGpuBytes(): number {
        return this.uniformBuffer?.size ?? 0;
    }

    public dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.liveLassoCssPoints = null;
        if (this.uniformBuffer) {
            this.uniformBuffer.destroy();
            this.uniformBuffer = null;
        }
        this.bindGroup = null;
        this.initialized = false;
    }
}
