export interface Camera2DPosition {
    x: number;
    y: number;
    z: number;
    set(x: number, y: number, z?: number): void;
}

export interface Camera2D {
    position: Camera2DPosition;
    left: number;
    right: number;
    top: number;
    bottom: number;
    zoom: number;
    updateProjectionMatrix(): void;
    updateMatrixWorld(): void;
}

const FRUSTUM_SIZE = 4.0;
const MIN_ZOOM = 0.05;
const MAX_ZOOM = 1_000_000.0;
const CLICK_MAX_TRAVEL_CSS_PX = 4.0;

/**
 * Native WebGPU Canvas, Swapchain, `GPUDevice`, and 2D Orthographic Pan/Zoom Controller (`D9`).
 *
 * - Directly owns `<canvas>`, `GPUDevice`, and `GPUCanvasContext` with zero scene-graph wrapper and zero global WebGPU monkey-patching.
 * - Forwards `powerPreference` (`'high-performance' | 'low-power'`) to `navigator.gpu.requestAdapter`.
 * - Implements `touch-action: none` pointer-capture drag panning, `< 4 CSS px` click gating (`Phase 7`), hover callbacks, and exact `Float64` wheel zoom-to-cursor.
 * - Unconfigures the canvas context and calls `device.destroy()` synchronously on `dispose()`.
 */
export class Renderer {
    public readonly canvas: HTMLCanvasElement;
    public readonly camera: Camera2D;
    public device: GPUDevice | null = null;
    public context: GPUCanvasContext | null = null;
    public canvasFormat: GPUTextureFormat = 'bgra8unorm';

    public dpr = 1.0;
    public baseZoom = 1.0;
    public zoomT = 0.0;
    /** Continuous unclamped `max(0, log2(camera.zoom / baseZoom))` for multiscale deep-zoom sizing & exposure (`Phase 8b`). */
    public logZoom = 0.0;
    public worldUnitsPerPixel = 0.001;

    public onCanvasClick?: (cssX: number, cssY: number, event: PointerEvent) => void;
    public onCanvasHover?: (cssX: number, cssY: number, event: PointerEvent) => void;
    public onCanvasPointerLeave?: () => void;

    private container: HTMLElement;
    private powerPreference: GPUPowerPreference;
    private resizeObserver: ResizeObserver | null = null;
    private rafId: number | null = null;
    private animationCallback: (() => void) | null = null;
    private disposed = false;

    private activePointerId: number | null = null;
    private activePointerButton = 0;
    private pointerDownX = 0;
    private pointerDownY = 0;
    private maxPointerTravelPx = 0;
    private lastPointerX = 0;
    private lastPointerY = 0;

    private boundResize: () => void;
    private boundWheel: (event: WheelEvent) => void;
    private boundPointerDown: (event: PointerEvent) => void;
    private boundPointerMove: (event: PointerEvent) => void;
    private boundPointerUp: (event: PointerEvent) => void;
    private boundPointerCancel: (event: PointerEvent) => void;
    private boundPointerLeave: () => void;
    private boundContextMenu: (event: MouseEvent) => void;

    constructor(
        container: HTMLElement,
        powerPreference: GPUPowerPreference = 'high-performance'
    ) {
        this.container = container;
        this.powerPreference = powerPreference;

        const canvas = document.createElement('canvas');
        canvas.style.display = 'block';
        canvas.style.touchAction = 'none';
        canvas.style.userSelect = 'none';
        this.canvas = canvas;
        container.appendChild(canvas);

        const pos: Camera2DPosition = {
            x: 0.0,
            y: 0.0,
            z: 5.0,
            set(x: number, y: number, z = pos.z) {
                pos.x = x;
                pos.y = y;
                pos.z = z;
            },
        };

        const { width, height } = this.getContainerSize();
        const aspect = width / height;
        this.camera = {
            position: pos,
            left: (-FRUSTUM_SIZE * aspect) / 2,
            right: (FRUSTUM_SIZE * aspect) / 2,
            top: FRUSTUM_SIZE / 2,
            bottom: -FRUSTUM_SIZE / 2,
            zoom: 1.61,
            updateProjectionMatrix() {},
            updateMatrixWorld() {},
        };

        this.syncCanvasDimensions(width, height);

        this.boundResize = () => this.onResize();
        this.boundWheel = (e) => this.onWheel(e);
        this.boundPointerDown = (e) => this.onPointerDown(e);
        this.boundPointerMove = (e) => this.onPointerMove(e);
        this.boundPointerUp = (e) => this.onPointerUp(e, true);
        this.boundPointerCancel = (e) => this.onPointerUp(e, false);
        this.boundPointerLeave = () => {
            if (!this.disposed && this.activePointerId === null) {
                this.onCanvasPointerLeave?.();
            }
        };
        this.boundContextMenu = (e) => e.preventDefault();

        window.addEventListener('resize', this.boundResize);
        if (typeof ResizeObserver !== 'undefined') {
            this.resizeObserver = new ResizeObserver(() => this.onResize());
            this.resizeObserver.observe(container);
        }

        canvas.addEventListener('wheel', this.boundWheel, { passive: false });
        canvas.addEventListener('pointerdown', this.boundPointerDown);
        canvas.addEventListener('pointermove', this.boundPointerMove);
        canvas.addEventListener('pointerup', this.boundPointerUp);
        canvas.addEventListener('pointercancel', this.boundPointerCancel);
        canvas.addEventListener('lostpointercapture', this.boundPointerCancel);
        canvas.addEventListener('pointerleave', this.boundPointerLeave);
        canvas.addEventListener('contextmenu', this.boundContextMenu);
    }

    public async init(): Promise<void> {
        if (this.disposed) return;
        if (!navigator.gpu) {
            throw new Error('WebGPU is not supported by your browser.');
        }

        const adapter =
            (await navigator.gpu.requestAdapter({ powerPreference: this.powerPreference })) ??
            (await navigator.gpu.requestAdapter());
        if (!adapter) {
            throw new Error('Failed to acquire a WebGPU adapter.');
        }
        if (this.disposed) return;

        const requiredLimits: Record<string, number> = {};
        const lim = adapter.limits;
        if (lim) {
            if (lim.maxStorageBufferBindingSize) {
                requiredLimits.maxStorageBufferBindingSize = lim.maxStorageBufferBindingSize;
            }
            if (lim.maxComputeWorkgroupStorageSize) {
                requiredLimits.maxComputeWorkgroupStorageSize = lim.maxComputeWorkgroupStorageSize;
            }
            if (lim.maxStorageBuffersPerShaderStage) {
                requiredLimits.maxStorageBuffersPerShaderStage = lim.maxStorageBuffersPerShaderStage;
            }
            if (lim.maxBufferSize) {
                requiredLimits.maxBufferSize = lim.maxBufferSize;
            }
            if (lim.maxTextureDimension2D) {
                requiredLimits.maxTextureDimension2D = lim.maxTextureDimension2D;
            }
        }

        const device = await adapter.requestDevice({ requiredLimits });
        if (this.disposed) {
            device.destroy();
            return;
        }

        const context = this.canvas.getContext('webgpu') as GPUCanvasContext | null;
        if (!context) {
            device.destroy();
            throw new Error('Failed to acquire WebGPU canvas context.');
        }

        this.device = device;
        this.context = context;
        this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();

        const { width, height } = this.getContainerSize();
        this.syncCanvasDimensions(width, height);
        this.configureContext();
    }

    private configureContext(): void {
        if (!this.device || !this.context || this.disposed) return;
        this.context.configure({
            device: this.device,
            format: this.canvasFormat,
            alphaMode: 'opaque',
        });
    }

    public getContainerSize(): { width: number; height: number } {
        const w = this.container.clientWidth || window.innerWidth || 1;
        const h = this.container.clientHeight || window.innerHeight || 1;
        return { width: Math.max(1, w), height: Math.max(1, h) };
    }

    public setBaseZoom(next: number): void {
        this.baseZoom = Number.isFinite(next) && next > 0 ? next : 1.0;
    }

    public clampZoom(zoom: number): number {
        const minZ = MIN_ZOOM * this.baseZoom;
        const maxZ = MAX_ZOOM * this.baseZoom;
        return Math.min(maxZ, Math.max(minZ, zoom));
    }

    private syncCanvasDimensions(cssWidth: number, cssHeight: number): void {
        const dpr = Math.max(1, window.devicePixelRatio || 1);
        this.dpr = dpr;
        const maxDim = this.device?.limits?.maxTextureDimension2D ?? 8192;
        const backingW = Math.min(maxDim, Math.max(1, Math.floor(cssWidth * dpr)));
        const backingH = Math.min(maxDim, Math.max(1, Math.floor(cssHeight * dpr)));

        this.canvas.style.width = `${cssWidth}px`;
        this.canvas.style.height = `${cssHeight}px`;
        if (this.canvas.width !== backingW) this.canvas.width = backingW;
        if (this.canvas.height !== backingH) this.canvas.height = backingH;

        const aspect = cssWidth / cssHeight;
        this.camera.left = (-FRUSTUM_SIZE * aspect) / 2;
        this.camera.right = (FRUSTUM_SIZE * aspect) / 2;
        this.camera.top = FRUSTUM_SIZE / 2;
        this.camera.bottom = -FRUSTUM_SIZE / 2;
    }

    private onResize(): void {
        if (this.disposed) return;
        const { width, height } = this.getContainerSize();
        const prevW = this.canvas.width;
        const prevH = this.canvas.height;
        this.syncCanvasDimensions(width, height);
        if (this.canvas.width !== prevW || this.canvas.height !== prevH) {
            this.configureContext();
        }
    }

    private onPointerDown(event: PointerEvent): void {
        if (this.disposed) return;
        if (event.button !== 0 && event.button !== 1 && event.button !== 2) return;
        if (this.activePointerId !== null) return;

        this.activePointerId = event.pointerId;
        this.activePointerButton = event.button;
        this.pointerDownX = event.clientX;
        this.pointerDownY = event.clientY;
        this.maxPointerTravelPx = 0;
        this.lastPointerX = event.clientX;
        this.lastPointerY = event.clientY;
        try {
            this.canvas.setPointerCapture(event.pointerId);
        } catch {
            // Synthetic test pointer events may omit active pointer registration
        }
    }

    private onPointerMove(event: PointerEvent): void {
        if (this.disposed) return;

        if (this.activePointerId === null) {
            if (this.onCanvasHover) {
                const rect = this.canvas.getBoundingClientRect();
                const cssX = event.clientX - rect.left;
                const cssY = event.clientY - rect.top;
                if (cssX >= 0 && cssY >= 0 && cssX <= rect.width && cssY <= rect.height) {
                    this.onCanvasHover(cssX, cssY, event);
                }
            }
            return;
        }

        if (this.activePointerId !== event.pointerId) return;

        const travel = Math.hypot(
            event.clientX - this.pointerDownX,
            event.clientY - this.pointerDownY
        );
        if (travel > this.maxPointerTravelPx) {
            this.maxPointerTravelPx = travel;
        }

        const dxPx = event.clientX - this.lastPointerX;
        const dyPx = event.clientY - this.lastPointerY;
        this.lastPointerX = event.clientX;
        this.lastPointerY = event.clientY;
        if (dxPx === 0 && dyPx === 0) return;

        const rect = this.canvas.getBoundingClientRect();
        const rw = Math.max(1, rect.width);
        const rh = Math.max(1, rect.height);

        const visibleWidth = (this.camera.right - this.camera.left) / this.camera.zoom;
        const visibleHeight = (this.camera.top - this.camera.bottom) / this.camera.zoom;

        this.camera.position.x -= (dxPx / rw) * visibleWidth;
        this.camera.position.y += (dyPx / rh) * visibleHeight;
    }

    private onPointerUp(event: PointerEvent, allowClick: boolean): void {
        if (this.activePointerId !== event.pointerId) return;
        const wasLeftButton = this.activePointerButton === 0;
        const finalTravel = Math.max(
            this.maxPointerTravelPx,
            Math.hypot(event.clientX - this.pointerDownX, event.clientY - this.pointerDownY)
        );

        this.activePointerId = null;
        try {
            if (this.canvas.hasPointerCapture(event.pointerId)) {
                this.canvas.releasePointerCapture(event.pointerId);
            }
        } catch {
            // Ignore release errors on synthetic events
        }

        const rect = this.canvas.getBoundingClientRect();
        const cssX = event.clientX - rect.left;
        const cssY = event.clientY - rect.top;
        const isInside = cssX >= 0 && cssY >= 0 && cssX <= rect.width && cssY <= rect.height;

        if (
            allowClick &&
            wasLeftButton &&
            finalTravel < CLICK_MAX_TRAVEL_CSS_PX &&
            this.onCanvasClick &&
            isInside
        ) {
            this.onCanvasClick(cssX, cssY, event);
        } else if (!isInside) {
            this.onCanvasPointerLeave?.();
        }
    }

    private onWheel(event: WheelEvent): void {
        if (this.disposed) return;
        event.preventDefault();
        event.stopPropagation();

        const zoomFactor = event.deltaY > 0 ? 1.1 : 0.9;
        const prevZoom = this.camera.zoom;
        const nextZoom = this.clampZoom(prevZoom / zoomFactor);
        if (nextZoom === prevZoom) return;

        const rect = this.canvas.getBoundingClientRect();
        const rw = Math.max(1, rect.width);
        const rh = Math.max(1, rect.height);

        const u = (event.clientX - rect.left) / rw;
        const v = (event.clientY - rect.top) / rh;

        const frustumW = this.camera.right - this.camera.left;
        const frustumH = this.camera.top - this.camera.bottom;
        const localX = this.camera.left + u * frustumW;
        const localY = this.camera.top - v * frustumH;

        const worldX = this.camera.position.x + localX / prevZoom;
        const worldY = this.camera.position.y + localY / prevZoom;

        this.camera.zoom = nextZoom;
        this.camera.position.x = worldX - localX / nextZoom;
        this.camera.position.y = worldY - localY / nextZoom;
    }

    public updateCameraUniforms(): void {
        const { height } = this.getContainerSize();
        const visibleHeight = (this.camera.top - this.camera.bottom) / this.camera.zoom;
        this.worldUnitsPerPixel = visibleHeight / height;
        this.dpr = Math.max(1, window.devicePixelRatio || 1);
        const relZoom = this.camera.zoom / Math.max(1e-30, this.baseZoom);
        const currentZoom = Math.log2(Math.max(0.1, relZoom));
        this.logZoom = Math.max(0.0, currentZoom);
        this.zoomT = Math.max(0.0, Math.min(1.0, currentZoom / 6.0));
    }

    public setAnimationLoop(callback: (() => void) | null): void {
        this.animationCallback = callback;
        if (!callback || this.disposed) {
            if (this.rafId !== null) {
                cancelAnimationFrame(this.rafId);
                this.rafId = null;
            }
            return;
        }
        if (this.rafId === null) {
            const tick = () => {
                if (this.disposed || !this.animationCallback) {
                    this.rafId = null;
                    return;
                }
                this.rafId = requestAnimationFrame(tick);
                this.animationCallback();
            };
            this.rafId = requestAnimationFrame(tick);
        }
    }

    public dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.setAnimationLoop(null);

        window.removeEventListener('resize', this.boundResize);
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = null;
        }

        this.canvas.removeEventListener('wheel', this.boundWheel);
        this.canvas.removeEventListener('pointerdown', this.boundPointerDown);
        this.canvas.removeEventListener('pointermove', this.boundPointerMove);
        this.canvas.removeEventListener('pointerup', this.boundPointerUp);
        this.canvas.removeEventListener('pointercancel', this.boundPointerCancel);
        this.canvas.removeEventListener('lostpointercapture', this.boundPointerCancel);
        this.canvas.removeEventListener('pointerleave', this.boundPointerLeave);
        this.canvas.removeEventListener('contextmenu', this.boundContextMenu);

        if (this.context) {
            try {
                this.context.unconfigure();
            } catch {
                // Ignore if context already lost
            }
            this.context = null;
        }

        if (this.device) {
            try {
                this.device.destroy();
            } catch {
                // Ignore if device already destroyed
            }
            this.device = null;
        }

        if (this.canvas.parentElement === this.container) {
            this.container.removeChild(this.canvas);
        }
    }
}
