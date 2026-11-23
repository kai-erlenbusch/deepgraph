/**
 * Demo & Test-Harness Real-GPU `GPUDevice` and `GPUBuffer` Lifecycle Instrumentation (`__gpuStats`).
 *
 * Lives strictly under `src/demo/` (never imported by `src/lib/` or `src/core/`) so the
 * embeddable `Deepgraph` library never monkey-patches browser WebGPU prototypes or mutates
 * `globalThis.__gpuStats` on host pages.
 */

export interface GpuLifecycleStats {
    liveDevices: number;
    totalDevicesCreated: number;
    totalDevicesDestroyed: number;
    liveBuffers: number;
    liveBytes: number;
    tileLiveBuffers: number;
    tileLiveBytes: number;
    totalCreatedBuffers: number;
    totalDestroyedBuffers: number;
    totalCreatedBytes: number;
    totalDestroyedBytes: number;
}

interface TrackedBufferMeta {
    bytes: number;
    isTileSlot: boolean;
}

const stats: GpuLifecycleStats = {
    liveDevices: 0,
    totalDevicesCreated: 0,
    totalDevicesDestroyed: 0,
    liveBuffers: 0,
    liveBytes: 0,
    tileLiveBuffers: 0,
    tileLiveBytes: 0,
    totalCreatedBuffers: 0,
    totalDestroyedBuffers: 0,
    totalCreatedBytes: 0,
    totalDestroyedBytes: 0,
};

let instrumented = false;
const destroyedDevices = new WeakSet<object>();
const destroyedBuffers = new WeakSet<object>();
const bufferMeta = new WeakMap<object, TrackedBufferMeta>();

export function installGpuStatsInstrumentation(): GpuLifecycleStats {
    if (typeof globalThis !== 'undefined') {
        (globalThis as any).__gpuStats = stats;
    }
    if (instrumented) {
        return stats;
    }
    if (
        typeof GPUAdapter === 'undefined' ||
        typeof GPUDevice === 'undefined' ||
        typeof GPUBuffer === 'undefined'
    ) {
        return stats;
    }

    instrumented = true;

    const origRequestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (...args: any[]): Promise<GPUDevice> {
        const device = await origRequestDevice.apply(this, args as any);
        if (device) {
            stats.liveDevices++;
            stats.totalDevicesCreated++;
        }
        return device;
    };

    const origDeviceDestroy = GPUDevice.prototype.destroy;
    GPUDevice.prototype.destroy = function (...args: any[]): undefined {
        if (this && !destroyedDevices.has(this)) {
            destroyedDevices.add(this);
            stats.liveDevices = Math.max(0, stats.liveDevices - 1);
            stats.totalDevicesDestroyed++;
        }
        origDeviceDestroy.apply(this, args as any);
        return undefined;
    };

    const origCreateBuffer = GPUDevice.prototype.createBuffer;
    GPUDevice.prototype.createBuffer = function (descriptor: GPUBufferDescriptor): GPUBuffer {
        const buf = origCreateBuffer.call(this, descriptor);
        if (buf) {
            const bytes = Number(buf.size ?? descriptor?.size ?? 0);
            const label = String(descriptor?.label ?? '');
            const isTileSlot = label.startsWith('TileSlot_');
            bufferMeta.set(buf, { bytes, isTileSlot });
            stats.liveBuffers++;
            stats.liveBytes += bytes;
            stats.totalCreatedBuffers++;
            stats.totalCreatedBytes += bytes;
            if (isTileSlot) {
                stats.tileLiveBuffers++;
                stats.tileLiveBytes += bytes;
            }
        }
        return buf;
    };

    const origBufferDestroy = GPUBuffer.prototype.destroy;
    GPUBuffer.prototype.destroy = function (...args: any[]): undefined {
        if (this && !destroyedBuffers.has(this)) {
            destroyedBuffers.add(this);
            const meta = bufferMeta.get(this);
            if (meta) {
                stats.liveBuffers = Math.max(0, stats.liveBuffers - 1);
                stats.liveBytes = Math.max(0, stats.liveBytes - meta.bytes);
                stats.totalDestroyedBuffers++;
                stats.totalDestroyedBytes += meta.bytes;
                if (meta.isTileSlot) {
                    stats.tileLiveBuffers = Math.max(0, stats.tileLiveBuffers - 1);
                    stats.tileLiveBytes = Math.max(0, stats.tileLiveBytes - meta.bytes);
                }
            }
        }
        origBufferDestroy.apply(this, args as any);
        return undefined;
    };

    return stats;
}

export function getGpuStats(): GpuLifecycleStats {
    return { ...stats };
}
