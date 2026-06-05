import * as THREE from 'three';
import * as WebGPU from 'three/webgpu';
console.log('THREE:', Object.keys(THREE).filter(k => k.toLowerCase().includes('storage')));
console.log('WebGPU:', Object.keys(WebGPU).filter(k => k.toLowerCase().includes('storage')));
