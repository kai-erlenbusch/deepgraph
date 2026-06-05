import * as THREE from 'three';
import { WebGPURenderer, MeshBasicNodeMaterial, StorageInstancedBufferAttribute, StorageBufferAttribute } from 'three/webgpu';
// @ts-ignore
import { storage, instanceIndex, tslFn, atomicAdd, atomicStore, int, float, vec3, max, positionLocal } from 'three/tsl';

export class ComputeController {
  private renderer: WebGPURenderer;
  private maxNodes: number;
  private maxEdges: number;
  private positionBuffer!: StorageInstancedBufferAttribute;
  private edgeBuffer!: StorageBufferAttribute;
  private forceBuffer!: StorageBufferAttribute;
  private edgeComputeNode: any;
  private nodeComputeNode: any;

  constructor(renderer: WebGPURenderer, maxNodes: number, maxEdges: number) {
    this.renderer = renderer;
    this.maxNodes = maxNodes;
    this.maxEdges = maxEdges;
  }

  public init(scene: THREE.Scene, initialPositions: Float32Array, edges: Uint32Array) {
    this.positionBuffer = new StorageInstancedBufferAttribute(initialPositions, 3);
    this.edgeBuffer = new StorageBufferAttribute(edges, 2);
    // Flat int array for atomic operations (x, y, z per node)
    this.forceBuffer = new StorageBufferAttribute(new Int32Array(this.maxNodes * 3), 1);

    const positionStorage = storage(this.positionBuffer, 'vec3', this.maxNodes);
    const edgeStorage = storage(this.edgeBuffer, 'uvec2', this.maxEdges);
    const forceStorage = (storage(this.forceBuffer, 'int', this.maxNodes * 3) as any).toAtomic();

    const edgeLogic = (tslFn as any)(() => {
      const edge = edgeStorage.element(instanceIndex);
      const sourceId = edge.x;
      const targetId = edge.y;

      const posSource = positionStorage.element(sourceId);
      const posTarget = positionStorage.element(targetId);

      const diff = posTarget.sub(posSource);
      const dist = diff.length();
      
      const restLength = float(2.0);
      const springK = float(0.01);

      const forceMagnitude = dist.sub(restLength).mul(springK);
      const safeDist = max(dist, 0.0001);
      const forceDir = diff.div(safeDist);
      
      const fSource = forceDir.mul(forceMagnitude);
      const fTarget = forceDir.mul(forceMagnitude.negate());

      const multiplier = float(100000.0);
      const fixSource = fSource.mul(multiplier);
      const fixTarget = fTarget.mul(multiplier);

      // Atomic Add for source
      atomicAdd(forceStorage.element(sourceId.mul(3).add(0)), int(fixSource.x));
      atomicAdd(forceStorage.element(sourceId.mul(3).add(1)), int(fixSource.y));
      atomicAdd(forceStorage.element(sourceId.mul(3).add(2)), int(fixSource.z));

      // Atomic Add for target
      atomicAdd(forceStorage.element(targetId.mul(3).add(0)), int(fixTarget.x));
      atomicAdd(forceStorage.element(targetId.mul(3).add(1)), int(fixTarget.y));
      atomicAdd(forceStorage.element(targetId.mul(3).add(2)), int(fixTarget.z));
    });

    this.edgeComputeNode = (edgeLogic() as any).compute(this.maxEdges);

    const nodeLogic = (tslFn as any)(() => {
      const pos = positionStorage.element(instanceIndex);
      
      const idxX = instanceIndex.mul(3).add(0);
      const idxY = instanceIndex.mul(3).add(1);
      const idxZ = instanceIndex.mul(3).add(2);

      // WGSL doesn't allow direct cast of atomic<i32> to f32.
      // We load the value by doing an atomicAdd of 0, which returns the current value.
      const fxInt = atomicAdd(forceStorage.element(idxX), 0);
      const fyInt = atomicAdd(forceStorage.element(idxY), 0);
      const fzInt = atomicAdd(forceStorage.element(idxZ), 0);

      const forceX = float(fxInt).div(100000.0);
      const forceY = float(fyInt).div(100000.0);
      const forceZ = float(fzInt).div(100000.0);

      const centerGravity = pos.negate().normalize().mul(0.005);
      
      pos.x.addAssign(forceX.add(centerGravity.x));
      pos.y.addAssign(forceY.add(centerGravity.y));
      pos.z.addAssign(forceZ.add(centerGravity.z));

      // Reset forces using atomicStore
      atomicStore(forceStorage.element(idxX), 0);
      atomicStore(forceStorage.element(idxY), 0);
      atomicStore(forceStorage.element(idxZ), 0);
    });

    this.nodeComputeNode = (nodeLogic() as any).compute(this.maxNodes);

    const geometry = new THREE.CircleGeometry(0.3, 16);
    const material = new MeshBasicNodeMaterial({
      color: 0x00ff00,
    });
    // @ts-ignore
    material.positionNode = positionLocal.add(positionStorage.element(instanceIndex));

    const instancedMesh = new THREE.InstancedMesh(geometry, material, this.maxNodes);
    instancedMesh.count = this.maxNodes;
    scene.add(instancedMesh);
  }

  public async compute() {
    if (this.edgeComputeNode && this.nodeComputeNode) {
      await this.renderer.computeAsync(this.edgeComputeNode);
      await this.renderer.computeAsync(this.nodeComputeNode);
    }
  }
}
