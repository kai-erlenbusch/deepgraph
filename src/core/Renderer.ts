import * as THREE from 'three';
import { WebGPURenderer } from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class Renderer {
  public renderer: WebGPURenderer;
  public scene: THREE.Scene;
  public camera: THREE.OrthographicCamera;
  public controls: OrbitControls;

  constructor(container: HTMLElement) {
    this.renderer = new WebGPURenderer({ antialias: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x111111);

    const aspect = window.innerWidth / window.innerHeight;
    const frustumSize = 40;
    this.camera = new THREE.OrthographicCamera(
      frustumSize * aspect / -2,
      frustumSize * aspect / 2,
      frustumSize / 2,
      frustumSize / -2,
      0.1,
      1000
    );
    this.camera.position.set(7.1, 1.8, 100);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(7.1, 1.8, 0);
    this.controls.enableDamping = true;
    this.controls.enableRotate = false; // 2D scatterplot only
    
    // Configure mouse buttons for 2D panning
    this.controls.mouseButtons = {
      LEFT: THREE.MOUSE.PAN,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN
    };

    window.addEventListener('resize', this.onWindowResize.bind(this));
  }

  private onWindowResize() {
    const aspect = window.innerWidth / window.innerHeight;
    const frustumSize = 40; // Base size, don't multiply by zoom here
    this.camera.left = -frustumSize * aspect / 2;
    this.camera.right = frustumSize * aspect / 2;
    this.camera.top = frustumSize / 2;
    this.camera.bottom = -frustumSize / 2;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  public getViewportBounds() {
    // Calculate current visible bounding box
    const aspect = window.innerWidth / window.innerHeight;
    const frustumSize = 40 / this.camera.zoom;
    const halfW = frustumSize * aspect / 2;
    const halfH = frustumSize / 2;
    return {
      minX: this.camera.position.x - halfW,
      maxX: this.camera.position.x + halfW,
      minY: this.camera.position.y - halfH,
      maxY: this.camera.position.y + halfH
    };
  }

  public render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  public async init() {
    await this.renderer.init();
  }
}
