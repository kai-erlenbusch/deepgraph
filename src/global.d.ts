export {};

declare global {
  interface Window {
    tileManagerInstance?: any;
    tileFetchTimes?: number[];
    perfAccum?: {
      frames: number;
      getVisibleTiles: number;
      updateTiles: number;
      updateCam: number;
      render: number;
      totalFrame: number;
      lastReport: number;
    };
    lastPerfReport?: {
      fps: number;
      tiles: number;
      getVisibleTiles: string;
      updateTiles: string;
      updateCam: string;
      render: string;
      totalFrame: string;
    };
  }
}
