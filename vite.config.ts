import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 5178,
    open: false,
    watch: {
      ignored: ['**/public/data/**']
    }
  },
  build: {
    target: 'esnext'
  }
});
