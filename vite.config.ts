import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    watch: {
      ignored: ['**/Gltf files/**', '**/*.zip', '**/dist/**', '**/.venv/**', '**/public/assets/**']
    },
    proxy: {
      '/randomize': 'http://localhost:8000',
      '/start': 'http://localhost:8000',
      '/step': 'http://localhost:8000',
      '/reset': 'http://localhost:8000',
      '/area_mapping': 'http://localhost:8000'
    }
  }
});

