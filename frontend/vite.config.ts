import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// In dev, /api is proxied to the FastAPI server (uvicorn on :8000).
export default defineConfig({
  // Relative asset URLs let the app run at / and under a sub-path (e.g. soldrinai.com/rag-search/).
  base: './',
  plugins: [react()],
  worker: { format: 'es' },
  server: { port: 5173, proxy: { '/api': 'http://127.0.0.1:8000' } },
  test: { environment: 'node' },
});
