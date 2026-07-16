import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
  server: {
    // `npm run dev` proxies API + ws to the running planefuse-server.
    proxy: {
      '/api': 'http://127.0.0.1:8425',
      '/ws': { target: 'ws://127.0.0.1:8425', ws: true }
    }
  }
});
