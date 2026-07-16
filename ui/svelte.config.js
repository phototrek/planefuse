import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

export default {
  preprocess: vitePreprocess(),
  kit: {
    adapter: adapter({
      // Emit the built SPA directly into the server's static mount.
      pages: '../server/src/planefuse_server/static',
      assets: '../server/src/planefuse_server/static',
      fallback: 'index.html',
      precompress: false,
      strict: false
    })
  }
};
