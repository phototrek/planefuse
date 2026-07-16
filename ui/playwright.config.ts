import { defineConfig } from '@playwright/test';

const PORT = 8525; // dedicated test port
const DEVICE = process.env.PLANEFUSE_E2E_DEVICE ?? 'cpu';

export default defineConfig({
  testDir: './tests',
  timeout: 120_000,
  use: { baseURL: `http://127.0.0.1:${PORT}` },
  webServer: {
    // Build the UI into the server static dir, then serve via the real server.
    command: `npm run build && uv run --frozen --extra cpu --extra raw --project .. python tests/fixtures/serve_test.py ${PORT}`,
    url: `http://127.0.0.1:${PORT}/api/system`,
    env: { ...process.env, PLANEFUSE_DEVICE: DEVICE },
    reuseExistingServer: false,
    timeout: 180_000
  }
});
