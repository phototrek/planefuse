import { test, expect } from '@playwright/test';
import { execSync } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

test('import -> PMax -> view -> export', async ({ page }) => {
  const work = mkdtempSync(join(tmpdir(), 'fs-e2e-'));
  const frames = join(work, 'frames');
  const projDir = join(work, 'proj');
  const outFile = join(work, 'out.tif');

  // Generate a synthetic stack via the engine (cwd is the ui/ package).
  execSync(`uv run --project .. python tests/fixtures/make_stack.py "${frames}"`, {
    stdio: 'inherit'
  });

  await page.goto('/');

  // Create project
  await page.getByTestId('new-project').click();
  await page.getByTestId('project-path').fill(projDir);
  await page.getByTestId('project-name').fill('E2E');
  await page.getByTestId('create-project').click();

  // Import frames
  await page.getByTestId('import-frames').click();
  await page.getByTestId('scan-path').fill(frames);
  await page.getByTestId('scan-go').click();
  await expect(page.getByTestId('scan-ok')).toBeVisible();

  // Stack with PMax
  await page.getByTestId('nav-stack').click();
  await page.getByTestId('algo-pmax').click();
  await page.getByTestId('stack-go').click();

  // Queue: wait for done
  await expect(page.getByTestId('job-status')).toHaveText('done', { timeout: 60_000 });

  // Viewer: a tile renders
  await page.getByTestId('nav-viewer').click();
  await expect(page.getByTestId('viewer-tile').first()).toBeVisible({ timeout: 30_000 });

  // Export
  await page.getByTestId('nav-export').click();
  await page.getByTestId('export-dest').fill(outFile);
  await page.getByTestId('export-go').click();
  await expect(page.getByTestId('export-done')).toBeVisible({ timeout: 60_000 });
});
