import { test, expect } from '@playwright/test';
import { mkdtempSync, rmSync, statSync } from 'node:fs';
import { execFileSync, execSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const cleanup: string[] = [];
test.afterEach(() => {
  for (const path of cleanup.splice(0)) rmSync(path, { recursive: true, force: true });
});

function workdir(prefix: string): string {
  const path = mkdtempSync(join(tmpdir(), prefix));
  cleanup.push(path);
  return path;
}

async function addFolder(page: import('@playwright/test').Page, folder: string) {
  await page.getByTestId('ws-add').click();
  await page.getByTestId('scan-path').fill(folder);
  await page.getByTestId('ws-add-confirm').click();
  await expect(page.getByTestId('ws-input')).toHaveCount(2);
}

async function waitForJob(page: import('@playwright/test').Page, jobId: string) {
  await expect.poll(async () => {
    const job = await (await page.request.get(`/api/jobs/${jobId}`)).json() as {
      status: string;
      error: string;
    };
    return job.error ? `${job.status}: ${job.error}` : job.status;
  }, { timeout: 60_000 }).toBe('done');
}

test('RAW mode stacks without baking and exports Capture One Linear DNG', async ({ page }) => {
  const work = workdir('fs-raw-e2e-');
  const frames = join(work, 'frames');
  const output = join(work, 'stack.dng');
  const companion = join(work, 'stack-scene-linear-float.tif');
  execSync(`uv run --project .. --extra raw python tests/fixtures/make_raw_stack.py "${frames}"`, {
    stdio: 'inherit'
  });

  await page.goto('/');
  await addFolder(page, frames);
  await expect(page.getByTestId('raw-mode-summary')).toContainText('No aesthetic development is baked');
  await expect(page.getByTestId('raw-mode-summary')).toContainText('scene-linear camera RGB');

  await page.getByRole('checkbox', { name: 'Align frames' }).uncheck();
  await page.getByTestId('algo-weighted').click();
  const response = page.waitForResponse((item) => item.request().method() === 'POST' && item.url().endsWith('/jobs'));
  await page.getByTestId('ws-run').click();
  const jobId = ((await (await response).json()) as { id: string }).id;
  await waitForJob(page, jobId);
  await expect(page.getByTestId('ws-result')).toHaveCount(1, { timeout: 30_000 });
  await page.getByTestId('ws-result').click();
  await expect(page.getByTestId('viewer-tile').first()).toBeVisible();
  await expect(page.getByTestId('histogram')).toBeVisible();

  // Preview tonemapping is on by default and only affects tile URLs, not data.
  const tonemap = page.getByTestId('tonemap-toggle');
  await expect(tonemap).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByTestId('viewer-tile').first()).toHaveAttribute('src', /display=1/);
  await tonemap.click();
  await expect(tonemap).toHaveAttribute('aria-pressed', 'false');
  await expect(page.getByTestId('viewer-tile').first()).not.toHaveAttribute('src', /display=1/);
  await tonemap.click();
  await expect(page.getByTestId('viewer-tile').first()).toHaveAttribute('src', /display=1/);

  await page.getByRole('button', { name: 'Result / source' }).click();
  await expect(page.getByTestId('compare-viewer')).toBeVisible();

  await page.getByTestId('ws-export').click();
  await page.getByLabel('Export format').selectOption('dng');
  await expect(page.getByTestId('dng-no-bake-summary')).toContainText('Capture One Pro');
  await expect(page.getByTestId('dng-no-bake-summary')).toBeInViewport({ ratio: 1 });
  await page.getByLabel('Also write an unclamped 32-bit float TIFF companion (not RAW)').check();
  await page.getByTestId('export-dest').fill(output);
  const exportResponse = page.waitForResponse(
    (item) => item.request().method() === 'POST' && item.url().endsWith('/export')
  );
  await page.getByTestId('export-go').click();
  const exportJob = ((await (await exportResponse).json()) as { id: string }).id;
  await waitForJob(page, exportJob);
  expect(statSync(output).size).toBeGreaterThan(0);
  expect(statSync(companion).size).toBeGreaterThan(0);
  execFileSync(
    'uv',
    [
      'run', '--project', '..', '--extra', 'raw', 'python', '-c',
      'import sys; from focusstack.io import validate_linear_dng; assert validate_linear_dng(sys.argv[1]).rawpy_validated',
      output
    ],
    { stdio: 'inherit' }
  );
});

test('mixed-camera RAW validation blocks Run and identifies the frame', async ({ page }) => {
  const work = workdir('fs-raw-invalid-e2e-');
  const frames = join(work, 'frames');
  execSync(`uv run --project .. --extra raw python tests/fixtures/make_raw_stack.py "${frames}" incompatible`, {
    stdio: 'inherit'
  });
  await page.goto('/');
  await addFolder(page, frames);
  await expect(page.getByTestId('validation-panel')).toContainText('Different camera');
  await page.getByTestId('algo-pmax').click();
  await expect(page.getByTestId('ws-run')).toBeDisabled();
});
