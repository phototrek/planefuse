import { test, expect } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

test.skip(process.env.FOCUSSTACK_DOCS_CAPTURE !== '1', 'documentation screenshot capture only');

test('capture the no-bake RAW workspace documentation', async ({ page }) => {
  const work = mkdtempSync(join(tmpdir(), 'focusstack-docs-'));
  const frames = join(work, 'same-camera-raw');
  const assets = join('..', 'docs', 'assets');
  mkdirSync(assets, { recursive: true });
  execFileSync(join('..', '.venv', 'bin', 'python'), ['tests/fixtures/make_raw_stack.py', frames], {
    stdio: 'inherit'
  });

  try {
    await page.setViewportSize({ width: 1600, height: 1000 });
    await page.goto('/');
    await page.getByTestId('ws-project-name').fill('RAW Product Stack');
    await page.getByTestId('ws-save').click();
    await expect(page.locator('.topbar .pname')).toHaveText('RAW Product Stack');
    await page.getByTestId('ws-add').click();
    await page.getByTestId('scan-path').fill(frames);
    await page.getByTestId('ws-add-confirm').click();
    await expect(page.getByTestId('ws-input')).toHaveCount(2);
    await expect(page.getByTestId('raw-mode-summary')).toContainText('No aesthetic development is baked');
    await expect(page.getByTestId('viewer-tile').first()).toBeVisible();
    await page.screenshot({ path: join(assets, 'raw-workspace.png'), animations: 'disabled' });

    await page.getByRole('checkbox', { name: 'Align frames' }).uncheck();
    await page.getByTestId('algo-weighted').click();
    const response = page.waitForResponse(
      (item) => item.request().method() === 'POST' && item.url().endsWith('/jobs')
    );
    await page.getByTestId('ws-run').click();
    const jobId = ((await (await response).json()) as { id: string }).id;
    await expect.poll(async () => {
      const job = (await (await page.request.get(`/api/jobs/${jobId}`)).json()) as {
        status: string;
        error?: string;
      };
      return job.error ? `${job.status}: ${job.error}` : job.status;
    }, { timeout: 60_000 }).toBe('done');
    await expect(page.getByTestId('ws-result')).toHaveCount(1, { timeout: 30_000 });
    await page.getByTestId('ws-result').click();
    await expect(page.getByTestId('viewer-tile').first()).toBeVisible();
    await page.getByRole('button', { name: 'Result / source' }).click();
    await expect(page.getByTestId('compare-viewer')).toBeVisible();
    await expect(page.getByTestId('histogram')).toBeVisible();
    await page.screenshot({ path: join(assets, 'compare-histogram.png'), animations: 'disabled' });

    await page.getByTestId('ws-export').click();
    await page.getByLabel('Export format').selectOption('dng');
    await page.getByTestId('export-dest').fill('/photos/RAW_Product_Stack_weighted.dng');
    await expect(page.getByTestId('dng-no-bake-summary')).toContainText('Capture One Pro');
    await page.screenshot({ path: join(assets, 'dng-export.png'), animations: 'disabled' });
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
});
