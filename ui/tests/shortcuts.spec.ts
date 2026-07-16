import { test, expect } from '@playwright/test';
import { execSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

test('workspace shortcuts step frames, zoom, show help, and respect text focus', async ({ page }) => {
  const work = mkdtempSync(join(tmpdir(), 'fs-shortcuts-e2e-'));
  const frames = join(work, 'frames');
  try {
    execSync(`uv run --project .. python tests/fixtures/make_stack.py "${frames}"`, {
      stdio: 'inherit'
    });
    await page.goto('/');
    await page.getByTestId('ws-add').click();
    await page.getByTestId('scan-path').fill(frames);
    await page.getByTestId('ws-add-confirm').click();
    await expect(page.getByTestId('ws-input').first()).toHaveClass(/active/);

    await page.keyboard.press('ArrowRight');
    await expect(page.getByTestId('ws-input').nth(1)).toHaveClass(/active/);
    await page.keyboard.press('z');
    await expect(page.getByText('100%').first()).toBeVisible();
    // Regression guard: a stale-prop feedback loop in DeepZoom's view-sync
    // effect used to revert this a moment later (the assertion above alone
    // caught the fleeting true state and passed even with that bug present).
    // Waiting past the deferred onmove round trip and re-asserting no longer
    // gives that revert anywhere to hide.
    await page.waitForTimeout(500);
    await expect(page.getByText('100%').first()).toBeVisible();
    await page.keyboard.press('f');

    await page.keyboard.press('?');
    await expect(page.getByLabel('Keyboard shortcuts')).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.getByLabel('Keyboard shortcuts')).toHaveCount(0);

    await page.getByTestId('ws-add').click();
    const input = page.getByTestId('scan-path');
    await input.focus();
    await page.keyboard.press('ArrowRight');
    await expect(page.getByTestId('ws-input').nth(1)).toHaveClass(/active/);
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
});
