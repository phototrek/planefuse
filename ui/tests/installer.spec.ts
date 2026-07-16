import { test, expect } from '@playwright/test';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';


test('guided installer switches platform, tracks progress, and resets', async ({ page }) => {
  await page.goto(pathToFileURL(resolve('..', 'installer', 'index.html')).href);

  await expect(page.getByRole('heading', { name: 'Your first focus stack is a few clicks away.' }))
    .toBeVisible();

  const windows = page.getByRole('tab', { name: /Windows/ });
  await windows.click();
  await expect(windows).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator('[data-platform-panel="windows"]')).toBeVisible();

  const panel = page.locator('[data-platform-panel="windows"]');
  await panel.getByRole('button', { name: 'Done — I can see the folder' }).click();
  await panel.getByRole('button', { name: 'I opened the launcher' }).click();
  await panel.getByRole('button', { name: 'PlaneFuse opened in my browser' }).click();

  const download = panel.getByRole('link', { name: 'Download for Windows' });
  await download.evaluate((element) => {
    element.addEventListener('click', (event) => event.preventDefault(), { once: true });
  });
  await download.click();

  await expect(page.locator('[data-progress-label]')).toHaveText('4 of 4');
  await expect(page.getByText('You’re ready to stack.')).toBeVisible();

  await page.getByRole('button', { name: 'Reset these steps' }).click();
  await expect(page.locator('[data-progress-label]')).toHaveText('0 of 4');
  await expect(page.getByText('You’re ready to stack.')).toBeHidden();

  await windows.press('ArrowLeft');
  await expect(page.getByRole('tab', { name: /macOS/ })).toHaveAttribute('aria-selected', 'true');
});
