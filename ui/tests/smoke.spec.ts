import { test, expect } from '@playwright/test';
import { execFileSync, execSync } from 'node:child_process';
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  rmSync,
  statSync,
  symlinkSync
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const zionStack =
  process.env.PLANEFUSE_REAL_STACK_DIR ??
  String.raw`G:\BackUpPhoto\USA_2018\2018-10-13 - Zion\STACK 2026`;

const cleanup: string[] = [];
test.afterEach(() => {
  for (const path of cleanup.splice(0)) rmSync(path, { recursive: true, force: true });
});

function tempWork(prefix: string): string {
  const path = mkdtempSync(join(tmpdir(), prefix));
  cleanup.push(path);
  return path;
}

function stageZionSources(work: string): string {
  const frames = join(work, 'frames');
  mkdirSync(frames);
  const names = readdirSync(zionStack).filter((name) => /^stack-pure-.*\.tif$/i.test(name));
  expect(names).toHaveLength(7);
  for (const name of names) {
    const source = join(zionStack, name);
    const dest = join(frames, name);
    try {
      symlinkSync(source, dest, 'file');
    } catch {
      copyFileSync(source, dest);
    }
  }
  return frames;
}

async function waitForJob(page: import('@playwright/test').Page, jobId: string, timeout = 60_000) {
  await expect
    .poll(
      async () => {
        const response = await page.request.get(`/api/jobs/${jobId}`);
        const job = (await response.json()) as { status: string; error: string };
        return job.error ? `${job.status}: ${job.error}` : job.status;
      },
      { timeout }
    )
    .toBe('done');
}

test('workspace: add -> PMax -> view -> export', async ({ page }) => {
  const work = tempWork('fs-e2e-');
  const frames = join(work, 'frames');
  const outFile = join(work, 'out.tif');

  // Generate a synthetic stack via the engine (cwd is the ui/ package).
  execSync(`uv run --project .. python tests/fixtures/make_stack.py "${frames}"`, {
    stdio: 'inherit'
  });

  // Workspace opens directly — auto scratch project, no create step.
  await page.goto('/');

  // Add frames: open add panel, fill the scan-path, confirm.
  await page.getByTestId('ws-add').click();
  await page.getByTestId('scan-path').fill(frames);
  await page.getByTestId('ws-add-confirm').click();
  await expect(page.getByTestId('ws-input').first()).toBeVisible();

  // Stack with PMax: capture the POST /jobs response to get jobId.
  await page.getByTestId('algo-pmax').click();
  const jobResponse = page.waitForResponse(
    (r) => r.request().method() === 'POST' && r.url().includes('/jobs')
  );
  await page.getByTestId('ws-run').click();
  const jobId = ((await (await jobResponse).json()) as { id: string }).id;
  await waitForJob(page, jobId);

  // Settings stay fully inside a compact viewport and are readable without
  // exposing raw JSON or adding an internal scroll area.
  await page.setViewportSize({ width: 640, height: 480 });
  await page.locator(`[data-testid="job-settings"][data-job-id="${jobId}"]`).click();
  const settings = page.locator(
    `[data-testid="job-settings-dialog"][data-job-id="${jobId}"]`
  );
  await expect(settings).toBeVisible();
  await expect(settings).toContainText('Focus stack settings');
  await expect(settings).toContainText('Stacking method');
  await expect(settings).toContainText('PMax');
  await expect(settings).toContainText('Alignment');
  await expect(settings.locator('pre')).toHaveCount(0);
  const metrics = await settings.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    return {
      top: rect.top,
      right: rect.right,
      bottom: rect.bottom,
      left: rect.left,
      viewportWidth: window.innerWidth,
      viewportHeight: window.innerHeight,
      clientWidth: element.clientWidth,
      scrollWidth: element.scrollWidth,
      clientHeight: element.clientHeight,
      scrollHeight: element.scrollHeight
    };
  });
  expect(metrics.left).toBeGreaterThanOrEqual(8);
  expect(metrics.top).toBeGreaterThanOrEqual(8);
  expect(metrics.right).toBeLessThanOrEqual(metrics.viewportWidth - 8);
  expect(metrics.bottom).toBeLessThanOrEqual(metrics.viewportHeight - 8);
  expect(metrics.scrollWidth).toBeLessThanOrEqual(metrics.clientWidth);
  expect(metrics.scrollHeight).toBeLessThanOrEqual(metrics.clientHeight);
  await page.getByRole('button', { name: 'Close settings' }).click();
  await page.setViewportSize({ width: 1280, height: 720 });

  // Result + viewer: a result thumb appears; click it; a DeepZoom tile renders.
  await expect(page.getByTestId('ws-result').first()).toBeVisible({ timeout: 30_000 });
  await page.getByTestId('ws-result').first().click();
  await expect(page.getByTestId('viewer-tile').first()).toBeVisible({ timeout: 30_000 });

  // Name the project so the template preview reflects it.
  await page.getByTestId('ws-project-name').fill('E2E');

  // Export: open export popover, fill dest, confirm, wait for done.
  await page.getByTestId('ws-export').click();
  await expect(page.getByTestId('export-preview')).toHaveText(/E2E_pmax\.tif/);
  await page.getByTestId('export-dest').fill(outFile);
  const exportResponse = page.waitForResponse(
    (r) => r.request().method() === 'POST' && r.url().endsWith('/export')
  );
  await page.getByTestId('export-go').click();
  const exportJobId = ((await (await exportResponse).json()) as { id: string }).id;
  await waitForJob(page, exportJobId);
  await expect(page.getByTestId('export-done')).toBeVisible();

  expect(statSync(outFile).size).toBeGreaterThan(0);
  execFileSync(
    'uv',
    [
      'run',
      '--project',
      '..',
      'python',
      '-c',
      'import sys; from planefuse.io import load_image; f=load_image(sys.argv[1]); assert f.pixels.shape == (64, 80, 3); assert f.bit_depth == 16',
      outFile
    ],
    { stdio: 'inherit' }
  );
});

test('workspace: multi-select runs one stack per toggled algorithm', async ({ page }) => {
  const work = tempWork('fs-multi-e2e-');
  const frames = join(work, 'frames');
  execSync(`uv run --project .. python tests/fixtures/make_stack.py "${frames}"`, {
    stdio: 'inherit'
  });

  await page.goto('/');
  await page.getByTestId('ws-add').click();
  await page.getByTestId('scan-path').fill(frames);
  await page.getByTestId('ws-add-confirm').click();
  await expect(page.getByTestId('ws-input').first()).toBeVisible();

  // Toggle two algorithms; Run should enqueue one stack each → two results.
  await page.getByTestId('algo-pmax').click();
  await page.getByTestId('algo-weighted').click();
  await page.getByTestId('ws-run').click();

  await expect(page.getByTestId('ws-result')).toHaveCount(2, { timeout: 90_000 });
});

test('workspace: retouch result -> paint -> undo/redo -> flatten -> export', async ({ page }) => {
  test.setTimeout(5 * 60_000);
  const work = tempWork('fs-retouch-e2e-');
  const frames = join(work, 'frames');
  const outFile = join(work, 'retouched.tif');

  execSync(`uv run --project .. python tests/fixtures/make_stack.py "${frames}"`, {
    stdio: 'inherit'
  });

  // Workspace opens directly.
  await page.goto('/');

  // Add frames.
  await page.getByTestId('ws-add').click();
  await page.getByTestId('scan-path').fill(frames);
  await page.getByTestId('ws-add-confirm').click();
  await expect(page.getByTestId('ws-input').first()).toBeVisible();

  // Stack pmax and weighted as two separate single-algo runs. Algo buttons are
  // toggles, so deselect the previous method before selecting the next.
  let prev: string | null = null;
  for (const method of ['pmax', 'weighted']) {
    if (prev) await page.getByTestId(`algo-${prev}`).click();
    await page.getByTestId(`algo-${method}`).click();
    const response = page.waitForResponse(
      (r) => r.request().method() === 'POST' && r.url().endsWith('/jobs')
    );
    await page.getByTestId('ws-run').click();
    const jobId = ((await (await response).json()) as { id: string }).id;
    await waitForJob(page, jobId);
    prev = method;
  }

  // Select the first result and launch retouch from it.
  await expect(page.getByTestId('ws-result').first()).toBeVisible({ timeout: 30_000 });
  await page.getByTestId('ws-result').first().click();
  await expect(page.getByTestId('viewer-tile').first()).toBeVisible({ timeout: 30_000 });
  await page.getByTestId('retouch-this-result').first().click();

  await expect(page).toHaveURL(/\/retouch$/);
  await page.getByTestId('retouch-source').first().click();
  const strokeResponse = page.waitForResponse(
    (r) => r.request().method() === 'POST' && /\/api\/retouch\/[^/]+\/stroke$/.test(r.url())
  );
  const canvas = page.getByTestId('retouch-canvas');
  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();
  await page.mouse.move(box!.x + box!.width * 0.4, box!.y + box!.height * 0.5);
  await page.mouse.down();
  await page.mouse.move(box!.x + box!.width * 0.6, box!.y + box!.height * 0.5, { steps: 5 });
  await page.mouse.up();

  const strokeRequest = (await strokeResponse).request().postDataJSON() as {
    points: [number, number, number][];
  };
  expect(strokeRequest.points.length).toBeGreaterThan(1);
  for (const [x, y, pressure] of strokeRequest.points) {
    expect(x).toBeGreaterThanOrEqual(0);
    expect(x).toBeLessThan(80);
    expect(y).toBeGreaterThanOrEqual(0);
    expect(y).toBeLessThan(64);
    expect(pressure).toBeGreaterThan(0);
  }

  await page.getByTestId('retouch-undo').click();
  await page.getByTestId('retouch-redo').click();
  await page.getByTestId('retouch-flatten-name').fill('E2E retouched');
  const flattenResponse = page.waitForResponse(
    (r) => r.request().method() === 'POST' && /\/api\/retouch\/[^/]+\/flatten$/.test(r.url())
  );
  await page.getByTestId('retouch-flatten').click();
  const flattenedId = ((await (await flattenResponse).json()) as { image_id: string }).image_id;

  // After flatten, expect to be back on the workspace with the viewer showing the flattened result.
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByTestId('viewer-tile').first()).toBeVisible({ timeout: 30_000 });
  const projects = (await page.request.get('/api/projects')).json() as Promise<
    { images: Record<string, { name?: string }> }[]
  >;
  const project = (await projects).find((item) => flattenedId in item.images);
  expect(project?.images[flattenedId].name).toBe('E2E retouched');

  // Export the retouched result.
  await page.getByTestId('ws-export').click();
  await page.getByTestId('export-dest').fill(outFile);
  const exportResponse = page.waitForResponse(
    (r) => r.request().method() === 'POST' && r.url().endsWith('/export')
  );
  await page.getByTestId('export-go').click();
  const exportJobId = ((await (await exportResponse).json()) as { id: string }).id;
  await waitForJob(page, exportJobId);
  await expect(page.getByTestId('export-done')).toBeVisible();

  expect(statSync(outFile).size).toBeGreaterThan(0);
  execFileSync(
    'uv',
    [
      'run',
      '--project',
      '..',
      'python',
      '-c',
      'import sys; from planefuse.io import load_image; f=load_image(sys.argv[1]); assert f.pixels.shape == (64, 80, 3); assert f.bit_depth == 16',
      outFile
    ],
    { stdio: 'inherit' }
  );
});

test('real Zion TIFFs: workspace: add -> PMax -> view -> export', async ({ page }) => {
  test.skip(!existsSync(zionStack), `real stack not available at ${zionStack}`);
  test.setTimeout(10 * 60_000);

  const work = tempWork('fs-zion-e2e-');
  const frames = stageZionSources(work);
  const outFile = join(work, 'zion-pmax.tif');

  // Workspace opens directly.
  await page.goto('/');

  // Add frames.
  await page.getByTestId('ws-add').click();
  await page.getByTestId('scan-path').fill(frames);
  await page.getByTestId('ws-add-confirm').click();
  await expect(page.getByTestId('ws-input').first()).toBeVisible({ timeout: 120_000 });
  await expect(page.getByText('7 frames · 5199×7795 · 16-bit')).toBeVisible();

  // Stack with PMax, assert align param.
  await page.getByTestId('algo-pmax').click();
  await expect(page.getByRole('checkbox', { name: 'Align frames' })).toBeChecked();
  const stackResponse = page.waitForResponse(
    (r) =>
      r.request().method() === 'POST' &&
      r.url().includes('/api/projects/') &&
      r.url().endsWith('/jobs')
  );
  await page.getByTestId('ws-run').click();
  const stackJobResponse = await stackResponse;
  const stackBody = stackJobResponse.request().postDataJSON() as {
    params: { align?: { max_long_edge: number } };
  };
  expect(stackBody.params.align).toMatchObject({
    max_long_edge: 2048,
    model: 'similarity',
    interp: 'lanczos3',
    normalize_brightness: true
  });
  const stackJobId = ((await stackJobResponse.json()) as { id: string }).id;
  await waitForJob(page, stackJobId, 5 * 60_000);

  // Result renders in viewer.
  await expect(page.getByTestId('ws-result').first()).toBeVisible({ timeout: 30_000 });
  await page.getByTestId('ws-result').first().click();
  await expect(page.getByTestId('viewer-tile').first()).toBeVisible({ timeout: 120_000 });

  // Export.
  await page.getByTestId('ws-export').click();
  await page.getByTestId('export-dest').fill(outFile);
  const exportResponse = page.waitForResponse(
    (r) =>
      r.request().method() === 'POST' &&
      r.url().includes('/api/projects/') &&
      r.url().endsWith('/export')
  );
  await page.getByTestId('export-go').click();
  const exportJobId = ((await (await exportResponse).json()) as { id: string }).id;
  await waitForJob(page, exportJobId, 120_000);
  await expect(page.getByTestId('export-done')).toBeVisible();

  expect(statSync(outFile).size).toBeGreaterThan(0);
  execFileSync(
    'uv',
    [
      'run',
      '--project',
      '..',
      'python',
      '-c',
      'import sys; from planefuse.io import load_image; f=load_image(sys.argv[1]); assert f.pixels.shape == (7795, 5199, 3); assert f.bit_depth == 16',
      outFile
    ],
    { stdio: 'inherit' }
  );
});
