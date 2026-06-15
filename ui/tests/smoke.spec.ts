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
  process.env.FOCUSSTACK_REAL_STACK_DIR ??
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

test('import -> PMax -> view -> export', async ({ page }) => {
  const work = tempWork('fs-e2e-');
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
  await expect(page.getByRole('checkbox', { name: 'Align frames' })).toBeChecked();
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

test('retouch result -> paint -> undo/redo -> flatten -> export', async ({ page }) => {
  test.setTimeout(5 * 60_000);
  const work = tempWork('fs-retouch-e2e-');
  const frames = join(work, 'frames');
  const projDir = join(work, 'proj');
  const outFile = join(work, 'retouched.tif');

  execSync(`uv run --project .. python tests/fixtures/make_stack.py "${frames}"`, {
    stdio: 'inherit'
  });

  await page.goto('/');
  await page.getByTestId('new-project').click();
  await page.getByTestId('project-path').fill(projDir);
  await page.getByTestId('project-name').fill('Retouch E2E');
  await page.getByTestId('create-project').click();
  await page.getByTestId('import-frames').click();
  await page.getByTestId('scan-path').fill(frames);
  await page.getByTestId('scan-go').click();
  await expect(page.getByTestId('scan-ok')).toBeVisible();

  for (const method of ['pmax', 'weighted']) {
    await page.getByTestId('nav-stack').click();
    await page.getByTestId(`algo-${method}`).click();
    const response = page.waitForResponse(
      (r) => r.request().method() === 'POST' && r.url().endsWith('/jobs')
    );
    await page.getByTestId('stack-go').click();
    const jobId = ((await (await response).json()) as { id: string }).id;
    await waitForJob(page, jobId);
  }

  await page.getByTestId('nav-viewer').click();
  await expect(page.getByTestId('viewer-tile').first()).toBeVisible();
  await page.getByTestId('retouch-this-result').click();

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

  await expect(page).toHaveURL(/\/viewer$/);
  await expect(page.getByTestId('viewer-tile').first()).toBeVisible();
  const projects = (await page.request.get('/api/projects')).json() as Promise<
    { images: Record<string, { name?: string }> }[]
  >;
  const project = (await projects).find((item) => flattenedId in item.images);
  expect(project?.images[flattenedId].name).toBe('E2E retouched');

  await page.getByTestId('nav-export').click();
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
      'import sys; from focusstack.io import load_image; f=load_image(sys.argv[1]); assert f.pixels.shape == (64, 80, 3); assert f.bit_depth == 16',
      outFile
    ],
    { stdio: 'inherit' }
  );
});

test('real Zion TIFFs: import -> PMax -> view -> export', async ({ page }) => {
  test.skip(!existsSync(zionStack), `real stack not available at ${zionStack}`);
  test.setTimeout(10 * 60_000);

  const work = tempWork('fs-zion-e2e-');
  const frames = stageZionSources(work);
  const projDir = join(work, 'proj');
  const outFile = join(work, 'zion-pmax.tif');

  await page.goto('/');

  await page.getByTestId('new-project').click();
  await page.getByTestId('project-path').fill(projDir);
  await page.getByTestId('project-name').fill('Zion real-data E2E');
  await page.getByTestId('create-project').click();

  await page.getByTestId('import-frames').click();
  await page.getByTestId('scan-path').fill(frames);
  await page.getByTestId('scan-go').click();
  await expect(page.getByTestId('scan-ok')).toBeVisible({ timeout: 120_000 });
  await expect(page.getByText('7 frames · 5199×7795 · 16-bit')).toBeVisible();

  await page.getByTestId('nav-stack').click();
  await page.getByTestId('algo-pmax').click();
  await expect(page.getByRole('checkbox', { name: 'Align frames' })).toBeChecked();
  const stackResponse = page.waitForResponse(
    (response) =>
      response.request().method() === 'POST' &&
      response.url().includes('/api/projects/') &&
      response.url().endsWith('/jobs')
  );
  await page.getByTestId('stack-go').click();
  const stackJobResponse = await stackResponse;
  const stackBody = stackJobResponse.request().postDataJSON() as {
    params: { align?: { max_long_edge: number } };
  };
  expect(stackBody.params.align).toEqual({ max_long_edge: 2048 });
  const stackJobId = ((await stackJobResponse.json()) as { id: string }).id;
  await waitForJob(page, stackJobId, 5 * 60_000);

  await page.getByTestId('nav-viewer').click();
  await expect(page.getByTestId('viewer-tile').first()).toBeVisible({ timeout: 120_000 });

  await page.getByTestId('nav-export').click();
  await page.getByTestId('export-dest').fill(outFile);
  const exportResponse = page.waitForResponse(
    (response) =>
      response.request().method() === 'POST' &&
      response.url().includes('/api/projects/') &&
      response.url().endsWith('/export')
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
      'import sys; from focusstack.io import load_image; f=load_image(sys.argv[1]); assert f.pixels.shape == (7795, 5199, 3); assert f.bit_depth == 16',
      outFile
    ],
    { stdio: 'inherit' }
  );
});
