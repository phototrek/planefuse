# FocusStack M5 Part 2 Retouch UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Viewer-launched Retouch editor and close M5 with a real browser paint, undo/redo, flatten, view, and export workflow.

**Architecture:** Reuse the existing retouch REST API and `DeepZoom` tile canvas. Viewer resolves or creates a session; `/retouch` owns brush state and sends full-image strokes; per-tile revision values refetch only dirty tiles.

**Tech Stack:** Svelte 5, TypeScript, SvelteKit SPA, Playwright, existing FastAPI retouch API.

---

### Task 1: Write The Browser Acceptance Test

**Files:**
- Modify: `ui/tests/smoke.spec.ts`

- [ ] **Step 1: Add a second stack and the complete retouch workflow**

Add a focused `test('retouch result -> paint -> undo/redo -> flatten -> export', ...)`
that creates a synthetic project, stacks `pmax`, stacks `weighted`, opens Viewer,
launches retouch, paints across the center of the canvas, verifies the stroke
payload contains full-image coordinates, undo/redoes, flattens as `E2E retouched`,
and exports the resulting TIFF.

Use response waits for exact stack, stroke, flatten, and export requests. Verify
the output with:

```ts
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
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
npm run test:e2e -- --grep "retouch result"
```

Expected: FAIL because `retouch-this-result` does not exist.

### Task 2: Add Typed API Calls And Viewer Launch

**Files:**
- Modify: `ui/src/lib/api.ts`
- Modify: `ui/src/routes/viewer/+page.svelte`

- [ ] **Step 1: Add retouch types and methods**

Add `RetouchSession`, `RetouchSource`, `RetouchMutation`, and `RetouchStroke`
types plus these client methods:

```ts
listRetouch: (id: string) =>
  req<{ sessions: RetouchSession[] }>('GET', `/api/projects/${id}/retouch`),
createRetouch: (id: string, targetImageId: string) =>
  req<{ session_id: string; working_image_id: string; sources: RetouchSource[] }>(
    'POST',
    `/api/projects/${id}/retouch`,
    { target_image_id: targetImageId }
  ),
retouchStroke: (id: string, stroke: RetouchStroke) =>
  req<RetouchMutation>('POST', `/api/retouch/${id}/stroke`, stroke),
retouchUndo: (id: string) =>
  req<RetouchMutation>('POST', `/api/retouch/${id}/undo`),
retouchRedo: (id: string) =>
  req<RetouchMutation>('POST', `/api/retouch/${id}/redo`),
flattenRetouch: (id: string, name: string) =>
  req<{ image_id: string }>('POST', `/api/retouch/${id}/flatten`, { name })
```

- [ ] **Step 2: Launch or resume from Viewer**

Add a `Retouch this result` button. On click, list sessions, select the last
matching `target_image_id`, otherwise create one, persist
`{ retouchSessionId }`, and `goto('/retouch')`.

- [ ] **Step 3: Run static checks**

Run:

```powershell
npm run check
```

Expected: PASS.

### Task 3: Make DeepZoom Paintable

**Files:**
- Modify: `ui/src/lib/viewer/DeepZoom.svelte`

- [ ] **Step 1: Add optional editing props**

Add:

```ts
tileVersion?: (z: number, x: number, y: number) => number;
brushRadius?: number;
onstroke?: (points: [number, number, number][]) => void;
```

Use `tileVersion` in each tile URL as `?rev=...`.

- [ ] **Step 2: Add brush and temporary pan behavior**

Convert pointer coordinates with:

```ts
const x = (screenX - tx) / scale;
const y = (screenY - ty) / scale;
```

Painting is active when `onstroke` exists and Space is not held. Clamp accepted
points to the image bounds, default pressure to `1`, submit on pointer-up, and
render a pointer-events-none brush circle with diameter
`2 * brushRadius * scale`.

- [ ] **Step 3: Run static checks**

Run `npm run check`.

Expected: PASS.

### Task 4: Build The Retouch Screen And Close M5

**Files:**
- Create: `ui/src/routes/retouch/+page.svelte`
- Modify: `ui/tests/smoke.spec.ts`
- Modify: `README.md`

- [ ] **Step 1: Load the persisted session**

Refresh the project, find `ui_state.retouchSessionId`, locate its session
record, read working-image dimensions from `project.images`, and select the
first available source.

- [ ] **Step 2: Render the minimal editor**

Use one source rail, one toolbar, native range controls, and the existing
`DeepZoom`. Provide test IDs:

```text
retouch-source
retouch-canvas
retouch-undo
retouch-redo
retouch-flatten-name
retouch-flatten
```

Register each source path with the existing viewer endpoint and show its
lowest-resolution tile as its thumbnail.

- [ ] **Step 3: Wire mutations and dirty tile revisions**

Send one stroke per pointer gesture. Keep a map keyed by `z/x/y`; after each
mutation assign the returned `rev` only to returned dirty tiles. Disable
commands while a request is active and preserve the current image on error.

- [ ] **Step 4: Wire shortcuts and flatten**

Support `Ctrl+Z`, `Ctrl+Shift+Z`, `Ctrl+Y`, `[`, `]`, and Space. After flatten,
refresh the project, register the new image path, persist it as `ui_state.view`,
and navigate to Viewer.

- [ ] **Step 5: Run the focused test until GREEN**

Run:

```powershell
npm run test:e2e -- --grep "retouch result"
```

Expected: 1 passed.

- [ ] **Step 6: Mark M5 complete**

Change README status to:

```text
- [x] M5 - retouch engine, server, UI, and browser workflow complete
```

- [ ] **Step 7: Run all completion gates**

```powershell
npm run check
npm run test:e2e
uv run pytest -q
uv run ruff check .
uv run mypy engine/src server/src
git diff --check
```

Expected: all commands exit 0.
