# Workspace UI — Part 2 (Frontend) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking. Validate every `.svelte` file you write/modify with the Svelte MCP `svelte-autofixer` tool before committing.

**Goal:** Replace the five-screen wizard with one workspace screen at `/` — left input list, right viewer, bottom render drawer, top toolbar — backed by an auto scratch project, additive frames, inline export, and a separate launched retouch mode.

**Architecture:** `/+page.svelte` becomes a 3-region grid shell composed of four focused components under `lib/workspace/`, reusing the existing `DeepZoom`, `FolderBrowser`, `ParamForm`, `ProgressBar`, and the `ws.ts` job feed. Backend Part 1 (scratch projects, `frames/add`/`remove`, `PATCH /projects/{id}`) is already merged on this branch and is consumed here. The deleted routes' logic is lifted into the components. Spec: `docs/superpowers/specs/2026-06-15-workspace-ui-design.md` (esp. §3–§8).

**Tech Stack:** Svelte 5 (runes: `$state`/`$derived`/`$props`/`$bindable`), TypeScript, SvelteKit SPA (`adapter-static`), Playwright.

**Conventions to follow (do not reinvent):**
- Match the existing CSS-variable design system (`--panel`, `--line`, `--accent`, `--text-dim`, `--radius`, `--font-mono`, etc. from `app.css`) and the panel/button styling used in the screens being deleted. Lift styles from those screens rather than inventing new ones.
- Errors: inline message, state unchanged on failure (same pattern as `routes/retouch/+page.svelte`).
- All server mutations go through `$lib/api`. All job state comes from `appState.jobs` (fed by `ws.ts`), never a second socket.

---

## Chunk 1: Foundation (state, API, layout, acceptance test)

### Task 1: API client methods + types

**Files:**
- Modify: `ui/src/lib/api.ts`

- [ ] **Step 1: Add types and methods**

Add to the `api` object (mirror the existing `req<T>` style):

```ts
createScratchProject: () => req<Project>('POST', '/api/projects', {}),
saveProject: (id: string, body: { name?: string; saved?: boolean }) =>
  req<{ id: string; name: string; ui_state: Record<string, unknown> }>(
    'PATCH', `/api/projects/${id}`, body),
addFrames: (id: string, paths: string[]) =>
  req<ScanReport>('POST', `/api/projects/${id}/frames/add`, { paths }),
removeFrames: (id: string, paths: string[]) =>
  req<ScanReport>('POST', `/api/projects/${id}/frames/remove`, { paths }),
```

`ScanReport` already exists and is the shape both endpoints return. Also add an `fsList`-based helper is NOT needed — `FolderBrowser` already calls `api.fsList`.

- [ ] **Step 2: Static check**

Run: `cd ui && npm run check`
Expected: PASS (no type errors; methods unused yet is fine).

- [ ] **Step 3: Commit**

```bash
git add ui/src/lib/api.ts
git commit -m "feat(ui): api client methods for scratch project + frame add/remove"
```

### Task 2: Workspace state + scratch bootstrap

**Files:**
- Modify: `ui/src/lib/stores.svelte.ts`

- [ ] **Step 1: Extend appState**

Replace the store with the workspace shape (keep `system`, `project`, `jobs`):

```ts
import type { FileStatus, Job, Project, SystemInfo } from './api';

export interface WorkspaceResult {
  id: string;          // image_id key in project.images
  label: string;       // name or method
  method: string;
  path: string;
  imageId?: string;    // registered viewer image id (lazy)
  thumb?: string;      // tile-0 url (lazy)
  levels?: number;
  width?: number;
  height?: number;
}

export type ViewerTarget =
  | { kind: 'input'; path: string }
  | { kind: 'result'; id: string }
  | { kind: 'job'; id: string }
  | null;

export const appState = $state<{
  system: SystemInfo | null;
  project: Project | null;
  jobs: Record<string, Job>;
  inputs: FileStatus[];
  results: WorkspaceResult[];
  viewer: ViewerTarget;
  drawerOpen: boolean;
  stackJobIds: string[];      // enqueued stack jobs to watch for result discovery
  exportJobId: string | null; // current export job, for the export-done indicator
}>({
  system: null,
  project: null,
  jobs: {},
  inputs: [],
  results: [],
  viewer: null,
  drawerOpen: true,
  stackJobIds: [],
  exportJobId: null
});
```

- [ ] **Step 2: Static check**

Run: `cd ui && npm run check`
Expected: FAIL — the old `routes/+page.svelte`, `stack`, `viewer`, `export`, `queue` still reference removed shapes / will be deleted in Chunk 2. **This is expected;** record the failures. (If `npm run check` errors are ONLY in files slated for deletion in Task 8, proceed. Otherwise fix.) Note: `routes/retouch/+page.svelte` is KEPT and only reads `appState.project`/`jobs` — it must stay green throughout; do not "fix" or delete it.

- [ ] **Step 3: Commit**

```bash
git add ui/src/lib/stores.svelte.ts
git commit -m "feat(ui): workspace app state (inputs, results, viewer target, drawer)"
```

### Task 3: Slim the layout shell (remove the wizard nav rail)

**Files:**
- Modify: `ui/src/routes/+layout.svelte`

- [ ] **Step 1: Remove the rail + reduce to a single-column shell**

Delete the `NAV` array, the `<nav class="rail">…</nav>` block, the `.rail*`/`.nav*`/`.ico*` styles and the icon mask `--m-*` definitions. Change the `.app` grid to a single content column under a full-width topbar:

```css
.app { display: grid; grid-template-rows: var(--topbar-h) 1fr; height: 100vh; }
.topbar { display: grid; grid-template-columns: auto 1fr auto; align-items: center;
          border-bottom: 1px solid var(--line);
          background: linear-gradient(180deg, var(--panel-2), var(--panel)); }
.main { overflow: hidden; }   /* workspace manages its own scrolling */
```

Keep the brand, the `project` block (shows `appState.project?.name`), and the `device` badge exactly as-is. Keep `onMount` fetching `appState.system` and calling `connectJobs()`. Remove the `$page` import if now unused.

- [ ] **Step 2: Static check**

Run: `cd ui && npm run check`
Expected: same as Task 2 (only deletion-pending files fail). Layout itself must be clean.

- [ ] **Step 3: Commit**

```bash
git add ui/src/routes/+layout.svelte
git commit -m "feat(ui): slim layout to single-screen shell (drop wizard rail)"
```

### Task 4: Write the workspace acceptance e2e (RED)

**Files:**
- Modify: `ui/tests/smoke.spec.ts`

- [ ] **Step 1: Rewrite the three tests to the workspace flow**

The workspace test-id contract (components in Chunk 2 must honor these):

| test-id | element |
|---------|---------|
| `ws-add` | InputList "Add…" button (opens FolderBrowser) |
| `scan-path` | path input inside the add browser (reuse existing id) |
| `ws-add-confirm` | confirm/add button in the add browser |
| `ws-input` | an input frame row |
| `algo-{name}` | algorithm select buttons (reuse existing ids) |
| `ws-run` | Run button |
| `ws-autogroup` | Auto-group button |
| `ws-result` | a result thumbnail in the drawer |
| `viewer-tile` | a rendered DeepZoom tile (already emitted by DeepZoom) |
| `retouch-this-result` | per-result Retouch action (keep id so retouch flow is stable) |
| `ws-export` | Export toolbar button (opens popover) |
| `export-dest` | full output path input (reuse existing id) |
| `export-go` | Export confirm (reuse) |
| `export-done` | export success indicator (reuse) |
| `ws-save` / `ws-project-name` | Save button / project name field |

Rewrite `test('workspace: add -> PMax -> view -> export', …)` to:
1. `await page.goto('/')` — workspace opens directly (auto scratch project; no create step).
2. Generate frames via the existing `execSync(... make_stack.py ...)` helper.
3. Add frames: click `ws-add`, fill `scan-path` with the frames folder, click `ws-add-confirm`; assert an `ws-input` row is visible (count matches).
4. Stack: click `algo-pmax`, capture the `POST …/jobs` response to get `jobId`, click `ws-run`, `await waitForJob(page, jobId)`.
5. Result + viewer: assert a `ws-result` thumb appears; click it; assert `viewer-tile` renders (`{ timeout: 30_000 }`).
6. Export: click `ws-export`, fill `export-dest` with `outFile`, capture `POST …/export` for its jobId, click `export-go`, `await waitForJob`, assert `export-done` visible. Keep the `load_image` dim/bit-depth assertion.

Update `test('retouch result -> paint -> …')` and `test('real Zion TIFFs …')` the same way: replace the create-project + `nav-stack`/`nav-viewer`/`nav-export` steps with the workspace add/run/select/export steps above. For retouch, after the result renders, click `retouch-this-result` (now a per-result action) and keep the rest of the retouch assertions unchanged. The Zion test keeps its `test.skip` guard and the `7 frames · 5199×7795 · 16-bit` report assertion (the report text still comes from `addFrames`).

Keep `waitForJob`, `tempWork`, `stageZionSources`, and the cleanup harness as-is.

- [ ] **Step 2: Run to verify RED**

Run: `cd ui && npm run test:e2e -- --grep "workspace:"`
Expected: FAIL (no `ws-add` etc. — workspace not built yet).

- [ ] **Step 3: Commit**

```bash
git add ui/tests/smoke.spec.ts
git commit -m "test(ui): workspace acceptance e2e (red)"
```

---

## Chunk 2: Components, shell, route deletion (GREEN)

> Build order: leaf components first, then the shell that composes them, then delete dead routes, then drive the e2e to green. Validate each `.svelte` with the Svelte MCP `svelte-autofixer` before committing.

### Task 5: FolderBrowser — allow selecting individual files

**Files:**
- Modify: `ui/src/lib/components/FolderBrowser.svelte`

- [ ] **Step 1: Add an optional multi-file selection mode**

Add a prop `selectFiles = false` and, when true, render the listing's **file** entries (not just dirs) each with a checkbox, tracking a bindable `selected: string[]` of file paths. `/api/fs/list` already returns file entries (`is_dir === false`); filter them the same way `dirs` is derived. When `selectFiles` is false the component behaves exactly as today (folder navigation only). Keep `value` (the current folder) and the `Browse`/`up`/`into` behavior.

Interface:
```ts
let { value = $bindable(''), selected = $bindable<string[]>([]),
      selectFiles = false, placeholder = '…', inputTestid = '' } = $props();
```

- [ ] **Step 2: Static check + autofix**

Run the Svelte autofixer MCP on the file, then `cd ui && npm run check`. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add ui/src/lib/components/FolderBrowser.svelte
git commit -m "feat(ui): FolderBrowser optional individual-file selection"
```

### Task 6: InputList component

**Files:**
- Create: `ui/src/lib/workspace/InputList.svelte`

**Responsibility:** show `appState.inputs`; add frames (a folder and/or individual files via `FolderBrowser` with `selectFiles`); remove a frame; clicking a row sets `appState.viewer = { kind:'input', path }`. Reflect per-frame validation status (color the row when `status !== 'ok'`, mirroring the old scan report styling).

- [ ] **Step 1: Implement**

Behavior:
- "Add…" (`data-testid="ws-add"`) toggles an add panel containing `<FolderBrowser bind:value={folder} bind:selected={files} selectFiles inputTestid="scan-path" />` (the `inputTestid` is required — the e2e fills `scan-path`) and a confirm button (`data-testid="ws-add-confirm"`).
- Confirm calls `api.addFrames(appState.project.id, paths)` where `paths = files.length ? files : [folder]` (folder if no individual files picked), then sets `appState.inputs = report.files` and `appState.project = await api.getProject(id)`. On first add, auto-select the first input into the viewer.
- Each row: `data-testid="ws-input"`, shows file name + status; a remove "✕" calls `api.removeFrames(id, [path])` and updates `appState.inputs` from the returned report.
- Errors inline; inputs unchanged on failure.

- [ ] **Step 2: Autofix + check**

Svelte autofixer MCP, then `cd ui && npm run check`. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add ui/src/lib/workspace/InputList.svelte
git commit -m "feat(ui): InputList (add folder/files, remove, select, status)"
```

### Task 7: Toolbar component

**Files:**
- Create: `ui/src/lib/workspace/Toolbar.svelte`

**Responsibility:** algorithm choice + options, Run, Auto-group, Export popover, project name + Save, plus reused align/select toggles. Lift the algorithm/params/preset/align/select logic and `buildParams()` from the deleted `routes/stack/+page.svelte`, and the export-popover logic (format/bit-depth/quality + `FolderBrowser` folder + filename → `dest`) from the deleted `routes/export/+page.svelte`.

- [ ] **Step 1: Implement**

- On mount: `api.algorithms()` and `api.listPresets()` (same as stack screen). `current` derived from selected `method`.
- Algorithm buttons reuse `data-testid="algo-{name}"`. Options behind a toggle using `<ParamForm specs={current.params} bind:values />`. Keep align (`max_long_edge`) + smart-select toggles and `buildParams()` verbatim from the stack screen.
- **Run** (`data-testid="ws-run"`): `api.enqueueStack(project.id, buildParams())`; set `appState.viewer = { kind:'job', id }` for the new job; ensure `appState.drawerOpen = true`.
- **Auto-group** (`data-testid="ws-autogroup"`): `const { groups } = await api.autoGroup(project.id)` then one `api.enqueueStack(project.id, { ...buildParams(), frames: g })` per group. Use the returned `groups` directly — do NOT lift the old `stackAllGroups`' dependency on `appState.project.ui_state.groups` (nothing populates that in the new flow). Track each enqueued job id as a stack job (see Task 8 result-discovery).
- **Export** (`data-testid="ws-export"`) toggles a popover: `<FolderBrowser bind:value={folder}/>` + filename + format/bit-depth/quality; full path input `data-testid="export-dest"`; `data-testid="export-go"` calls `api.export`. Export targets the currently selected result (`appState.viewer.kind==='result'`) else the latest result; disable with a hint if none. (The job result/done indicator `export-done` lives in RenderDrawer/Task 8 wiring — see note.)
- **Save** (`data-testid="ws-save"`): `api.saveProject(project.id, { name, saved:true })`; project name editable via `data-testid="ws-project-name"` bound to a local that defaults to `appState.project.name`.
- Device badge already lives in the layout topbar — do NOT duplicate it here.

- [ ] **Step 2: Autofix + check**

Svelte autofixer MCP, then `cd ui && npm run check`. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add ui/src/lib/workspace/Toolbar.svelte
git commit -m "feat(ui): Toolbar (algo/options/run/auto-group/export/save)"
```

### Task 8: ViewerPane + RenderDrawer + workspace shell; delete old routes

**Files:**
- Create: `ui/src/lib/workspace/ViewerPane.svelte`
- Create: `ui/src/lib/workspace/RenderDrawer.svelte`
- Rewrite: `ui/src/routes/+page.svelte`
- Delete: `ui/src/routes/stack/`, `ui/src/routes/viewer/`, `ui/src/routes/export/`, `ui/src/routes/queue/`

- [ ] **Step 1: ViewerPane**

Renders based on `appState.viewer`. **Note `registerView` takes `(projectId, path)`** — always pass `appState.project.id` first (see `viewer/+page.svelte:48`); `registerView(path)` alone is wrong.
- `kind:'result'`: ensure the result is registered (lazy `api.registerView(appState.project.id, path)` → store `imageId/levels/width/height` on the `WorkspaceResult`), then `<DeepZoom …/>`. Reuse the cached-`view` + `patchUiState({view})` persistence from the deleted viewer screen so reopening restores it.
- `kind:'input'`: register + show the input frame the same way via `registerView(appState.project.id, path)`. (`/viewer/register` accepts any path; this is best-effort and not covered by the e2e, which only asserts `viewer-tile` after selecting a result. If registration of a raw frame path fails, show the error inline and keep the prior image.)
- `kind:'job'`: show the job's `ProgressBar` + message/params (no DeepZoom) — lift the look from the queue screen's job card.
- `null`: a muted empty state ("Add frames and Run to see a result").
DeepZoom already emits `data-testid="viewer-tile"` per tile — keep using DeepZoom unchanged.

- [ ] **Step 2: RenderDrawer**

Collapsible strip (`data-testid="ws-drawer-toggle"`, honor `appState.drawerOpen`). Shows:
- Result thumbnails (`data-testid="ws-result"`) — thumbnail = lazily `api.registerView(appState.project.id, path)` then `api.tileUrl(reg.image_id, 0, 0, 0)` (NOT `frame-thumb` — results are not frames; and `registerView` needs the project id first). Note `WorkspaceResult.id` is the `project.images` key used for selection, while `imageId` (the `registerView` return) is what `tileUrl` needs — they differ. Click → `appState.viewer = { kind:'result', id }`. Per-result actions: **Retouch** (`data-testid="retouch-this-result"`) lifting the create-or-resume-session + `goto('/retouch')` logic from the deleted viewer screen.
- In-flight jobs (`data-testid="ws-job"`) from `appState.jobs` (active = pending/running) with `<ProgressBar/>`; click → `appState.viewer = { kind:'job', id }`.
- Export-done indicator: render `data-testid="export-done"` here when the export job (tracked by id from Toolbar) reaches `status==='done'` (lift from the export screen's status block). Use a shared piece of state — see Step 3.

- [ ] **Step 3: Workspace shell `+page.svelte`**

The grid shell (toolbar row; `InputList | ViewerPane`; `RenderDrawer`):
- On mount: if `!appState.project`, `appState.project = await api.createScratchProject()`; hydrate `appState.inputs` from `project.frames` (call `addFrames(id, [])`? No — instead `GET` project and build a report). Simplest: on mount, if `project.frames` non-empty, set inputs by calling a read: reuse `api.addFrames(id, [])` returns the full report with empty add (no-op append) — acceptable and documented. Then derive `appState.results` from `project.images` (entries with `kind==='result'`).
- **Result discovery (spec §7.5):** the Toolbar Run/Auto-group records each enqueued **stack** job id into a tracked set on `appState` (add `stackJobIds: string[]` to the store). A `$effect` watches `appState.jobs`; when a job whose id is in `stackJobIds` flips to `done`, `appState.project = await api.getProject(id)`, rebuild `appState.results` from `project.images`, and register + auto-select the newest result (`appState.viewer = { kind:'result', id }`). Tracking the stack-job set (separate from the export job) prevents an export `done` event from triggering a needless re-fetch/auto-select. Do NOT read the result off the WS event (it omits it).
- Share the export-job id between Toolbar and RenderDrawer via `appState` (add `exportJobId: string | null`) so `export-done` can render. Add both `stackJobIds: string[]` (init `[]`) and `exportJobId: string | null` (init `null`) to the Task 2 store interface — keep it minimal.
- Layout CSS: grid `grid-template-rows: auto 1fr auto; grid-template-columns: 240px 1fr;` with the toolbar and drawer spanning both columns; mirror panel styling.

- [ ] **Step 4: Delete the dead routes**

```bash
git rm -r ui/src/routes/stack ui/src/routes/viewer ui/src/routes/export ui/src/routes/queue
```

- [ ] **Step 5: Autofix + static check (now must be fully clean)**

Run the Svelte autofixer MCP on the three new/rewritten `.svelte` files, then:
Run: `cd ui && npm run check`
Expected: PASS with zero errors (all references to deleted routes/old state gone).

- [ ] **Step 6: Commit**

```bash
git add ui/src/lib/workspace ui/src/routes/+page.svelte
git commit -m "feat(ui): ViewerPane, RenderDrawer, workspace shell; remove wizard routes"
```

### Task 9: Drive the acceptance e2e to GREEN + full gate

- [ ] **Step 1: Run the workspace e2e until green**

Run: `cd ui && npm run test:e2e -- --grep "workspace:"`
Expected: 1 passed. Fix component/test-id mismatches until green. (Iterate on data-testids and the result-discovery effect — these are the usual culprits.)

- [ ] **Step 2: Run the full e2e suite**

Run: `cd ui && npm run test:e2e`
Expected: synthetic workspace + retouch tests pass; the Zion test passes if the real stack is present, else skips.

- [ ] **Step 3: Full static + build gate**

```bash
cd ui && npm run check && npm run build
cd .. && uv run pytest -q && uv run ruff check . && uv run mypy engine/src server/src
git diff --check
```
Expected: all exit 0.

- [ ] **Step 4: Commit any fixups**

```bash
git commit -am "fix(ui): workspace e2e green + gate"   # only if needed
```

- [ ] **Step 5: Update README screen list**

In `README.md`, replace the "Web UI" wizard description with a one-paragraph note that the UI is a single workspace screen (add inputs → choose algorithm → Run → results in the drawer → view → export; retouch launches from a result). Keep the `serve`/dev commands. Commit:

```bash
git add README.md && git commit -m "docs: describe single-screen workspace UI"
```
