# PlaneFuse Single-Screen Workspace — Design Spec

**Date:** 2026-06-15
**Status:** ✅ Approved by spec-document-reviewer (pass 2); pending user spec review

## 1. Goal & Motivation

Replace the multi-screen wizard (`/` import → `/stack` → `/viewer` → `/export`,
plus `/queue`) with a single **workspace** screen that holds the whole core loop:
add inputs → pick algorithm → run → watch progress → view results → export. No
explicit "create a project" step; a project is created automatically and only
named when the user chooses to **Save**. The retouch editor (`/retouch`) stays a
separate mode launched from a result.

## 2. Decisions (settled during brainstorming)

| Topic | Decision |
|-------|----------|
| Ingest | **Reference by path** (no upload/copy). Inputs are absolute server-side paths. |
| Drag-drop | Literal OS drag-drop **deferred** to a future desktop (Electron/Tauri) shell — a browser can't expose real paths. "Add" today = server-side folder browser / file picker. |
| Project | **Auto scratch, persist on Save.** App opens into an auto-created "Untitled" project under the server data dir; Save names it and marks it kept. Unsaved scratch projects are pruned later. |
| Export | Inline toolbar action with an options popover (no separate screen). |
| Retouch | Stays its own full-screen mode at `/retouch`, launched from a selected result. |
| Grouping | Optional **Auto-group** toolbar action → one queued render per bracketed group (multiple results into the drawer). Single Run (no auto-group) stacks all inputs as one group. |
| Code structure | Rewrite `/` into the workspace; decompose into components under `lib/workspace/`; delete `/stack`, `/viewer`, `/export`, `/queue`; keep `/retouch`. |

## 3. Layout

```
┌─────────────────────────────────────────────────────────────┐
│ [Untitled ▸ Save]  Algo:[PMax ▾] [Options…] [Run] [Auto-grp]  │
│                                       [Export]   device:cuda  │
├───────────────┬─────────────────────────────────────────────┤
│ INPUTS        │                                             │
│ [+ Add…]      │              VIEWER (DeepZoom)              │
│ ▢ img001.tif  │       (selected input, result, or          │
│ ▢ img002.tif  │        a running job's progress)           │
│ … 24 frames   │                                             │
├───────────────┴─────────────────────────────────────────────┤
│ RENDERS ▾(hide)  [▣ PMax ✓] [▣ DMap ⟳ 62%] [▣ Weighted ✓]   │
└─────────────────────────────────────────────────────────────┘
```

Three regions in a CSS grid: left input rail (fixed width), right viewer (fills),
bottom render drawer (collapsible). Toolbar spans the top.

## 4. Backend changes (Plan Part 1)

The server stays project-centric; we make a project effortless to obtain and add
the frame-management verbs the additive UI needs. All changes are in
`server/src/planefuse_server/`.

### 4.1 Scratch project
- `ProjectStore.create(directory, name)` accepts `directory=None` → server creates
  the project under `data_dir/scratch/<id>/` (with `cache/`). Saved projects and
  user-chosen directories are unaffected.
- New projects set `ui_state.saved = False`.
- `POST /api/projects` accepts an optional/empty `path`; when empty, the scratch
  path is used. Client calls this once on load if it has no current project.

### 4.2 Save / rename
- New `PATCH /api/projects/{id}` body `{ name?, saved? }`. `name` is the
  **top-level** `Project.name` field; `saved` is stored in **`ui_state.saved`**
  (consistent with §4.1) — the route sets `proj.name` and/or
  `proj.ui_state["saved"]` accordingly and persists. (This is distinct from the
  existing `PATCH /api/projects/{id}/ui-state`, which only merges `ui_state`.)
- Directory move is **out of scope**; saved scratch projects keep living under
  data_dir — they are simply retained (prune skips them).
- Add `api.saveProject(id, { name, saved })` to the client.

### 4.3 Additive frame management
- `POST /api/projects/{id}/frames/add` body `{ paths: [...] }`: each path may be a
  directory (expanded to its images via the existing `_scan_paths`) or an
  individual image file. Append to `proj.frames`, de-duplicate preserving order,
  validate via `validate_stack`, persist, and return the same report shape as
  `scan` (`{ ok, width, height, bit_depth, files }`) computed over the **full**
  frame list.
- `POST /api/projects/{id}/frames/remove` body `{ paths: [...] }`: drop those
  paths, persist, return the updated report.
- `scan` is kept but reimplemented to call the same core (replace semantics) so
  existing tests and any callers keep working.
- Known cost: re-validating the **full** frame list on every add re-reads every
  frame (same as `scan` does today). Acceptable for now; revisit with per-frame
  caching if large stacks feel slow.

### 4.4 Prune unsaved scratch
- On server startup (`create_app`), delete scratch project directories under
  `data_dir/scratch/` whose `project.json` has `ui_state.saved` falsy **and**
  whose **`project.json` mtime** (deterministic across platforms, unlike dir
  mtime) is older than 7 days; unregister them. Never touches dirs outside
  `data_dir/scratch/`. Bounded, best-effort, logged.

## 5. Client state (`stores.svelte.ts`)

`appState` extends to:
- `project: Project | null` — the current (scratch or saved) project.
- `inputs: FileStatus[]` — current frame list with per-frame status (from the
  add/remove report).
- `results: { id, label, method, thumb, path }[]` — **derived** from
  `project.images` (single source of truth: the project), enriched with the
  registered `image_id` + tile-0 `thumb` once each is registered. Not stored
  independently of the project.
- `jobs: Record<string, Job>` — live job state, updated over the WebSocket.
- `viewerTarget: { kind: 'input' | 'result' | 'job', id: string } | null`.
- `drawerOpen: boolean`.

Selection drives the viewer; the viewer is stateless about *what* it shows beyond
the target descriptor.

## 6. Components (`lib/workspace/`)

| Component | Responsibility | Depends on |
|-----------|----------------|------------|
| `Toolbar.svelte` | Project name + Save; algorithm `<select>`; Options popover (renders `ParamSpec[]` from `/api/algorithms`); Run; Auto-group; Export popover; device badge. Emits run/export/save intents. | `api`, `appState` |
| `InputList.svelte` | "Add…" (opens `FolderBrowser`); frame rows with status; remove; click selects into viewer. | `api`, `FolderBrowser`, `appState` |

**Add — folder and individual files:** the backend `frames/add` already accepts a
mix of folder and file paths (§4.3). The current `FolderBrowser` only *picks a
folder*. Part 2 enhances it so individual image files in the listing
(`/api/fs/list` already returns file entries) are selectable (checkbox per file
row) and returns the chosen path set — satisfying "a bunch of images **or** a
folder." Picking a folder still adds all its images.
| `ViewerPane.svelte` | Renders `DeepZoom` for the current target; for a `job` target shows progress + params/info instead. | `DeepZoom`, `api`, `appState` |
| `RenderDrawer.svelte` | Collapsible. Thumbnails of results + chips for in-flight jobs with progress bars. Click result → viewer; click running job → viewer progress. Per-result actions: Export, Retouch (→ `/retouch`). | `api`, `appState` |

**Export popover (in `Toolbar`):** because the dedicated export screen is deleted
and there is no native save dialog, the popover must still let the user pick the
**output folder** via `FolderBrowser` plus a filename, then build the `dest`
absolute path the existing `POST /{id}/export` requires. Format / bit-depth /
JPEG-quality controls sit alongside.

**Result thumbnails (in `RenderDrawer`):** `frame-thumb` only serves paths in
`proj.frames`; results live in `proj.images` and are **not** frames. So a result
thumbnail is its lowest-resolution registered tile — `registerView(path)` then
`tileUrl(image_id, 0, 0, 0)` — the same trick the `/retouch` source rail already
uses. Register each result's path once when it first appears.

`+page.svelte` is the shell: the grid, mounts the four components, ensures a
scratch project on mount, and opens the WebSocket to feed `appState.jobs`.

## 7. Data flow

1. **Mount:** ensure scratch project (`POST /api/projects` empty path if none);
   open `/ws`.
2. **Add:** `InputList` → `frames/add` → updates `inputs`; first add auto-selects
   a frame into the viewer.
3. **Run:** `Toolbar` enqueues a stack job (`enqueueStack`) with chosen
   algorithm + options. **Auto-group:** `auto-group` then one `enqueueStack` per
   group.
4. **Progress:** WS events update `appState.jobs`; `RenderDrawer` shows bars; a
   `job`-target viewer shows live progress.
5. **Done:** the WS feed carries status/percent but **deliberately omits the job
   `result`** (`jobs.py::_emit`; `ws.ts` preserves the prior `result`). So on a
   `status: done` event the client **re-fetches the project** (`api.getProject`)
   to pick up the newly added `project.images` entry, registers its path
   (`registerView`), shows its tile-0 thumbnail in the drawer, and auto-selects
   it into the viewer. (Equivalently it may `getJob(jid)` for `result.image_id`;
   the spec mandates re-fetch as the source of truth since `images` is keyed
   there.) Listening to WS alone is insufficient and must not be the mechanism.
6. **Export:** popover (folder via `FolderBrowser` + filename → `dest`, plus
   format/bit-depth) → `export` writes the file (existing endpoint).
7. **Retouch:** result action → persist `retouchSessionId` (existing create flow)
   → `goto('/retouch')`.
8. **Save:** `Toolbar` → `PATCH /projects/{id}` with name + `saved=true`.

## 8. Error handling

- Per-frame validation reflected in `InputList` (invalid frames flagged, mirroring
  today's scan report).
- Job failures: error chip in `RenderDrawer`, detail in `ViewerPane` when that job
  is the target.
- Add/remove/save failures: inline message near the toolbar/list; state unchanged
  on error.
- Viewer keeps the last good image on transient tile/registration errors (same
  pattern as the current `/retouch` screen).

## 9. Testing

**Backend (pytest):**
- scratch create (empty path) lands under `data_dir/scratch/` with `saved=false`.
- `frames/add` appends + dedupes folders and individual files; report covers the
  full list. `frames/remove` drops paths.
- `PATCH` rename sets name + `saved=true`.
- prune deletes only old unsaved scratch dirs; keeps saved + recent.

**UI (Playwright):** rewrite `smoke.spec.ts` into one workspace flow — add frames
→ Run PMax → drawer progress → **result becomes selectable and renders in the
viewer** (explicitly covers the §7-step-5 re-fetch path) → Export TIFF (verify
output dims/bit-depth as today). The existing retouch e2e continues to launch
from a result.

## 10. Out of scope / deferred

- Literal OS drag-drop and any file upload (future desktop shell).
- Moving a saved project's files out of the data dir.
- A manual group editor (Auto-group is one-shot; no per-group reassignment UI).
- Multi-window / comparison view, histogram, export naming templates (these are
  the separate M6 "polish" items).

## 11. Plan split

- **Part 1 — backend:** scratch project, frames add/remove, save/rename, prune,
  with pytest. No UI change; existing screens keep working.
- **Part 2 — UI:** the workspace screen, components, state, route deletions, and
  the reworked Playwright flow.
