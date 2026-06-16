# Export Naming Templates Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the export form build the output filename from SPEC §130 tokens (`{stack_name}`, `{method}`, `{frames}`, `{date}`, `{seq}`) with a live preview, replacing the plain stem field.

**Architecture:** Token substitution is client-side — a pure `applyTemplate` helper does string replacement; the UI already assembles the `dest` path it POSTs. One server line records the frame count so `{frames}` is reachable. Chosen template persists in `ui_state`.

**Tech Stack:** Python/FastAPI/pytest (server), Svelte 5 runes/TypeScript/Playwright (UI). No new dependencies.

**Read first:** `docs/superpowers/specs/2026-06-16-export-naming-templates-design.md`.

**Branch:** `feat/export-templates` (already created, design committed).

---

## Chunk 1: Implementation

### Task 1: Record frame count in stack result metadata

The stack runner records `{kind, path, method}` for each result; add `frames` so the UI can resolve `{frames}`. Pure pytest TDD.

**Files:**
- Modify: `server/src/focusstack_server/runners.py:34`
- Test: `tests/server/test_runners.py` (create if absent; else append)

- [ ] **Step 1: Write the failing test** — assert a stack result records its frame count.

```python
# tests/server/test_runners.py
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from focusstack.io import save_image
from focusstack_server.main import create_app


def _proj_with_frames(tmp_path, n=3):
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    pid = c.post("/api/projects", json={"path": str(tmp_path / "proj"), "name": "P"}).json()["id"]
    src = tmp_path / "frames"
    src.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        save_image(rng.uniform(0, 1, (32, 40, 3)).astype(np.float32),
                   src / f"f_{i:03d}.tif", bit_depth=16)
    c.post(f"/api/projects/{pid}/frames/scan", json={"path": str(src)})
    return c, pid


def test_stack_result_records_frame_count(tmp_path):
    c, pid = _proj_with_frames(tmp_path, n=3)
    jid = c.post(f"/api/projects/{pid}/jobs",
                 json={"type": "stack", "params": {"method": "pmax", "device": "cpu"}}).json()["id"]
    # Drain the job queue synchronously.
    for _ in range(200):
        job = c.get(f"/api/jobs/{jid}").json()
        if job["status"] in ("done", "error"):
            break
    assert job["status"] == "done", job
    proj = c.get(f"/api/projects/{pid}").json()
    results = [img for img in proj["images"].values() if img.get("kind") == "result"]
    assert results, proj["images"]
    assert results[0]["frames"] == 3
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/server/test_runners.py::test_stack_result_records_frame_count -v` → FAIL (KeyError `frames`).

- [ ] **Step 3: Implement** — `server/src/focusstack_server/runners.py:34`, add `frames` to the recorded dict:

```python
        proj.images[image_id] = {"kind": "result", "path": str(out_path),
                                 "method": method, "frames": len(paths)}
```

(`paths` is already computed at the top of the runner: `paths = [Path(p) for p in (params.get("frames") or proj.frames)]`.)

- [ ] **Step 4: Run to verify it passes** — `uv run pytest tests/server/test_runners.py -v` → PASS.

- [ ] **Step 5: Server gates** — `uv run pytest tests/server -q` → green; `uv run ruff check server tests/server` → clean; `uv run mypy server/src` → Success.

- [ ] **Step 6: Commit**

```bash
git add server/src/focusstack_server/runners.py tests/server/test_runners.py
git commit -m "feat: record frame count in stack result metadata for {frames} token"
```
End commit body with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

### Task 2: `applyTemplate` pure helper + unit test

A pure function that substitutes the five tokens and sanitizes the result. Tested with Playwright (the test runner already present — no new framework).

**Files:**
- Create: `ui/src/lib/workspace/template.ts`
- Create: `ui/tests/template.spec.ts`

- [ ] **Step 1: Write the failing test** — `ui/tests/template.spec.ts`:

```typescript
import { test, expect } from '@playwright/test';
import { applyTemplate } from '../src/lib/workspace/template';

const ctx = { stack_name: 'Zion', method: 'pmax', frames: 12, date: '2026-06-16', seq: 1 };

test('applyTemplate: substitutes known tokens', () => {
  expect(applyTemplate('{stack_name}_{method}', ctx)).toBe('Zion_pmax');
  expect(applyTemplate('{stack_name}_{method}_{frames}f_{date}', ctx)).toBe('Zion_pmax_12f_2026-06-16');
});

test('applyTemplate: pads seq to 3 digits', () => {
  expect(applyTemplate('shot_{seq}', ctx)).toBe('shot_001');
});

test('applyTemplate: leaves unknown tokens literal', () => {
  expect(applyTemplate('{stack_name}_{bogus}', ctx)).toBe('Zion_{bogus}');
});

test('applyTemplate: sanitizes filesystem-illegal characters', () => {
  expect(applyTemplate('{stack_name}', { ...ctx, stack_name: 'a/b:c*d' })).toBe('a_b_c_d');
});

test('applyTemplate: missing frames resolves to empty', () => {
  expect(applyTemplate('{stack_name}{frames}', { ...ctx, frames: undefined })).toBe('Zion');
});
```

- [ ] **Step 2: Run to verify it fails** — `cd ui && npm run test:e2e -- --grep applyTemplate` → FAIL (module not found). Note: the webServer in `playwright.config.ts` still starts; that's fine.

- [ ] **Step 3: Implement** — `ui/src/lib/workspace/template.ts`:

```typescript
// Client-side filename templating for export (SPEC §130).
// ponytail: 5 fixed tokens, plain string replace — no template-engine dep.
export interface TemplateCtx {
  stack_name: string;
  method: string;
  frames: number | undefined;
  date: string;   // YYYY-MM-DD
  seq: number;    // 1-based
}

const ILLEGAL = /[\\/:*?"<>|]/g;

export function applyTemplate(template: string, ctx: TemplateCtx): string {
  const values: Record<string, string> = {
    stack_name: ctx.stack_name,
    method: ctx.method,
    frames: ctx.frames === undefined ? '' : String(ctx.frames),
    date: ctx.date,
    seq: String(ctx.seq).padStart(3, '0')
  };
  const out = template.replace(/\{(\w+)\}/g, (m, key) =>
    key in values ? values[key] : m // unknown token stays literal
  );
  return out.replace(ILLEGAL, '_');
}

export function todayISO(): string {
  return new Date().toLocaleDateString('sv-SE'); // 'sv-SE' → YYYY-MM-DD, local tz
}
```

- [ ] **Step 4: Run to verify it passes** — `cd ui && npm run test:e2e -- --grep applyTemplate` → 5 passed.

- [ ] **Step 5: Commit**

```bash
git add ui/src/lib/workspace/template.ts ui/tests/template.spec.ts
git commit -m "feat: applyTemplate helper for export filename tokens + unit tests"
```
End commit body with the Co-Authored-By trailer.

---

### Task 3: Wire the template field into the Toolbar export popover

Replace the plain stem input with a template input + live preview; surface `frames` on results; persist the template to `ui_state`.

**Files:**
- Modify: `ui/src/lib/stores.svelte.ts` (add `frames?` to `WorkspaceResult`)
- Modify: `ui/src/routes/+page.svelte:12-18` and `ui/src/routes/retouch/+page.svelte:141-146` (carry `frames` through `buildResults`)
- Modify: `ui/src/lib/workspace/Toolbar.svelte` (template state, preview, persistence; replace stem input)
- Modify: `ui/tests/smoke.spec.ts` (assert preview resolves a token)

- [ ] **Step 1: Add `frames` to the result type** — `ui/src/lib/stores.svelte.ts`, in `WorkspaceResult`:

```typescript
  frames?: number;     // frame count from result metadata (for {frames} token)
```

- [ ] **Step 2: Carry `frames` through result discovery** — in `ui/src/routes/+page.svelte` `buildResults` (around line 18) add to the mapped object:

```typescript
        frames: typeof img.frames === 'number' ? img.frames : undefined,
```

Apply the same one-line addition to the inline `buildResults`-equivalent in `ui/src/routes/retouch/+page.svelte` (the `.map` around line 146).

- [ ] **Step 3: Replace the stem input with a template input + preview** in `ui/src/lib/workspace/Toolbar.svelte`.

  Replace the `exportStem`/`exportFilename` derivation (lines ~127-133) with template-based logic:

```typescript
  import { applyTemplate, todayISO, type TemplateCtx } from './template';

  let exportTemplate = $state('{stack_name}_{method}');

  // Load the persisted template when the project changes.
  $effect(() => {
    const t = appState.project?.ui_state?.exportTemplate;
    if (typeof t === 'string' && t) exportTemplate = t;
  });

  function exportCtx(): TemplateCtx {
    const id = getExportImageId();
    const idx = appState.results.findIndex((r) => r.id === id);
    const result = idx >= 0 ? appState.results[idx] : undefined;
    return {
      stack_name: appState.project?.name ?? 'stacked',
      method: result?.method ?? '',
      frames: result?.frames,
      date: todayISO(),
      seq: (idx >= 0 ? idx : 0) + 1
    };
  }

  let exportStem = $derived(applyTemplate(exportTemplate, exportCtx()) || 'stacked');
  let exportFilename = $derived(`${exportStem}.${exportFormat}`);
```

(Keep the existing `exportAutoDest`/`exportDest` derivations below unchanged — they consume `exportFilename`.)

  Replace the "Filename" `<label>` (lines ~299-302) with a template field + preview:

```svelte
            <label class="tpl">
              <span class="lbl">Name template</span>
              <input type="text" bind:value={exportTemplate} data-testid="export-template" />
            </label>
            <div class="tpl-preview faint mono" data-testid="export-preview">{exportFilename}</div>
```

  Add a tokens hint (cheap, helps discoverability — not a dropdown):

```svelte
            <div class="tpl-hint faint">{`{stack_name} {method} {frames} {date} {seq}`}</div>
```

- [ ] **Step 4: Persist the template on export** — in `doExport()` (after a successful `api.export`, near line 161), fire-and-forget the PATCH:

```typescript
      api.patchUiState(appState.project.id, { exportTemplate }).catch(() => {});
```

- [ ] **Step 5: svelte-autofixer + check** — run the Svelte MCP `svelte-autofixer` on `Toolbar.svelte` until clean, then `cd ui && npm run check` → 0 errors.

- [ ] **Step 6: Assert the preview in the smoke** — in `ui/tests/smoke.spec.ts`, in the existing `workspace: add -> PMax -> view -> export` test, before filling `export-dest`, assert the preview resolved the project name + method:

```typescript
  await expect(page.getByTestId('export-preview')).toHaveText(/E2E_pmax\.tif/);
```

(The project is created with name `E2E` and stacked with `pmax`; `export-dest` is still filled directly afterward, so the override path keeps the rest of the test unchanged.)

- [ ] **Step 7: Commit**

```bash
git add ui/src/lib/stores.svelte.ts ui/src/routes/+page.svelte ui/src/routes/retouch/+page.svelte ui/src/lib/workspace/Toolbar.svelte ui/tests/smoke.spec.ts
git commit -m "feat: export name-template field with live preview and ui_state persistence"
```
End commit body with the Co-Authored-By trailer.

---

### Task 4: Full gate + finish

- [ ] **Step 1: UI gate** — `cd ui && npm run check && npm run build` → 0 errors, build OK.
- [ ] **Step 2: e2e** — `cd ui && npm run test:e2e` → all pass (applyTemplate unit specs + workspace/retouch/Zion smokes).
- [ ] **Step 3: Server gate** — from repo root: `uv run pytest -q && uv run ruff check . && uv run mypy engine/src server/src` → green; `git diff --check` → clean.
- [ ] **Step 4: Finish the branch** — use superpowers-extended-cc:finishing-a-development-branch to merge `feat/export-templates` into `main` (no-ff).

---

## Known caveats (documented, not blocking)

- **`{frames}` on old results:** results created before Task 1 have no `frames` key → the token resolves to empty string. Acceptable; no migration.
- **`{seq}` semantics:** 1-based index of the result within the current `appState.results` list, padded to 3. Meaningful when several results exist; for a single result it's `001`.
- **Test runner:** `applyTemplate` is unit-tested through Playwright (already installed) rather than adding Vitest — the webServer starts even for the pure test, a small time cost accepted to avoid a new dependency.
