# Phase 3 Product UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the professional workspace UI, including RAW clarity, validation, comparison, histogram, queue, batch, selection review, shortcuts, and export controls.

**Architecture:** Extend typed API/state models, keep image analysis server-side, and compose focused workspace components rather than growing `Toolbar.svelte` and `ViewerPane.svelte`. All viewer modes share one transform state for synchronized navigation.

**Tech Stack:** Svelte 5, TypeScript 6, Vite 8, FastAPI, Playwright.

## Global Constraints

- The UI remains a single local workspace and requires no accounts.
- RAW mode must plainly say that no aesthetic development is baked.
- Every control has an accessible label, keyboard path, validation state, and Playwright coverage.
- Large-image histogram/compare operations use server tiles or compact analysis payloads, not full images in browser memory.

---

### Task 1: Split workspace controls into focused components

**Files:**
- Create: `ui/src/lib/workspace/AlgorithmControls.svelte`
- Create: `ui/src/lib/workspace/ExportPanel.svelte`
- Create: `ui/src/lib/workspace/ProjectControls.svelte`
- Modify: `ui/src/lib/workspace/Toolbar.svelte`
- Modify: `ui/tests/smoke.spec.ts`

- [ ] Add component behavior tests through Playwright before moving markup.
- [ ] Extract existing behavior without changing API calls or test IDs.
- [ ] Run Svelte diagnostics and smoke tests after each extraction.
- [ ] Commit with `refactor: split workspace controls`.

### Task 2: Add typed validation and RAW compatibility UX

**Files:**
- Modify: `ui/src/lib/api.ts`
- Modify: `ui/src/lib/stores.svelte.ts`
- Modify: `ui/src/lib/workspace/InputList.svelte`
- Create: `ui/src/lib/workspace/ValidationPanel.svelte`
- Modify: `ui/tests/smoke.spec.ts`

- [ ] Write browser tests for wrong dimensions/bit depth, mixed camera/CFA, unreadable RAW, and valid no-bake RAW summary.
- [ ] Render report-level domain/camera/decoder details and per-file actionable badges.
- [ ] Disable Run until blocking issues are resolved; link each error to the affected input.
- [ ] Commit with `feat: add professional input validation`.

### Task 3: Add alignment-quality and smart-selection review

**Files:**
- Create: `ui/src/lib/workspace/AlignmentReview.svelte`
- Create: `ui/src/lib/workspace/SelectionReview.svelte`
- Modify: `ui/src/lib/workspace/InputList.svelte`
- Modify: `ui/src/lib/workspace/Toolbar.svelte`
- Modify: `ui/src/lib/api.ts`
- Modify: `ui/tests/smoke.spec.ts`

- [ ] Write tests for low-correlation keep/exclude decisions, recovered links, proposed selected frames, dimmed exclusions, coverage summary, and >40-frame recommendation.
- [ ] Persist accepted decisions in stack parameters/project state.
- [ ] Ensure batch mode shows and confirms group proposals before enqueue.
- [ ] Commit with `feat: review alignment and frame selection`.

### Task 4: Complete queue, history, and estimates

**Files:**
- Modify: `ui/src/lib/workspace/RenderDrawer.svelte`
- Create: `ui/src/lib/workspace/JobQueue.svelte`
- Create: `ui/src/lib/workspace/EstimateBadge.svelte`
- Modify: `ui/src/lib/api.ts`
- Modify: `ui/tests/smoke.spec.ts`

- [ ] Write tests for cancel, reorder, progress/log levels, history, rerun same/edited parameters, and stack-all ordering.
- [ ] Add server-provided memory/time estimate display with an explicitly approximate label.
- [ ] Keep finished results and active jobs visually distinct.
- [ ] Commit with `feat: complete queue and job history`.

### Task 5: Add histogram and clipping indicators

**Files:**
- Create: `server/src/planefuse_server/api/analysis.py`
- Modify: `server/src/planefuse_server/main.py`
- Create: `tests/server/test_analysis.py`
- Modify: `ui/src/lib/api.ts`
- Create: `ui/src/lib/viewer/Histogram.svelte`
- Modify: `ui/src/lib/workspace/ViewerPane.svelte`
- Modify: `ui/tests/smoke.spec.ts`

- [ ] Write API tests for 256-bin RGB/luminance histograms and per-channel shadow/highlight clipping counts from viewable images.
- [ ] Implement cached analysis keyed by image id/revision.
- [ ] Write UI tests for histogram rendering and independent channel indicators.
- [ ] Commit with `feat: add histogram and clipping analysis`.

### Task 6: Add synchronized compare modes and overlays

**Files:**
- Modify: `ui/src/lib/viewer/DeepZoom.svelte`
- Create: `ui/src/lib/viewer/CompareViewer.svelte`
- Create: `ui/src/lib/viewer/ViewerControls.svelte`
- Modify: `ui/src/lib/workspace/ViewerPane.svelte`
- Modify: `ui/src/lib/stores.svelte.ts`
- Modify: `ui/tests/smoke.spec.ts`

- [ ] Write tests for result/source split, result/result side-by-side, draggable divider, synchronized pan/zoom, before/after toggle, depth opacity, and timed alignment blink.
- [ ] Lift transform state into the compare parent and make DeepZoom controlled/uncontrolled compatible.
- [ ] Preserve tile culling and revision cache-busting in all modes.
- [ ] Commit with `feat: add synchronized viewer comparison`.

### Task 7: Complete keyboard and accessibility behavior

**Files:**
- Create: `ui/src/lib/shortcuts.ts`
- Modify: `ui/src/routes/+page.svelte`
- Modify: `ui/src/routes/retouch/+page.svelte`
- Modify: `ui/src/lib/viewer/DeepZoom.svelte`
- Create: `ui/tests/shortcuts.spec.ts`

- [ ] Write tests for F, Z, arrows, backslash, bracket brush sizing, undo/redo, Escape, and focus guards in text fields.
- [ ] Centralize platform-aware Ctrl/Cmd matching and conflict prevention.
- [ ] Add visible shortcut help and accessible pressed/expanded states.
- [ ] Commit with `feat: complete keyboard workflow`.

### Task 8: Complete rendered and RAW export UX

**Files:**
- Modify: `ui/src/lib/workspace/ExportPanel.svelte`
- Modify: `ui/src/lib/workspace/template.ts`
- Modify: `ui/src/lib/api.ts`
- Create: `ui/tests/export.spec.ts`

- [ ] Write tests for TIFF compression, JPEG quality/bit-depth constraints, 16-bit PNG, depth companion, Linear DNG availability only for RAW results, optional float TIFF, naming tokens, destination validation, progress, and conformance failure.
- [ ] Display the exact no-bake DNG metadata/decoder summary and Capture One target.
- [ ] Prevent impossible format/bit-depth combinations before enqueue.
- [ ] Commit with `feat: complete professional export controls`.

### Task 9: Phase 3 verification

- [ ] Run `npm run check`, production build, all Playwright tests on CPU, targeted MPS RAW/browser workflow, and server API tests.
- [ ] Capture stable screenshot fixtures only after all functional tests pass.
