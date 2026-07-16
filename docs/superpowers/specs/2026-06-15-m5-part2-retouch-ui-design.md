# PlaneFuse M5 Part 2 - Retouch UI Design

**Date:** 2026-06-15
**Status:** Approved
**Depends on:** M5 Part 1 retouch engine and server.
**Spec refs:** SPEC sections 10, 11, 13.2, and 14.

## Goal

Complete M5 with a browser retouch workflow: launch from the result currently
shown in Viewer, paint another result into it, undo/redo, flatten to a new
result, then view and export that result.

## Decisions

- Viewer owns target selection through a **Retouch this result** button.
- An existing session for the target resumes; otherwise the UI creates one.
- The editor is a dedicated `/retouch` route, not a Viewer mode or modal.
- Painting is the default interaction. Holding `Space` temporarily pans.
- Flatten creates a new named result and opens it in Viewer.
- Aligned-frame sources, split view, and sharpness-at-cursor remain deferred to
  M6, as established by the Part 1 design.

## Files And Responsibilities

- `ui/src/lib/api.ts`: typed retouch REST methods and response types.
- `ui/src/lib/viewer/DeepZoom.svelte`: per-tile revision query parameters,
  image-coordinate pointer callbacks, temporary pan mode, and brush preview.
- `ui/src/routes/viewer/+page.svelte`: launch or resume retouch for the visible
  result.
- `ui/src/routes/retouch/+page.svelte`: session loading, source and brush
  controls, stroke collection, history, flatten, shortcuts, and errors.
- `ui/tests/smoke.spec.ts`: browser acceptance flow.

No navigation-rail entry is added. Retouch is entered from Viewer and returns
there after flatten.

## Viewer Launch

Viewer continues to track both:

- `srcId`: the original `result` image ID used by retouch and export.
- `imageId`: the generated tile-pyramid ID used for display.

**Retouch this result** performs:

1. `GET /api/projects/{pid}/retouch`.
2. Find the newest session whose `target_image_id` equals `srcId`.
3. If absent, call `POST /api/projects/{pid}/retouch` with `target_image_id`.
4. Store the session ID in project UI state as `retouchSessionId`.
5. Navigate to `/retouch`.

The button is disabled while the request is running and displays any structured
API error without leaving Viewer.

## Retouch Screen

The screen uses the existing darkroom-instrument visual language:

- A narrow left panel lists candidate result sources with a thumbnail and their
  method or saved name. The selected source is visually explicit.
- A compact top toolbar contains Normal/Erase, Undo, Redo, and Flatten.
- Radius, hardness, and opacity use native range inputs with numeric readouts.
- The remaining area is the existing tiled canvas, showing the session's
  `working_image_id`.

On load the page refreshes the project, resolves `retouchSessionId`, and reads
the persisted session record. The working image metadata (`levels`, `width`,
`height`) comes from `project.images[working_image_id]`. If the session is
missing, the page shows an actionable link back to Viewer.

Source selection is required for Normal mode. Erase sends the target ID as
`source_id`; the server ignores it for compositing but the payload remains
schema-valid.

Each source already has a result path. The page uses the existing viewer
registration endpoint once per source and displays its lowest-resolution tile
as the thumbnail. No new thumbnail endpoint is added.

## Canvas Interaction

`DeepZoom` gains optional editing props while preserving its current Viewer
behavior:

- `tileVersion(z, x, y)` supplies the cache-busting revision for each tile.
- `brushRadius` renders a non-interactive circular cursor in screen space.
- `onstroke(points)` receives full-image `(x, y, pressure)` coordinates.
- `Space` or an explicit `pan` prop selects pan behavior.

Pointer down starts a stroke, pointer move appends clamped image coordinates,
and pointer up submits one stroke. Pressure defaults to `1` when the browser
reports `0` or omits it. Points outside the image are ignored. A click still
produces a one-point stroke.

While `Space` is held, pointer gestures pan and never create a stroke. Wheel
zoom remains centered on the cursor. The brush preview diameter is
`2 * radius * scale`.

The page initializes every tile at the persisted session revision. After a
successful stroke, undo, or redo, it assigns the response revision only to the
returned dirty tile coordinates. Only those tile URLs change and refetch;
clean tiles keep their cached URL.

## Commands And Shortcuts

- `Ctrl+Z`: undo.
- `Ctrl+Shift+Z` or `Ctrl+Y`: redo.
- `[` and `]`: decrease or increase radius within the control limits.
- `Space`: temporary pan.
- `F` and `Z`: retain fit and 100% viewer behavior.

Commands are disabled while their request is in flight to keep stroke order
deterministic. A failed request leaves the current tiles and controls intact and
shows the error in the toolbar.

## Flatten

Flatten asks for a result name, defaulting to `Retouched`. On success:

1. Refresh the project to obtain the new result path.
2. Register its viewer pyramid through the existing viewer endpoint.
3. Persist it as the current `ui_state.view`.
4. Navigate to `/viewer`.

The original result and retouch session remain unchanged.

## Testing

The Playwright M5 test uses the existing synthetic fixture:

1. Import once and run PMax.
2. Run a second stack method to create a paintable source.
3. Open Viewer and click **Retouch this result**.
4. Select the other result and paint one visible stroke.
5. Assert the stroke request contains full-image coordinates and the canvas
   revision advances.
6. Undo and redo.
7. Flatten to a named result and assert Viewer opens it.
8. Export the flattened result and verify the TIFF exists and opens.

Done gates:

- `npm run check`
- `npm run test:e2e`
- `uv run pytest -q`
- `uv run ruff check .`
- `uv run mypy engine/src server/src`

## Out Of Scope

- Aligned-frame retouch sources.
- Split or synced source view.
- Sharpness-at-cursor hints.
- New server routes or frontend dependencies.
