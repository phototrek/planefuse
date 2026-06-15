# FocusStack M5 Part 1 — Retouch Engine + Server Design

**Date:** 2026-06-15
**Status:** Approved (brainstorming)
**Depends on:** M4 complete — merged to `main`.
**Spec refs:** SPEC §2 (engine has zero server imports; layout), §7.1 (unclamped float32 until export), §9 (retouch API), §11 (retouching), §13.2 (undo/redo determinism + retouch Playwright), §14 (M5).

## Goal

Implement server-side retouch compositing (SPEC §11): a retouch session targets a stacked result and paints another stacked result of the same stack into it, stroke by stroke, with undo/redo and flatten — all over HTTP, covered by engine + server tests including undo/redo determinism. **No UI in this plan** — the Retouch screen (screen 5) + retouch Playwright smoke are M5 Part 2; together with this part's determinism suite they form the M5 done-gate.

## Decisions (locked)

- **Two parts** (mirrors M4): this is **Part 1** = engine retouch + server sessions/strokes/undo/redo/flatten + API/determinism tests. Part 2 = the Retouch UI screen + retouch Playwright.
- **Sources = other stacked results** of the same stack (the SPEC headline workflow: paint DMap detail into PMax). The source picker is a general `image_id` list, so more source kinds (aligned frames) can be added later. Aligned-frame sources are deferred (they need per-frame `image_id` registration + pyramids).
- **Sharpness-at-cursor hint deferred to M6** (it needs the DMap index map; not part of the M5 done-gate).
- **Compositing is numpy/CPU**, float32, unclamped (SPEC §7.1). Deterministic; no device parity required for retouch (§13.2 parity covers ops/algorithms, not compositing).
- **Maintained working composite + checkpoints every 20 strokes** for O(affected-area) undo, per SPEC §11.

## Architecture

```
engine/src/focusstack/retouch/        # pure, zero server imports
  __init__.py
  brush.py        # stroke -> (gaussian-dab mask, bbox)
  session.py      # RetouchSession: composite, strokes, checkpoints, undo/redo

server/src/focusstack_server/
  retouch.py      # session persistence (project.json) + in-memory cache + tile rebuild
  api/retouch.py  # routes
tests/
  engine/test_retouch_brush.py
  engine/test_retouch_session.py
  server/test_retouch_api.py
```

### Engine: `retouch/brush.py`

- `Stroke` dataclass: `source_id: str`, `points: list[tuple[float, float, float]]` (x, y, pressure; pressure defaults 1.0), `radius: float`, `hardness: float (0–1)`, `opacity: float (0–1)`, `mode: "normal" | "erase"`.
- `stroke_mask(h, w, stroke) -> tuple[np.ndarray, tuple[int,int,int,int]]`: rasterizes the stroke as a chain of Gaussian dabs along `points` (deterministic dab spacing ≈ radius/4), hardness sets falloff sigma, per-point pressure scales that dab's strength; returns a float32 mask (only within bbox, zeros elsewhere or a cropped mask + bbox) and the affected `(y0, x0, y1, x1)` bbox clamped to the image. The full-image opacity is applied by the compositor, not the mask.

### Engine: `retouch/session.py`

- `RetouchSession`:
  - state: `base: np.ndarray` (target result, float32 HxWx3), `sources: dict[str, np.ndarray]`, `composite: np.ndarray`, `strokes: list[Stroke]`, `redo: list[Stroke]`, `checkpoints: list[tuple[int, np.ndarray]]` (stroke-count → exact composite copy; index 0 = base).
  - `apply(stroke) -> bbox`: `mask, bbox = stroke_mask(...)`; `a = mask * stroke.opacity` within bbox; `src = sources[stroke.source_id]` (normal) or `base` (erase); `composite[bbox] = composite[bbox]*(1-a) + src[bbox]*a`; append to `strokes`; clear `redo`; checkpoint a copy when `len(strokes) % 20 == 0`.
  - `undo() -> bbox`: move last stroke to `redo`; rebuild `composite` from the nearest checkpoint ≤ new length, replay remaining strokes; return the undone stroke's bbox.
  - `redo() -> bbox`: re-apply the next redo stroke.
  - `rebuild()`: full replay from base (used on rehydrate).
  - Determinism: checkpoints are exact copies and replay is pure, so equal stroke lists yield byte-identical composites.

### Server: `retouch.py`

- Session record persisted in `project.json` under `retouch: [{id, target_image_id, working_image_id, sources: [image_id], strokes: [stroke-dict]}]`.
- `RetouchManager(data_dir)`:
  - in-memory `dict[session_id -> RetouchSession]`; `_load(session_id)` rehydrates by reading the record, loading `base` + source arrays, replaying strokes (rebuilds composite + checkpoints + working pyramid).
  - `create(pid, target_image_id)`: base = target result array; sources = the project's other `result` images; `working_image_id = uuid`; build the working pyramid (= base) under `cache/tiles/<working_id>`; register `project.images[working_image_id] = {kind: "retouch-working", levels, width, height}`; persist the session.
  - `stroke(session_id, stroke)`: apply → bbox; rebuild the working pyramid tiles intersecting bbox across all levels; persist the stroke; bump `rev`; return `(rev, dirty_tiles)`.
  - `undo/redo(session_id)`: same, over the affected bbox.
  - `flatten(session_id, name)`: save composite to `cache/<id>.tif`; register a new `result` image; return its `image_id` (target untouched).
  - session-id → project lookup scans projects (same pattern as the existing tile route).
  - tile rebuild helper: given a bbox + the working composite, recompute exactly the tiles overlapping it at each zoom level (reuse `tiles.build_pyramid` math; add a `rebuild_region` that writes only affected tiles).

### Server: `api/retouch.py`

| Route | Behavior |
|---|---|
| `POST /api/projects/{pid}/retouch` `{target_image_id}` | create → `{session_id, working_image_id, sources:[{image_id, method}]}` |
| `GET /api/projects/{pid}/retouch` | list sessions |
| `DELETE /api/retouch/{session_id}` | remove session (original result preserved) |
| `POST /api/retouch/{session_id}/stroke` `{stroke}` | `{rev, dirty_tiles:[{z,x,y}]}` |
| `POST /api/retouch/{session_id}/undo` / `/redo` | `{rev, dirty_tiles}` |
| `POST /api/retouch/{session_id}/flatten` `{name}` | `{image_id}` |

Working-composite tiles are served by the **existing** `GET /api/viewer/{image_id}/tile/{z}/{x}/{y}` route (the working image is a normal pyramid). The client busts cache with `?rev=N`; the tile route ignores query params. Structured errors: unknown session/target → 404; missing/dim-mismatched source → 400.

## Data flow

create → base + working pyramid → client views `working_image_id` via tile route. stroke/undo/redo → server mutates composite, rebuilds dirty tiles, returns `{rev, dirty_tiles}` → (Part 2 UI) refetches only those tiles with the new `rev`. flatten → new `result` image, viewable/exportable through existing routes.

## Error handling

- Unknown `session_id` / `target_image_id` → `JSONResponse(404, {"error","detail"})`.
- Source not among candidates, or source/target shape mismatch → 400.
- Atomic writes for composite + project.json (existing helper); cancellation/crash leaves strokes consistent (rehydrate replays the persisted list).

## Testing (Part 1 done-gate)

**Engine** (`tests/engine/`):
- `test_retouch_brush.py`: deterministic mask; bbox tightly bounds the non-zero mask; hardness changes falloff; pressure scales dab strength.
- `test_retouch_session.py`: `normal` moves composite toward source, `erase` restores base; **undo/redo determinism** — apply 25 strokes (spans a checkpoint), hash composite; `undo`×25 → equals base; `redo`×25 → equals the snapshot hash; undo-to-middle equals a fresh replay to that length.

**Server** (`tests/server/test_retouch_api.py`, TestClient):
- create session on a stacked result → working tiles fetchable via `/api/viewer/.../tile/...`.
- `stroke` → non-empty `dirty_tiles`; a working tile's bytes change.
- **undo/redo determinism over HTTP**: stroke→flatten→array hash A; undo→redo→flatten→hash B; assert A == B.
- `flatten` registers a new `result` image; original target intact.
- session persists across `create_app` restart (same data dir → listed, strokes replay).
- structured errors (unknown session 404, dim-mismatch 400).

**Gates:** `uv run pytest` (engine + server) green; `ruff` + `mypy` clean.

## Out of scope (later)

- **Retouch UI screen (5) + retouch Playwright** — M5 Part 2 (closes the milestone done-gate).
- **Aligned-frame sources**, **sharpness-at-cursor hint**, **split/synced source view** — M6 / follow-ups.
