# PlaneFuse M5 Part 1 — Retouch Engine + Server Design

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
engine/src/planefuse/retouch/        # pure, zero server imports
  __init__.py
  brush.py        # stroke -> (gaussian-dab mask, bbox)
  session.py      # RetouchSession: composite, strokes, checkpoints, undo/redo

server/src/planefuse_server/
  retouch.py      # session persistence (project.json) + in-memory cache + tile rebuild
  api/retouch.py  # routes
tests/
  engine/test_retouch_brush.py
  engine/test_retouch_session.py
  server/test_retouch_api.py
```

### Engine: `retouch/brush.py`

- `Stroke` dataclass: `source_id: str`, `points: list[tuple[float, float, float]]` (x, y, pressure; pressure defaults 1.0), `radius: float`, `hardness: float (0–1)`, `opacity: float (0–1)`, `mode: "normal" | "erase"`.
- `stroke_mask(h, w, stroke) -> tuple[np.ndarray, tuple[int,int,int,int]]`: rasterizes the stroke as a chain of Gaussian dabs along `points` with a fixed dab spacing `max(1, round(radius/4))` px (deterministic, version-independent integer rule), hardness sets falloff sigma (`sigma = radius * (1 - hardness) + epsilon`, fixed formula), per-point pressure scales that dab's strength. Returns a **cropped** float32 mask of shape `(y1−y0, x1−x0)` (NOT full-image) plus the affected bbox `(y0, x0, y1, x1)` clamped to the image — the compositor slices `composite[y0:y1, x0:x1]` so the mask must match that shape. Stroke-level `opacity` is applied by the compositor, not the mask.
- **Empty stroke:** if the clamped bbox has zero area (all points off-image, or radius rounds to <1 px), `stroke_mask` returns an empty bbox `(0,0,0,0)`; the session treats it as a no-op (see below).

### Engine: `retouch/session.py`

- `RetouchSession`:
  - state: `base: np.ndarray` (target result, float32 HxWx3), `sources: dict[str, np.ndarray]`, `composite: np.ndarray`, `strokes: list[Stroke]`, `redo: list[Stroke]`, `checkpoints: list[tuple[int, np.ndarray]]` (stroke-count → exact composite copy; index 0 = base).
  - `apply(stroke) -> bbox`: `mask, bbox = stroke_mask(...)`. **If bbox is empty (zero area): no-op — return it without recording the stroke or advancing the checkpoint counter** (keeps replay/determinism clean and yields empty `dirty_tiles`). Otherwise `a = mask * stroke.opacity` (shape = bbox); `src = sources[stroke.source_id]` (normal) or `base` (erase); `composite[y0:y1,x0:x1] = composite[y0:y1,x0:x1]*(1-a[...,None]) + src[y0:y1,x0:x1]*a[...,None]`; append to `strokes`; clear `redo`; checkpoint an exact copy when `len(strokes) % 20 == 0`.
  - `undo() -> bbox`: move last stroke to `redo`; rebuild `composite` from the nearest checkpoint ≤ new length, replay remaining strokes; return the undone stroke's bbox.
  - `redo() -> bbox`: re-apply the next redo stroke.
  - `rebuild()`: full replay from base (used on rehydrate).
  - Determinism: checkpoints are exact copies and replay is pure, so equal stroke lists yield byte-identical composites.

### Server: `retouch.py`

- Session record persisted in `project.json` under `retouch: [{id, target_image_id, working_image_id, sources: [image_id], strokes: [stroke-dict], rev: int}]`.
- **`rev` contract:** a **persisted, monotonically increasing** counter, incremented on every mutating op (stroke/undo/redo/flatten) and **never decremented** (so undo→redo cannot reissue a stale `?rev`). It is loaded on rehydrate, so the cache-bust value keeps advancing across an app restart. (Distinct from `len(strokes)`, which goes down on undo.)
- `RetouchManager(data_dir)`:
  - in-memory `dict[session_id -> RetouchSession]`; `_load(session_id)` rehydrates by reading the record, loading `base` + source arrays, replaying strokes (rebuilds composite + checkpoints + working pyramid).
  - `create(pid, target_image_id)`: base = target result array; sources = the project's other `result` images; `working_image_id = uuid`; build the working pyramid (= base) under `cache/tiles/<working_id>`; register `project.images[working_image_id] = {kind: "retouch-working", levels, width, height}` (tile-only — no `path`; see note); `rev = 0`; persist the session.
  - `stroke(session_id, stroke)`: apply → bbox; if bbox empty, return `(rev, [])` unchanged; else `rebuild_region` the working tiles intersecting bbox; persist the stroke; `rev += 1`; return `(rev, dirty_tiles)`.
  - `undo/redo(session_id)`: same, over the affected bbox; `rev += 1`.
  - `flatten(session_id, name)`: write composite (float32, clamped at the I/O boundary) to `cache/<new_image_id>.tif` via `save_image(..., bit_depth=16)` — a **fresh** image_id each call (re-flatten never clobbers a prior one); register `project.images[new_image_id] = {kind: "result", path, method: "retouch", name}`; `rev += 1`; return `{image_id}`. Target untouched.
  - session-id → project lookup scans `store.list()` (same pattern as the viewer tile route); O(projects) per call, acceptable at single-user scale.

#### Tile rebuild: `rebuild_region(composite, tiles_root/<working_id>, levels, bbox) -> list[{z,x,y}]`

The existing `tiles.build_pyramid` resizes the full image per level then writes every tile. `rebuild_region` mirrors its level math (`scale = 2**(levels-1-z)`, `lw = W//scale`, `lh = H//scale`, 256-px tiles) but writes **only** the tiles whose extent overlaps the bbox:
- For each level `z`: map the base-res bbox into level coords (`bbox / scale`), expand to whole tile indices, and for each overlapping `(x, y)` re-crop+save just that tile.
- It re-resizes the **full** composite to the level once (same call `build_pyramid` already makes), then writes only the dirty tiles — this avoids bilinear resample-edge bleed at tile borders (a region-only resize would mis-sample at seams) while still bounding disk writes to the affected tiles. The per-level full resize of a single result image is cheap for a local single-user app; the "O(affected area)" benefit is in the write/refetch count, not the resize.
- `build_pyramid` is refactored to share the per-level tile-writing helper so both paths agree byte-for-byte.

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
- `retouch-working` images are **tile-only** (no `path`): they are viewed via the tile route and never exported directly (export targets the flattened `result`, which has a `path`). Add a small guard in `make_export_runner` to raise a structured `400` (not a bare `KeyError`) if `info` lacks a `path`, so an accidental working-image export fails cleanly.

## Testing (Part 1 done-gate)

**Engine** (`tests/engine/`):
- `test_retouch_brush.py`: deterministic mask; bbox tightly bounds the non-zero mask; hardness changes falloff; pressure scales dab strength.
- `test_retouch_session.py`: `normal` moves composite toward source, `erase` restores base; **undo/redo determinism** — apply 25 strokes (spans a checkpoint), hash composite; `undo`×25 → equals base; `redo`×25 → equals the snapshot hash; undo-to-middle equals a fresh replay to that length.

**Server** (`tests/server/test_retouch_api.py`, TestClient):
- fixture stacks the same frames **twice** (e.g. `pmax` and `dmap`) so the project has ≥2 `result` images — one as the target, the other as a paintable source.
- create session on a stacked result → returned `sources` includes the other result; working tiles fetchable via `/api/viewer/.../tile/...`.
- `stroke` → non-empty `dirty_tiles`; a working tile's bytes change.
- **undo/redo determinism over HTTP**: stroke→flatten→array hash A; undo→redo→flatten→hash B; assert A == B.
- `flatten` registers a new `result` image; original target intact.
- session persists across `create_app` restart (same data dir → listed, strokes replay).
- structured errors (unknown session 404, dim-mismatch 400).

**Gates:** `uv run pytest` (engine + server) green; `ruff` + `mypy` clean.

## Out of scope (later)

- **Retouch UI screen (5) + retouch Playwright** — M5 Part 2 (closes the milestone done-gate).
- **Aligned-frame sources**, **sharpness-at-cursor hint**, **split/synced source view** — M6 / follow-ups.
