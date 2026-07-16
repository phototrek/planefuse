# PlaneFuse M5 Part 1 — Retouch Engine + Server Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement server-side retouch compositing (SPEC §11) — a session paints one stacked result into another with undo/redo/flatten, all over HTTP — covered by engine + server tests including undo/redo determinism.

**Architecture:** A pure-numpy engine module `planefuse.retouch` (zero server imports) owns brush rasterization and the deterministic `RetouchSession` (maintained composite + checkpoints every 20 strokes). The server adds a `RetouchManager` that persists sessions in `project.json`, rehydrates by replay, rebuilds only the dirty tiles of a working-image pyramid (reusing the viewer tile route), and exposes the retouch REST routes. No UI here (M5 Part 2).

**Tech Stack:** Python 3.12, numpy, Pillow (tiles), FastAPI + httpx TestClient, the existing `planefuse`/`planefuse-server` packages, uv, pytest, ruff, mypy.

**Read first:** `docs/superpowers/specs/2026-06-15-m5-retouch-design.md` (the approved design — spec wins on conflict; flag conflicts rather than deviating). SPEC §2 (engine-has-zero-server-imports hard rule), §7.1 (unclamped float32 until export), §9/§11 (retouch API + behavior), §13.2 (undo/redo determinism), §14 (M5).

**Preconditions:** M4 + the batch/re-run follow-up merged to `main`; `uv run pytest` green (167 passed, 13 skipped); `git log --oneline -1` shows the `feat/m4-batch-rerun` merge. CPU-only machine.

**Conventions:**
- Branch: `feat/m5-retouch` off `main`. All commands from repo root via `uv run …`.
- Engine tests in `tests/engine/`, server tests in `tests/server/`.
- Structured errors: routes map `LookupError -> 404`, `ValueError -> 400` as `JSONResponse(status_code, {"error": type, "detail": msg})`.
- End every commit body with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

**Module layout:**
```
engine/src/planefuse/retouch/
  __init__.py          # exports Stroke, stroke_mask, RetouchSession
  brush.py             # Stroke dataclass + stroke_mask (gaussian dabs -> cropped mask + bbox)
  session.py           # RetouchSession (apply/undo/redo, checkpoints, determinism)
server/src/planefuse_server/
  tiles.py             # + rebuild_region + shared tile-writing helper (refactor build_pyramid)
  retouch.py           # RetouchManager (persistence, rehydrate, tile rebuild, flatten)
  api/retouch.py       # routes
  main.py              # app-scoped RetouchManager + include router
  projects.py          # Project gains a `retouch: list[dict]` field
  runners.py           # make_export_runner: guard missing path -> ValueError
tests/engine/test_retouch_brush.py
tests/engine/test_retouch_session.py
tests/server/test_retouch_api.py
```

---

## Chunk 1: Engine retouch

### Task 1: Branch + brush (`stroke_mask`)

**Files:**
- Create: branch `feat/m5-retouch`
- Create: `engine/src/planefuse/retouch/__init__.py`, `engine/src/planefuse/retouch/brush.py`
- Test: `tests/engine/test_retouch_brush.py`

- [ ] **Step 1: Branch**

```bash
git checkout -b feat/m5-retouch
```

- [ ] **Step 2: Write the failing tests** — `tests/engine/test_retouch_brush.py`:

```python
import numpy as np

from planefuse.retouch.brush import Stroke, stroke_mask


def _stroke(**kw):
    base = dict(source_id="s", points=[(50.0, 40.0, 1.0)], radius=10.0,
                hardness=0.5, opacity=1.0, mode="normal")
    base.update(kw)
    return Stroke(**base)


def test_mask_shape_matches_bbox():
    mask, (y0, x0, y1, x1) = stroke_mask(100, 120, _stroke())
    assert mask.shape == (y1 - y0, x1 - x0)
    assert mask.dtype == np.float32
    assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0


def test_mask_peaks_at_center():
    mask, (y0, x0, y1, x1) = stroke_mask(100, 120, _stroke())
    # center (50,40) is inside the bbox and ~1.0
    cy, cx = 40 - y0, 50 - x0
    assert mask[cy, cx] > 0.9


def test_bbox_tightly_bounds_nonzero():
    mask, (y0, x0, y1, x1) = stroke_mask(100, 120, _stroke(radius=8.0, hardness=0.9))
    # every nonzero pixel lies within the returned bbox (mask is cropped, so trivially true),
    # and the bbox is not the whole image
    assert (x1 - x0) < 120 and (y1 - y0) < 100


def test_deterministic():
    a, ba = stroke_mask(100, 120, _stroke())
    b, bb = stroke_mask(100, 120, _stroke())
    assert ba == bb
    assert np.array_equal(a, b)


def test_offscreen_stroke_is_empty():
    mask, bbox = stroke_mask(100, 120, _stroke(points=[(-50.0, -50.0, 1.0)], radius=5.0))
    assert bbox == (0, 0, 0, 0)
    assert mask.size == 0


def test_pressure_scales_strength():
    full, _ = stroke_mask(100, 120, _stroke(points=[(50.0, 40.0, 1.0)]))
    half, _ = stroke_mask(100, 120, _stroke(points=[(50.0, 40.0, 0.5)]))
    assert float(half.max()) < float(full.max())
```

- [ ] **Step 3: Run to verify failure** — `uv run pytest tests/engine/test_retouch_brush.py -q` → FAIL (module missing).

- [ ] **Step 4: Implement** `engine/src/planefuse/retouch/brush.py`:

```python
"""Brush stroke rasterization (SPEC §11). Pure numpy, deterministic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EMPTY_BBOX = (0, 0, 0, 0)


@dataclass
class Stroke:
    source_id: str
    points: list[tuple[float, float, float]]  # (x, y, pressure); pressure defaults 1.0
    radius: float
    hardness: float = 0.5  # 0 = soft, 1 = hard
    opacity: float = 1.0
    mode: str = "normal"  # "normal" | "erase"


def _norm_points(points) -> list[tuple[float, float, float]]:
    out = []
    for p in points:
        if len(p) >= 3:
            out.append((float(p[0]), float(p[1]), float(p[2])))
        else:
            out.append((float(p[0]), float(p[1]), 1.0))
    return out


def _dab_centers(points: list[tuple[float, float, float]], radius: float):
    """Stamp centers along the polyline at fixed integer spacing (deterministic)."""
    spacing = max(1, round(radius / 4))
    if not points:
        return []
    centers = [points[0]]
    for (x0, y0, p0), (x1, y1, p1) in zip(points, points[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(seg / spacing))
        for i in range(1, n + 1):
            t = i / n
            centers.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, p0 + (p1 - p0) * t))
    return centers


def stroke_mask(h: int, w: int, stroke: Stroke):
    """Return (cropped float32 mask of shape (y1-y0, x1-x0), bbox=(y0,x0,y1,x1)).

    Mask = max over Gaussian dabs (so overlaps don't exceed pressure), clamped [0,1].
    Stroke-level opacity is applied by the compositor, not here. Empty -> ((0,0)-array, EMPTY_BBOX).
    """
    centers = _dab_centers(_norm_points(stroke.points), stroke.radius)
    if not centers:
        return np.zeros((0, 0), np.float32), EMPTY_BBOX

    reach = max(1, math.ceil(stroke.radius))
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    x0 = max(0, int(math.floor(min(xs) - reach)))
    y0 = max(0, int(math.floor(min(ys) - reach)))
    x1 = min(w, int(math.ceil(max(xs) + reach)) + 1)
    y1 = min(h, int(math.ceil(max(ys) + reach)) + 1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0), np.float32), EMPTY_BBOX

    sigma = stroke.radius * (1.0 - stroke.hardness) + 1e-3
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    mask = np.zeros((y1 - y0, x1 - x0), np.float32)
    inv = 1.0 / (2.0 * sigma * sigma)
    for cx, cy, pr in centers:
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        g = (np.exp(-d2 * inv).astype(np.float32)) * np.float32(pr)
        np.maximum(mask, g, out=mask)
    np.clip(mask, 0.0, 1.0, out=mask)
    return mask, (y0, x0, y1, x1)
```

`engine/src/planefuse/retouch/__init__.py`:

```python
"""Retouch stroke-compositing engine (SPEC §11)."""

from planefuse.retouch.brush import EMPTY_BBOX, Stroke, stroke_mask
from planefuse.retouch.session import RetouchSession

__all__ = ["EMPTY_BBOX", "Stroke", "stroke_mask", "RetouchSession"]
```

Note: `__init__` imports `RetouchSession` (built in Task 2). To keep Task 1 importable on its own, temporarily omit the `session` import + `RetouchSession` from `__all__`, then add them in Task 2.

- [ ] **Step 5: Run to verify pass** — `uv run pytest tests/engine/test_retouch_brush.py -q` → PASS.

- [ ] **Step 6: ruff + mypy + commit**

```bash
uv run ruff check engine/src/planefuse/retouch tests/engine/test_retouch_brush.py
uv run mypy engine/src
git add engine/src/planefuse/retouch tests/engine/test_retouch_brush.py
git commit -m "feat: retouch brush stroke rasterization (gaussian dabs)"
```

---

### Task 2: `RetouchSession` (apply / undo / redo / checkpoints)

**Files:**
- Create: `engine/src/planefuse/retouch/session.py`
- Modify: `engine/src/planefuse/retouch/__init__.py` (add session import)
- Test: `tests/engine/test_retouch_session.py`

- [ ] **Step 1: Write the failing tests** — `tests/engine/test_retouch_session.py`:

```python
import numpy as np

from planefuse.retouch import RetouchSession, Stroke


def _session():
    base = np.zeros((64, 80, 3), np.float32)            # black target
    src = np.ones((64, 80, 3), np.float32)              # white source
    return RetouchSession(base, {"white": src})


def _stroke(**kw):
    base = dict(source_id="white", points=[(40.0, 32.0, 1.0)], radius=12.0,
                hardness=0.5, opacity=1.0, mode="normal")
    base.update(kw)
    return Stroke(**base)


def test_normal_blends_source_in():
    s = _session()
    s.apply(_stroke())
    assert s.composite[32, 40, 0] > 0.5          # painted toward white at center
    assert s.composite[0, 0, 0] == 0.0           # corner untouched


def test_erase_restores_base():
    s = _session()
    s.apply(_stroke())
    s.apply(_stroke(mode="erase"))
    assert s.composite[32, 40, 0] < 0.5          # erased back toward base (black)


def test_empty_stroke_not_recorded():
    s = _session()
    bbox = s.apply(_stroke(points=[(-99.0, -99.0, 1.0)], radius=4.0))
    assert bbox == (0, 0, 0, 0)
    assert len(s.strokes) == 0


def test_undo_redo_determinism_across_checkpoint():
    s = _session()
    rng = np.random.default_rng(0)
    for _ in range(25):                           # spans the 20-stroke checkpoint
        x, y = rng.uniform(5, 75), rng.uniform(5, 59)
        s.apply(_stroke(points=[(float(x), float(y), 1.0)]))
    snap = s.composite.copy()
    for _ in range(25):
        s.undo()
    assert np.array_equal(s.composite, s.base)    # fully undone == base
    for _ in range(25):
        s.redo()
    assert np.array_equal(s.composite, snap)      # fully redone == snapshot


def test_undo_to_middle_equals_fresh_replay():
    base = np.zeros((64, 80, 3), np.float32)
    src = {"white": np.ones((64, 80, 3), np.float32)}
    strokes = [Stroke("white", [(float(10 + i), 32.0, 1.0)], 8.0) for i in range(23)]

    a = RetouchSession(base, src)
    for st in strokes:
        a.apply(st)
    for _ in range(8):                            # undo back to 15 strokes
        a.undo()

    b = RetouchSession(base, src)
    for st in strokes[:15]:
        b.apply(st)

    assert np.array_equal(a.composite, b.composite)
```

- [ ] **Step 2: Run to verify failure** — FAIL (no `RetouchSession`).

- [ ] **Step 3: Implement** `engine/src/planefuse/retouch/session.py`:

```python
"""Deterministic retouch session: maintained composite + checkpoints (SPEC §11)."""

from __future__ import annotations

import numpy as np

from planefuse.retouch.brush import EMPTY_BBOX, Stroke, stroke_mask

CHECKPOINT_EVERY = 20

Bbox = tuple[int, int, int, int]


class RetouchSession:
    def __init__(self, base: np.ndarray, sources: dict[str, np.ndarray]):
        self.base = np.ascontiguousarray(base, dtype=np.float32)
        self.sources = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in sources.items()}
        self.composite = self.base.copy()
        self.strokes: list[Stroke] = []
        self.redo_stack: list[Stroke] = []
        self.checkpoints: list[tuple[int, np.ndarray]] = [(0, self.base.copy())]

    def _blend(self, stroke: Stroke) -> Bbox:
        h, w = self.base.shape[:2]
        mask, bbox = stroke_mask(h, w, stroke)
        y0, x0, y1, x1 = bbox
        if y1 <= y0 or x1 <= x0:
            return EMPTY_BBOX
        a = (mask * np.float32(stroke.opacity))[..., None]
        src = self.base if stroke.mode == "erase" else self.sources[stroke.source_id]
        region = self.composite[y0:y1, x0:x1]
        self.composite[y0:y1, x0:x1] = region * (1.0 - a) + src[y0:y1, x0:x1] * a
        return bbox

    def _commit(self, stroke: Stroke) -> Bbox:
        bbox = self._blend(stroke)
        if bbox == EMPTY_BBOX:
            return bbox
        self.strokes.append(stroke)
        if len(self.strokes) % CHECKPOINT_EVERY == 0:
            self.checkpoints.append((len(self.strokes), self.composite.copy()))
        return bbox

    def apply(self, stroke: Stroke) -> Bbox:
        bbox = self._commit(stroke)
        if bbox != EMPTY_BBOX:
            self.redo_stack.clear()
        return bbox

    def _rebuild_to(self, n: int) -> None:
        cp = max((c for c in self.checkpoints if c[0] <= n), key=lambda c: c[0])
        self.composite = cp[1].copy()
        for s in self.strokes[cp[0]:n]:
            self._blend(s)

    def undo(self) -> Bbox:
        if not self.strokes:
            return EMPTY_BBOX
        s = self.strokes.pop()
        self.redo_stack.append(s)
        self.checkpoints = [c for c in self.checkpoints if c[0] <= len(self.strokes)]
        bbox = stroke_mask(self.base.shape[0], self.base.shape[1], s)[1]
        self._rebuild_to(len(self.strokes))
        return bbox

    def redo(self) -> Bbox:
        if not self.redo_stack:
            return EMPTY_BBOX
        return self._commit(self.redo_stack.pop())
```

Then update `__init__.py` to import `RetouchSession` (as shown in Task 1).

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/engine/test_retouch_session.py -q` → PASS.

- [ ] **Step 5: ruff + mypy + commit**

```bash
uv run ruff check engine/src/planefuse/retouch tests/engine/test_retouch_session.py
uv run mypy engine/src
git add engine/src/planefuse/retouch tests/engine/test_retouch_session.py
git commit -m "feat: deterministic RetouchSession with checkpoints and undo/redo"
```

---

## Chunk 2: Server — tiles, sessions, routes

### Task 3: Tile `rebuild_region` + shared helper

**Files:**
- Modify: `server/src/planefuse_server/tiles.py`
- Test: `tests/server/test_tiles_rebuild.py`

- [ ] **Step 1: Write the failing tests** — `tests/server/test_tiles_rebuild.py`:

```python
import numpy as np

from planefuse_server.tiles import build_pyramid, rebuild_region, tile_path


def test_rebuild_region_changes_only_dirty_tiles(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 1, (600, 800, 3)).astype(np.float32)
    root = tmp_path / "tiles" / "img"
    levels = build_pyramid(img, root)

    # bytes of a full-res tile far from the edit and one under the edit
    far = tile_path(tmp_path / "tiles", "img", levels - 1, 0, 0).read_bytes()

    # edit a region near the bottom-right at full res
    img2 = img.copy()
    img2[560:600, 760:800] = 0.0
    dirty = rebuild_region(img2, root, levels, (560, 760, 600, 800))

    assert dirty, "expected at least one dirty tile"
    assert all({"z", "x", "y"} <= set(d) for d in dirty)
    # the far top-left full-res tile is unchanged on disk
    assert tile_path(tmp_path / "tiles", "img", levels - 1, 0, 0).read_bytes() == far


def test_build_pyramid_unchanged_output(tmp_path):
    # refactor must not change build_pyramid's byte output (viewer tests depend on it)
    img = np.random.default_rng(1).uniform(0, 1, (300, 400, 3)).astype(np.float32)
    levels = build_pyramid(img, tmp_path / "a")
    levels2 = build_pyramid(img, tmp_path / "b")
    assert levels == levels2
    for z in range(levels):
        for f in (tmp_path / "a" / str(z)).glob("*.jpg"):
            assert f.read_bytes() == (tmp_path / "b" / str(z) / f.name).read_bytes()
```

- [ ] **Step 2: Run to verify failure** — FAIL (no `rebuild_region`).

- [ ] **Step 3: Refactor + implement** — replace `tiles.py` body (keep `TILE`, `tile_path` signatures):

```python
"""Deep-zoom viewer tile pyramid (SPEC §9). 256px JPEG tiles per zoom level.
Stored as cache/tiles/<image_id>/<z>/<x>_<y>.jpg; z=0 coarsest."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

TILE = 256


def _to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))


def _level_dims(w: int, h: int, levels: int, z: int) -> tuple[int, int, int]:
    scale = 2 ** (levels - 1 - z)
    return max(1, w // scale), max(1, h // scale), scale


def _save_tile(lvl_img: Image.Image, zdir: Path, tx: int, ty: int, lw: int, lh: int) -> None:
    box = (tx * TILE, ty * TILE, min((tx + 1) * TILE, lw), min((ty + 1) * TILE, lh))
    lvl_img.crop(box).save(zdir / f"{tx}_{ty}.jpg", quality=85)


def _levels_for(w: int, h: int) -> int:
    return max(1, math.ceil(math.log2(max(1, max(h, w) / TILE))) + 1)


def build_pyramid(image: np.ndarray, out_dir: Path) -> int:
    """Build all tiles for a float32 (H,W,3) image. Returns the number of zoom levels."""
    out_dir = Path(out_dir)
    h, w = image.shape[:2]
    base = _to_pil(image)
    levels = _levels_for(w, h)
    for z in range(levels):
        lw, lh, _ = _level_dims(w, h, levels, z)
        lvl = base.resize((lw, lh), Image.Resampling.BILINEAR)
        zdir = out_dir / str(z)
        zdir.mkdir(parents=True, exist_ok=True)
        for ty in range(math.ceil(lh / TILE)):
            for tx in range(math.ceil(lw / TILE)):
                _save_tile(lvl, zdir, tx, ty, lw, lh)
    return levels


def rebuild_region(image: np.ndarray, out_dir: Path, levels: int,
                   bbox: tuple[int, int, int, int]) -> list[dict]:
    """Rewrite only the tiles overlapping a base-resolution bbox (y0,x0,y1,x1).
    Re-resizes the full image per level (avoids resample edge bleed), writes only
    dirty tiles. Returns the dirty tile list [{z,x,y}]."""
    out_dir = Path(out_dir)
    h, w = image.shape[:2]
    base = _to_pil(image)
    y0, x0, y1, x1 = bbox
    dirty: list[dict] = []
    if y1 <= y0 or x1 <= x0:
        return dirty
    for z in range(levels):
        lw, lh, scale = _level_dims(w, h, levels, z)
        lvl = base.resize((lw, lh), Image.Resampling.BILINEAR)
        zdir = out_dir / str(z)
        zdir.mkdir(parents=True, exist_ok=True)
        cols = math.ceil(lw / TILE)
        rows = math.ceil(lh / TILE)
        tx_lo = max(0, (x0 // scale) // TILE)
        ty_lo = max(0, (y0 // scale) // TILE)
        tx_hi = min(cols - 1, ((math.ceil(x1 / scale)) - 1) // TILE)
        ty_hi = min(rows - 1, ((math.ceil(y1 / scale)) - 1) // TILE)
        for ty in range(ty_lo, ty_hi + 1):
            for tx in range(tx_lo, tx_hi + 1):
                _save_tile(lvl, zdir, tx, ty, lw, lh)
                dirty.append({"z": z, "x": tx, "y": ty})
    return dirty


def tile_path(tiles_root: Path, image_id: str, z: int, x: int, y: int) -> Path:
    return tiles_root / image_id / str(z) / f"{x}_{y}.jpg"
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/server/test_tiles_rebuild.py tests/server/test_viewer.py -q` → PASS (existing viewer tests still pass — byte output unchanged).

- [ ] **Step 5: ruff/mypy + commit**

```bash
uv run ruff check server/src/planefuse_server/tiles.py tests/server/test_tiles_rebuild.py
uv run mypy server/src
git add server/src/planefuse_server/tiles.py tests/server/test_tiles_rebuild.py
git commit -m "feat: rebuild_region for partial tile updates; factor tile helpers"
```

---

### Task 4: `Project.retouch` field + `RetouchManager`

**Files:**
- Modify: `server/src/planefuse_server/projects.py` (add `retouch` field)
- Create: `server/src/planefuse_server/retouch.py`
- Test: `tests/server/test_retouch_manager.py`

- [ ] **Step 1: Add the `retouch` field to `Project`** in `projects.py`:
  - In the `Project` dataclass, add: `retouch: list[dict] = field(default_factory=list)`.
  - In `asdict_no_props`, add `"retouch": proj.retouch` to the returned dict.
  - In `_load`, add `retouch=d.get("retouch", [])` to the `Project(...)` constructor.

- [ ] **Step 2: Write the failing tests** — `tests/server/test_retouch_manager.py`:

```python
import numpy as np

from planefuse.io import save_image
from planefuse_server.projects import ProjectStore
from planefuse_server.retouch import RetouchManager


def _project_with_two_results(tmp_path):
    store = ProjectStore(tmp_path / "data")
    proj = store.create(tmp_path / "proj", "P")
    target = np.zeros((64, 80, 3), np.float32)
    source = np.ones((64, 80, 3), np.float32)
    for iid, arr in (("res_target", target), ("res_source", source)):
        p = proj.cache / f"{iid}.tif"
        save_image(arr, p, bit_depth=16)
        proj.images[iid] = {"kind": "result", "path": str(p), "method": "pmax"}
    store.save(proj)
    return store, proj


def _stroke_dict(source_id="res_source"):
    return {"source_id": source_id, "points": [[40, 32, 1.0]], "radius": 14.0,
            "hardness": 0.5, "opacity": 1.0, "mode": "normal"}


def test_create_session_lists_sources_and_builds_working(tmp_path):
    store, proj = _project_with_two_results(tmp_path)
    mgr = RetouchManager(tmp_path / "data")
    out = mgr.create(proj.id, "res_target")
    assert "session_id" in out and "working_image_id" in out
    assert any(s["image_id"] == "res_source" for s in out["sources"])
    # working pyramid exists
    from planefuse_server.tiles import tile_path
    wid = out["working_image_id"]
    assert tile_path(proj.cache / "tiles", wid, 0, 0, 0).exists()


def test_stroke_returns_dirty_tiles_and_persists(tmp_path):
    store, proj = _project_with_two_results(tmp_path)
    mgr = RetouchManager(tmp_path / "data")
    sid = mgr.create(proj.id, "res_target")["session_id"]
    res = mgr.stroke(sid, _stroke_dict())
    assert res["dirty_tiles"]
    assert res["rev"] == 1
    # persisted in project.json
    reloaded = ProjectStore(tmp_path / "data").get(proj.id)
    rec = next(r for r in reloaded.retouch if r["id"] == sid)
    assert len(rec["strokes"]) == 1


def test_undo_redo_flatten_determinism(tmp_path):
    store, proj = _project_with_two_results(tmp_path)
    mgr = RetouchManager(tmp_path / "data")
    sid = mgr.create(proj.id, "res_target")["session_id"]
    mgr.stroke(sid, _stroke_dict())
    id_a = mgr.flatten(sid, "A")["image_id"]
    mgr.undo(sid)
    mgr.redo(sid)
    id_b = mgr.flatten(sid, "B")["image_id"]

    from planefuse.io import load_image
    proj2 = ProjectStore(tmp_path / "data").get(proj.id)
    a = load_image(proj2.images[id_a]["path"]).pixels
    b = load_image(proj2.images[id_b]["path"]).pixels
    assert np.array_equal(a, b)


def test_session_rehydrates_after_restart(tmp_path):
    store, proj = _project_with_two_results(tmp_path)
    mgr = RetouchManager(tmp_path / "data")
    sid = mgr.create(proj.id, "res_target")["session_id"]
    mgr.stroke(sid, _stroke_dict())

    mgr2 = RetouchManager(tmp_path / "data")          # fresh manager, empty cache
    res = mgr2.stroke(sid, _stroke_dict())            # must rehydrate from project.json
    assert res["rev"] == 2                            # rev persisted + advanced


def test_unknown_session_raises_lookup(tmp_path):
    _project_with_two_results(tmp_path)
    mgr = RetouchManager(tmp_path / "data")
    import pytest
    with pytest.raises(LookupError):
        mgr.stroke("nope", _stroke_dict())
```

- [ ] **Step 3: Run to verify failure** — FAIL (no `retouch` module).

- [ ] **Step 4: Implement** `server/src/planefuse_server/retouch.py`:

```python
"""Retouch session manager (SPEC §11): persistence, rehydrate, tile rebuild."""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np

from planefuse.io import load_image, save_image
from planefuse.retouch import RetouchSession, Stroke
from planefuse_server.projects import Project, ProjectStore
from planefuse_server.tiles import build_pyramid, rebuild_region


def _stroke_from_dict(d: dict) -> Stroke:
    pts = [(float(p[0]), float(p[1]), float(p[2]) if len(p) >= 3 else 1.0) for p in d["points"]]
    return Stroke(source_id=d["source_id"], points=pts, radius=float(d["radius"]),
                  hardness=float(d.get("hardness", 0.5)), opacity=float(d.get("opacity", 1.0)),
                  mode=d.get("mode", "normal"))


class RetouchManager:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._cache: dict[str, RetouchSession] = {}

    def _store(self) -> ProjectStore:
        return ProjectStore(self.data_dir)

    def _find(self, session_id: str) -> tuple[ProjectStore, Project, dict]:
        store = self._store()
        for proj in store.list():
            for rec in proj.retouch:
                if rec["id"] == session_id:
                    return store, proj, rec
        raise LookupError(f"unknown retouch session {session_id}")

    def _load_session(self, proj: Project, rec: dict) -> RetouchSession:
        sid = rec["id"]
        if sid in self._cache:
            return self._cache[sid]
        base = load_image(Path(proj.images[rec["target_image_id"]]["path"])).pixels
        sources = {iid: load_image(Path(proj.images[iid]["path"])).pixels for iid in rec["sources"]}
        session = RetouchSession(base, sources)
        for sd in rec["strokes"]:
            session.apply(_stroke_from_dict(sd))
        self._cache[sid] = session
        return session

    def create(self, pid: str, target_image_id: str) -> dict:
        store = self._store()
        proj = store.get(pid)
        if proj is None:
            raise LookupError(f"unknown project {pid}")
        if target_image_id not in proj.images:
            raise ValueError(f"unknown image {target_image_id}")
        base = load_image(Path(proj.images[target_image_id]["path"])).pixels
        sources = [iid for iid, info in proj.images.items()
                   if info.get("kind") == "result" and iid != target_image_id]
        for iid in sources:
            shp = load_image(Path(proj.images[iid]["path"])).pixels.shape
            if shp != base.shape:
                raise ValueError(f"source {iid} shape {shp} != target {base.shape}")
        sid = uuid.uuid4().hex[:12]
        working_id = uuid.uuid4().hex[:12]
        levels = build_pyramid(base, proj.cache / "tiles" / working_id)
        h, w = base.shape[:2]
        proj.images[working_id] = {"kind": "retouch-working", "levels": levels,
                                   "width": w, "height": h}
        rec = {"id": sid, "target_image_id": target_image_id, "working_image_id": working_id,
               "sources": sources, "strokes": [], "rev": 0}
        proj.retouch.append(rec)
        store.save(proj)
        sources_arr = {iid: load_image(Path(proj.images[iid]["path"])).pixels for iid in sources}
        self._cache[sid] = RetouchSession(base, sources_arr)
        return {"session_id": sid, "working_image_id": working_id,
                "sources": [{"image_id": iid, "method": proj.images[iid].get("method")}
                            for iid in sources]}

    def list_sessions(self, pid: str) -> list[dict]:
        proj = self._store().get(pid)
        if proj is None:
            raise LookupError(f"unknown project {pid}")
        return proj.retouch

    def _rebuild_and_save(self, store: ProjectStore, proj: Project, rec: dict,
                          composite: np.ndarray, bbox: tuple[int, int, int, int]) -> dict:
        if bbox != (0, 0, 0, 0):
            tiles_root = proj.cache / "tiles" / rec["working_image_id"]
            levels = int(proj.images[rec["working_image_id"]]["levels"])
            dirty = rebuild_region(composite, tiles_root, levels, bbox)
        else:
            dirty = []
        rec["rev"] = int(rec.get("rev", 0)) + 1
        store.save(proj)
        return {"rev": rec["rev"], "dirty_tiles": dirty}

    def stroke(self, session_id: str, stroke: dict) -> dict:
        store, proj, rec = self._find(session_id)
        session = self._load_session(proj, rec)
        bbox = session.apply(_stroke_from_dict(stroke))
        if bbox != (0, 0, 0, 0):
            rec["strokes"].append(stroke)   # preserve the exact client dict
        return self._rebuild_and_save(store, proj, rec, session.composite, bbox)

    def _mutate(self, session_id: str, op) -> dict:
        """undo/redo: run op, then resync the persisted stroke list to the engine's."""
        store, proj, rec = self._find(session_id)
        session = self._load_session(proj, rec)
        bbox = op(session)
        rec["strokes"] = [_stroke_to_dict(s) for s in session.strokes]  # keep persistence in sync
        return self._rebuild_and_save(store, proj, rec, session.composite, bbox)

    def undo(self, session_id: str) -> dict:
        return self._mutate(session_id, lambda s: s.undo())

    def redo(self, session_id: str) -> dict:
        return self._mutate(session_id, lambda s: s.redo())

    def flatten(self, session_id: str, name: str) -> dict:
        store, proj, rec = self._find(session_id)
        session = self._load_session(proj, rec)
        new_id = uuid.uuid4().hex[:12]
        path = proj.cache / f"{new_id}.tif"
        save_image(session.composite, path, bit_depth=16)
        proj.images[new_id] = {"kind": "result", "path": str(path), "method": "retouch", "name": name}
        rec["rev"] = int(rec.get("rev", 0)) + 1
        store.save(proj)
        return {"image_id": new_id}

    def delete(self, session_id: str) -> None:
        store, proj, rec = self._find(session_id)
        proj.retouch = [r for r in proj.retouch if r["id"] != session_id]
        proj.images.pop(rec["working_image_id"], None)
        store.save(proj)
        self._cache.pop(session_id, None)
```

Add a `_stroke_to_dict` helper next to `_stroke_from_dict`:

```python
def _stroke_to_dict(s: Stroke) -> dict:
    return {"source_id": s.source_id, "points": [list(p) for p in s.points], "radius": s.radius,
            "hardness": s.hardness, "opacity": s.opacity, "mode": s.mode}
```

Note: `stroke` records the exact incoming client dict; `undo`/`redo` go through `_mutate`, which resyncs `rec["strokes"]` from the engine's `session.strokes` so the persisted list never drifts from the in-memory session (critical for rehydrate determinism after an undo + restart).

- [ ] **Step 5: Run to verify pass** — `uv run pytest tests/server/test_retouch_manager.py -q` → PASS. Fix any drift (the determinism + rehydrate tests are the important ones).

- [ ] **Step 6: ruff/mypy + commit**

```bash
uv run ruff check server/src/planefuse_server/retouch.py server/src/planefuse_server/projects.py tests/server/test_retouch_manager.py
uv run mypy server/src
git add server/src/planefuse_server/retouch.py server/src/planefuse_server/projects.py tests/server/test_retouch_manager.py
git commit -m "feat: RetouchManager — sessions, persistence, rehydrate, flatten"
```

---

### Task 5: Retouch routes + wiring + export guard; finish

**Files:**
- Create: `server/src/planefuse_server/api/retouch.py`
- Modify: `server/src/planefuse_server/main.py` (app-scoped manager + include router)
- Modify: `server/src/planefuse_server/runners.py` (export guard)
- Test: `tests/server/test_retouch_api.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing tests** — `tests/server/test_retouch_api.py`:

```python
import numpy as np
from fastapi.testclient import TestClient

from planefuse.io import save_image
from planefuse_server.main import create_app


def _client_with_results(tmp_path):
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    pid = c.post("/api/projects", json={"path": str(tmp_path / "proj"), "name": "P"}).json()["id"]
    proj_cache = tmp_path / "proj" / "cache"
    proj_cache.mkdir(parents=True, exist_ok=True)
    # register two results directly via the viewer register route (builds pyramids + image ids)
    ids = {}
    for name, val in (("target", 0.2), ("source", 0.8)):
        p = proj_cache / f"{name}.tif"
        save_image(np.full((48, 60, 3), val, np.float32), p, bit_depth=16)
        # register as a *result* by writing through the project file is complex; use viewer register
        r = c.post(f"/api/projects/{pid}/viewer/register", json={"path": str(p)})
        ids[name] = r.json()["image_id"]
    return c, pid, ids


def _stroke(source_id: str):
    return {"source_id": source_id, "points": [[30, 24, 1.0]], "radius": 12.0,
            "hardness": 0.5, "opacity": 1.0, "mode": "normal"}
```

> **Note for the implementer — fixture must produce two `result`-kind images.** `viewer/register` produces `kind="view"`, which `RetouchManager.create` does NOT treat as a source. Replace the `_client_with_results` stub above with the **stack-job approach**: import a synthetic stack (reuse `generate_stack` as in `tests/server/test_job_api.py`), enqueue a `pmax` job and a `dmap` job on the same frames, wait for both `done` (the enqueue+poll-`/api/jobs/{id}` helper is in `test_job_api.py`), then read the two `result` `image_id`s from `GET /api/projects/{pid}` (`images` where `kind == "result"`). pmax and dmap of the same frames are identical-shape, so they validate as sources. Always pass a real `source_id` (the other result id) to `_stroke(...)`.

```python
def test_retouch_lifecycle_and_determinism(tmp_path):
    # Build a project with two results via stack jobs, then:
    #   create session(target) -> sources includes the other result
    #   working tiles fetchable via the viewer tile route
    #   stroke -> dirty_tiles non-empty
    #   flatten A; undo; redo; flatten B; assert A == B (array equality)
    ...


def test_unknown_session_404(tmp_path):
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    assert c.post("/api/retouch/nope/stroke", json={"source_id": "x", "points": [[1, 1]],
                                                     "radius": 4.0}).status_code == 404
```

Flesh out `test_retouch_lifecycle_and_determinism` using the stack-job helper (reuse `generate_stack`, enqueue `pmax` then `dmap`, wait for both `done`, read the two result ids from `GET /api/projects/{pid}`). Assertions:
- `POST /api/projects/{pid}/retouch {target_image_id}` → 200; `sources` includes the other result id; `working_image_id` returned.
- `GET /api/viewer/{working_image_id}/tile/{levels-1}/0/0` → 200 (use levels from the working image entry, or just request `/0/0/0`).
- `POST /api/retouch/{sid}/stroke {stroke with source_id=other result}` → 200, `dirty_tiles` non-empty, `rev==1`.
- flatten→A, undo, redo, flatten→B; load both result tifs; `np.array_equal(A, B)`.

- [ ] **Step 2: Run to verify failure** — FAIL (no routes).

- [ ] **Step 3: Implement** `server/src/planefuse_server/api/retouch.py`:

```python
"""Retouch session routes (SPEC §9, §11)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api")


class CreateBody(BaseModel):
    target_image_id: str


class StrokeBody(BaseModel):
    source_id: str
    points: list[list[float]]
    radius: float
    hardness: float = 0.5
    opacity: float = 1.0
    mode: str = "normal"


class FlattenBody(BaseModel):
    name: str = "retouched"


def _mgr(request: Request):
    return request.app.state.retouch


def _guard(fn):
    try:
        return JSONResponse(content=fn())
    except LookupError as e:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": str(e)})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": "bad_request", "detail": str(e)})


@router.post("/projects/{pid}/retouch")
def create(pid: str, body: CreateBody, request: Request) -> JSONResponse:
    return _guard(lambda: _mgr(request).create(pid, body.target_image_id))


@router.get("/projects/{pid}/retouch")
def list_sessions(pid: str, request: Request) -> JSONResponse:
    return _guard(lambda: {"sessions": _mgr(request).list_sessions(pid)})


@router.delete("/retouch/{sid}")
def delete(sid: str, request: Request) -> JSONResponse:
    def go():
        _mgr(request).delete(sid)
        return {"deleted": sid}
    return _guard(go)


@router.post("/retouch/{sid}/stroke")
def stroke(sid: str, body: StrokeBody, request: Request) -> JSONResponse:
    return _guard(lambda: _mgr(request).stroke(sid, body.model_dump()))


@router.post("/retouch/{sid}/undo")
def undo(sid: str, request: Request) -> JSONResponse:
    return _guard(lambda: _mgr(request).undo(sid))


@router.post("/retouch/{sid}/redo")
def redo(sid: str, request: Request) -> JSONResponse:
    return _guard(lambda: _mgr(request).redo(sid))


@router.post("/retouch/{sid}/flatten")
def flatten(sid: str, body: FlattenBody, request: Request) -> JSONResponse:
    return _guard(lambda: _mgr(request).flatten(sid, body.name))
```

In `main.py` `create_app`: after the job queue setup, add:

```python
    from planefuse_server.retouch import RetouchManager
    app.state.retouch = RetouchManager(data_dir)
```
and `from planefuse_server.api import retouch` + `app.include_router(retouch.router)` (before the SPA catch-all mount).

In `runners.py` `make_export_runner.run`, replace the `if info is None` guard:

```python
        if info is None:
            raise ValueError(f"unknown image_id {image_id}")
        if "path" not in info:
            raise ValueError(f"image {image_id} is not exportable (no path)")
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/server/test_retouch_api.py -q` → PASS.

- [ ] **Step 5: Full suite + gates**

```bash
uv run pytest -q          # engine + server all green
uv run ruff check .
uv run mypy engine/src server/src
```
Fix any issues.

- [ ] **Step 6: README + commit**

Update `README.md` status: `- [~] M5 — retouch engine + server done; retouch UI (Part 2) pending`.

```bash
git add server/src/planefuse_server/api/retouch.py server/src/planefuse_server/main.py server/src/planefuse_server/runners.py tests/server/test_retouch_api.py README.md
git commit -m "feat: retouch REST routes, app-scoped manager, export guard; M5 Part 1 done"
```

- [ ] **Step 7: Finish the branch**

Use superpowers-extended-cc:finishing-a-development-branch to merge `feat/m5-retouch` into `main` (no-ff).

---

## Known caveats (documented, not blocking)

- **Retouch UI (screen 5) + retouch Playwright** — M5 Part 2 (closes the milestone done-gate per SPEC §14).
- **Aligned-frame sources, sharpness-at-cursor hint, split/synced view** — deferred (M6 / follow-ups).
- **Source arrays held in memory** per session (a few results) — fine for local single-user; revisit if many large sources.
- **`rebuild_region` re-resizes the full composite per level** — cheap for one result image; the saving is in tile writes + client refetch, not the resize.
