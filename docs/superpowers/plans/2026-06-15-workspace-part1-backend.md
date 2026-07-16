# Workspace UI — Part 1 (Backend) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the server the project + frame primitives the single-screen workspace needs — auto scratch projects, additive frame add/remove, a save/rename endpoint, and startup pruning of stale unsaved scratch projects — with no change to existing UI.

**Architecture:** The server stays project-centric (a project is a directory with `project.json` + `cache/`). We make a project obtainable with no user input (scratch dir under the data dir), add additive frame verbs alongside the existing replace-style `scan`, add a top-level project `PATCH`, and prune abandoned scratch dirs on startup. All changes live in `server/src/planefuse_server/`. Spec: `docs/superpowers/specs/2026-06-15-workspace-ui-design.md` §4.

**Tech Stack:** Python 3.12, FastAPI, pydantic, pytest + `fastapi.testclient`. Run via `uv run pytest`.

**Scope note:** This part is backend-only and fully covered by pytest. The matching client (`api.ts`) methods and all UI live in Part 2.

---

## Chunk 1: Backend primitives

### Task 1: Scratch project create (empty path) + unsaved default

**Files:**
- Modify: `server/src/planefuse_server/projects.py` (`ProjectStore.create`)
- Modify: `server/src/planefuse_server/api/projects.py` (`CreateProject`, `create`)
- Test: `tests/server/test_workspace_projects.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/server/test_workspace_projects.py`:

```python
from fastapi.testclient import TestClient

from planefuse_server.main import create_app


def _c(tmp_path):
    return TestClient(create_app(data_dir=tmp_path / "data"))


def test_create_scratch_project_with_empty_path(tmp_path):
    c = _c(tmp_path)
    r = c.post("/api/projects", json={})           # no path, no name
    assert r.status_code == 200, r.text
    proj = r.json()
    # scratch dir lives under <data_dir>/scratch/<id>
    assert (tmp_path / "data" / "scratch" / proj["id"]).is_dir()
    assert proj["ui_state"]["saved"] is False
    # still listable/gettable like any project
    assert c.get(f"/api/projects/{proj['id']}").json()["id"] == proj["id"]


def test_create_with_explicit_path_still_works(tmp_path):
    c = _c(tmp_path)
    proj_dir = tmp_path / "mine"
    proj = c.post("/api/projects", json={"path": str(proj_dir), "name": "Mine"}).json()
    assert (proj_dir / "project.json").exists()
    assert proj["name"] == "Mine"
    assert proj["ui_state"]["saved"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/server/test_workspace_projects.py -q`
Expected: FAIL (empty body rejected — `path` required — and no `scratch/` dir / `saved` key).

- [ ] **Step 3: Implement `ProjectStore.create`**

First confirm the route is the only caller of the method whose signature changes
(should print exactly the one call in `api/projects.py`):

```bash
grep -rn "\.create(" server/src
```

In `server/src/planefuse_server/projects.py`, replace the `create` method:

```python
    def create(self, directory: Path | None, name: str) -> Project:
        pid = uuid.uuid4().hex[:12]
        if directory is None:
            directory = self.data_dir / "scratch" / pid
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "cache").mkdir(exist_ok=True)
        proj = Project(id=pid, name=name, directory=str(directory),
                       ui_state={"saved": False})
        self._save(proj)
        reg = self._read_registry()
        reg[pid] = str(directory)
        self._write_registry(reg)
        return proj
```

- [ ] **Step 4: Implement the route**

In `server/src/planefuse_server/api/projects.py`, change the model + route:

```python
class CreateProject(BaseModel):
    path: str = ""
    name: str = "Untitled"


@router.post("")
def create(body: CreateProject, request: Request) -> dict:
    directory = Path(body.path) if body.path else None
    proj = _store(request).create(directory, body.name)
    return asdict_no_props(proj)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/server/test_workspace_projects.py tests/server/test_projects.py -q`
Expected: PASS (new tests pass; existing project tests still pass).

- [ ] **Step 6: Commit**

```bash
git add server/src/planefuse_server/projects.py server/src/planefuse_server/api/projects.py tests/server/test_workspace_projects.py
git commit -m "feat: scratch projects (empty-path create) with unsaved default"
```

### Task 2: Save / rename endpoint

**Files:**
- Modify: `server/src/planefuse_server/api/projects.py` (add `SaveBody`, `save` route)
- Test: `tests/server/test_workspace_projects.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/server/test_workspace_projects.py`:

```python
def test_patch_project_name_and_saved(tmp_path):
    c = _c(tmp_path)
    pid = c.post("/api/projects", json={}).json()["id"]
    r = c.patch(f"/api/projects/{pid}", json={"name": "Beetle stack", "saved": True})
    assert r.status_code == 200, r.text
    got = c.get(f"/api/projects/{pid}").json()
    assert got["name"] == "Beetle stack"        # top-level field
    assert got["ui_state"]["saved"] is True      # flag in ui_state


def test_patch_unknown_project_404(tmp_path):
    assert _c(tmp_path).patch("/api/projects/nope", json={"name": "x"}).status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/server/test_workspace_projects.py -q -k patch`
Expected: FAIL (405/404 — no PATCH `/{pid}` route).

- [ ] **Step 3: Implement the route**

In `server/src/planefuse_server/api/projects.py`, add (after `patch_ui_state`):

```python
class SaveBody(BaseModel):
    name: str | None = None
    saved: bool | None = None


@router.patch("/{pid}")
def save_project(pid: str, body: SaveBody, request: Request) -> JSONResponse:
    store = _store(request)
    proj = store.get(pid)
    if proj is None:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": pid})
    if body.name is not None:
        proj.name = body.name
    if body.saved is not None:
        proj.ui_state["saved"] = body.saved
    store.save(proj)
    return JSONResponse(content={"id": proj.id, "name": proj.name, "ui_state": proj.ui_state})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/server/test_workspace_projects.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/src/planefuse_server/api/projects.py tests/server/test_workspace_projects.py
git commit -m "feat: PATCH /projects/{id} for save/rename (name + ui_state.saved)"
```

### Task 3: Additive frame add / remove (+ shared report helper)

**Files:**
- Modify: `server/src/planefuse_server/api/frames.py`
- Test: `tests/server/test_workspace_frames.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/server/test_workspace_frames.py`:

```python
import numpy as np
from fastapi.testclient import TestClient

from planefuse.io import save_image
from planefuse_server.main import create_app


def _proj(tmp_path):
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    pid = c.post("/api/projects", json={}).json()["id"]
    return c, pid


def _frames(c, pid):
    return c.get(f"/api/projects/{pid}").json()["frames"]


def test_add_folder_then_dedupe_then_file(tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    for i in range(3):
        save_image(np.zeros((16, 16, 3), np.float32), src / f"f_{i:03d}.tif", bit_depth=16)
    extra = tmp_path / "extra.tif"
    save_image(np.zeros((16, 16, 3), np.float32), extra, bit_depth=16)

    c, pid = _proj(tmp_path)
    r = c.post(f"/api/projects/{pid}/frames/add", json={"paths": [str(src)]})
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True
    assert len(_frames(c, pid)) == 3

    # re-adding the same folder must not duplicate
    c.post(f"/api/projects/{pid}/frames/add", json={"paths": [str(src)]})
    assert len(_frames(c, pid)) == 3

    # an individual file appends
    c.post(f"/api/projects/{pid}/frames/add", json={"paths": [str(extra)]})
    assert len(_frames(c, pid)) == 4


def test_remove_frames(tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    paths = []
    for i in range(3):
        p = src / f"f_{i:03d}.tif"
        save_image(np.zeros((16, 16, 3), np.float32), p, bit_depth=16)
        paths.append(str(p))
    c, pid = _proj(tmp_path)
    c.post(f"/api/projects/{pid}/frames/add", json={"paths": [str(src)]})
    r = c.post(f"/api/projects/{pid}/frames/remove", json={"paths": [paths[0]]})
    assert r.status_code == 200, r.text
    remaining = _frames(c, pid)
    assert paths[0] not in remaining
    assert len(remaining) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/server/test_workspace_frames.py -q`
Expected: FAIL (404/405 — no add/remove routes).

- [ ] **Step 3: Refactor + implement in `frames.py`**

In `server/src/planefuse_server/api/frames.py`, add a `PathsBody`, an `_expand` helper, and a shared `_report` builder; reimplement `scan` on top of `_report`; add the two routes. Full additions/changes:

```python
class PathsBody(BaseModel):
    paths: list[str]


def _expand(raw: list[str]) -> list[str]:
    """Expand a mix of folders and image files into image-file paths."""
    out: list[str] = []
    for r in raw:
        p = Path(r)
        if p.is_dir():
            out.extend(str(x) for x in _scan_paths(p))
        elif p.is_file() and p.suffix.lower() in _IMAGE_EXT:
            out.append(str(p))
    return out


def _report(proj) -> dict:
    """Validate the project's full frame list and shape the scan-style report."""
    report = validate_stack([Path(p) for p in proj.frames])
    return {
        "ok": report.ok,
        "width": report.width, "height": report.height, "bit_depth": report.bit_depth,
        "files": [{"name": Path(f.path).name, "path": str(f.path),
                   "status": f.status, "message": f.message} for f in report.files],
    }
```

Replace the existing `scan` body (keep replace semantics, but build the report via `_report`):

```python
@router.post("/{pid}/frames/scan")
def scan(pid: str, body: ScanBody, request: Request) -> JSONResponse:
    store = _store(request)
    proj = store.get(pid)
    if proj is None:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": pid})
    folder = Path(body.path)
    if not folder.is_dir():
        return JSONResponse(status_code=400, content={"error": "not_a_directory", "detail": body.path})
    proj.frames = [str(p) for p in _scan_paths(folder)]
    store.save(proj)
    return JSONResponse(content=_report(proj))
```

Add the two new routes (after `scan`):

```python
@router.post("/{pid}/frames/add")
def add_frames(pid: str, body: PathsBody, request: Request) -> JSONResponse:
    store = _store(request)
    proj = store.get(pid)
    if proj is None:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": pid})
    frames = list(proj.frames)
    for p in _expand(body.paths):
        if p not in frames:
            frames.append(p)
    proj.frames = frames
    store.save(proj)
    return JSONResponse(content=_report(proj))


@router.post("/{pid}/frames/remove")
def remove_frames(pid: str, body: PathsBody, request: Request) -> JSONResponse:
    store = _store(request)
    proj = store.get(pid)
    if proj is None:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": pid})
    drop = set(body.paths)
    proj.frames = [f for f in proj.frames if f not in drop]
    store.save(proj)
    return JSONResponse(content=_report(proj))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/server/test_workspace_frames.py tests/server/test_frames.py -q`
Expected: PASS (new add/remove tests pass; existing scan tests still pass).

- [ ] **Step 5: Commit**

```bash
git add server/src/planefuse_server/api/frames.py tests/server/test_workspace_frames.py
git commit -m "feat: additive frames/add and frames/remove (folders + files, dedup)"
```

### Task 4: Prune stale unsaved scratch projects on startup

**Files:**
- Modify: `server/src/planefuse_server/main.py` (`_prune_scratch` + call in `create_app`)
- Test: `tests/server/test_workspace_prune.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/server/test_workspace_prune.py`:

```python
import os
import time

from fastapi.testclient import TestClient

from planefuse_server.main import create_app


def _client(tmp_path):
    return TestClient(create_app(data_dir=tmp_path / "data"))


def _age(data_dir, pid, days):
    pj = data_dir / "scratch" / pid / "project.json"
    old = time.time() - days * 86400
    os.utime(pj, (old, old))


def test_prune_removes_old_unsaved_scratch_only(tmp_path):
    data = tmp_path / "data"
    c = _client(tmp_path)
    old_unsaved = c.post("/api/projects", json={}).json()["id"]
    old_saved = c.post("/api/projects", json={}).json()["id"]
    recent_unsaved = c.post("/api/projects", json={}).json()["id"]

    c.patch(f"/api/projects/{old_saved}", json={"saved": True})
    _age(data, old_unsaved, 30)
    _age(data, old_saved, 30)
    # recent_unsaved keeps its fresh mtime

    c2 = _client(tmp_path)  # new app over same data_dir triggers prune on startup
    ids = {p["id"] for p in c2.get("/api/projects").json()}
    assert old_unsaved not in ids                      # pruned
    assert not (data / "scratch" / old_unsaved).exists()
    assert old_saved in ids                            # saved -> kept
    assert recent_unsaved in ids                       # recent -> kept
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/server/test_workspace_prune.py -q`
Expected: FAIL (nothing pruned — `old_unsaved` still present).

- [ ] **Step 3: Implement prune in `main.py`**

Add imports at the top of `server/src/planefuse_server/main.py` (alongside existing imports):

```python
import json
import shutil
import time
```

Add the helper (module level, e.g. after `_static_dir`). It reads `project.json`
directly rather than via `ProjectStore.get` because it needs the raw on-disk
`ui_state` **and** the file mtime, which the store API doesn't expose:

```python
def _prune_scratch(data_dir: Path, max_age_days: int = 7) -> None:
    """Delete abandoned scratch projects: those under data_dir/scratch whose
    project.json is unsaved and older than max_age_days. Best-effort; never
    touches dirs outside data_dir/scratch."""
    scratch = Path(data_dir) / "scratch"
    if not scratch.is_dir():
        return
    from planefuse_server.projects import ProjectStore

    store = ProjectStore(data_dir)
    cutoff = time.time() - max_age_days * 86400
    for d in scratch.iterdir():
        pj = d / "project.json"
        if not pj.is_file():
            continue
        try:
            data = json.loads(pj.read_text())
            if data.get("ui_state", {}).get("saved"):
                continue
            if pj.stat().st_mtime >= cutoff:
                continue
            shutil.rmtree(d, ignore_errors=True)
            pid = data.get("id")
            if pid:
                store.unregister(pid)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            log.warning("prune: skipped %s", d, exc_info=True)
```

Call it inside `create_app`, right after `data_dir.mkdir(...)`:

```python
    data_dir.mkdir(parents=True, exist_ok=True)
    _prune_scratch(data_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/server/test_workspace_prune.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/src/planefuse_server/main.py tests/server/test_workspace_prune.py
git commit -m "feat: prune stale unsaved scratch projects on startup"
```

### Task 5: Full backend gate

- [ ] **Step 1: Run the whole backend suite + static checks**

```bash
uv run pytest -q
uv run ruff check .
uv run mypy engine/src server/src
```

Expected: all exit 0. Fix anything that regressed before handing off to Part 2.

- [ ] **Step 2: Commit any lint/type fixups**

```bash
git commit -am "chore: lint/type fixups for workspace backend"   # only if needed
```
