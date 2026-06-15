"""Frame import (scan) + batch auto-grouping (SPEC §9, §12)."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from focusstack.io import validate_stack
from focusstack_server.projects import Project, ProjectStore

router = APIRouter(prefix="/api/projects")
_IMAGE_EXT = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


class ScanBody(BaseModel):
    path: str


class PathsBody(BaseModel):
    paths: list[str]


def _store(request: Request) -> ProjectStore:
    return ProjectStore(request.app.state.data_dir)


def _scan_paths(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in _IMAGE_EXT)


def _expand(raw: list[str]) -> list[str]:
    """Expand a mix of folders and image files into canonical image-file paths.
    Paths are resolved so dedup/removal use a stable key (the same file via a
    different spelling or case — e.g. on Windows — collapses to one entry)."""
    out: list[str] = []
    for r in raw:
        p = Path(r)
        if p.is_dir():
            out.extend(str(x.resolve()) for x in _scan_paths(p))
        elif p.is_file() and p.suffix.lower() in _IMAGE_EXT:
            out.append(str(p.resolve()))
    return out


def _report(proj: Project) -> dict:
    """Validate the project's full frame list and shape the scan-style report.
    Re-reads every frame on each call (same cost as `scan`); fine for now."""
    report = validate_stack([Path(p) for p in proj.frames])
    return {
        "ok": report.ok,
        "width": report.width, "height": report.height, "bit_depth": report.bit_depth,
        "files": [{"name": Path(f.path).name, "path": str(f.path),
                   "status": f.status, "message": f.message} for f in report.files],
    }


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
    # Expand folders to their files (symmetric with add) and also accept the raw
    # strings, so a stored entry can be dropped even if the file no longer exists.
    drop = set(_expand(body.paths)) | set(body.paths)
    proj.frames = [f for f in proj.frames if f not in drop]
    store.save(proj)
    return JSONResponse(content=_report(proj))


@router.post("/{pid}/frames/auto-group")
def auto_group(pid: str, request: Request, max_gap: int = 3) -> JSONResponse:
    """Group the project's frames into stacks by gaps in their trailing file
    number (EXIF-time grouping needs pyexiv2 — M6). A gap > max_gap starts a new
    group. Returns proposed groups for UI confirmation."""
    store = _store(request)
    proj = store.get(pid)
    if proj is None:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": pid})

    def _num(p: str) -> int | None:
        m = re.search(r"(\d+)(?=\.[^.]+$)", Path(p).name)
        return int(m.group(1)) if m else None

    groups: list[list[str]] = []
    cur: list[str] = []
    last: int | None = None
    for fp in proj.frames:
        n = _num(fp)
        if last is not None and n is not None and (n - last) > max_gap and cur:
            groups.append(cur)
            cur = []
        cur.append(fp)
        last = n
    if cur:
        groups.append(cur)
    return JSONResponse(content={"groups": groups})
