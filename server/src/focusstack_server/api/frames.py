"""Frame import (scan) + batch auto-grouping (SPEC §9, §12)."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from focusstack.io import validate_stack
from focusstack_server.projects import ProjectStore

router = APIRouter(prefix="/api/projects")
_IMAGE_EXT = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


class ScanBody(BaseModel):
    path: str


def _store(request: Request) -> ProjectStore:
    return ProjectStore(request.app.state.data_dir)


def _scan_paths(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in _IMAGE_EXT)


@router.post("/{pid}/frames/scan")
def scan(pid: str, body: ScanBody, request: Request) -> JSONResponse:
    store = _store(request)
    proj = store.get(pid)
    if proj is None:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": pid})
    folder = Path(body.path)
    if not folder.is_dir():
        return JSONResponse(status_code=400, content={"error": "not_a_directory", "detail": body.path})
    paths = _scan_paths(folder)
    report = validate_stack(paths)
    proj.frames = [str(p) for p in paths]
    store.save(proj)
    return JSONResponse(content={
        "ok": report.ok,
        "width": report.width, "height": report.height, "bit_depth": report.bit_depth,
        "files": [{"name": Path(f.path).name, "path": str(f.path),
                   "status": f.status, "message": f.message} for f in report.files],
    })


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
