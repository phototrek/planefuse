"""Server-side folder browser + native OS picker (SPEC §9)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/fs")

_IMAGE_EXT = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


def _entry(child: Path) -> dict | None:
    """Describe one directory child, or None if it can't be read.

    OneDrive "files on-demand" placeholders, reparse points and locked system
    entries raise OSError on is_dir(); skip them rather than 500 the listing.
    """
    try:
        is_dir = child.is_dir()
    except OSError:
        return None
    return {"name": child.name, "path": str(child), "is_dir": is_dir}


@router.get("/list")
def list_dir(path: str) -> JSONResponse:
    p = Path(path)
    try:
        if not p.is_dir():
            return JSONResponse(status_code=400,
                                content={"error": "not_a_directory", "detail": path})
        children = list(p.iterdir())
    except OSError as e:
        return JSONResponse(status_code=400, content={"error": "cannot_list", "detail": str(e)})

    entries = []
    image_count = 0
    for child in children:
        info = _entry(child)
        if info is None:
            continue
        entries.append(info)
        if not info["is_dir"] and child.suffix.lower() in _IMAGE_EXT:
            image_count += 1
    # Dirs first, then case-insensitive by name.
    entries.sort(key=lambda info: (not info["is_dir"], info["name"].lower()))
    return JSONResponse(content={"path": str(p), "entries": entries, "image_count": image_count})


class PickBody(BaseModel):
    mode: str = "directory"  # "directory" | "files"


def _run_picker(mode: str) -> list[str]:
    """Open a native OS file/folder dialog in a subprocess and return the chosen
    absolute paths ([] if cancelled). The dialog opens on the host running the
    server — localhost desktop use only."""
    proc = subprocess.run(
        [sys.executable, "-m", "focusstack_server.picker", mode],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[:500] or "picker subprocess failed")
    return json.loads(proc.stdout.strip() or "[]")


@router.post("/pick")
def pick(body: PickBody) -> JSONResponse:
    mode = "files" if body.mode == "files" else "directory"
    try:
        paths = _run_picker(mode)
    except Exception as e:  # subprocess failure, no display, timeout, bad JSON
        return JSONResponse(status_code=500, content={"error": "picker_failed", "detail": str(e)})
    return JSONResponse(content={"paths": paths})
