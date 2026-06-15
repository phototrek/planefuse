"""Server-side folder browser (SPEC §9)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/fs")

_IMAGE_EXT = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


@router.get("/list")
def list_dir(path: str) -> JSONResponse:
    p = Path(path)
    if not p.is_dir():
        return JSONResponse(status_code=400, content={"error": "not_a_directory", "detail": path})
    entries = []
    image_count = 0
    try:
        for child in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
            is_dir = child.is_dir()
            entries.append({"name": child.name, "path": str(child), "is_dir": is_dir})
            if not is_dir and child.suffix.lower() in _IMAGE_EXT:
                image_count += 1
    except PermissionError:
        return JSONResponse(status_code=403, content={"error": "permission_denied", "detail": path})
    return JSONResponse(content={"path": str(p), "entries": entries, "image_count": image_count})
