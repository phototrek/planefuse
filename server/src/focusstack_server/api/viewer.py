"""Viewer tile serving + image registration/pyramid build (SPEC §9)."""

from __future__ import annotations

import io
import threading
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image
from pydantic import BaseModel

from focusstack.io import ProcessingDomain, load_image, tonemap_preview
from focusstack.io.display import DISPLAY_TONEMAP_REVISION
from focusstack_server.image_data import load_project_frame
from focusstack_server.projects import ProjectStore
from focusstack_server.tiles import TILE, build_pyramid, tile_path

router = APIRouter(prefix="/api")
_registration_lock = threading.Lock()


class RegisterBody(BaseModel):
    path: str


def _registered_view(proj, path: str) -> tuple[str, dict] | None:
    for image_id, info in proj.images.items():
        if info.get("kind") == "view" and info.get("path") == path:
            return image_id, info
    return None


def _ensure_display_variant(store: ProjectStore, proj, image_id: str, info: dict) -> None:
    """(Re)build the tonemapped pyramid for views whose variant predates the
    current tonemap math (or the feature entirely)."""
    if info.get("display_rev") == DISPLAY_TONEMAP_REVISION:
        return
    pixels, domain, metadata = load_project_frame(proj, Path(str(info["path"])))
    tonemapped = domain is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB
    if tonemapped:
        build_pyramid(
            tonemap_preview(pixels, metadata),
            proj.cache / "tiles" / image_id / "display",
        )
    info["display_tonemap"] = tonemapped
    info["display_rev"] = DISPLAY_TONEMAP_REVISION
    store.save(proj)


@router.post("/projects/{pid}/viewer/register")
def register(pid: str, body: RegisterBody, request: Request) -> JSONResponse:
    # Thumbnail and main-view effects may register the same image concurrently.
    # Serialize the read/build/merge sequence so returned ids stay in project.json.
    with _registration_lock:
        return _register(pid, body, request)


def _register(pid: str, body: RegisterBody, request: Request) -> JSONResponse:
    store = ProjectStore(request.app.state.data_dir)
    proj = store.get(pid)
    if proj is None:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": pid})
    existing = _registered_view(proj, body.path)
    if existing is not None:
        existing_id, info = existing
        _ensure_display_variant(store, proj, existing_id, info)
        return JSONResponse(
            content={
                "image_id": existing_id,
                "levels": info["levels"],
                "width": info["width"],
                "height": info["height"],
            }
        )
    image_id = uuid.uuid4().hex[:12]
    img, domain, metadata = load_project_frame(proj, Path(body.path))
    height, width = img.shape[:2]
    tiles_root = proj.cache / "tiles"
    levels = build_pyramid(img, tiles_root / image_id)
    tonemapped = domain is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB
    if tonemapped:
        build_pyramid(tonemap_preview(img, metadata), tiles_root / image_id / "display")
    # Registration can race between thumbnail and main-view effects. Merge into
    # the latest project snapshot and reuse a winner that appeared meanwhile so
    # an image id returned to the UI is never clobbered by a stale save.
    latest = store.get(pid) or proj
    existing = _registered_view(latest, body.path)
    if existing is not None:
        existing_id, info = existing
        _ensure_display_variant(store, latest, existing_id, info)
        return JSONResponse(
            content={
                "image_id": existing_id,
                "levels": info["levels"],
                "width": info["width"],
                "height": info["height"],
            }
        )
    latest.images[image_id] = {"kind": "view", "path": body.path, "levels": levels,
                               "width": width, "height": height,
                               "display_tonemap": tonemapped,
                               "display_rev": DISPLAY_TONEMAP_REVISION}
    store.save(latest)
    return JSONResponse(content={"image_id": image_id, "levels": levels,
                                 "width": width, "height": height})


@router.get("/projects/{pid}/frame-thumb", response_model=None)
def frame_thumb(pid: str, path: str, request: Request, display: int = 0) -> Response:
    store = ProjectStore(request.app.state.data_dir)
    proj = store.get(pid)
    if proj is None:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": pid})
    if path not in set(proj.frames):
        return JSONResponse(status_code=400,
                            content={"error": "not_a_project_frame", "detail": path})
    frame = load_image(Path(path))
    pixels = frame.pixels
    if display and frame.domain is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB:
        pixels = tonemap_preview(pixels, frame.metadata)
    arr = (np.clip(pixels, 0.0, 1.0) * 255.0 + 0.5).astype("uint8")
    im = Image.fromarray(arr)
    im.thumbnail((TILE, TILE), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


@router.get("/viewer/{image_id}/tile/{z}/{x}/{y}", response_model=None)
def tile(image_id: str, z: int, x: int, y: int, request: Request,
         display: int = 0) -> JSONResponse | FileResponse:
    # search all projects' caches for this image_id's tiles
    store = ProjectStore(request.app.state.data_dir)
    for proj in store.list():
        if display:
            # Fall back to the base tile: rendered-domain images have no
            # display variant, so the toggle is a no-op for them.
            p = tile_path(proj.cache / "tiles", image_id, z, x, y, display=True)
            if p.exists():
                return FileResponse(p, media_type="image/jpeg")
        p = tile_path(proj.cache / "tiles", image_id, z, x, y)
        if p.exists():
            return FileResponse(p, media_type="image/jpeg")
    return JSONResponse(status_code=404, content={"error": "tile_not_found",
                                                  "detail": f"{image_id}/{z}/{x}/{y}"})
