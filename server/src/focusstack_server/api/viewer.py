"""Viewer tile serving + image registration/pyramid build (SPEC §9)."""

from __future__ import annotations

import io
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image
from pydantic import BaseModel

from focusstack.io import load_image
from focusstack_server.projects import ProjectStore
from focusstack_server.tiles import TILE, build_pyramid, tile_path

router = APIRouter(prefix="/api")


class RegisterBody(BaseModel):
    path: str


@router.post("/projects/{pid}/viewer/register")
def register(pid: str, body: RegisterBody, request: Request) -> JSONResponse:
    store = ProjectStore(request.app.state.data_dir)
    proj = store.get(pid)
    if proj is None:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": pid})
    image_id = uuid.uuid4().hex[:12]
    img = load_image(Path(body.path)).pixels
    tiles_root = proj.cache / "tiles"
    levels = build_pyramid(img, tiles_root / image_id)
    proj.images[image_id] = {"kind": "view", "path": body.path, "levels": levels}
    store.save(proj)
    return JSONResponse(content={"image_id": image_id, "levels": levels})


@router.get("/projects/{pid}/frame-thumb", response_model=None)
def frame_thumb(pid: str, path: str, request: Request) -> Response:
    store = ProjectStore(request.app.state.data_dir)
    proj = store.get(pid)
    if proj is None:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": pid})
    if path not in set(proj.frames):
        return JSONResponse(status_code=400,
                            content={"error": "not_a_project_frame", "detail": path})
    pixels = load_image(Path(path)).pixels
    arr = (np.clip(pixels, 0.0, 1.0) * 255.0 + 0.5).astype("uint8")
    im = Image.fromarray(arr)
    im.thumbnail((TILE, TILE), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


@router.get("/viewer/{image_id}/tile/{z}/{x}/{y}", response_model=None)
def tile(image_id: str, z: int, x: int, y: int, request: Request) -> JSONResponse | FileResponse:
    # search all projects' caches for this image_id's tiles
    store = ProjectStore(request.app.state.data_dir)
    for proj in store.list():
        p = tile_path(proj.cache / "tiles", image_id, z, x, y)
        if p.exists():
            return FileResponse(p, media_type="image/jpeg")
    return JSONResponse(status_code=404, content={"error": "tile_not_found",
                                                  "detail": f"{image_id}/{z}/{x}/{y}"})
