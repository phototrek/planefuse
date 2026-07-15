"""Compact server-side image histograms and clipping analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from focusstack_server.image_data import load_project_pixels
from focusstack_server.projects import ProjectStore

router = APIRouter(prefix="/api")


def _histogram(values: np.ndarray) -> list[int]:
    counts, _edges = np.histogram(np.clip(values, 0.0, 1.0), bins=256, range=(0.0, 1.0))
    return [int(value) for value in counts]


@router.get("/viewer/{image_id}/analysis")
def image_analysis(image_id: str, request: Request, revision: int = 0) -> JSONResponse:
    store = ProjectStore(request.app.state.data_dir)
    for project in store.list():
        info = project.images.get(image_id)
        if info is None or not info.get("path"):
            continue
        path = Path(str(info["path"]))
        try:
            stat = path.stat()
        except OSError as error:
            return JSONResponse(
                status_code=404,
                content={"error": "image_not_found", "detail": str(error)},
            )
        key = (image_id, revision, stat.st_mtime_ns, stat.st_size)
        cached = request.app.state.analysis_cache.get(key)
        if cached is not None:
            return JSONResponse(content=cached)
        pixels = load_project_pixels(project, path)
        channels = {
            "red": pixels[..., 0],
            "green": pixels[..., 1],
            "blue": pixels[..., 2],
        }
        luminance = (
            pixels[..., 0] * 0.2126
            + pixels[..., 1] * 0.7152
            + pixels[..., 2] * 0.0722
        )
        payload = {
            "image_id": image_id,
            "revision": revision,
            "pixels": int(pixels.shape[0] * pixels.shape[1]),
            "histograms": {
                **{name: _histogram(values) for name, values in channels.items()},
                "luminance": _histogram(luminance),
            },
            "clipping": {
                "shadows": {
                    name: int(np.count_nonzero(values <= 0.0))
                    for name, values in channels.items()
                },
                "highlights": {
                    name: int(np.count_nonzero(values >= 1.0))
                    for name, values in channels.items()
                },
            },
        }
        request.app.state.analysis_cache[key] = payload
        return JSONResponse(content=payload)
    return JSONResponse(
        status_code=404,
        content={"error": "image_not_found", "detail": image_id},
    )
