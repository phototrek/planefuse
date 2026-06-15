"""Engine-calling job runners (SPEC §9). Each returns a closure (progress, cancel)->result."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from focusstack.align import AlignParams
from focusstack.io import load_image, save_image
from focusstack.pipeline import stack_frames
from focusstack.select import SelectParams
from focusstack_server.projects import Project, ProjectStore


def make_stack_runner(store: ProjectStore, proj: Project, params: dict[str, Any]) -> Callable:
    paths = [Path(p) for p in proj.frames]
    method = params.get("method", "pmax")
    device = params.get("device", "auto")
    align = AlignParams(**params["align"]) if params.get("align") else None
    select = SelectParams(**params["select"]) if params.get("select") else None
    algo_params = params.get("algo_params", {})

    def run(progress: Callable[[str, float], None], cancel: Callable[[], bool]) -> dict:
        result = stack_frames(paths, method=method, params=algo_params, device_pref=device,
                              align=align, select=select, cache_dir=proj.cache,
                              progress=progress, cancel=cancel)
        image_id = uuid.uuid4().hex[:12]
        out_path = proj.cache / f"{image_id}.tif"
        icc = load_image(paths[len(paths) // 2]).icc
        save_image(result.image, out_path, bit_depth=16, icc=icc)
        proj.images[image_id] = {"kind": "result", "path": str(out_path), "method": method}
        if "depth" in result.aux:
            depth_id = uuid.uuid4().hex[:12]
            dpath = proj.cache / f"{depth_id}.tif"
            save_image(result.aux["depth"], dpath, bit_depth=16)
            proj.images[depth_id] = {"kind": "depth", "path": str(dpath)}
        store.save(proj)
        return {"image_id": image_id}

    return run
