"""Retouch session persistence, rehydration, and tile updates."""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np

from planefuse.io import load_image, save_image
from planefuse.retouch import EMPTY_BBOX, RetouchSession, Stroke
from planefuse_server.projects import Project, ProjectStore
from planefuse_server.tiles import build_pyramid, rebuild_region


def _stroke_from_dict(data: dict) -> Stroke:
    mode = data.get("mode", "normal")
    if mode not in {"normal", "erase"}:
        raise ValueError(f"unknown stroke mode {mode!r}")
    radius = float(data["radius"])
    hardness = float(data.get("hardness", 0.5))
    opacity = float(data.get("opacity", 1.0))
    if radius < 0 or not 0.0 <= hardness <= 1.0 or not 0.0 <= opacity <= 1.0:
        raise ValueError("radius must be non-negative; hardness and opacity must be within [0, 1]")
    points = [
        (float(point[0]), float(point[1]), float(point[2]) if len(point) >= 3 else 1.0)
        for point in data["points"]
    ]
    return Stroke(
        source_id=str(data["source_id"]),
        points=points,
        radius=radius,
        hardness=hardness,
        opacity=opacity,
        mode=mode,
    )


def _stroke_to_dict(stroke: Stroke) -> dict:
    return {
        "source_id": stroke.source_id,
        "points": [list(point) for point in stroke.points],
        "radius": stroke.radius,
        "hardness": stroke.hardness,
        "opacity": stroke.opacity,
        "mode": stroke.mode,
    }


class RetouchManager:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._cache: dict[str, RetouchSession] = {}

    def _store(self) -> ProjectStore:
        return ProjectStore(self.data_dir)

    def _find(self, session_id: str) -> tuple[ProjectStore, Project, dict]:
        store = self._store()
        for project in store.list():
            for record in project.retouch:
                if record["id"] == session_id:
                    return store, project, record
        raise LookupError(f"unknown retouch session {session_id}")

    @staticmethod
    def _pixels(project: Project, image_id: str) -> np.ndarray:
        info = project.images.get(image_id)
        if info is None or "path" not in info:
            raise ValueError(f"image {image_id} has no pixel source")
        return load_image(Path(info["path"])).pixels

    def _load_session(self, project: Project, record: dict) -> RetouchSession:
        session_id = record["id"]
        if session_id in self._cache:
            return self._cache[session_id]

        base = self._pixels(project, record["target_image_id"])
        sources = {
            image_id: self._pixels(project, image_id)
            for image_id in record["sources"]
        }
        session = RetouchSession(base, sources)
        session.strokes = [_stroke_from_dict(stroke) for stroke in record["strokes"]]
        session.rebuild()
        build_pyramid(session.composite, project.cache / "tiles" / record["working_image_id"])
        self._cache[session_id] = session
        return session

    def create(self, project_id: str, target_image_id: str) -> dict:
        store = self._store()
        project = store.get(project_id)
        if project is None:
            raise LookupError(f"unknown project {project_id}")
        target = project.images.get(target_image_id)
        if target is None or target.get("kind") != "result":
            raise LookupError(f"unknown result image {target_image_id}")

        base = self._pixels(project, target_image_id)
        source_ids = [
            image_id
            for image_id, info in project.images.items()
            if image_id != target_image_id and info.get("kind") == "result"
        ]
        sources = {image_id: self._pixels(project, image_id) for image_id in source_ids}
        for image_id, source in sources.items():
            if source.shape != base.shape:
                raise ValueError(f"source {image_id} shape {source.shape} != target {base.shape}")

        session_id = uuid.uuid4().hex[:12]
        working_id = uuid.uuid4().hex[:12]
        levels = build_pyramid(base, project.cache / "tiles" / working_id)
        height, width = base.shape[:2]
        project.images[working_id] = {
            "kind": "retouch-working",
            "levels": levels,
            "width": width,
            "height": height,
        }
        record = {
            "id": session_id,
            "target_image_id": target_image_id,
            "working_image_id": working_id,
            "sources": source_ids,
            "strokes": [],
            "rev": 0,
        }
        project.retouch.append(record)
        store.save(project)
        self._cache[session_id] = RetouchSession(base, sources)
        return {
            "session_id": session_id,
            "working_image_id": working_id,
            "sources": [
                {"image_id": image_id, "method": project.images[image_id].get("method")}
                for image_id in source_ids
            ],
        }

    def list_sessions(self, project_id: str) -> list[dict]:
        project = self._store().get(project_id)
        if project is None:
            raise LookupError(f"unknown project {project_id}")
        return project.retouch

    def _save_mutation(
        self,
        store: ProjectStore,
        project: Project,
        record: dict,
        session: RetouchSession,
        bbox: tuple[int, int, int, int],
    ) -> dict:
        if bbox == EMPTY_BBOX:
            return {"rev": int(record["rev"]), "dirty_tiles": []}
        working_id = record["working_image_id"]
        levels = int(project.images[working_id]["levels"])
        dirty = rebuild_region(
            session.composite,
            project.cache / "tiles" / working_id,
            levels,
            bbox,
        )
        record["rev"] = int(record["rev"]) + 1
        store.save(project)
        return {"rev": record["rev"], "dirty_tiles": dirty}

    def stroke(self, session_id: str, data: dict) -> dict:
        store, project, record = self._find(session_id)
        stroke = _stroke_from_dict(data)
        if stroke.mode == "normal" and stroke.source_id not in record["sources"]:
            raise ValueError(f"source {stroke.source_id} is not available in session {session_id}")
        session = self._load_session(project, record)
        bbox = session.apply(stroke)
        if bbox != EMPTY_BBOX:
            record["strokes"].append(_stroke_to_dict(stroke))
        return self._save_mutation(store, project, record, session, bbox)

    def _history_mutation(self, session_id: str, operation: str) -> dict:
        store, project, record = self._find(session_id)
        session = self._load_session(project, record)
        bbox = session.undo() if operation == "undo" else session.redo()
        if bbox != EMPTY_BBOX:
            record["strokes"] = [_stroke_to_dict(stroke) for stroke in session.strokes]
        return self._save_mutation(store, project, record, session, bbox)

    def undo(self, session_id: str) -> dict:
        return self._history_mutation(session_id, "undo")

    def redo(self, session_id: str) -> dict:
        return self._history_mutation(session_id, "redo")

    def flatten(self, session_id: str, name: str) -> dict:
        store, project, record = self._find(session_id)
        session = self._load_session(project, record)
        image_id = uuid.uuid4().hex[:12]
        path = project.cache / f"{image_id}.tif"
        save_image(session.composite, path, bit_depth=16)
        project.images[image_id] = {
            "kind": "result",
            "path": str(path),
            "method": "retouch",
            "name": name,
        }
        record["rev"] = int(record["rev"]) + 1
        store.save(project)
        return {"image_id": image_id}

    def delete(self, session_id: str) -> None:
        store, project, record = self._find(session_id)
        project.retouch = [item for item in project.retouch if item["id"] != session_id]
        project.images.pop(record["working_image_id"], None)
        store.save(project)
        self._cache.pop(session_id, None)
