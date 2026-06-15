"""Project model + registry (SPEC §9). A project is a user directory with
project.json + a deletable cache/. The store keeps a registry of project dirs
under the server data dir; DELETE only unregisters (never touches user files)."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Project:
    id: str
    name: str
    directory: str
    frames: list[str] = field(default_factory=list)        # absolute frame paths
    images: dict[str, dict] = field(default_factory=dict)  # image_id -> {kind, path/...}
    jobs: list[dict] = field(default_factory=list)         # job history
    ui_state: dict = field(default_factory=dict)
    retouch: list[dict] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return Path(self.directory)

    @property
    def cache(self) -> Path:
        return self.path / "cache"


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


class ProjectStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._registry = self.data_dir / "projects.json"

    def _read_registry(self) -> dict[str, str]:
        if self._registry.exists():
            return json.loads(self._registry.read_text())
        return {}

    def _write_registry(self, reg: dict[str, str]) -> None:
        _atomic_write_json(self._registry, reg)

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

    def _save(self, proj: Project) -> None:
        _atomic_write_json(proj.path / "project.json", asdict_no_props(proj))

    def save(self, proj: Project) -> None:
        self._save(proj)

    def list(self) -> list[Project]:
        out = []
        for _pid, directory in self._read_registry().items():
            pj = Path(directory) / "project.json"
            if pj.exists():
                out.append(_load(pj))
        return out

    def get(self, pid: str) -> Project | None:
        directory = self._read_registry().get(pid)
        if directory is None:
            return None
        pj = Path(directory) / "project.json"
        return _load(pj) if pj.exists() else None

    def unregister(self, pid: str) -> bool:
        reg = self._read_registry()
        if pid not in reg:
            return False
        del reg[pid]
        self._write_registry(reg)
        return True


def asdict_no_props(proj: Project) -> dict[str, Any]:
    return {"id": proj.id, "name": proj.name, "directory": proj.directory,
            "frames": proj.frames, "images": proj.images, "jobs": proj.jobs,
            "ui_state": proj.ui_state, "retouch": proj.retouch}


def _load(pj: Path) -> Project:
    d = json.loads(pj.read_text())
    return Project(id=d["id"], name=d["name"], directory=d["directory"],
                   frames=d.get("frames", []), images=d.get("images", {}),
                   jobs=d.get("jobs", []), ui_state=d.get("ui_state", {}),
                   retouch=d.get("retouch", []))
