"""Named parameter presets, global JSON in the server data dir (SPEC §9)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/presets")


class Preset(BaseModel):
    name: str
    params: dict


def _file(request: Request) -> Path:
    return Path(request.app.state.data_dir) / "presets.json"


def _read(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def _write(path: Path, data: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


@router.get("")
def list_presets(request: Request) -> list[dict]:
    return _read(_file(request))


@router.post("")
def add_preset(body: Preset, request: Request) -> dict:
    path = _file(request)
    data = [p for p in _read(path) if p["name"] != body.name]
    data.append(body.model_dump())
    _write(path, data)
    return body.model_dump()


@router.delete("/{name}")
def delete_preset(name: str, request: Request) -> JSONResponse:
    path = _file(request)
    data = [p for p in _read(path) if p["name"] != name]
    _write(path, data)
    return JSONResponse(content={"deleted": name})
