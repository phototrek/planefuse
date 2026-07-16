"""Project CRUD (SPEC §9). DELETE only unregisters; user files are never removed."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from planefuse_server.projects import ProjectStore, asdict_no_props

router = APIRouter(prefix="/api/projects")


class CreateProject(BaseModel):
    path: str = ""
    name: str = "Untitled"


def _store(request: Request) -> ProjectStore:
    return ProjectStore(request.app.state.data_dir)


@router.post("")
def create(body: CreateProject, request: Request) -> dict:
    directory = Path(body.path) if body.path else None
    proj = _store(request).create(directory, body.name)
    return asdict_no_props(proj)


@router.get("")
def list_projects(request: Request) -> list[dict]:
    return [asdict_no_props(p) for p in _store(request).list()]


@router.get("/{pid}")
def get(pid: str, request: Request) -> JSONResponse:
    proj = _store(request).get(pid)
    if proj is None:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": pid})
    return JSONResponse(content=asdict_no_props(proj))


@router.delete("/{pid}")
def delete(pid: str, request: Request) -> JSONResponse:
    if not _store(request).unregister(pid):
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": pid})
    return JSONResponse(content={"unregistered": pid})


@router.patch("/{pid}/ui-state")
def patch_ui_state(pid: str, body: dict, request: Request) -> JSONResponse:
    store = _store(request)
    proj = store.get(pid)
    if proj is None:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": pid})
    proj.ui_state.update(body)
    store.save(proj)
    return JSONResponse(content={"ui_state": proj.ui_state})


class SaveBody(BaseModel):
    name: str | None = None
    saved: bool | None = None


@router.patch("/{pid}")
def save_project(pid: str, body: SaveBody, request: Request) -> JSONResponse:
    store = _store(request)
    proj = store.get(pid)
    if proj is None:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": pid})
    if body.name is not None:
        proj.name = body.name
    if body.saved is not None:
        proj.ui_state["saved"] = body.saved
    store.save(proj)
    return JSONResponse(content={"id": proj.id, "name": proj.name, "ui_state": proj.ui_state})
