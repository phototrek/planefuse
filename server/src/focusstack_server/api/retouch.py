"""Retouch session routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api")


class CreateBody(BaseModel):
    target_image_id: str


class StrokeBody(BaseModel):
    source_id: str
    points: list[tuple[float, float] | tuple[float, float, float]]
    radius: float = Field(ge=0)
    hardness: float = Field(default=0.5, ge=0, le=1)
    opacity: float = Field(default=1.0, ge=0, le=1)
    mode: Literal["normal", "erase"] = "normal"


class FlattenBody(BaseModel):
    name: str = "retouched"


def _manager(request: Request):
    return request.app.state.retouch


def _guard(operation: Callable[[], Any]) -> JSONResponse:
    try:
        return JSONResponse(content=operation())
    except LookupError as error:
        return JSONResponse(
            status_code=404,
            content={"error": "not_found", "detail": str(error)},
        )
    except ValueError as error:
        return JSONResponse(
            status_code=400,
            content={"error": "bad_request", "detail": str(error)},
        )


@router.post("/projects/{project_id}/retouch")
def create(project_id: str, body: CreateBody, request: Request) -> JSONResponse:
    return _guard(lambda: _manager(request).create(project_id, body.target_image_id))


@router.get("/projects/{project_id}/retouch")
def list_sessions(project_id: str, request: Request) -> JSONResponse:
    return _guard(lambda: {"sessions": _manager(request).list_sessions(project_id)})


@router.delete("/retouch/{session_id}")
def delete(session_id: str, request: Request) -> JSONResponse:
    def operation() -> dict:
        _manager(request).delete(session_id)
        return {"deleted": session_id}

    return _guard(operation)


@router.post("/retouch/{session_id}/stroke")
def stroke(session_id: str, body: StrokeBody, request: Request) -> JSONResponse:
    return _guard(lambda: _manager(request).stroke(session_id, body.model_dump()))


@router.post("/retouch/{session_id}/undo")
def undo(session_id: str, request: Request) -> JSONResponse:
    return _guard(lambda: _manager(request).undo(session_id))


@router.post("/retouch/{session_id}/redo")
def redo(session_id: str, request: Request) -> JSONResponse:
    return _guard(lambda: _manager(request).redo(session_id))


@router.post("/retouch/{session_id}/flatten")
def flatten(session_id: str, body: FlattenBody, request: Request) -> JSONResponse:
    return _guard(lambda: _manager(request).flatten(session_id, body.name))
