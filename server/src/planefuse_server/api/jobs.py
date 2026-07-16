"""Job enqueue/list/cancel/reorder (SPEC §9)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from planefuse_server.projects import ProjectStore
from planefuse_server.runners import make_export_runner, make_select_runner, make_stack_runner

router = APIRouter(prefix="/api")


class EnqueueBody(BaseModel):
    type: str
    params: dict = {}


def _job_dict(job) -> dict:
    return {"id": job.id, "type": job.type, "status": job.status,
            "percent": job.percent, "message": job.message,
            "result": job.result, "error": job.error, "error_code": job.error_code,
            "params": job.params}


@router.post("/projects/{pid}/jobs")
def enqueue(pid: str, body: EnqueueBody, request: Request) -> JSONResponse:
    store = ProjectStore(request.app.state.data_dir)
    proj = store.get(pid)
    if proj is None:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": pid})
    q = request.app.state.jobs
    if body.type == "stack":
        work = make_stack_runner(store, proj, body.params)
    elif body.type == "select":
        work = make_select_runner(store, proj, body.params)
    elif body.type == "export":
        work = make_export_runner(store, proj, body.params)
    else:
        return JSONResponse(status_code=400,
                            content={"error": "unsupported_job", "detail": body.type})
    jid = q.submit(body.type, body.params, work)
    return JSONResponse(content={"id": jid})


@router.get("/jobs")
def list_jobs(request: Request) -> list[dict]:
    return [_job_dict(j) for j in request.app.state.jobs.list_jobs()]


@router.get("/jobs/{jid}")
def get_job(jid: str, request: Request) -> JSONResponse:
    job = request.app.state.jobs.get(jid)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": jid})
    return JSONResponse(content=_job_dict(job))


@router.delete("/jobs/{jid}")
def cancel_job(jid: str, request: Request) -> JSONResponse:
    if not request.app.state.jobs.cancel(jid):
        return JSONResponse(status_code=404, content={"error": "not_cancellable", "detail": jid})
    return JSONResponse(content={"cancelling": jid})


class ReorderBody(BaseModel):
    order: list[str]


@router.post("/jobs/reorder")
def reorder(body: ReorderBody, request: Request) -> dict:
    request.app.state.jobs.reorder(body.order)
    return {"ok": True}


class ExportBody(BaseModel):
    image_id: str
    dest: str
    format: str = "tif"
    bit_depth: int = 16
    jpeg_quality: int = 95
    compression: str = "zlib"
    float_tiff_dest: str | None = None
    depth_dest: str | None = None


@router.post("/projects/{pid}/export")
def export(pid: str, body: ExportBody, request: Request) -> JSONResponse:
    return enqueue(pid, EnqueueBody(type="export", params=body.model_dump()), request)
