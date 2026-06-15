"""FastAPI app factory + uvicorn launcher (SPEC §9, §12)."""

from __future__ import annotations

import json
import logging
import shutil
import socket
import time
import webbrowser
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse

from focusstack_server.api import frames, fs, jobs, presets, projects, retouch, system, viewer
from focusstack_server.jobs import JobQueue
from focusstack_server.retouch import RetouchManager
from focusstack_server.ws import WsHub

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8425


def create_app(data_dir: Path) -> FastAPI:
    """Build the app. All on-disk state lives under `data_dir` (hermetic in tests)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    _prune_scratch(data_dir)
    hub = WsHub()
    app = FastAPI(title="FocusStack", version="0.1.0")
    app.state.data_dir = data_dir
    app.state.ws = hub
    # App-scoped job queue; progress events fan out over the WebSocket hub.
    app.state.jobs = JobQueue(on_event=hub.publish)
    app.state.retouch = RetouchManager(data_dir)
    app.include_router(system.router)
    app.include_router(fs.router)
    app.include_router(projects.router)
    app.include_router(frames.router)
    app.include_router(jobs.router)
    app.include_router(viewer.router)
    app.include_router(presets.router)
    app.include_router(retouch.router)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await hub.connect(ws)
        try:
            while True:
                await ws.receive_text()  # keepalive; clients may send pings
        except Exception:  # noqa: BLE001 - disconnect
            hub.disconnect(ws)

    # Serve the built UI if present (M4 Part 2 builds it); harmless if absent.
    # /api and /ws are matched by their routers first; this catch-all serves real
    # files, else the SPA shell so client-routed deep links resolve.
    static = _static_dir()
    if static.is_dir():

        @app.get("/{full_path:path}", response_model=None)
        def spa(full_path: str) -> FileResponse | JSONResponse:
            # Never shadow unmatched API/ws routes — let them 404 as JSON.
            if full_path == "api" or full_path.startswith("api/") or full_path == "ws":
                return JSONResponse(status_code=404, content={"error": "not_found", "detail": full_path})
            candidate = static / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            index = static / "index.html"
            if index.is_file():
                return FileResponse(index)
            return JSONResponse(status_code=404, content={"error": "not_found", "detail": full_path})

    return app


def _static_dir() -> Path:
    return Path(__file__).parent / "static"


def _prune_scratch(data_dir: Path, max_age_days: int = 7) -> None:
    """Delete abandoned scratch projects: those under data_dir/scratch whose
    project.json is unsaved and older than max_age_days. Best-effort; never
    touches dirs outside data_dir/scratch."""
    scratch = Path(data_dir) / "scratch"
    if not scratch.is_dir():
        return
    from focusstack_server.projects import ProjectStore

    store = ProjectStore(data_dir)
    cutoff = time.time() - max_age_days * 86400
    for d in scratch.iterdir():
        pj = d / "project.json"
        if not pj.is_file():
            continue
        try:
            data = json.loads(pj.read_text())
            if data.get("ui_state", {}).get("saved"):
                continue
            if pj.stat().st_mtime >= cutoff:
                continue
            shutil.rmtree(d, ignore_errors=True)
            pid = data.get("id")
            if pid:
                store.unregister(pid)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            log.warning("prune: skipped %s", d, exc_info=True)


def _find_free_port(host: str, start: int) -> int:
    """SPEC §12: if the port is busy, try the next one."""
    port = start
    for _ in range(20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) != 0:
                return port
        port += 1
    return start


def run(host: str | None = None, port: int | None = None) -> None:  # pragma: no cover
    import os

    import uvicorn
    from platformdirs import user_data_dir

    # Env overrides let the Docker image bind 0.0.0.0 and use a mounted data dir (§16).
    host = host or os.environ.get("FOCUSSTACK_HOST", DEFAULT_HOST)
    port = port or int(os.environ.get("FOCUSSTACK_PORT", str(DEFAULT_PORT)))
    data_env = os.environ.get("FOCUSSTACK_DATA_DIR")
    data_dir = Path(data_env) if data_env else Path(user_data_dir("focusstack", "focusstack"))
    port = _find_free_port(host, port)
    url = f"http://{host}:{port}"
    log.info("FocusStack server on %s", url)
    if host in ("127.0.0.1", "localhost"):  # don't pop a browser in a headless container
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    uvicorn.run(create_app(data_dir), host=host, port=port)
