"""FastAPI app factory + uvicorn launcher (SPEC §9, §12)."""

from __future__ import annotations

import logging
import socket
import webbrowser
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from focusstack_server.api import fs, projects, system

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8425


def create_app(data_dir: Path) -> FastAPI:
    """Build the app. All on-disk state lives under `data_dir` (hermetic in tests)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="FocusStack", version="0.1.0")
    app.state.data_dir = data_dir
    app.include_router(system.router)
    app.include_router(fs.router)
    app.include_router(projects.router)
    # Mount the built UI if it exists (M4 Part 2 builds it); harmless if absent.
    ui_dir = Path(__file__).parent / "static"
    if ui_dir.is_dir():
        app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")
    return app


def _find_free_port(host: str, start: int) -> int:
    """SPEC §12: if the port is busy, try the next one."""
    port = start
    for _ in range(20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) != 0:
                return port
        port += 1
    return start


def run(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:  # pragma: no cover
    import uvicorn
    from platformdirs import user_data_dir

    data_dir = Path(user_data_dir("focusstack", "focusstack"))
    port = _find_free_port(host, port)
    url = f"http://{host}:{port}"
    log.info("FocusStack server on %s", url)
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass
    uvicorn.run(create_app(data_dir), host=host, port=port)
