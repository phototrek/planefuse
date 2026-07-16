"""WebSocket progress hub (SPEC §9). Thread-safe fan-out of job events to clients."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket


class WsHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, ws: WebSocket) -> None:
        # Capture the serving loop here (we're on it); avoids depending on a
        # startup hook that TestClient skips unless used as a context manager.
        self._loop = asyncio.get_running_loop()
        await ws.accept()
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    def publish(self, event: dict[str, Any]) -> None:
        """Called from the worker thread; schedules sends on the event loop."""
        loop = self._loop
        if loop is None:
            return
        for ws in list(self._clients):
            asyncio.run_coroutine_threadsafe(self._safe_send(ws, event), loop)

    async def _safe_send(self, ws: WebSocket, event: dict[str, Any]) -> None:
        try:
            await ws.send_json(event)
        except Exception:  # noqa: BLE001 - client gone
            self.disconnect(ws)
