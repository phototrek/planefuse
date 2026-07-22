#!/usr/bin/env python3
"""Boot a ready-folder ZIP exactly as a first-time user would."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_app(port: int, timeout: float, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + timeout
    system_url = f"http://127.0.0.1:{port}/api/system"
    page_url = f"http://127.0.0.1:{port}/"
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"PlaneFuse exited early with status {process.returncode}")
        try:
            with urllib.request.urlopen(system_url, timeout=3) as response:
                system = json.load(response)
            with urllib.request.urlopen(page_url, timeout=3) as response:
                page = response.read()
            if system.get("device") != "cpu":
                raise RuntimeError(f"expected CPU smoke device, got {system.get('device')!r}")
            if b"<title>PlaneFuse</title>" not in page:
                raise RuntimeError("compiled PlaneFuse UI was not served")
            return
        except (OSError, RuntimeError, urllib.error.URLError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(1)

    raise RuntimeError(f"PlaneFuse did not become healthy within {timeout:.0f}s: {last_error}")


def _stop_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
        )
    else:
        getattr(os, "killpg")(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            getattr(os, "killpg")(process.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        process.kill()
        process.wait(timeout=10)


def smoke(bundle: Path, timeout: float) -> None:
    executable = "uv.exe" if os.name == "nt" else "uv"
    uv = (bundle / "tools" / executable).resolve()
    if not uv.is_file():
        raise RuntimeError(f"bundled uv executable not found: {uv}")
    if not (bundle / "RELEASE.json").is_file():
        raise RuntimeError("ready folder has no RELEASE.json provenance")

    port = _free_port()
    command = [
        str(uv),
        "run",
        "--frozen",
        "--no-dev",
        "--extra",
        "cpu",
        "--extra",
        "raw",
        "planefuse",
        "serve",
    ]

    with (
        tempfile.TemporaryDirectory(prefix="planefuse-smoke-data-") as data_dir,
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log,
    ):
        environment = {
            **os.environ,
            "PLANEFUSE_DATA_DIR": data_dir,
            "PLANEFUSE_DEVICE": "cpu",
            "PLANEFUSE_NO_BROWSER": "1",
            "PLANEFUSE_PORT": str(port),
            "UV_NO_PROGRESS": "1",
            "UV_PYTHON_PREFERENCE": "only-managed",
        }
        process = subprocess.Popen(
            command,
            cwd=bundle,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=os.name != "nt",
        )
        try:
            _wait_for_app(port, timeout, process)
        except Exception:
            log.seek(0)
            output = log.read()
            raise RuntimeError(f"ready-folder smoke failed\n{output[-12000:]}") from None
        finally:
            _stop_process_tree(process)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    arguments = parser.parse_args()
    smoke(arguments.bundle.resolve(), arguments.timeout)
    print("ready-folder API and UI are healthy")


if __name__ == "__main__":
    main()
