import errno
import json
import os
import select
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def _run_in_pty(command):
    master_fd, slave_fd = os.openpty()
    process = None
    try:
        process = subprocess.Popen(
            command,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
        )
        os.close(slave_fd)
        slave_fd = -1
        os.write(master_fd, b"\n")

        output = bytearray()
        while True:
            readable, _, _ = select.select([master_fd], [], [], 5)
            if not readable:
                raise AssertionError("launcher did not exit within 5 seconds")
            try:
                chunk = os.read(master_fd, 4096)
            except OSError as error:
                if error.errno != errno.EIO:
                    raise
                break
            if not chunk:
                break
            output.extend(chunk)

        return_code = process.wait(timeout=5)
        return return_code, output.decode(errors="replace")
    finally:
        if slave_fd >= 0:
            os.close(slave_fd)
        os.close(master_fd)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()


def test_distribution_files_enforce_platform_line_endings():
    attributes = (ROOT / ".gitattributes").read_text().splitlines()
    assert "*.sh text eol=lf" in attributes
    assert "*.command text eol=lf" in attributes
    assert "*.bat text eol=crlf" in attributes
    assert "*.cmd text eol=crlf" in attributes


def test_launchers_use_locked_raw_install_and_reproducible_ui_build():
    launchers = [
        ROOT / "scripts/start-macos.sh",
        ROOT / "scripts/start-windows-cpu.bat",
        ROOT / "scripts/start-windows-gpu.bat",
    ]
    for launcher in launchers:
        text = launcher.read_text()
        assert "npm ci" in text
        assert "--frozen" in text
        assert "--extra raw" in text
        assert "Node.js 22+" in text
    assert "exec uv run" in launchers[0].read_text()
    assert "exit /b %errorlevel%" in launchers[1].read_text()
    assert "nvidia-smi" in launchers[2].read_text()


def test_container_configuration_keeps_localhost_boundary_and_raw_support():
    dockerfile = (ROOT / "Dockerfile").read_text()
    compose = (ROOT / "docker-compose.yml").read_text()
    assert "ghcr.io/astral-sh/uv:0.11.28" in dockerfile
    assert "node:22.23.1-slim" in dockerfile
    assert dockerfile.count('"raw"') >= 2
    assert '"127.0.0.1:8425:8425"' in compose
    assert ":/photos:ro" in compose


def test_benchmark_records_runtime_versions():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/benchmark.py"),
            "--height", "16",
            "--width", "16",
            "--frames", "2",
            "--method", "weighted",
            "--device", "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["python_version"].startswith("3.12")
    assert report["torch_version"]


def test_root_macos_launcher_delegates_and_keeps_failures_visible():
    launcher = ROOT / "Launch FocusStack.command"
    text = launcher.read_text()
    assert os.access(launcher, os.X_OK)
    assert 'scripts/start-macos.sh' in text
    assert 'status=$?' in text
    assert 'read -r' in text
    assert 'exit "$status"' in text


@pytest.mark.skipif(
    os.name != "posix",
    reason="macOS launcher behavior requires a POSIX pseudo-terminal",
)
def test_root_macos_launcher_pauses_only_on_failure_and_preserves_status(tmp_path):
    project = tmp_path / "focus-stacker"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)

    launcher = project / "Launch FocusStack.command"
    launcher.write_bytes((ROOT / launcher.name).read_bytes())
    launcher.chmod(0o755)
    starter = scripts / "start-macos.sh"

    def run_with_stub(exit_code):
        starter.write_text(f"#!/usr/bin/env bash\nexit {exit_code}\n")
        starter.chmod(0o755)
        return _run_in_pty([str(launcher)])

    success_code, success_output = run_with_stub(0)
    assert success_code == 0
    assert "Press Return to close this window" not in success_output

    failure_code, failure_output = run_with_stub(23)
    assert failure_code == 23
    assert "Press Return to close this window" in failure_output


def test_root_windows_launcher_prefers_gpu_and_falls_back_to_cpu():
    text = (ROOT / "Launch FocusStack.bat").read_text().lower()
    probe = text.index("nvidia-smi")
    gpu = text.index("scripts\\start-windows-gpu.bat")
    cpu = text.index("scripts\\start-windows-cpu.bat")
    assert probe < gpu < cpu
    assert "if errorlevel 1 goto cpu" in text
    assert "exit /b %focusstack_exit%" in text

    lines = [line.strip() for line in text.splitlines()]
    status_capture = lines.index('set "focusstack_exit=%errorlevel%"')
    failure_check = lines.index('if not "%focusstack_exit%"=="0" (')
    pause_lines = [index for index, line in enumerate(lines) if line == "pause"]
    failure_close = lines.index(")", failure_check + 1)
    exit_line = lines.index("exit /b %focusstack_exit%")
    assert len(pause_lines) == 1
    assert status_capture < failure_check < pause_lines[0] < failure_close < exit_line


def test_docs_advertise_both_root_one_click_launchers():
    for document in (ROOT / "README.md", ROOT / "docs/INSTALL.md"):
        text = document.read_text()
        assert "Launch FocusStack.command" in text
        assert "Launch FocusStack.bat" in text
    install = (ROOT / "docs/INSTALL.md").read_text()
    assert "NVIDIA" in install
    assert "fall" in install.lower() and "CPU" in install
