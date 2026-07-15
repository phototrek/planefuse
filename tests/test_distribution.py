import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


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


def test_root_windows_launcher_prefers_gpu_and_falls_back_to_cpu():
    text = (ROOT / "Launch FocusStack.bat").read_text().lower()
    probe = text.index("nvidia-smi")
    gpu = text.index("scripts\\start-windows-gpu.bat")
    cpu = text.index("scripts\\start-windows-cpu.bat")
    assert probe < gpu < cpu
    assert "if errorlevel 1 goto cpu" in text
    assert "pause" in text
    assert "exit /b %focusstack_exit%" in text
