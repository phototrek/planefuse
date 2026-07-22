import errno
import io
import json
import os
import re
import select
import subprocess
import sys
import tarfile
import zipfile
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
    assert 'exec "$UV_BIN" run' in launchers[0].read_text()
    assert "exit /b %errorlevel%" in launchers[1].read_text()
    assert "nvidia-smi" in launchers[2].read_text()
    assert launchers[0].read_text().index("tools/uv") < launchers[0].read_text().index("command -v node")
    for launcher in launchers[1:]:
        text = launcher.read_text()
        assert text.index(r"tools\uv.exe") < text.index("where node")


def test_container_configuration_keeps_localhost_boundary_and_raw_support():
    dockerfile = (ROOT / "Dockerfile").read_text()
    compose = (ROOT / "docker-compose.yml").read_text()
    assert "ghcr.io/astral-sh/uv:0.11.28" in dockerfile
    assert "node:22.23.1-slim" in dockerfile
    assert dockerfile.count('"raw"') >= 2
    assert '"127.0.0.1:8425:8425"' in compose
    assert ":/photos:ro" in compose


def test_guided_installer_is_visual_platform_specific_and_release_backed():
    page = (ROOT / "installer/index.html").read_text(encoding="utf-8")
    script = (ROOT / "installer/setup.js").read_text(encoding="utf-8")
    installer_favicon = (ROOT / "installer/favicon.svg").read_text(encoding="utf-8")
    app_favicon = (ROOT / "ui/static/favicon.svg").read_text(encoding="utf-8")
    app_shell = (ROOT / "ui/src/app.html").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    dependabot = (ROOT / ".github/dependabot.yml").read_text(encoding="utf-8")

    assert '<link rel="icon" href="favicon.svg" type="image/svg+xml">' in page
    assert '<link rel="icon" href="/favicon.svg" type="image/svg+xml" />' in app_shell
    assert installer_favicon == app_favicon
    assert "<svg" in installer_favicon and "#ff8c42" in installer_favicon
    assert 'data-platform-panel="macos"' in page
    assert 'data-platform-panel="windows"' in page
    assert page.count('data-step="') == 8
    assert "PlaneFuse-macOS-Apple-Silicon.zip" in page
    assert "PlaneFuse-Windows-x64.zip" in page
    assert page.count("/releases/download/continuous/") == 2
    assert 'role="progressbar"' in page
    assert "navigator.userAgentData" in script
    assert "window.localStorage" in script
    assert "actions/deploy-pages" in workflow
    assert "Detect GitHub Pages availability" in workflow
    assert "$GITHUB_API_URL/repos/$GITHUB_REPOSITORY/pages" in workflow
    assert "steps.pages_site.outputs.enabled == 'true'" in workflow
    assert "uv-aarch64-apple-darwin.tar.gz" in workflow
    assert "uv-x86_64-pc-windows-msvc.zip" in workflow
    assert "sha256sum -c" in workflow
    assert "ACTIONLINT_SHA256" in workflow
    assert "UV_MACOS_SHA256" in workflow
    assert "UV_WINDOWS_SHA256" in workflow
    assert "./actionlint -color" in workflow
    assert "bash -n scripts/start-macos.sh" in workflow
    assert "cmp release/PlaneFuse-macOS-Apple-Silicon.zip" in workflow
    assert "runs-on: macos-15" in workflow
    assert "runs-on: windows-2025" in workflow
    assert workflow.count("python scripts/smoke_ready_bundle.py") == 2
    assert "refs/heads/main" in workflow and "continuous" in workflow
    assert "pyinstaller" not in workflow.lower()
    actions = re.findall(r"uses:\s+\S+@([0-9a-f]{40})", workflow)
    assert len(actions) >= 10
    assert workflow.count("uses:") == len(actions)
    assert {"github-actions", "uv", "npm", "docker"} <= set(
        re.findall(r"package-ecosystem:\s+([a-z-]+)", dependabot)
    )


def test_ready_bundle_contains_compiled_ui_bundled_uv_and_executable_launcher(tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<title>PlaneFuse</title>")

    uv_archive = tmp_path / "uv-aarch64-apple-darwin.tar.gz"
    payload = b"#!/bin/sh\nexit 0\n"
    with tarfile.open(uv_archive, "w:gz") as archive:
        info = tarfile.TarInfo("uv-aarch64-apple-darwin/uv")
        info.mode = 0o755
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    output = tmp_path / "PlaneFuse-macOS-Apple-Silicon.zip"
    output_repeated = tmp_path / "PlaneFuse-macOS-Apple-Silicon-repeated.zip"
    environment = {
        **os.environ,
        "GITHUB_SHA": "f" * 40,
        "SOURCE_DATE_EPOCH": "1700000000",
    }
    command = [
        sys.executable,
        str(ROOT / "scripts/build_ready_bundle.py"),
        "--platform",
        "macos",
        "--uv-archive",
        str(uv_archive),
        "--static-dir",
        str(static),
    ]
    for destination in (output, output_repeated):
        subprocess.run(
            [*command, "--output", str(destination)],
            cwd=ROOT,
            env=environment,
            check=True,
        )

    assert output.read_bytes() == output_repeated.read_bytes()

    with zipfile.ZipFile(output) as bundle:
        names = set(bundle.namelist())
        identity = json.loads(bundle.read("PlaneFuse/RELEASE.json"))
        assert "PlaneFuse/tools/uv" in names
        assert "PlaneFuse/server/src/planefuse_server/static/index.html" in names
        assert "PlaneFuse/Launch PlaneFuse.command" in names
        assert "PlaneFuse/Launch PlaneFuse.bat" not in names
        assert "PlaneFuse/ui/package.json" not in names
        assert identity == {
            "commit": "f" * 40,
            "platform": "macos",
            "source_date": "2023-11-14T22:13:20Z",
            "uv": "0.11.28",
            "version": "0.1.0",
        }
        uv_mode = bundle.getinfo("PlaneFuse/tools/uv").external_attr >> 16
        launcher_mode = bundle.getinfo("PlaneFuse/Launch PlaneFuse.command").external_attr >> 16
        assert uv_mode & 0o111
        assert launcher_mode & 0o111


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
    launcher = ROOT / "Launch PlaneFuse.command"
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

    launcher = project / "Launch PlaneFuse.command"
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
    text = (ROOT / "Launch PlaneFuse.bat").read_text().lower()
    probe = text.index("nvidia-smi")
    gpu = text.index("scripts\\start-windows-gpu.bat")
    cpu = text.index("scripts\\start-windows-cpu.bat")
    assert probe < gpu < cpu
    assert "if errorlevel 1 goto cpu" in text
    assert "exit /b %planefuse_exit%" in text

    lines = [line.strip() for line in text.splitlines()]
    status_capture = lines.index('set "planefuse_exit=%errorlevel%"')
    failure_check = lines.index('if not "%planefuse_exit%"=="0" (')
    pause_lines = [index for index, line in enumerate(lines) if line == "pause"]
    failure_close = lines.index(")", failure_check + 1)
    exit_line = lines.index("exit /b %planefuse_exit%")
    assert len(pause_lines) == 1
    assert status_capture < failure_check < pause_lines[0] < failure_close < exit_line


def test_docs_advertise_both_root_one_click_launchers():
    for document in (ROOT / "README.md", ROOT / "docs/INSTALL.md"):
        text = document.read_text()
        assert "Launch PlaneFuse.command" in text
        assert "Launch PlaneFuse.bat" in text
    install = (ROOT / "docs/INSTALL.md").read_text()
    assert "NVIDIA" in install
    assert "fall" in install.lower() and "CPU" in install
