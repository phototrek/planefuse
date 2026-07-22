#!/usr/bin/env python3
"""Build a photographer-facing source bundle with uv and the compiled UI."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
import tomllib
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).parents[1]
TOP_LEVEL_FILES = {
    ".python-version",
    "LICENSE",
    "README.md",
    "Launch PlaneFuse.bat",
    "Launch PlaneFuse.command",
    "pyproject.toml",
    "uv.lock",
}
START_SCRIPTS = {
    "scripts/start-macos.sh",
    "scripts/start-windows-cpu.bat",
    "scripts/start-windows-gpu.bat",
}
MINIMUM_ZIP_EPOCH = 315532800  # 1980-01-01, the earliest ZIP timestamp.


def _is_bundle_file(relative: str) -> bool:
    return (
        relative in TOP_LEVEL_FILES
        or relative in START_SCRIPTS
        or relative.startswith("engine/")
        or relative.startswith("server/")
        or (relative.startswith("docs/") and not relative.startswith("docs/superpowers/"))
    )


def _tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [
        ROOT / relative
        for relative in output.decode().split("\0")
        if relative and _is_bundle_file(relative)
    ]


def _copy_sources(destination: Path, platform: str) -> None:
    for source in _tracked_files():
        relative = source.relative_to(ROOT)
        if platform == "macos" and relative.suffix == ".bat":
            continue
        if platform == "windows" and (
            relative.name == "Launch PlaneFuse.command" or relative.name == "start-macos.sh"
        ):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _copy_uv(archive: Path, destination: Path, platform: str) -> None:
    executable = "uv.exe" if platform == "windows" else "uv"
    payload: bytes | None = None

    if archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as source:
            tar_member = next(
                (item for item in source.getmembers() if PurePosixPath(item.name).name == executable),
                None,
            )
            if tar_member is not None:
                extracted = source.extractfile(tar_member)
                if extracted is not None:
                    payload = extracted.read()
    else:
        with zipfile.ZipFile(archive) as source:
            zip_member = next(
                (name for name in source.namelist() if PurePosixPath(name).name == executable),
                None,
            )
            if zip_member is not None:
                payload = source.read(zip_member)

    if payload is None:
        raise SystemExit(f"{archive} does not contain {executable}")

    target = destination / "tools" / executable
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    target.chmod(0o755)


def _git_value(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def _release_identity(platform: str) -> tuple[dict[str, str], int]:
    commit = os.environ.get("GITHUB_SHA") or _git_value("rev-parse", "HEAD")
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH") or _git_value("show", "-s", "--format=%ct", "HEAD"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    identity = {
        "commit": commit,
        "platform": platform,
        "source_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch)),
        "uv": "0.11.28",
        "version": project["project"]["version"],
    }
    return identity, max(epoch, MINIMUM_ZIP_EPOCH)


def _write_release_identity(destination: Path, identity: dict[str, str]) -> None:
    (destination / "RELEASE.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalise_launchers(destination: Path) -> None:
    for path in destination.rglob("*"):
        if not path.is_file() or path.suffix not in {".bat", ".command", ".sh"}:
            continue
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        newline = "\r\n" if path.suffix == ".bat" else "\n"
        path.write_bytes(text.replace("\n", newline).encode())
        if path.suffix in {".command", ".sh"}:
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _zip_tree(source: Path, output: Path, epoch: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    date_time = time.gmtime(epoch)[:6]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = Path("PlaneFuse") / path.relative_to(source)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=date_time)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = path.stat().st_mode & 0xFFFF
            if path.name == "uv" or path.suffix in {".command", ".sh"}:
                mode |= stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            info.external_attr = mode << 16
            with path.open("rb") as handle:
                archive.writestr(info, handle.read(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build(platform: str, uv_archive: Path, static_dir: Path, output: Path) -> None:
    index = static_dir / "index.html"
    if not index.is_file():
        raise SystemExit(f"compiled UI not found at {index}")

    with tempfile.TemporaryDirectory(prefix="planefuse-ready-") as temporary:
        bundle = Path(temporary) / "PlaneFuse"
        bundle.mkdir()
        identity, epoch = _release_identity(platform)
        _copy_sources(bundle, platform)
        shutil.copytree(static_dir, bundle / "server/src/planefuse_server/static")
        _copy_uv(uv_archive, bundle, platform)
        _write_release_identity(bundle, identity)
        _normalise_launchers(bundle)
        _zip_tree(bundle, output, epoch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=("macos", "windows"), required=True)
    parser.add_argument("--uv-archive", type=Path, required=True)
    parser.add_argument(
        "--static-dir",
        type=Path,
        default=ROOT / "server/src/planefuse_server/static",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    build(arguments.platform, arguments.uv_archive, arguments.static_dir, arguments.output)


if __name__ == "__main__":
    main()
