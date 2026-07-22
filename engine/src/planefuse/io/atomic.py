"""Atomic output publication shared by rendered and DNG writers."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def atomic_output(destination: Path) -> Iterator[Path]:
    destination = Path(destination)
    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {parent}")
    fd, name = tempfile.mkstemp(prefix=f".{destination.stem}-", suffix=destination.suffix, dir=parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        try:
            dir_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
