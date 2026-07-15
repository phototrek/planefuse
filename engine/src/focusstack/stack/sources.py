"""Frame sources: stream frames from memory or disk (SPEC §7 streaming)."""

from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np

from focusstack.io import ImageMetadata, ProcessingDomain, load_image

Region = tuple[int, int, int, int]  # (y0, x0, y1, x1)


def _crop(arr: np.ndarray, region: Region | None) -> np.ndarray:
    if region is None:
        return arr
    y0, x0, y1, x1 = region
    return arr[y0:y1, x0:x1]


class ArrayFrameSource:
    """In-memory frames (tests, small stacks)."""

    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        domain: ProcessingDomain = ProcessingDomain.RENDERED_RGB,
        metadata: ImageMetadata | None = None,
    ):
        self._frames = frames
        self.domain = domain
        self.metadata = metadata
        self.source_hash = ""

    def __len__(self) -> int:
        return len(self._frames)

    def read(self, idx: int, region: Region | None = None) -> np.ndarray:
        return _crop(self._frames[idx], region)


class DirFrameSource:
    """Frames loaded lazily from disk paths; re-reads on every call (streaming)."""

    def __init__(self, paths: list[Path], *, reference_index: int | None = None):
        self.paths = [Path(p) for p in paths]
        reference = reference_index if reference_index is not None else len(self.paths) // 2
        frame = load_image(self.paths[reference])
        self.domain = frame.domain
        self.metadata = frame.metadata
        digest = hashlib.sha256()
        self.source_hashes: list[str] = []
        for path in self.paths:
            source_digest = hashlib.sha256()
            digest.update(path.name.encode("utf-8", errors="surrogateescape"))
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    source_digest.update(chunk)
            source_hash = source_digest.hexdigest()
            self.source_hashes.append(source_hash)
            digest.update(source_hash.encode("ascii"))
        self.source_hash = digest.hexdigest()

    def __len__(self) -> int:
        return len(self.paths)

    def read(self, idx: int, region: Region | None = None) -> np.ndarray:
        return _crop(load_image(self.paths[idx]).pixels, region)


class _SubsetSource:
    """Exposes only a subset of another source's frames, re-indexed to 0..k-1."""

    def __init__(self, inner, indices: list[int]):
        self._inner = inner
        self._idx = list(indices)
        self.domain = inner.domain
        self.metadata = inner.metadata
        self.source_hash = inner.source_hash

    def __len__(self) -> int:
        return len(self._idx)

    def read(self, idx: int, region: Region | None = None) -> np.ndarray:
        return self._inner.read(self._idx[idx], region=region)
