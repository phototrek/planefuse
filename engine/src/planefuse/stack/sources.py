"""Frame sources: stream frames from memory or disk (SPEC §7 streaming)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import os

import numpy as np
import torch

from planefuse.io import Frame, ImageMetadata, ProcessingDomain, load_image

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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stack_digest(paths: list[Path], source_hashes: list[str]) -> str:
    """Stack identity: sha256 over (file name, file sha256) in stack order."""
    digest = hashlib.sha256()
    for path, source_hash in zip(paths, source_hashes, strict=True):
        digest.update(path.name.encode("utf-8", errors="surrogateescape"))
        digest.update(source_hash.encode("ascii"))
    return digest.hexdigest()


def default_decode_workers(n: int) -> int:
    return max(1, min(n, os.cpu_count() or 1))


def decode_frames(paths: list[Path], *, workers: int | None = None) -> list[Frame | BaseException]:
    """Decode every path concurrently (LibRaw releases the GIL while decoding).

    Returns one entry per path, in order: the Frame, or the exception its
    decode raised, so validation can classify failures exactly as a serial
    decode would.
    """
    paths = [Path(p) for p in paths]

    def _one(path: Path) -> Frame | BaseException:
        try:
            return load_image(path)
        except Exception as exc:  # noqa: BLE001 - handed to validate_stack to classify
            return exc

    count = workers if workers is not None else default_decode_workers(len(paths))
    if count <= 1 or len(paths) <= 1:
        return [_one(p) for p in paths]
    with ThreadPoolExecutor(max_workers=count, thread_name_prefix="planefuse-decode") as pool:
        return list(pool.map(_one, paths))


class DirFrameSource:
    """Frames loaded lazily from disk paths; re-reads on every call (streaming)."""

    def __init__(self, paths: list[Path], *, reference_index: int | None = None):
        self.paths = [Path(p) for p in paths]
        reference = reference_index if reference_index is not None else len(self.paths) // 2
        frame = load_image(self.paths[reference])
        self.domain = frame.domain
        self.metadata = frame.metadata
        self.source_hashes: list[str] = [_file_sha256(path) for path in self.paths]
        self.source_hash = _stack_digest(self.paths, self.source_hashes)

    def __len__(self) -> int:
        return len(self.paths)

    def read(self, idx: int, region: Region | None = None) -> np.ndarray:
        return _crop(load_image(self.paths[idx]).pixels, region)


class MemoryFrameSource:
    """Every frame decoded once and held in RAM as float32 (H, W, 3).

    Same identity as DirFrameSource over the same paths (domain and metadata
    of the reference frame, identical source hashes). `read` returns the held
    array itself, not a copy: callers must treat it as read-only.
    """

    def __init__(
        self,
        paths: list[Path],
        frames: list[Frame],
        *,
        reference_index: int | None = None,
        workers: int | None = None,
    ):
        self.paths = [Path(p) for p in paths]
        if len(frames) != len(self.paths):
            raise ValueError(f"{len(frames)} frames for {len(self.paths)} paths")
        reference = reference_index if reference_index is not None else len(self.paths) // 2
        self.domain = frames[reference].domain
        self.metadata = frames[reference].metadata
        self._frames: list[np.ndarray | None] = [frame.pixels for frame in frames]
        count = workers if workers is not None else default_decode_workers(len(self.paths))
        # The thread count the host budget allows for frame-sized work on
        # these frames; later concurrent stages on this source honor it too.
        self.workers = max(1, count)
        if count <= 1:
            self.source_hashes: list[str] = [_file_sha256(p) for p in self.paths]
        else:
            with ThreadPoolExecutor(max_workers=count, thread_name_prefix="planefuse-hash") as pool:
                self.source_hashes = list(pool.map(_file_sha256, self.paths))
        self.source_hash = _stack_digest(self.paths, self.source_hashes)

    def __len__(self) -> int:
        return len(self._frames)

    def read(self, idx: int, region: Region | None = None) -> np.ndarray:
        frame = self._frames[idx]
        if frame is None:
            raise RuntimeError("MemoryFrameSource frames were released")
        return _crop(frame, region)

    def release_frame(self, idx: int) -> None:
        """Drop one frame's pixels; reading it afterwards raises."""
        self._frames[idx] = None

    def release(self) -> None:
        """Drop the pixels (identity and hashes stay valid)."""
        self._frames = [None] * len(self._frames)


def read_tensor(source, idx: int, region: Region | None = None) -> torch.Tensor | None:
    """The frame as the tensor a source holds ((3, H, W) float32, or (H, W) bool
    for a mask source), cropped to `region`; None when the source holds numpy
    frames, in which case callers use read()."""
    reader = getattr(source, "read_tensor", None)
    return reader(idx, region) if reader is not None else None


class TensorFrameSource:
    """Frames held as torch tensors (aligned frames kept on the compute device,
    or in host memory), so a stacker can consume them without a disk round
    trip or a re-upload.

    Frames are (3, H, W) float32; with `as_mask`, (H, W) bool. read() returns
    host numpy in the FrameSource layout; read_tensor() returns the held tensor
    (a view when cropped). Both must be treated as read-only.
    """

    def __init__(
        self,
        tensors: list[torch.Tensor],
        *,
        as_mask: bool = False,
        domain: ProcessingDomain = ProcessingDomain.RENDERED_RGB,
        metadata: ImageMetadata | None = None,
        source_hash: str = "",
    ):
        self._tensors = list(tensors)
        self._as_mask = as_mask
        self.domain = domain
        self.metadata = metadata
        self.source_hash = source_hash

    def __len__(self) -> int:
        return len(self._tensors)

    def read_tensor(self, idx: int, region: Region | None = None) -> torch.Tensor:
        t = self._tensors[idx]
        if region is None:
            return t
        y0, x0, y1, x1 = region
        return t[..., y0:y1, x0:x1]

    def read(self, idx: int, region: Region | None = None) -> np.ndarray:
        t = self.read_tensor(idx, region)
        if self._as_mask:
            return t.cpu().numpy()
        return t.detach().permute(1, 2, 0).contiguous().cpu().numpy()

    def offload(self) -> None:
        """Move every held tensor to host memory (frees the device)."""
        self._tensors = [t.cpu() for t in self._tensors]


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

    def read_tensor(self, idx: int, region: Region | None = None) -> torch.Tensor | None:
        return read_tensor(self._inner, self._idx[idx], region)
