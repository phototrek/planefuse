# FocusStack M1 — Engine Core + PMax + CLI Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the FocusStack engine foundation — device abstraction (cuda/mps/cpu), image I/O, the synthetic test-stack generator, the PMax stacking algorithm with streaming fold and halo control, tiled processing, and a working `focusstack stack` CLI — per `docs/SPEC.md` Milestone M1.

**Architecture:** One device-agnostic PyTorch implementation of every image op (`backend/ops.py`); algorithms stream frames from a `FrameSource` so memory is independent of frame count; tiling and CPU fallback make OOM impossible. uv workspace with the `focusstack` engine package; tests are CPU-first with device-parametrized parity tests that auto-skip absent accelerators.

**Tech Stack:** Python 3.12, PyTorch 2.x, tifffile, Pillow, typer, pytest, scikit-image (tests only), uv.

**Read first:** `docs/SPEC.md` §2 (stack/hard rules), §4 (device abstraction), §5 (I/O), §7.1 (PMax), §8 (tiling), §13 (testing), §16 (uv). Spec wins on any conflict; flag conflicts rather than silently deviating.

**Preconditions:** the repo at `C:\Users\Adrien\Documents\CODE\focus-stacker` is already a git repository containing `docs/` (verify with `git log --oneline`). EXIF/XMP metadata copying (SPEC §5, pyexiv2) is deliberately **not** in M1 — it lands with export jobs in M4; only ICC passthrough is in scope here.

**Conventions used throughout:**
- Tensors: images are `torch.float32`, shape `(C, H, W)`, values nominally `[0, 1]` but **never clamped** except at export (SPEC §7.1).
- NumPy interchange: `(H, W, C)` float32.
- All commands run from repo root `C:\Users\Adrien\Documents\CODE\focus-stacker` via `uv run …`.
- Every test file starts CPU-only; device parametrization comes from the `devices` fixture (Task 2).

---

## Chunk 1: Workspace, device layer, ops, I/O

### Task 1: uv workspace scaffold

**Files:**
- Create: `pyproject.toml` (workspace root)
- Create: `engine/pyproject.toml`
- Create: `engine/src/focusstack/__init__.py`
- Create: `engine/src/focusstack/errors.py`
- Create: `tests/__init__.py` (empty), `tests/engine/__init__.py` (empty)
- Create: `.gitignore`

- [ ] **Step 1: Write root `pyproject.toml`** (uv workspace + torch index pattern from SPEC §16)

```toml
[project]
name = "focusstack-dev"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["focusstack"]

[project.optional-dependencies]
cpu = ["torch>=2.4"]
cu12x = ["torch>=2.4"]

[tool.uv]
conflicts = [[{ extra = "cpu" }, { extra = "cu12x" }]]

[tool.uv.sources]
focusstack = { workspace = true }
torch = [
  { index = "pytorch-cpu", extra = "cpu" },
  { index = "pytorch-cu128", extra = "cu12x" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.workspace]
members = ["engine"]

[dependency-groups]
dev = [
  "pytest>=8.0",
  "scikit-image>=0.24",
  "ruff>=0.8",
  "mypy>=1.13",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "cuda: tests requiring an NVIDIA GPU",
  "mps: tests requiring Apple silicon",
]

[tool.ruff]
line-length = 110
target-version = "py312"

[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true
check_untyped_defs = true
```

On macOS arm64 the plain `uv sync` resolves the default PyPI torch wheel (which includes MPS). On Windows/Linux always pass an extra: `uv sync --extra cu12x` (NVIDIA dev machine) or `--extra cpu` (CI/containers). SPEC §16.

- [ ] **Step 2: Write `engine/pyproject.toml`**

```toml
[project]
name = "focusstack"
version = "0.1.0"
description = "Professional GPU-accelerated focus stacking engine"
requires-python = ">=3.12"
dependencies = [
  "torch>=2.4",
  "numpy>=1.26",
  "tifffile>=2024.8.30",
  "imagecodecs>=2024.6.1",
  "pillow>=10.4",
  "typer>=0.12",
  "psutil>=6.0",
]

[project.scripts]
focusstack = "focusstack.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/focusstack"]
```

- [ ] **Step 3: Write `engine/src/focusstack/errors.py`**

```python
"""Typed engine exceptions (SPEC §12)."""


class FocusStackError(Exception):
    """Base for all engine errors."""


class ValidationError(FocusStackError):
    """Input frames failed validation (size/bit-depth/format)."""


class BackendError(FocusStackError):
    """Device unavailable or device-level failure."""


class AlignmentError(FocusStackError):
    """Alignment failed (M2; defined now so the hierarchy is complete)."""
```

- [ ] **Step 4: Write `engine/src/focusstack/__init__.py`**

```python
"""FocusStack engine — see docs/SPEC.md."""

__version__ = "0.1.0"
```

- [ ] **Step 5: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
dist/
ui/node_modules/
ui/dist/
```

- [ ] **Step 6: Sync and verify the right torch wheel landed**

Run: `uv sync --extra cu12x` (this machine has NVIDIA; use `--extra cpu` if not)
Expected: resolves and installs torch + deps, creates `uv.lock` and `.venv`.

Run: `uv run python -c "import focusstack, torch; print(focusstack.__version__, torch.__version__, torch.version.cuda, torch.cuda.is_available())"`
Expected on this machine: `0.1.0 2.x.y+cu128 12.8 True`. **`None False` means the index pin did not reach the engine package's torch dependency** (the extra-conditional `[tool.uv.sources]` lives in the root project; if uv resolves engine's torch from default PyPI instead, duplicate the `cpu`/`cu12x` extras + `[tool.uv.sources]` torch block into `engine/pyproject.toml` and re-sync). Do not proceed until this prints `True` — every CUDA parity test in this plan auto-skips without it, and the suite would silently go green CPU-only.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock engine/ tests/ .gitignore
git commit -m "feat: uv workspace scaffold with focusstack engine package"
```

---

### Task 2: Device abstraction

**Files:**
- Create: `engine/src/focusstack/backend/__init__.py`
- Create: `engine/src/focusstack/backend/device.py`
- Create: `tests/engine/conftest.py`
- Test: `tests/engine/test_device.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_device.py`:

```python
import pytest
import torch

from focusstack.backend import Device, free_memory, get_device
from focusstack.errors import BackendError


def test_cpu_device_always_available():
    d = get_device("cpu")
    assert d.kind == "cpu"
    assert d.torch_device == torch.device("cpu")
    assert d.name


def test_auto_returns_a_device():
    d = get_device("auto")
    assert d.kind in ("cuda", "mps", "cpu")


def test_unknown_preference_raises():
    with pytest.raises(ValueError):
        get_device("tpu")


def test_explicit_unavailable_accelerator_raises():
    if not torch.cuda.is_available():
        with pytest.raises(BackendError):
            get_device("cuda")
    if not torch.backends.mps.is_available():
        with pytest.raises(BackendError):
            get_device("mps")


def test_free_memory_positive():
    assert free_memory(get_device("cpu")) > 0


def test_free_memory_on_auto_device():
    assert free_memory(get_device("auto")) > 0
```

`tests/engine/conftest.py` (the device fixture every parity test uses):

```python
import pytest
import torch

from focusstack.backend import get_device


def _available_kinds() -> list[str]:
    kinds = ["cpu"]
    if torch.cuda.is_available():
        kinds.append("cuda")
    if torch.backends.mps.is_available():
        kinds.append("mps")
    return kinds


@pytest.fixture(params=_available_kinds())
def device(request):
    """Parametrizes a test over every device present on this machine."""
    return get_device(request.param)


@pytest.fixture(params=[k for k in _available_kinds() if k != "cpu"])
def accel_device(request):
    """Accelerators only; tests using this auto-skip on CPU-only machines."""
    return get_device(request.param)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/engine/test_device.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'focusstack.backend'`

- [ ] **Step 3: Implement `engine/src/focusstack/backend/device.py`**

```python
"""Runtime device selection (SPEC §4). One implementation, three devices."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from focusstack.errors import BackendError

log = logging.getLogger(__name__)

_VALID = ("auto", "cuda", "mps", "cpu")


@dataclass(frozen=True)
class Device:
    kind: str  # "cuda" | "mps" | "cpu"
    torch_device: torch.device
    name: str


def get_device(prefer: str = "auto") -> Device:
    if prefer not in _VALID:
        raise ValueError(f"unknown device preference {prefer!r}; expected one of {_VALID}")
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        d = Device("cuda", torch.device("cuda:0"), torch.cuda.get_device_name(0))
        log.info("using device: %s (%s)", d.kind, d.name)
        return d
    if prefer == "cuda":
        raise BackendError("CUDA requested but no CUDA device is available")
    if prefer in ("auto", "mps") and torch.backends.mps.is_available():
        d = Device("mps", torch.device("mps"), "Apple silicon (MPS)")
        log.info("using device: %s", d.kind)
        return d
    if prefer == "mps":
        raise BackendError("MPS requested but not available (requires Apple silicon macOS)")
    d = Device("cpu", torch.device("cpu"), "CPU")
    log.info("using device: cpu")
    return d


def free_memory(device: Device) -> int:
    """Bytes of memory available on the device (SPEC §4 memory discipline)."""
    if device.kind == "cuda":
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    if device.kind == "mps":
        return max(0, int(torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory()))
    import psutil

    return int(psutil.virtual_memory().available)


def empty_cache(device: Device) -> None:
    """Release cached allocations after OOM, before fallback (SPEC §4)."""
    if device.kind == "cuda":
        torch.cuda.empty_cache()
    elif device.kind == "mps":
        torch.mps.empty_cache()
```

`engine/src/focusstack/backend/__init__.py`:

```python
from focusstack.backend.device import Device, empty_cache, free_memory, get_device

__all__ = ["Device", "empty_cache", "free_memory", "get_device"]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/engine/test_device.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/backend tests/engine
git commit -m "feat: device abstraction with cuda/mps/cpu selection and memory probes"
```

---

### Task 3: Core image ops (device-agnostic torch)

**Files:**
- Create: `engine/src/focusstack/backend/ops.py`
- Test: `tests/engine/test_ops.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_ops.py`:

```python
import numpy as np
import pytest
import torch

from focusstack.backend import get_device, ops


def _rand_img(c=3, h=64, w=80, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.uniform(0, 1, (c, h, w)).astype(np.float32))


def test_to_tensor_roundtrip():
    arr = np.random.default_rng(0).uniform(0, 1, (40, 50, 3)).astype(np.float32)
    t = ops.to_tensor(arr, get_device("cpu"))
    assert t.shape == (3, 40, 50)
    assert t.dtype == torch.float32
    back = ops.to_numpy(t)
    assert back.shape == (40, 50, 3)
    np.testing.assert_allclose(back, arr, atol=1e-7)


def test_gaussian_blur_preserves_mean_and_shape():
    img = _rand_img()
    out = ops.gaussian_blur(img, sigma=2.0)
    assert out.shape == img.shape
    assert abs(out.mean().item() - img.mean().item()) < 1e-2


def test_gaussian_blur_reduces_variance():
    img = _rand_img()
    out = ops.gaussian_blur(img, sigma=2.0)
    assert out.var().item() < img.var().item()


def test_downsample2_halves_dims():
    img = _rand_img(h=64, w=80)
    out = ops.downsample2(img)
    assert out.shape == (3, 32, 40)


def test_upsample_to_exact_shape():
    img = _rand_img(h=32, w=40)
    out = ops.upsample_to(img, (63, 81))
    assert out.shape == (3, 63, 81)


def test_box_filter3_constant_invariant():
    img = torch.full((3, 16, 16), 0.5)
    out = ops.box_filter3(img)
    assert torch.allclose(out, img, atol=1e-6)


def test_median_filter2d_removes_speckle():
    x = torch.zeros(21, 21, dtype=torch.int32)
    x[10, 10] = 7  # lone speckle
    out = ops.median_filter2d(x, radius=1)
    assert out.dtype == torch.int32
    assert out[10, 10] == 0


def test_median_filter2d_preserves_constant_regions():
    x = torch.full((16, 16), 3, dtype=torch.int32)
    out = ops.median_filter2d(x, radius=2)
    assert torch.equal(out, x)


# --- device parity (SPEC §13.2): compares each available accelerator to CPU ---

@pytest.mark.parametrize("opname,kwargs", [
    ("gaussian_blur", {"sigma": 1.5}),
    ("downsample2", {}),
    ("box_filter3", {}),
])
def test_op_parity(accel_device, opname, kwargs):
    img = _rand_img(h=96, w=112, seed=42)
    fn = getattr(ops, opname)
    cpu_out = fn(img, **kwargs)
    dev_out = fn(img.to(accel_device.torch_device), **kwargs)
    torch.testing.assert_close(dev_out.cpu(), cpu_out, atol=1e-3, rtol=1e-3)


def test_upsample_parity(accel_device):
    img = _rand_img(h=48, w=56, seed=7)
    cpu_out = ops.upsample_to(img, (96, 112))
    dev_out = ops.upsample_to(img.to(accel_device.torch_device), (96, 112))
    torch.testing.assert_close(dev_out.cpu(), cpu_out, atol=1e-3, rtol=1e-3)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/engine/test_ops.py -v`
Expected: FAIL — `cannot import name 'ops'`

- [ ] **Step 3: Implement `engine/src/focusstack/backend/ops.py`**

```python
"""Device-agnostic image operations — the one true implementation (SPEC §4).

All image tensors are float32 (C, H, W); integer maps are (H, W).
The device is wherever the input tensor lives; ops never move tensors
between devices except documented explicit CPU round-trips for MPS gaps.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from focusstack.backend.device import Device


def to_tensor(arr: np.ndarray, device: Device) -> torch.Tensor:
    """(H, W, C) float32 numpy -> (C, H, W) float32 tensor on device."""
    if arr.ndim != 3:
        raise ValueError(f"expected (H, W, C) array, got shape {arr.shape}")
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).to(device.torch_device)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) tensor -> (H, W, C) float32 numpy on host."""
    return t.detach().permute(1, 2, 0).contiguous().cpu().numpy()


def gaussian_kernel1d(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, math.ceil(3.0 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur with reflect padding. img: (C, H, W)."""
    k = gaussian_kernel1d(sigma, img.device)
    r = (k.numel() - 1) // 2
    c = img.shape[0]
    x = img.unsqueeze(0)
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1).contiguous(), groups=c)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1).contiguous(), groups=c)
    return x.squeeze(0)


def downsample2(img: torch.Tensor) -> torch.Tensor:
    """Anti-aliased x2 downsample: blur(sigma=1.0) then stride-2 decimation."""
    return gaussian_blur(img, 1.0)[..., ::2, ::2]


def upsample_to(img: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Bilinear upsample to an exact (H, W)."""
    return F.interpolate(
        img.unsqueeze(0), size=shape, mode="bilinear", align_corners=False
    ).squeeze(0)


def box_filter3(img: torch.Tensor) -> torch.Tensor:
    """3x3 mean filter with reflect padding. img: (C, H, W)."""
    c = img.shape[0]
    w = torch.full((c, 1, 3, 3), 1.0 / 9.0, dtype=torch.float32, device=img.device)
    x = F.pad(img.unsqueeze(0), (1, 1, 1, 1), mode="reflect")
    return F.conv2d(x, w, groups=c).squeeze(0)


def median_filter2d(x: torch.Tensor, radius: int) -> torch.Tensor:
    """(2r+1)^2 median on an integer (H, W) map via unfold (SPEC §4: no native 2D median).

    Median of an odd-count window of integers is an integer, so the float
    round-trip is exact for the int32 winner maps this is used on.
    """
    if radius < 1:
        return x
    k = 2 * radius + 1
    f = x.to(torch.float32).unsqueeze(0).unsqueeze(0)
    f = F.pad(f, (radius, radius, radius, radius), mode="replicate")
    patches = f.unfold(2, k, 1).unfold(3, k, 1)
    med = patches.reshape(*patches.shape[:4], -1).median(dim=-1).values
    return med.squeeze(0).squeeze(0).round().to(x.dtype)


def local_energy(x: torch.Tensor) -> torch.Tensor:
    """Per-pixel energy = 3x3-smoothed mean |x| over channels -> (1, H, W). SPEC §7.1."""
    return box_filter3(x.abs().mean(dim=0, keepdim=True))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/engine/test_ops.py -v`
Expected: PASS (accelerator parity tests run on this machine's CUDA device; MPS auto-skips).

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/backend/ops.py tests/engine/test_ops.py
git commit -m "feat: device-agnostic core image ops with parity tests"
```

---

### Task 4: Image loading + stack validation

**Files:**
- Create: `engine/src/focusstack/io/__init__.py`
- Create: `engine/src/focusstack/io/loader.py`
- Test: `tests/engine/test_loader.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_loader.py`:

```python
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from focusstack.errors import ValidationError
from focusstack.io.loader import load_image, validate_stack


@pytest.fixture
def tmp_images(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    arr16 = rng.integers(0, 65535, (40, 60, 3), dtype=np.uint16)
    tifffile.imwrite(tmp_path / "a_16bit.tif", arr16)
    arr8 = rng.integers(0, 255, (40, 60, 3), dtype=np.uint8)
    Image.fromarray(arr8).save(tmp_path / "b_8bit.jpg", quality=95)
    Image.fromarray(arr8).save(tmp_path / "c_8bit.png")
    return tmp_path


def test_load_tiff16(tmp_images):
    f = load_image(tmp_images / "a_16bit.tif")
    assert f.pixels.shape == (40, 60, 3)
    assert f.pixels.dtype == np.float32
    assert f.bit_depth == 16
    assert 0.0 <= f.pixels.min() and f.pixels.max() <= 1.0


def test_load_jpeg_and_png(tmp_images):
    for name, depth in [("b_8bit.jpg", 8), ("c_8bit.png", 8)]:
        f = load_image(tmp_images / name)
        assert f.pixels.shape == (40, 60, 3)
        assert f.bit_depth == depth


def test_grayscale_rejected(tmp_path):
    gray = np.zeros((20, 20), dtype=np.uint8)
    Image.fromarray(gray, mode="L").save(tmp_path / "gray.png")
    with pytest.raises(ValidationError, match="RGB"):
        load_image(tmp_path / "gray.png")


def test_icc_passthrough(tmp_path):
    icc = b"\x00\x00\x02\x00fake-icc-profile-bytes"
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    Image.fromarray(arr).save(tmp_path / "with_icc.jpg", icc_profile=icc)
    f = load_image(tmp_path / "with_icc.jpg")
    assert f.icc == icc


def test_validate_stack_ok(tmp_images):
    report = validate_stack(sorted(tmp_images.glob("*8bit*")))
    assert report.ok
    assert all(s.status == "ok" for s in report.files)


def test_validate_stack_mixed_sizes(tmp_path):
    a = np.zeros((20, 20, 3), dtype=np.uint8)
    b = np.zeros((30, 20, 3), dtype=np.uint8)
    Image.fromarray(a).save(tmp_path / "a.png")
    Image.fromarray(b).save(tmp_path / "b.png")
    report = validate_stack([tmp_path / "a.png", tmp_path / "b.png"])
    assert not report.ok
    statuses = {s.path.name: s.status for s in report.files}
    assert statuses["a.png"] == "ok"
    assert statuses["b.png"] == "wrong_size"


def test_validate_stack_unreadable(tmp_path):
    bad = tmp_path / "bad.tif"
    bad.write_bytes(b"not a tiff at all")
    report = validate_stack([bad])
    assert not report.ok
    assert report.files[0].status == "unreadable"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/engine/test_loader.py -v`
Expected: FAIL — `No module named 'focusstack.io'`

- [ ] **Step 3: Implement `engine/src/focusstack/io/loader.py`**

```python
"""Image loading and stack validation (SPEC §5, §12).

Working format: float32 (H, W, C) in [0, 1], source gamma kept, ICC bytes
carried through untouched. RGB only; grayscale and alpha are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from focusstack.errors import ValidationError

SUPPORTED = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
_ICC_TAG = 34675


@dataclass
class Frame:
    pixels: np.ndarray  # float32 (H, W, 3) in [0, 1]
    bit_depth: int      # 8 or 16
    icc: bytes | None
    path: Path


@dataclass
class FileStatus:
    path: Path
    status: str   # ok | wrong_size | wrong_bit_depth | unreadable | unsupported | not_rgb
    message: str = ""


@dataclass
class ValidationReport:
    files: list[FileStatus] = field(default_factory=list)
    width: int = 0
    height: int = 0
    bit_depth: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.files) and all(s.status == "ok" for s in self.files)


def _scale(arr: np.ndarray) -> tuple[np.ndarray, int]:
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0, 8
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0, 16
    raise ValidationError(f"unsupported sample type {arr.dtype}; expected uint8 or uint16")


def load_image(path: Path) -> Frame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValidationError(f"{path.name}: unsupported format {suffix}")
    if suffix in (".tif", ".tiff"):
        with tifffile.TiffFile(path) as tf:
            arr = tf.pages[0].asarray()
            icc_tag = tf.pages[0].tags.get(_ICC_TAG)
            icc = bytes(icc_tag.value) if icc_tag is not None else None
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValidationError(f"{path.name}: RGB input required (got shape {arr.shape})")
        pixels, depth = _scale(arr)
        return Frame(pixels, depth, icc, path)
    with Image.open(path) as im:
        if im.mode != "RGB":
            raise ValidationError(f"{path.name}: RGB input required (got mode {im.mode})")
        icc = im.info.get("icc_profile")
        arr = np.asarray(im)
    pixels, depth = _scale(arr)
    return Frame(pixels, depth, bytes(icc) if icc else None, path)


def validate_stack(paths: list[Path]) -> ValidationReport:
    """Per-file validation report (SPEC §12). Never raises for bad files."""
    report = ValidationReport()
    for p in paths:
        p = Path(p)
        try:
            frame = load_image(p)
        except ValidationError as e:
            if p.suffix.lower() not in SUPPORTED:
                kind = "unsupported"
            elif "RGB" in str(e):
                kind = "not_rgb"
            else:
                kind = "unsupported"  # e.g. float32 TIFF samples
            report.files.append(FileStatus(p, kind, str(e)))
            continue
        except Exception as e:  # noqa: BLE001 - corrupted files land here by design
            report.files.append(FileStatus(p, "unreadable", str(e)))
            continue
        h, w = frame.pixels.shape[:2]
        if report.width == 0:
            report.width, report.height, report.bit_depth = w, h, frame.bit_depth
        if (w, h) != (report.width, report.height):
            report.files.append(FileStatus(p, "wrong_size", f"{w}x{h} != {report.width}x{report.height}"))
        elif frame.bit_depth != report.bit_depth:
            report.files.append(FileStatus(p, "wrong_bit_depth", f"{frame.bit_depth} != {report.bit_depth}"))
        else:
            report.files.append(FileStatus(p, "ok"))
    return report
```

`engine/src/focusstack/io/__init__.py`:

```python
from focusstack.io.loader import Frame, ValidationReport, load_image, validate_stack

__all__ = ["Frame", "ValidationReport", "load_image", "validate_stack"]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/engine/test_loader.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/io tests/engine/test_loader.py
git commit -m "feat: image loading with validation report and ICC passthrough"
```

---

### Task 5: Image writing (16-bit TIFF / JPEG / PNG)

**Files:**
- Create: `engine/src/focusstack/io/writer.py`
- Modify: `engine/src/focusstack/io/__init__.py` (export `save_image`)
- Test: `tests/engine/test_writer.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_writer.py`:

```python
import numpy as np
import pytest

from focusstack.io import load_image
from focusstack.io.writer import save_image


@pytest.fixture
def img():
    rng = np.random.default_rng(1)
    return rng.uniform(0, 1, (32, 48, 3)).astype(np.float32)


def test_tiff16_roundtrip(tmp_path, img):
    out = tmp_path / "r.tif"
    save_image(img, out, bit_depth=16, compression="zlib")
    back = load_image(out)
    assert back.bit_depth == 16
    np.testing.assert_allclose(back.pixels, img, atol=1.0 / 65535 + 1e-6)


def test_tiff_icc_embedded(tmp_path, img):
    icc = b"\x00\x00\x02\x00fake-icc"
    out = tmp_path / "icc.tif"
    save_image(img, out, bit_depth=16, icc=icc)
    assert load_image(out).icc == icc


def test_unclamped_input_is_clamped_at_export(tmp_path, img):
    hot = img.copy()
    hot[0, 0] = 1.7   # PMax overshoot (SPEC §7.1: clamp only at I/O boundary)
    hot[1, 1] = -0.3
    out = tmp_path / "c.tif"
    save_image(hot, out, bit_depth=16)
    back = load_image(out)
    assert back.pixels.max() <= 1.0
    assert back.pixels.min() >= 0.0


def test_jpeg_quality_and_icc(tmp_path, img):
    out = tmp_path / "q.jpg"
    save_image(img, out, jpeg_quality=90, icc=b"\x00\x00\x02\x00fake-icc")
    back = load_image(out)
    assert back.pixels.shape == img.shape
    assert back.icc == b"\x00\x00\x02\x00fake-icc"


def test_png8(tmp_path, img):
    out = tmp_path / "p.png"
    save_image(img, out, bit_depth=8)
    back = load_image(out)
    assert back.bit_depth == 8


def test_png16_rejected_until_m6(tmp_path, img):
    with pytest.raises(ValueError, match="M6"):
        save_image(img, tmp_path / "p16.png", bit_depth=16)


def test_unknown_extension_raises(tmp_path, img):
    with pytest.raises(ValueError):
        save_image(img, tmp_path / "x.webp")
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/engine/test_writer.py -v`
Expected: FAIL — `No module named 'focusstack.io.writer'`

- [ ] **Step 3: Implement `engine/src/focusstack/io/writer.py`**

```python
"""Image export (SPEC §5). Clamping to [0, 1] happens HERE and only here."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

_ICC_TAG = 34675
_TIFF_COMPRESSION = {"none": None, "lzw": "lzw", "zlib": "zlib"}


def save_image(
    arr: np.ndarray,
    path: Path,
    *,
    bit_depth: int = 16,
    icc: bytes | None = None,
    compression: str = "zlib",
    jpeg_quality: int = 95,
) -> None:
    """arr: float32 (H, W, 3), possibly outside [0, 1] (clamped here, SPEC §7.1)."""
    path = Path(path)
    clipped = np.clip(arr, 0.0, 1.0)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        if bit_depth == 16:
            data = (clipped * 65535.0 + 0.5).astype(np.uint16)
        elif bit_depth == 8:
            data = (clipped * 255.0 + 0.5).astype(np.uint8)
        else:
            raise ValueError(f"unsupported TIFF bit depth {bit_depth}")
        if compression not in _TIFF_COMPRESSION:
            raise ValueError(f"unknown compression {compression!r}")
        extratags = [(_ICC_TAG, 7, len(icc), icc, False)] if icc else []  # type 7 = UNDEFINED, per TIFF spec for ICC
        tifffile.imwrite(
            path, data, compression=_TIFF_COMPRESSION[compression], extratags=extratags
        )
        return
    if suffix in (".jpg", ".jpeg"):
        im = Image.fromarray((clipped * 255.0 + 0.5).astype(np.uint8))
        im.save(path, quality=jpeg_quality, icc_profile=icc)
        return
    if suffix == ".png":
        if bit_depth == 16:
            raise ValueError("16-bit PNG output lands in M6 (SPEC §5); use TIFF for 16-bit now")
        im = Image.fromarray((clipped * 255.0 + 0.5).astype(np.uint8))
        im.save(path, icc_profile=icc)
        return
    raise ValueError(f"unsupported output format {suffix!r} (use .tif/.jpg/.png)")
```

Update `engine/src/focusstack/io/__init__.py`:

```python
from focusstack.io.loader import Frame, ValidationReport, load_image, validate_stack
from focusstack.io.writer import save_image

__all__ = ["Frame", "ValidationReport", "load_image", "save_image", "validate_stack"]
```

Note: 16-bit PNG output is deferred to M6 per plan scope (SPEC §5 lists it; Pillow lacks 16-bit RGB PNG — M6 adds it via OpenCV, which arrives as a dependency in M2 anyway).

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/engine/test_writer.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/io tests/engine/test_writer.py
git commit -m "feat: image export with export-boundary clamping and ICC embedding"
```

---

## Chunk 2: Synthetic generator, pyramid, registry, PMax

### Task 6: Synthetic stack generator

**Files:**
- Create: `tests/synthetic/__init__.py`
- Create: `tests/synthetic/generate.py`
- Test: `tests/engine/test_synthetic.py`

This is the foundation of all quality testing (SPEC §13.1) — every later algorithm test consumes it.

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_synthetic.py`:

```python
import numpy as np

from tests.synthetic.generate import generate_stack, make_scene


def test_scene_shapes_and_ranges():
    sharp, depth = make_scene(96, 128, seed=3)
    assert sharp.shape == (96, 128, 3)
    assert depth.shape == (96, 128)
    assert sharp.dtype == np.float32 and depth.dtype == np.float32
    assert 0.0 <= depth.min() and depth.max() <= 1.0
    assert sharp.std() > 0.05  # must have texture for sharpness metrics


def test_stack_basic_properties():
    stack = generate_stack(h=96, w=128, n_frames=8, seed=3)
    assert len(stack.frames) == 8
    assert all(f.shape == (96, 128, 3) for f in stack.frames)
    assert stack.sharp.shape == (96, 128, 3)
    assert len(stack.transforms) == 8
    np.testing.assert_allclose(stack.transforms[0], np.eye(3), atol=1e-6)


def test_each_frame_sharpest_at_its_focus_depth():
    """Frame p must be the sharpest frame at pixels whose depth ~ p/(n-1)."""
    n = 8
    stack = generate_stack(h=96, w=128, n_frames=n, max_sigma=5.0, seed=3)

    def grad_energy(img, mask):
        gray = img.mean(axis=2)
        gy, gx = np.gradient(gray)
        return float(((gx**2 + gy**2) * mask).sum() / max(mask.sum(), 1))

    for p in [0, n // 2, n - 1]:
        focus = p / (n - 1)
        band = np.abs(stack.depth - focus) < 0.04
        if band.sum() < 200:
            continue
        energies = [grad_energy(f, band) for f in stack.frames]
        assert int(np.argmax(energies)) == p


def test_brightness_flicker_and_noise():
    a = generate_stack(h=64, w=64, n_frames=4, seed=1, flicker=0.1, noise=0.02)
    b = generate_stack(h=64, w=64, n_frames=4, seed=1, flicker=0.0, noise=0.0)
    assert not np.allclose(a.frames[1], b.frames[1])


def test_geometric_jitter_changes_frames():
    a = generate_stack(h=64, w=64, n_frames=4, seed=1, scale_step=0.002)
    b = generate_stack(h=64, w=64, n_frames=4, seed=1)
    np.testing.assert_allclose(a.frames[0], b.frames[0], atol=1e-6)  # frame 0 = identity
    assert not np.allclose(a.frames[3], b.frames[3])
    assert a.transforms[3][0, 0] != 1.0  # scale recorded in ground truth
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/engine/test_synthetic.py -v`
Expected: FAIL — `No module named 'tests.synthetic.generate'`

- [ ] **Step 3: Implement `tests/synthetic/generate.py`**

```python
"""Synthetic focus-stack generator with ground truth (SPEC §13.1).

Renders a textured scene with a known depth map, then simulates a focus
sweep: frame p focuses at depth p/(n-1); per-pixel defocus sigma is
max_sigma * |depth - focus_p|, applied via quantized Gaussian blurs.
Optional per-frame similarity jitter (focus breathing), brightness
flicker, and sensor noise — all recorded as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from focusstack.backend import get_device, ops


@dataclass
class SyntheticStack:
    frames: list[np.ndarray]        # float32 (H, W, 3)
    sharp: np.ndarray               # ground-truth all-in-focus (H, W, 3)
    depth: np.ndarray               # ground-truth depth (H, W) in [0, 1]
    transforms: list[np.ndarray]    # 3x3 ground-truth output->input pixel transforms


def make_scene(h: int, w: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Multi-octave textured image + depth map (ramp with two plateaus)."""
    rng = np.random.default_rng(seed)
    dev = get_device("cpu")
    t = torch.zeros(3, h, w)
    for amplitude, sigma in [(1.0, 1.0), (0.5, 4.0), (0.25, 12.0)]:
        noise = torch.from_numpy(rng.uniform(0, 1, (h, w, 3)).astype(np.float32))
        t = t + amplitude * ops.gaussian_blur(ops.to_tensor(noise.numpy(), dev), sigma)
    t = (t - t.min()) / (t.max() - t.min()) * 0.8 + 0.1
    depth = np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1))
    depth[h // 6 : h // 3, w // 6 : w // 3] = 0.15      # near plateau
    depth[h // 2 : 5 * h // 6, w // 2 : 5 * w // 6] = 0.85  # far plateau
    return ops.to_numpy(t), depth


def _apply_affine(img: torch.Tensor, m: np.ndarray) -> torch.Tensor:
    """Warp (C, H, W) by a 3x3 output->input pixel-coordinate transform."""
    c, h, w = img.shape
    s = np.array([[2.0 / (w - 1), 0, -1], [0, 2.0 / (h - 1), -1], [0, 0, 1]], dtype=np.float64)
    mn = (s @ m @ np.linalg.inv(s)).astype(np.float32)
    theta = torch.from_numpy(mn[:2]).unsqueeze(0)
    grid = F.affine_grid(theta, (1, c, h, w), align_corners=True)
    return F.grid_sample(
        img.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True
    ).squeeze(0)


def _similarity(scale: float, angle: float, tx: float, ty: float, h: int, w: int) -> np.ndarray:
    """Similarity about the image center, output->input convention."""
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ca, sa = np.cos(angle), np.sin(angle)
    rot = np.array([[scale * ca, -scale * sa, 0], [scale * sa, scale * ca, 0], [0, 0, 1]])
    pre = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    post = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], dtype=np.float64)
    return post @ rot @ pre


def _depth_blur(sharp: torch.Tensor, sigma_map: torch.Tensor, step: float = 0.5) -> torch.Tensor:
    out = sharp.clone()
    max_s = float(sigma_map.max())
    s = step
    while s <= max_s + step:
        blurred = ops.gaussian_blur(sharp, s)
        sel = (sigma_map >= s - step / 2) & (sigma_map < s + step / 2)
        out = torch.where(sel.unsqueeze(0), blurred, out)
        s += step
    return out


def generate_stack(
    h: int = 128,
    w: int = 160,
    n_frames: int = 10,
    max_sigma: float = 5.0,
    seed: int = 0,
    scale_step: float = 0.0,   # focus breathing per frame, e.g. 0.002 = 0.2%/frame
    rot_jitter: float = 0.0,   # radians, uniform +/-
    trans_jitter: float = 0.0, # pixels, uniform +/-
    flicker: float = 0.0,      # multiplicative brightness jitter, uniform +/-
    noise: float = 0.0,        # gaussian sensor noise sigma
) -> SyntheticStack:
    rng = np.random.default_rng(seed + 1000)
    sharp_np, depth = make_scene(h, w, seed)
    dev = get_device("cpu")
    sharp = ops.to_tensor(sharp_np, dev)
    depth_t = torch.from_numpy(depth)
    frames: list[np.ndarray] = []
    transforms: list[np.ndarray] = []
    for p in range(n_frames):
        focus = p / max(n_frames - 1, 1)
        sigma_map = (depth_t - focus).abs() * max_sigma
        frame = _depth_blur(sharp, sigma_map)
        if p == 0 or (scale_step == 0 and rot_jitter == 0 and trans_jitter == 0):
            m = np.eye(3)
        else:
            m = _similarity(
                1.0 + scale_step * p,
                rng.uniform(-rot_jitter, rot_jitter),
                rng.uniform(-trans_jitter, trans_jitter),
                rng.uniform(-trans_jitter, trans_jitter),
                h, w,
            )
            frame = _apply_affine(frame, m)
        if flicker:
            frame = frame * (1.0 + rng.uniform(-flicker, flicker))
        if noise:
            frame = frame + torch.from_numpy(rng.normal(0, noise, (3, h, w)).astype(np.float32))
        frames.append(ops.to_numpy(frame))
        transforms.append(m.astype(np.float64))
    return SyntheticStack(frames=frames, sharp=sharp_np, depth=depth, transforms=transforms)
```

`tests/synthetic/__init__.py`: empty file.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/engine/test_synthetic.py -v`
Expected: all PASS. If `test_each_frame_sharpest_at_its_focus_depth` is flaky, widen the band to `< 0.05` — but investigate first; it usually means the blur quantization step is too coarse.

- [ ] **Step 5: Commit**

```bash
git add tests/synthetic tests/engine/test_synthetic.py
git commit -m "feat: synthetic focus-stack generator with ground truth"
```

---

### Task 7: Laplacian pyramid

**Files:**
- Create: `engine/src/focusstack/stack/__init__.py` (empty for now)
- Create: `engine/src/focusstack/stack/pyramid.py`
- Test: `tests/engine/test_pyramid.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_pyramid.py`:

```python
import numpy as np
import torch

from focusstack.backend import get_device, ops
from focusstack.stack.pyramid import build_laplacian, collapse_laplacian, pyramid_depth


def test_pyramid_depth_formula():
    # SPEC §7.1: depth = floor(log2(min(H,W))) - 5, coarsest level >= 32 px
    assert pyramid_depth(1024, 1536) == 5
    assert pyramid_depth(4000, 6000) == 6
    assert pyramid_depth(64, 64) == 1
    assert pyramid_depth(40, 40) == 1  # never below 1


def test_build_collapse_is_exact_identity():
    rng = np.random.default_rng(0)
    img = torch.from_numpy(rng.uniform(0, 1, (3, 96, 128)).astype(np.float32))
    depth = pyramid_depth(96, 128)
    lap, residual = build_laplacian(img, depth)
    assert len(lap) == depth
    rec = collapse_laplacian(lap, residual)
    torch.testing.assert_close(rec, img, atol=1e-5, rtol=1e-5)


def test_level_shapes_halve():
    img = torch.zeros(3, 96, 128)
    lap, residual = build_laplacian(img, 3)
    assert lap[0].shape == (3, 96, 128)
    assert lap[1].shape == (3, 48, 64)
    assert lap[2].shape == (3, 24, 32)
    assert residual.shape == (3, 12, 16)


def test_pyramid_parity(accel_device):
    rng = np.random.default_rng(5)
    img = torch.from_numpy(rng.uniform(0, 1, (3, 96, 128)).astype(np.float32))
    lap_c, res_c = build_laplacian(img, 3)
    lap_d, res_d = build_laplacian(img.to(accel_device.torch_device), 3)
    for lc, ld in zip(lap_c, lap_d):
        torch.testing.assert_close(ld.cpu(), lc, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(res_d.cpu(), res_c, atol=1e-3, rtol=1e-3)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/engine/test_pyramid.py -v`
Expected: FAIL — `No module named 'focusstack.stack.pyramid'`

- [ ] **Step 3: Implement `engine/src/focusstack/stack/pyramid.py`**

```python
"""Laplacian pyramid build/collapse (SPEC §7.1).

Build subtracts the upsampled next-coarser Gaussian level, so
collapse(build(x)) == x exactly (up to float error) by construction.
"""

from __future__ import annotations

import math

import torch

from focusstack.backend import ops


def pyramid_depth(h: int, w: int) -> int:
    """SPEC §7.1: floor(log2(min(H, W))) - 5, clamped to >= 1 (coarsest >= 32 px)."""
    return max(1, int(math.floor(math.log2(min(h, w)))) - 5)


def build_laplacian(img: torch.Tensor, depth: int) -> tuple[list[torch.Tensor], torch.Tensor]:
    lap: list[torch.Tensor] = []
    cur = img
    for _ in range(depth):
        down = ops.downsample2(cur)
        up = ops.upsample_to(down, (cur.shape[-2], cur.shape[-1]))
        lap.append(cur - up)
        cur = down
    return lap, cur


def collapse_laplacian(lap: list[torch.Tensor], residual: torch.Tensor) -> torch.Tensor:
    cur = residual
    for level in reversed(lap):
        cur = ops.upsample_to(cur, (level.shape[-2], level.shape[-1])) + level
    return cur
```

`engine/src/focusstack/stack/__init__.py`: empty file for now (populated in Task 8).

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/engine/test_pyramid.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/stack tests/engine/test_pyramid.py
git commit -m "feat: exact-reconstruction Laplacian pyramid"
```

---

### Task 8: Algorithm registry + frame sources

**Files:**
- Create: `engine/src/focusstack/stack/base.py`
- Create: `engine/src/focusstack/stack/sources.py`
- Modify: `engine/src/focusstack/stack/__init__.py`
- Test: `tests/engine/test_registry.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_registry.py`:

```python
import numpy as np
import pytest

from focusstack.io import save_image
from focusstack.stack.base import REGISTRY, ParamSpec, StackResult, get_algorithm, register
from focusstack.stack.sources import ArrayFrameSource, DirFrameSource


def test_register_and_lookup():
    @register("dummy")
    class Dummy:
        name = "dummy"

        @staticmethod
        def params():
            return [ParamSpec("k", "K", "int", default=1, min=0, max=5, tooltip="test")]

        def run(self, source, device, params, progress=None, cancel=None):
            return StackResult(image=source.read(0))

    assert "dummy" in REGISTRY
    algo = get_algorithm("dummy")
    assert algo.params()[0].name == "k"
    del REGISTRY["dummy"]


def test_unknown_algorithm_raises():
    with pytest.raises(KeyError, match="nope"):
        get_algorithm("nope")


def test_array_source_read_and_region():
    frames = [np.full((20, 30, 3), i / 10, dtype=np.float32) for i in range(4)]
    src = ArrayFrameSource(frames)
    assert len(src) == 4
    assert src.read(2).shape == (20, 30, 3)
    crop = src.read(1, region=(5, 10, 15, 25))  # (y0, x0, y1, x1)
    assert crop.shape == (10, 15, 3)
    np.testing.assert_allclose(crop, 0.1)


def test_dir_source(tmp_path):
    for i in range(3):
        arr = np.full((16, 24, 3), i / 4, dtype=np.float32)
        save_image(arr, tmp_path / f"f{i}.tif", bit_depth=16)
    src = DirFrameSource(sorted(tmp_path.glob("*.tif")))
    assert len(src) == 3
    assert src.read(0).shape == (16, 24, 3)
    crop = src.read(2, region=(0, 0, 8, 8))
    assert crop.shape == (8, 8, 3)
    np.testing.assert_allclose(crop, 0.5, atol=1e-3)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/engine/test_registry.py -v`
Expected: FAIL — `No module named 'focusstack.stack.base'`

- [ ] **Step 3: Implement `engine/src/focusstack/stack/base.py`**

```python
"""Algorithm registry with typed parameters (SPEC §7).

The UI and CLI build their parameter forms from ParamSpec metadata, so
adding an algorithm requires no UI changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

ProgressFn = Callable[[str, float], None]  # (message, fraction 0..1)
CancelFn = Callable[[], bool]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    type: str  # "int" | "float" | "choice" | "bool"
    default: Any
    min: Any = None
    max: Any = None
    choices: tuple[str, ...] | None = None
    tooltip: str = ""


@dataclass
class StackResult:
    image: np.ndarray                 # float32 (H, W, 3), UNclamped (SPEC §7.1)
    aux: dict[str, np.ndarray] = field(default_factory=dict)


class FrameSource(Protocol):
    def __len__(self) -> int: ...

    def read(self, idx: int, region: tuple[int, int, int, int] | None = None) -> np.ndarray:
        """float32 (H, W, 3); region=(y0, x0, y1, x1) returns that crop."""
        ...


class StackAlgorithm(Protocol):
    name: str

    @staticmethod
    def params() -> list[ParamSpec]: ...

    def run(
        self,
        source: FrameSource,
        device: Any,
        params: dict[str, Any],
        progress: ProgressFn | None = None,
        cancel: CancelFn | None = None,
    ) -> StackResult: ...


REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        REGISTRY[name] = cls
        return cls

    return deco


def get_algorithm(name: str) -> StackAlgorithm:
    if name not in REGISTRY:
        raise KeyError(f"unknown stacking algorithm {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]()
```

`engine/src/focusstack/stack/sources.py`:

```python
"""Frame sources: stream frames from memory or disk (SPEC §7 streaming)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from focusstack.io import load_image

Region = tuple[int, int, int, int]  # (y0, x0, y1, x1)


def _crop(arr: np.ndarray, region: Region | None) -> np.ndarray:
    if region is None:
        return arr
    y0, x0, y1, x1 = region
    return arr[y0:y1, x0:x1]


class ArrayFrameSource:
    """In-memory frames (tests, small stacks)."""

    def __init__(self, frames: list[np.ndarray]):
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    def read(self, idx: int, region: Region | None = None) -> np.ndarray:
        return _crop(self._frames[idx], region)


class DirFrameSource:
    """Frames loaded lazily from disk paths; re-reads on every call (streaming)."""

    def __init__(self, paths: list[Path]):
        self.paths = [Path(p) for p in paths]

    def __len__(self) -> int:
        return len(self.paths)

    def read(self, idx: int, region: Region | None = None) -> np.ndarray:
        return _crop(load_image(self.paths[idx]).pixels, region)
```

Update `engine/src/focusstack/stack/__init__.py`:

```python
from focusstack.stack.base import (
    REGISTRY,
    FrameSource,
    ParamSpec,
    StackResult,
    get_algorithm,
    register,
)
from focusstack.stack.sources import ArrayFrameSource, DirFrameSource

__all__ = [
    "REGISTRY",
    "ArrayFrameSource",
    "DirFrameSource",
    "FrameSource",
    "ParamSpec",
    "StackResult",
    "get_algorithm",
    "register",
]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/engine/test_registry.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/stack tests/engine/test_registry.py
git commit -m "feat: algorithm registry with typed params and streaming frame sources"
```

---

### Task 9: PMax — streaming fold + halo control + validity masks

**Files:**
- Create: `engine/src/focusstack/stack/pmax.py`
- Modify: `engine/src/focusstack/stack/__init__.py` (import pmax so it registers)
- Test: `tests/engine/test_pmax.py`

This is SPEC §7.1 verbatim — read it before implementing. Key invariants: memory independent of frame count (streaming fold), winner maps are **int32**, halo control is a median filter on winner maps + a second streaming pass, result stays unclamped.

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_pmax.py`:

```python
import numpy as np
import pytest
import torch
from skimage.metrics import structural_similarity

from focusstack.backend import get_device
from focusstack.stack import ArrayFrameSource, get_algorithm
from tests.synthetic.generate import generate_stack


@pytest.fixture(scope="module")
def synth():
    return generate_stack(h=192, w=256, n_frames=10, max_sigma=5.0, seed=7)


def _ssim(a, b):
    return structural_similarity(a.clip(0, 1), b.clip(0, 1), channel_axis=2, data_range=1.0)


def test_pmax_registered_with_params():
    algo = get_algorithm("pmax")
    names = [p.name for p in algo.params()]
    assert "selection_smoothing" in names


def test_pmax_beats_every_input_frame(synth):
    algo = get_algorithm("pmax")
    res = algo.run(ArrayFrameSource(synth.frames), get_device("cpu"), {"selection_smoothing": 1})
    result_ssim = _ssim(res.image, synth.sharp)
    best_frame_ssim = max(_ssim(f, synth.sharp) for f in synth.frames)
    assert result_ssim > best_frame_ssim


def test_pmax_quality_gate(synth):
    # SPEC §13.2: SSIM > 0.97 vs ground-truth sharp image
    algo = get_algorithm("pmax")
    res = algo.run(ArrayFrameSource(synth.frames), get_device("cpu"), {"selection_smoothing": 1})
    assert _ssim(res.image, synth.sharp) > 0.97


def test_pmax_progress_and_cancel(synth):
    algo = get_algorithm("pmax")
    calls = []
    algo.run(
        ArrayFrameSource(synth.frames), get_device("cpu"), {"selection_smoothing": 0},
        progress=lambda msg, frac: calls.append(frac),
    )
    assert calls and calls[-1] == pytest.approx(1.0)
    with pytest.raises(InterruptedError):
        algo.run(
            ArrayFrameSource(synth.frames), get_device("cpu"), {"selection_smoothing": 0},
            cancel=lambda: True,
        )


def test_pmax_validity_mask_excludes_region(synth):
    """A frame whose mask invalidates a region must never win there."""
    frames = [f.copy() for f in synth.frames]
    frames[3][:, :, :] = 5.0  # absurd hot frame; would dominate energy everywhere
    masks = [np.ones(f.shape[:2], dtype=bool) for f in frames]
    masks[3][:, :] = False    # ...but it is fully invalid
    algo = get_algorithm("pmax")
    res = algo.run(
        ArrayFrameSource(frames), get_device("cpu"),
        {"selection_smoothing": 0}, masks=ArrayMaskSource(masks),
    )
    assert res.image.max() < 2.0  # the hot frame never selected


def test_pmax_smoothing_changes_output(synth):
    algo = get_algorithm("pmax")
    src = ArrayFrameSource(synth.frames)
    a = algo.run(src, get_device("cpu"), {"selection_smoothing": 0}).image
    b = algo.run(src, get_device("cpu"), {"selection_smoothing": 2}).image
    assert not np.allclose(a, b)
    assert _ssim(b, synth.sharp) > 0.97  # smoothing must not wreck quality


def test_pmax_device_parity(accel_device, synth):
    algo = get_algorithm("pmax")
    src = ArrayFrameSource(synth.frames)
    cpu = algo.run(src, get_device("cpu"), {"selection_smoothing": 1}).image
    dev = algo.run(src, accel_device, {"selection_smoothing": 1}).image
    np.testing.assert_allclose(dev, cpu, atol=1e-3)


class ArrayMaskSource:
    def __init__(self, masks):
        self._masks = masks

    def __len__(self):
        return len(self._masks)

    def read(self, idx, region=None):
        m = self._masks[idx]
        if region is None:
            return m
        y0, x0, y1, x1 = region
        return m[y0:y1, x0:x1]
```

(Move `ArrayMaskSource` above its first use in the file when writing it.)

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/engine/test_pmax.py -v`
Expected: FAIL — `unknown stacking algorithm 'pmax'`

- [ ] **Step 3: Implement `engine/src/focusstack/stack/pmax.py`**

```python
"""PMax: Laplacian pyramid, max-energy selection, streaming fold (SPEC §7.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from focusstack.backend import Device, ops
from focusstack.stack.base import (
    CancelFn,
    FrameSource,
    ParamSpec,
    ProgressFn,
    StackResult,
    register,
)
from focusstack.stack.pyramid import build_laplacian, collapse_laplacian, pyramid_depth

_NEG_INF = float("-inf")


@dataclass
class _FoldState:
    lap: list[torch.Tensor]          # best coefficients per level (C, h, w)
    energy: list[torch.Tensor]       # best energy per level (1, h, w)
    winner: list[torch.Tensor]       # winning frame index per level (h, w) int32
    res_num: torch.Tensor            # residual weighted sum (C, h, w)
    res_den: torch.Tensor            # residual weight sum (1, h, w)


def _frame_pyramid(frame_np: np.ndarray, device: Device, depth: int):
    t = ops.to_tensor(frame_np, device)
    lap, residual = build_laplacian(t, depth)
    energies = [ops.local_energy(lvl) for lvl in lap]
    res_weight = ops.box_filter3((residual - ops.box_filter3(residual)).abs().mean(0, keepdim=True)) + 1e-6
    return lap, energies, residual, res_weight


def _mask_levels(mask_np: np.ndarray, device: Device, depth: int) -> list[torch.Tensor]:
    """Downsample a validity mask through the pyramid; valid = coverage > 0.5."""
    m = torch.from_numpy(mask_np.astype(np.float32))[None].to(device.torch_device)
    levels = []
    for _ in range(depth + 1):  # one per lap level + residual
        levels.append(m > 0.5)
        m = ops.downsample2(m)
    return levels


@register("pmax")
class PMax:
    @staticmethod
    def params() -> list[ParamSpec]:
        return [
            ParamSpec(
                "selection_smoothing", "Selection smoothing", "int",
                default=1, min=0, max=3,
                tooltip=(
                    "Median-filters the per-level winner maps to suppress halos. "
                    "0 = off (fastest). PMax can amplify noise and contrast; that is "
                    "expected and matches Zerene."
                ),
            ),
        ]

    def run(
        self,
        source: FrameSource,
        device: Device,
        params: dict[str, Any],
        progress: ProgressFn | None = None,
        cancel: CancelFn | None = None,
        masks: FrameSource | None = None,
    ) -> StackResult:
        smoothing = int(params.get("selection_smoothing", 1))
        n = len(source)
        first = source.read(0)
        h, w = first.shape[:2]
        # tiles must fuse with the FULL image's pyramid depth or the tiled and
        # untiled frequency-band splits differ structurally (SPEC §8); the tiler
        # passes the full-image depth via this private param.
        depth = int(params.get("_pyramid_depth") or pyramid_depth(h, w))
        # second pass (if smoothing) costs another n frame reads
        total_steps = n * (2 if smoothing > 0 else 1) + 1
        state: _FoldState | None = None

        def _tick(i: int, msg: str) -> None:
            if cancel is not None and cancel():
                raise InterruptedError("stack job cancelled")
            if progress is not None:
                progress(msg, i / total_steps)

        for idx in range(n):
            _tick(idx, f"PMax fold {idx + 1}/{n}")
            frame = first if idx == 0 else source.read(idx)
            lap, energies, residual, res_w = _frame_pyramid(frame, device, depth)
            mlv = _mask_levels(masks.read(idx), device, depth) if masks is not None else None
            if mlv is not None:
                energies = [e.masked_fill(~m, _NEG_INF) for e, m in zip(energies, mlv)]
                res_w = res_w * mlv[depth].to(res_w.dtype)
            if state is None:
                state = _FoldState(
                    lap=lap,
                    energy=energies,
                    winner=[
                        torch.zeros(e.shape[-2:], dtype=torch.int32, device=e.device)
                        for e in energies
                    ],
                    res_num=residual * res_w,
                    res_den=res_w.clone(),
                )
                continue
            for lvl in range(depth):
                better = energies[lvl] > state.energy[lvl]  # (1, h, w)
                state.lap[lvl] = torch.where(better, lap[lvl], state.lap[lvl])
                state.energy[lvl] = torch.where(better, energies[lvl], state.energy[lvl])
                state.winner[lvl] = torch.where(
                    better.squeeze(0), torch.full_like(state.winner[lvl], idx), state.winner[lvl]
                )
            state.res_num = state.res_num + residual * res_w
            state.res_den = state.res_den + res_w

        assert state is not None

        if smoothing > 0:
            filtered = [ops.median_filter2d(wm, smoothing) for wm in state.winner]
            changed = [f != wm for f, wm in zip(filtered, state.winner)]
            if any(c.any() for c in changed):
                for idx in range(n):
                    _tick(n + idx, f"PMax halo pass {idx + 1}/{n}")
                    lap, _, _, _ = _frame_pyramid(source.read(idx), device, depth)
                    mlv = _mask_levels(masks.read(idx), device, depth) if masks is not None else None
                    for lvl in range(depth):
                        sel = changed[lvl] & (filtered[lvl] == idx)
                        if mlv is not None:
                            # never paste a frame's coefficients where it is invalid
                            sel = sel & mlv[lvl].squeeze(0)
                        if sel.any():
                            state.lap[lvl] = torch.where(sel.unsqueeze(0), lap[lvl], state.lap[lvl])

        residual = state.res_num / state.res_den.clamp_min(1e-8)
        result = collapse_laplacian(state.lap, residual)
        _tick(total_steps - 1, "collapse")
        if progress is not None:
            progress("done", 1.0)
        return StackResult(
            image=ops.to_numpy(result),
            # SPEC §7.1: per-level selection visualization (debug toggle in the UI later)
            aux={f"winner_l{i}": wm.cpu().numpy() for i, wm in enumerate(state.winner)},
        )
```

Append to `engine/src/focusstack/stack/__init__.py`:

```python
import focusstack.stack.pmax  # noqa: E402,F401  (registers "pmax")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/engine/test_pmax.py -v`
Expected: all PASS. The quality gate (SSIM > 0.97) is the test most likely to need iteration — if it fails marginally, inspect whether the residual weighted-average is washing out detail (try energy-max selection of the residual's high-energy regions) **before** touching the threshold. The threshold is a SPEC §13.2 requirement; do not lower it.

If `test_pmax_device_parity` fails with isolated large diffs, that is argmax tie-sensitivity (a near-tie energy comparison flipping winners between CPU and CUDA changes a whole coefficient), not a bug: add a deterministic tie margin (`better = energies[lvl] > state.energy[lvl] + 1e-7`) rather than loosening the tolerance.

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/stack tests/engine/test_pmax.py
git commit -m "feat: PMax streaming fold with halo control and validity masks"
```

---

## Chunk 3: Tiling, pipeline, CLI, M1 acceptance

### Task 10: Memory estimation + tiled processing

**Files:**
- Create: `engine/src/focusstack/tiles.py`
- Test: `tests/engine/test_tiles.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_tiles.py`:

```python
import numpy as np
from skimage.metrics import structural_similarity

from focusstack.backend import get_device
from focusstack.stack import ArrayFrameSource, get_algorithm
from focusstack.tiles import estimate_stack_bytes, plan_tiles, stack_tiled
from tests.synthetic.generate import generate_stack


def test_plan_tiles_covers_image_exactly():
    tiles = plan_tiles(500, 700, tile=256, overlap=32)
    covered = np.zeros((500, 700), dtype=int)
    for t in tiles:
        covered[t.y0 : t.y1, t.x0 : t.x1] += 1
    assert (covered >= 1).all()


def test_plan_tiles_single_when_image_fits():
    tiles = plan_tiles(200, 200, tile=256, overlap=32)
    assert len(tiles) == 1
    t = tiles[0]
    assert (t.y0, t.x0, t.y1, t.x1) == (0, 0, 200, 200)


def test_estimate_scales_with_pixels():
    small = estimate_stack_bytes(1000, 1500)
    big = estimate_stack_bytes(4000, 6000)
    assert big > small * 10
    assert small > 1000 * 1500 * 3 * 4  # at least one frame's worth


def test_tiled_matches_untiled():
    # SPEC §8: tiled and untiled agree within atol=2e-3
    synth = generate_stack(h=192, w=256, n_frames=8, max_sigma=4.0, seed=11)
    device = get_device("cpu")
    algo = get_algorithm("pmax")
    src = ArrayFrameSource(synth.frames)
    params = {"selection_smoothing": 0}
    untiled = algo.run(src, device, params).image
    tiled = stack_tiled("pmax", src, device, params, tile=96, overlap=48)
    np.testing.assert_allclose(tiled, untiled, atol=2e-3)
    # belt and braces: quality preserved
    s = structural_similarity(
        tiled.clip(0, 1), synth.sharp, channel_axis=2, data_range=1.0
    )
    assert s > 0.95
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/engine/test_tiles.py -v`
Expected: FAIL — `No module named 'focusstack.tiles'`

- [ ] **Step 3: Implement `engine/src/focusstack/tiles.py`**

```python
"""Tiled processing with feathered overlap blending (SPEC §8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from focusstack.backend import Device
from focusstack.stack.base import FrameSource
from focusstack.stack.pyramid import pyramid_depth


@dataclass(frozen=True)
class Tile:
    y0: int
    x0: int
    y1: int
    x1: int


def plan_tiles(h: int, w: int, tile: int, overlap: int) -> list[Tile]:
    if overlap >= tile:
        raise ValueError(f"overlap {overlap} must be smaller than tile {tile}")
    if h <= tile and w <= tile:
        return [Tile(0, 0, h, w)]
    step = tile - overlap
    ys = list(range(0, max(h - overlap, 1), step))
    xs = list(range(0, max(w - overlap, 1), step))
    out = []
    for y in ys:
        for x in xs:
            out.append(Tile(y, x, min(y + tile, h), min(x + tile, w)))
    return out


def estimate_stack_bytes(h: int, w: int, c: int = 3) -> int:
    """Rough upper bound of device memory for an untiled PMax fold (SPEC §4).

    State: pyramid coefficients (4/3 * C floats) + energy + winner per pixel,
    plus ~3 transient frame-sized buffers during the per-frame pyramid build.
    Frame count does not appear: the fold is streaming.
    """
    px = h * w
    pyr = px * 4 / 3
    state = pyr * (c * 4 + 4 + 4)
    transient = 3 * px * c * 4 + pyr * (c * 4 + 4)
    return int(state + transient)


def _feather(t: Tile, h: int, w: int, overlap: int) -> np.ndarray:
    """Per-tile weights: 1 in the interior, linear ramp across overlap,
    full weight at image borders (no neighbor there)."""
    th, tw = t.y1 - t.y0, t.x1 - t.x0
    ramp = max(overlap, 1)

    def axis_w(n: int, lo_edge: bool, hi_edge: bool) -> np.ndarray:
        a = np.ones(n, dtype=np.float32)
        r = min(ramp, n)
        if not lo_edge:
            a[:r] = np.linspace(1.0 / (r + 1), 1.0, r, dtype=np.float32)
        if not hi_edge:
            a[n - r :] = np.linspace(1.0, 1.0 / (r + 1), r, dtype=np.float32)
        return a

    wy = axis_w(th, t.y0 == 0, t.y1 == h)
    wx = axis_w(tw, t.x0 == 0, t.x1 == w)
    return np.outer(wy, wx)[..., None]  # (th, tw, 1)


class _RegionSource:
    """Restricts a FrameSource to one tile rectangle."""

    def __init__(self, inner: FrameSource, t: Tile):
        self._inner = inner
        self._t = t

    def __len__(self) -> int:
        return len(self._inner)

    def read(self, idx: int, region=None) -> np.ndarray:
        t = self._t
        if region is None:
            return self._inner.read(idx, region=(t.y0, t.x0, t.y1, t.x1))
        y0, x0, y1, x1 = region
        return self._inner.read(idx, region=(t.y0 + y0, t.x0 + x0, t.y0 + y1, t.x0 + x1))


def stack_tiled(
    method: str,
    source: FrameSource,
    device: Device,
    params: dict[str, Any],
    tile: int = 2048,
    overlap: int = 128,
    progress=None,
    cancel=None,
    masks: FrameSource | None = None,
) -> np.ndarray:
    from focusstack.stack.base import get_algorithm  # local import avoids cycle

    probe = source.read(0)
    h, w, c = probe.shape
    del probe
    # Tiles must fuse at the FULL image's pyramid depth (passed to the algorithm
    # below) or the tiled/untiled frequency-band splits differ structurally.
    full_depth = pyramid_depth(h, w)
    # SPEC §8: pyramid algorithms need overlap >= 2^depth to avoid seams...
    overlap = max(overlap, 2**full_depth)
    # ...and plan_tiles needs tile > overlap or the tile grid degenerates.
    tile = max(tile, 4 * overlap)
    params = {**params, "_pyramid_depth": full_depth}
    tiles = plan_tiles(h, w, tile, overlap)
    num = np.zeros((h, w, c), dtype=np.float32)
    den = np.zeros((h, w, 1), dtype=np.float32)
    algo = get_algorithm(method)
    for i, t in enumerate(tiles):
        if progress is not None:
            progress(f"tile {i + 1}/{len(tiles)}", i / len(tiles))
        sub = _RegionSource(source, t)
        sub_masks = _RegionSource(masks, t) if masks is not None else None
        kwargs = {"masks": sub_masks} if sub_masks is not None else {}
        res = algo.run(sub, device, params, cancel=cancel, **kwargs)
        fw = _feather(t, h, w, overlap)
        num[t.y0 : t.y1, t.x0 : t.x1] += res.image * fw
        den[t.y0 : t.y1, t.x0 : t.x1] += fw
    if progress is not None:
        progress("done", 1.0)
    return num / np.maximum(den, 1e-8)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/engine/test_tiles.py -v`
Expected: all PASS. `test_tiled_matches_untiled` is sensitive to feathering and overlap; if it fails, check overlap ≥ 2^depth of the *tile* pyramid first.

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/tiles.py tests/engine/test_tiles.py
git commit -m "feat: tiled stacking with feathered blending and memory estimation"
```

---

### Task 11: Pipeline orchestration with OOM fallback chain

**Files:**
- Create: `engine/src/focusstack/pipeline.py`
- Test: `tests/engine/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_pipeline.py`:

```python
import numpy as np
import pytest

from focusstack.errors import ValidationError
from focusstack.io import save_image
from focusstack.pipeline import stack_frames
from tests.synthetic.generate import generate_stack


@pytest.fixture
def stack_dir(tmp_path):
    synth = generate_stack(h=96, w=128, n_frames=6, max_sigma=4.0, seed=2)
    for i, f in enumerate(synth.frames):
        save_image(f, tmp_path / f"frame_{i:03d}.tif", bit_depth=16)
    return tmp_path, synth


def test_stack_frames_from_dir(stack_dir):
    d, synth = stack_dir
    result = stack_frames(sorted(d.glob("*.tif")), method="pmax", device_pref="cpu")
    assert result.image.shape == synth.sharp.shape


def test_validation_failure_raises_with_filenames(stack_dir):
    d, _ = stack_dir
    bad = np.zeros((10, 10, 3), dtype=np.float32)
    save_image(bad, d / "zz_wrong_size.tif")
    with pytest.raises(ValidationError, match="zz_wrong_size"):
        stack_frames(sorted(d.glob("*.tif")), method="pmax", device_pref="cpu")


def test_forced_tiled_mode_matches_direct(stack_dir):
    d, _ = stack_dir
    paths = sorted(d.glob("*.tif"))
    direct = stack_frames(paths, method="pmax", device_pref="cpu", tile_mode="never")
    tiled = stack_frames(paths, method="pmax", device_pref="cpu", tile_mode="always", tile=64)
    np.testing.assert_allclose(tiled.image, direct.image, atol=2e-3)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/engine/test_pipeline.py -v`
Expected: FAIL — `No module named 'focusstack.pipeline'`

- [ ] **Step 3: Implement `engine/src/focusstack/pipeline.py`**

```python
"""Job orchestration: validate -> stack (direct/tiled) -> fallback chain (SPEC §4, §12).

M1 scope: alignment is M2; frames are assumed pre-aligned here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from focusstack.backend import Device, empty_cache, free_memory, get_device
from focusstack.errors import ValidationError
from focusstack.io import validate_stack
from focusstack.stack import DirFrameSource, StackResult, get_algorithm
from focusstack.tiles import estimate_stack_bytes, stack_tiled

log = logging.getLogger(__name__)


def stack_frames(
    paths: list[Path],
    *,
    method: str = "pmax",
    params: dict[str, Any] | None = None,
    device_pref: str = "auto",
    tile_mode: str = "auto",  # auto | always | never
    tile: int = 2048,
    progress=None,
    cancel=None,
) -> StackResult:
    params = dict(params or {})
    report = validate_stack(paths)
    if not report.ok:
        bad = [f"{s.path.name}: {s.status}" for s in report.files if s.status != "ok"]
        raise ValidationError("stack validation failed:\n  " + "\n  ".join(bad))
    if len(paths) < 2:
        raise ValidationError("a stack needs at least 2 frames")

    device = get_device(device_pref)
    source = DirFrameSource(list(paths))

    use_tiled = tile_mode == "always"
    if tile_mode == "auto" and device.kind != "cpu":
        needed = estimate_stack_bytes(report.height, report.width)
        budget = int(free_memory(device) * 0.8)
        use_tiled = needed > budget
        if use_tiled:
            log.warning("estimated %d MB > budget %d MB; using tiled mode",
                        needed >> 20, budget >> 20)

    def _run(dev: Device, tiled: bool) -> StackResult:
        if tiled:
            img = stack_tiled(method, source, dev, params, tile=tile,
                              progress=progress, cancel=cancel)
            return StackResult(image=img)
        return get_algorithm(method).run(source, dev, params,
                                         progress=progress, cancel=cancel)

    def _is_oom(e: Exception) -> bool:
        # CUDA raises a dedicated type; MPS allocation failures are plain
        # RuntimeErrors and ONLY on MPS may a message check classify them —
        # on CUDA a "memory" string can also mean illegal access (a real bug).
        if isinstance(e, torch.cuda.OutOfMemoryError):
            return True
        return (
            device.kind == "mps"
            and isinstance(e, RuntimeError)
            and "memory" in str(e).lower()
        )

    # SPEC §4 fallback chain: direct -> tiled -> CPU. A job never fails on OOM.
    try:
        return _run(device, use_tiled)
    except Exception as e:
        if not _is_oom(e):
            raise
        log.warning("device OOM; retrying tiled")
        empty_cache(device)
    try:
        return _run(device, True)
    except Exception as e:
        if not _is_oom(e):
            raise
        log.warning("device OOM even tiled; falling back to CPU")
        empty_cache(device)
        return _run(get_device("cpu"), True)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/engine/test_pipeline.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/pipeline.py tests/engine/test_pipeline.py
git commit -m "feat: pipeline orchestration with validation and OOM fallback chain"
```

---

### Task 12: CLI

**Files:**
- Create: `engine/src/focusstack/cli.py`
- Test: `tests/engine/test_cli.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_cli.py`:

```python
import numpy as np
from skimage.metrics import structural_similarity
from typer.testing import CliRunner

from focusstack.cli import app
from focusstack.io import load_image, save_image
from tests.synthetic.generate import generate_stack

runner = CliRunner()


def test_stack_command_end_to_end(tmp_path):
    synth = generate_stack(h=96, w=128, n_frames=6, max_sigma=4.0, seed=4)
    for i, f in enumerate(synth.frames):
        save_image(f, tmp_path / f"f_{i:03d}.tif", bit_depth=16)
    out = tmp_path / "stacked.tif"
    result = runner.invoke(
        app, ["stack", str(tmp_path), "-o", str(out), "--method", "pmax", "--device", "cpu"]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    img = load_image(out)
    s = structural_similarity(img.pixels, synth.sharp.clip(0, 1), channel_axis=2, data_range=1.0)
    assert s > 0.95
    assert "cpu" in result.output.lower()


def _err_text(result) -> str:
    # click >= 8.2 captures stderr separately (result.stderr); older click
    # mixed it into result.output. Handle both so the suite isn't pinned.
    try:
        return result.output + result.stderr
    except ValueError:
        return result.output


def test_stack_command_no_images(tmp_path):
    result = runner.invoke(app, ["stack", str(tmp_path), "-o", str(tmp_path / "x.tif")])
    assert result.exit_code != 0
    assert "no input images" in _err_text(result).lower()


def test_stack_command_validation_error(tmp_path):
    save_image(np.zeros((10, 10, 3), dtype=np.float32), tmp_path / "a.tif")
    save_image(np.zeros((20, 20, 3), dtype=np.float32), tmp_path / "b.tif")
    result = runner.invoke(app, ["stack", str(tmp_path), "-o", str(tmp_path / "x.tif")])
    assert result.exit_code != 0
    assert "wrong_size" in _err_text(result)


def test_serve_hint_without_server_package():
    result = runner.invoke(app, ["serve"])
    assert result.exit_code != 0
    assert "focusstack-server" in _err_text(result)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/engine/test_cli.py -v`
Expected: FAIL — `No module named 'focusstack.cli'`

- [ ] **Step 3: Implement `engine/src/focusstack/cli.py`**

```python
"""FocusStack CLI (SPEC §14 M1). `focusstack stack DIR -o out.tif --method pmax`."""

from __future__ import annotations

import time
from pathlib import Path

import typer

from focusstack.errors import FocusStackError
from focusstack.io import load_image, save_image
from focusstack.pipeline import stack_frames
from focusstack.stack import REGISTRY

app = typer.Typer(no_args_is_help=True, add_completion=False)

EXTENSIONS = ("*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png")


@app.command()
def stack(
    input_dir: Path = typer.Argument(..., help="Directory of aligned stack frames"),
    output: Path = typer.Option(..., "-o", "--output", help="Output image (.tif/.jpg/.png)"),
    method: str = typer.Option("pmax", help=f"Stacking method: {sorted(REGISTRY)}"),
    device: str = typer.Option("auto", help="Compute device: auto|cuda|mps|cpu"),
    selection_smoothing: int = typer.Option(1, min=0, max=3, help="PMax halo control (0=off)"),
    tile_size: int = typer.Option(2048, help="Tile size when tiling is needed"),
    jpeg_quality: int = typer.Option(95, min=1, max=100),
):
    """Stack pre-aligned frames into a single all-in-focus image."""
    paths: list[Path] = []
    for pattern in EXTENSIONS:
        paths.extend(input_dir.glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        typer.echo(f"error: no input images found in {input_dir}", err=True)
        raise typer.Exit(1)
    typer.echo(f"{len(paths)} frames, method={method}, device={device}")

    last = {"msg": ""}

    def progress(msg: str, frac: float) -> None:
        if msg != last["msg"]:
            typer.echo(f"  [{frac * 100:5.1f}%] {msg}")
            last["msg"] = msg

    t0 = time.perf_counter()
    try:
        result = stack_frames(
            paths,
            method=method,
            params={"selection_smoothing": selection_smoothing},
            device_pref=device,
            tile=tile_size,
            progress=progress,
        )
    except FocusStackError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from None
    # ICC from the reference (middle) frame, SPEC §5
    icc = load_image(paths[len(paths) // 2]).icc
    depth_out = 16 if output.suffix.lower() in (".tif", ".tiff") else 8
    save_image(result.image, output, bit_depth=depth_out, icc=icc, jpeg_quality=jpeg_quality)
    typer.echo(f"wrote {output} in {time.perf_counter() - t0:.1f}s")


@app.command()
def serve():
    """Launch the FocusStack server + web UI (requires the focusstack-server package)."""
    try:
        from focusstack_server.main import run  # type: ignore[import-not-found]
    except ImportError:
        typer.echo(
            "error: the web UI is not installed. Install the focusstack-server package "
            "(coming in milestone M4): uv sync (workspace) or pip install focusstack-server",
            err=True,
        )
        raise typer.Exit(1) from None
    run()
```

Note: the CLI prints the *requested* device; `stack_frames` logs the resolved one. For the test that asserts `"cpu" in output`, passing `--device cpu` covers it.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/engine/test_cli.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/cli.py tests/engine/test_cli.py
git commit -m "feat: focusstack CLI with stack command and serve stub"
```

---

### Task 13: M1 acceptance gate

**Files:**
- Create: `README.md`
- Modify: anything the gates below flag

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests pass (device-parity tests exercise CUDA on this machine).

- [ ] **Step 2: Lint and type-check**

Run: `uv run ruff check .` — Expected: clean (fix anything it flags).
Run: `uv run ruff format --check .` — Expected: clean (run `uv run ruff format .` if not).
Run: `uv run mypy engine/src` — Expected: no errors.

- [ ] **Step 3: Real-stack smoke test (manual demo, SPEC M1 acceptance)**

Generate a demo stack on disk and run the real CLI binary:

```bash
uv run python -c "
from pathlib import Path
from tests.synthetic.generate import generate_stack
from focusstack.io import save_image
d = Path('demo_stack'); d.mkdir(exist_ok=True)
s = generate_stack(h=1024, w=1536, n_frames=20, max_sigma=6.0, seed=42)
[save_image(f, d / f'frame_{i:03d}.tif', bit_depth=16) for i, f in enumerate(s.frames)]
save_image(s.sharp, d / 'ground_truth.tif')
print('wrote', len(s.frames), 'frames')
"
uv run focusstack stack demo_stack -o demo_stack/result_pmax.tif --method pmax
```

Expected: completes without error in seconds on GPU; open `result_pmax.tif` next to `ground_truth.tif` and confirm the result is sharp everywhere. Then delete `demo_stack/` (do not commit it).

- [ ] **Step 4: Write `README.md`** (quick start per SPEC §16)

```markdown
# FocusStack

Professional GPU-accelerated focus stacking. CUDA (NVIDIA), MPS (Apple silicon), CPU fallback.
See `docs/SPEC.md` for the full specification.

## Quick start

    uv sync --extra cu12x        # NVIDIA (Windows/Linux); --extra cpu without a GPU; plain uv sync on Apple silicon
    uv run focusstack stack ./my_stack -o result.tif --method pmax

## Status

- [x] M1 — engine core, PMax, CLI
- [ ] M2 — alignment
- [ ] M3 — DMap, weighted, slabbing, smart frame selection
- [ ] M4 — server + web UI
- [ ] M5 — retouching
- [ ] M6 — polish, Docker, CI

## Development

    uv run pytest          # CPU + whatever accelerator this machine has
    uv run ruff check .
    uv run mypy engine/src
```

- [ ] **Step 5: Final commit**

```bash
git add README.md
git commit -m "docs: README with M1 status and quick start"
```

- [ ] **Step 6: Verify milestone acceptance against SPEC §14 M1**

Checklist (all must be true): parity tests pass ✓ / PMax quality gate (SSIM > 0.97) ✓ / tiled-matches-untiled ✓ / CLI stacks a real pre-aligned stack ✓ / ruff + mypy clean ✓. If any box is unchecked, M1 is not done — fix before declaring completion.
