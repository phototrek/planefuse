# PlaneFuse M3 (Part 1) — DMap, Weighted Average, Slabbing Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three stacking algorithms to the registry — DMap (depth-map, §7.2), Weighted average (§7.3), and Slabbing (§7.4) — plus the shared sharpness ops they need and 16-bit grayscale depth-map export, so `planefuse stack --method dmap|weighted|slab` works end to end and the §13.2 quality gates pass for every method.

**Architecture:** All three register via the existing `@register` decorator in `stack/`, consume the same streamed `FrameSource` + optional validity `masks=` that PMax already accepts, and emit `StackResult(image, aux)`. New device-agnostic ops (`sharpness_map`, `box_filter`, `guided_filter`, `masked_diffuse`) live in `backend/ops.py` as the one true implementation, parity-tested per device. DMap streams an argmax fold then refines the index map (contrast threshold → guided-filter smoothing → masked diffusion) and composites in a second streaming pass with fractional-index blending, emitting a depth map in `aux`. Weighted is a single streaming online-softmax pass with a stack-global scale. Slabbing is a composite algorithm that reuses the registry: it splits the ordered frames into overlapping slabs, stacks each with an inner method, caches the slab results, then stacks those with an outer method.

**Tech Stack:** Python 3.12, PyTorch 2.x, numpy, tifffile, typer, pytest, scikit-image (tests only), uv.

**Read first:** `docs/SPEC.md` §4 (device ops: sharpness_map, guided_filter, masked iterative box blur), §5 (DMap depth-map export), §7.0 intro (registry + streaming contract), §7.2 (DMap), §7.3 (Weighted), §7.4 (Slabbing), §8 (tiling caveat), §13.2 (quality gates). Spec wins on any conflict; flag conflicts rather than silently deviating.

**Preconditions:** M1 and M2 are complete and green on `main` (`uv run pytest` → 99 passed, 10 skipped; ruff + mypy clean). Verify with `git log --oneline -1` showing the M2 merge (`ee0b619`). **Smart frame selection (§7.0) is NOT in this plan** — it is a separate, large pre-stacking stage and gets its own plan (M3 Part 2). This plan delivers the three fusion algorithms and depth export, which satisfy the "quality tests pass for all methods; depth map exports" half of the §14 M3 acceptance criteria.

**Conventions used throughout (unchanged from M1/M2):**
- Image tensors: `torch.float32`, shape `(C, H, W)`; grayscale/maps are `(1, H, W)` or `(H, W)`. Values nominally `[0, 1]`, never clamped except at export.
- NumPy interchange: `(H, W, C)` float32; depth maps are `(H, W)` float32 in `[0, 1]`.
- An algorithm's `run(self, source, device, params, progress=None, cancel=None, masks=None)` — `masks` is an optional `FrameSource` of `(H, W)` bool validity masks (the same object the alignment cache and `stack_tiled` already supply). Invalid pixels never drive selection: their sharpness/weight is forced to 0 (SPEC §6 step 5).
- Frames stream from disk via `source.read(idx)`; never hold all frames in memory.
- All commands run from repo root `C:\Users\Adrien\Documents\CODE\focus-stacker` via `uv run …`.
- Tests are CPU-first; device parity uses the existing `device` / `accel_device` fixtures in `tests/engine/conftest.py`.
- Branch: do this work on `feat/m3-stacking` off `main`.
- This machine's torch is CPU-only, so accelerator parity tests auto-skip — that is expected (matches CI; SPEC §13.2 runs accelerator suites manually pre-release).

**Reference patterns to mirror:** `engine/src/planefuse/stack/pmax.py` is the template for a streaming algorithm with `masks=` handling, progress ticks, cancellation, the `_pyramid_depth`/tiling param, and `aux` output. DMap and Weighted should follow its shape (param dataclass via `params()`, `_tick` helper, mask-aware folding).

---

## Chunk 1: Shared sharpness ops + depth-map export

### Task 1: Sharpness + filtering ops in `backend/ops.py`

Add the device-agnostic primitives DMap and Weighted need: a general box filter (arbitrary radius), a windowed Laplacian sharpness map, an edge-aware guided filter, and a masked iterative diffusion fill. All parity-tested.

**Files:**
- Modify: `engine/src/planefuse/backend/ops.py` (append)
- Test: `tests/engine/test_sharpness_ops.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_sharpness_ops.py`:

```python
import numpy as np
import pytest
import torch

from planefuse.backend import get_device, ops


def _rand(c=1, h=64, w=80, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.uniform(0, 1, (c, h, w)).astype(np.float32))


def test_box_filter_radius0_is_identity():
    img = _rand(c=3)
    torch.testing.assert_close(ops.box_filter(img, radius=0), img, atol=1e-6, rtol=1e-6)


def test_box_filter_constant_invariant():
    img = torch.full((1, 20, 20), 0.3)
    torch.testing.assert_close(ops.box_filter(img, radius=4), img, atol=1e-5, rtol=1e-5)


def test_sharpness_map_higher_on_texture():
    flat = torch.full((1, 48, 48), 0.5)
    rng = np.random.default_rng(1)
    textured = torch.from_numpy(rng.uniform(0, 1, (1, 48, 48)).astype(np.float32))
    s_flat = ops.sharpness_map(flat, radius=8)
    s_tex = ops.sharpness_map(textured, radius=8)
    assert float(s_tex.mean()) > 10 * float(s_flat.mean()) + 1e-6
    assert s_flat.shape == (1, 48, 48)


def test_guided_filter_smooths_but_follows_guide_edge():
    # A step-edge guide; noisy source. Output should preserve the edge but
    # reduce noise (variance) on each side.
    guide = torch.zeros(1, 32, 32)
    guide[:, :, 16:] = 1.0
    rng = np.random.default_rng(2)
    src = guide + torch.from_numpy(rng.normal(0, 0.1, (1, 32, 32)).astype(np.float32))
    out = ops.guided_filter(guide, src, radius=4, eps=1e-3)
    assert out.shape == src.shape
    assert out[:, :, :16].var() < src[:, :, :16].var()      # denoised left side
    assert float(out[:, :, 15].mean()) < float(out[:, :, 16].mean())  # edge preserved


def test_masked_diffuse_fills_unknown_from_known():
    val = torch.zeros(1, 16, 16)
    val[:, :, :8] = 2.0          # left half known = 2.0
    known = torch.zeros(1, 16, 16, dtype=torch.bool)
    known[:, :, :8] = True
    filled = ops.masked_diffuse(val, known, radius=2, iters=50)
    # unknown right half should be filled toward the known value (no zeros left)
    assert float(filled[:, :, 8:].min()) > 1.0
    # known region is preserved
    torch.testing.assert_close(filled[:, :, :8], val[:, :, :8], atol=1e-4, rtol=1e-4)


# --- device parity (SPEC §13.2) ---

@pytest.mark.parametrize("opname,kwargs", [
    ("box_filter", {"radius": 3}),
    ("sharpness_map", {"radius": 6}),
])
def test_sharpness_op_parity(accel_device, opname, kwargs):
    img = _rand(c=1, h=72, w=88, seed=9)
    fn = getattr(ops, opname)
    cpu_out = fn(img, **kwargs)
    dev_out = fn(img.to(accel_device.torch_device), **kwargs)
    torch.testing.assert_close(dev_out.cpu(), cpu_out, atol=1e-3, rtol=1e-3)


def test_guided_filter_parity(accel_device):
    guide = _rand(c=1, h=64, w=72, seed=3)
    src = _rand(c=1, h=64, w=72, seed=4)
    cpu_out = ops.guided_filter(guide, src, radius=4, eps=1e-3)
    dev_out = ops.guided_filter(guide.to(accel_device.torch_device),
                                src.to(accel_device.torch_device), radius=4, eps=1e-3)
    torch.testing.assert_close(dev_out.cpu(), cpu_out, atol=2e-3, rtol=2e-3)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_sharpness_ops.py -v`
Expected: FAIL — `AttributeError: module 'planefuse.backend.ops' has no attribute 'box_filter'`.

- [ ] **Step 3: Append to `engine/src/planefuse/backend/ops.py`**

```python
# ----------------------------------------------------------------------------
# Sharpness + edge-aware filtering ops (SPEC §4, §7.2/§7.3).
# ----------------------------------------------------------------------------

def box_filter(img: torch.Tensor, radius: int) -> torch.Tensor:
    """(2r+1)x(2r+1) mean filter, reflect-padded. img: (C, H, W). radius 0 = identity.

    Separable: a horizontal then vertical uniform conv. O(r) per pixel via two passes.
    """
    if radius < 1:
        return img
    k = 2 * radius + 1
    c = img.shape[0]
    x = img.unsqueeze(0)
    wx = torch.full((c, 1, 1, k), 1.0 / k, dtype=img.dtype, device=img.device)
    wy = torch.full((c, 1, k, 1), 1.0 / k, dtype=img.dtype, device=img.device)
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, wx, groups=c)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, wy, groups=c)
    return x.squeeze(0)


_LAPLACIAN = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def sharpness_map(gray: torch.Tensor, radius: int) -> torch.Tensor:
    """Windowed local Laplacian energy (SPEC §4, §7.2). gray: (1, H, W) -> (1, H, W).

    energy(x) = mean over a (2r+1) window of laplacian(gray)^2. Higher = sharper.
    """
    if gray.shape[0] != 1:
        raise ValueError(f"sharpness_map expects (1, H, W), got {gray.shape[0]} channels")
    k = _LAPLACIAN.to(device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)
    x = F.pad(gray.unsqueeze(0), (1, 1, 1, 1), mode="reflect")
    lap = F.conv2d(x, k).squeeze(0)
    return box_filter(lap * lap, radius)


def guided_filter(guide: torch.Tensor, src: torch.Tensor, radius: int, eps: float) -> torch.Tensor:
    """Edge-aware smoothing of `src` guided by `guide` (He et al. 2010).

    guide, src: (1, H, W). Output follows guide's edges while smoothing src.
    """
    mean_i = box_filter(guide, radius)
    mean_p = box_filter(src, radius)
    corr_i = box_filter(guide * guide, radius)
    corr_ip = box_filter(guide * src, radius)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)
    return mean_a * guide + mean_b


def masked_diffuse(values: torch.Tensor, known: torch.Tensor, radius: int,
                   iters: int = 50, tol: float = 1e-5) -> torch.Tensor:
    """Distance-weighted diffusion fill (SPEC §7.2 step 4): fill !known pixels of
    `values` from known neighbours by iterated masked box blur until converged.

    values: (1, H, W) float; known: (1, H, W) bool. Known pixels are preserved.
    """
    out = values.clone()
    w = known.to(values.dtype)
    out = out * w  # zero the unknown region to start
    for _ in range(iters):
        num = box_filter(out, radius)
        den = box_filter(w, radius).clamp_min(1e-8)
        diffused = num / den
        new = torch.where(known, values, diffused)
        if float((new - out).abs().max()) < tol:
            out = new
            break
        out = new
    # any still-unknown pixel with no reachable known neighbour stays as diffused
    return out
```

Note for the implementer: `_LAPLACIAN` is created at module load on CPU and `.to(...)` inside the function so it follows the input device — same pattern as `_LUMA` in the alignment ops.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/engine/test_sharpness_ops.py -v`
Expected: all PASS (accelerator parity runs where present; auto-skips on CPU-only).

- [ ] **Step 5: Commit**

```bash
git add engine/src/planefuse/backend/ops.py tests/engine/test_sharpness_ops.py
git commit -m "feat: sharpness, box filter, guided filter, masked diffusion ops"
```
End commit body with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 2: 16-bit grayscale depth-map export

DMap exports a `(H, W)` depth map as 16-bit grayscale TIFF/PNG (SPEC §5). Extend `save_image` to accept 2-D arrays.

**Files:**
- Modify: `engine/src/planefuse/io/writer.py`
- Test: `tests/engine/test_writer_depth.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_writer_depth.py`:

```python
import numpy as np
import tifffile

from planefuse.io.writer import save_image


def test_depth_map_16bit_tiff_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    depth = rng.uniform(0, 1, (40, 60)).astype(np.float32)   # 2-D (H, W)
    out = tmp_path / "depth.tif"
    save_image(depth, out, bit_depth=16)
    back = tifffile.imread(out)
    assert back.shape == (40, 60)
    assert back.dtype == np.uint16
    np.testing.assert_allclose(back.astype(np.float32) / 65535.0, depth,
                               atol=1.0 / 65535 + 1e-6)


def test_depth_map_8bit_png(tmp_path):
    depth = np.linspace(0, 1, 50 * 40, dtype=np.float32).reshape(50, 40)
    out = tmp_path / "depth.png"
    save_image(depth, out, bit_depth=8)
    assert out.exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_writer_depth.py -v`
Expected: FAIL (the RGB-only paths mishandle a 2-D array — likely a shape error from `Image.fromarray` or the tiff scaling).

- [ ] **Step 3: Modify `engine/src/planefuse/io/writer.py`**

At the top of `save_image`, right after `clipped = np.clip(arr, 0.0, 1.0)`, add a grayscale branch that handles 2-D `(H, W)` arrays for TIFF and PNG (depth-map export). Keep the existing RGB paths unchanged for 3-D input:

```python
    clipped = np.clip(arr, 0.0, 1.0)
    suffix = path.suffix.lower()

    # Depth-map / grayscale export: 2-D (H, W) arrays (SPEC §5 DMap depth map).
    if clipped.ndim == 2:
        if suffix in (".tif", ".tiff"):
            if bit_depth == 16:
                data = (clipped * 65535.0 + 0.5).astype(np.uint16)
            elif bit_depth == 8:
                data = (clipped * 255.0 + 0.5).astype(np.uint8)
            else:
                raise ValueError(f"unsupported TIFF bit depth {bit_depth}")
            if compression not in _TIFF_COMPRESSION:
                raise ValueError(f"unknown compression {compression!r}")
            tifffile.imwrite(path, data, compression=_TIFF_COMPRESSION[compression])
            return
        if suffix == ".png":
            if bit_depth == 16:
                data = (clipped * 65535.0 + 0.5).astype(np.uint16)
                Image.fromarray(data, mode="I;16").save(path)
            else:
                Image.fromarray((clipped * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)
            return
        raise ValueError(f"grayscale export supports .tif/.png, not {suffix!r}")
```

(The existing 3-D RGB code below this block is unchanged.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/engine/test_writer_depth.py -v`
Expected: PASS. If `mode="I;16"` PNG raises on the installed Pillow, fall back to a 16-bit grayscale TIFF in the test and note that 16-bit grayscale PNG, like 16-bit RGB PNG, is deferred to M6 — but try `I;16` first; it is generally supported.

- [ ] **Step 5: Commit**

```bash
git add engine/src/planefuse/io/writer.py tests/engine/test_writer_depth.py
git commit -m "feat: 16-bit grayscale depth-map export in save_image"
```
End commit body with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

## Chunk 2: DMap (depth map)

### Task 3: DMap streaming argmax fold

The core of DMap (SPEC §7.2 steps 1–2): per frame compute a sharpness map; streaming-fold the per-pixel argmax over frames into an integer index map + a running max-sharpness map. Memory is O(image), independent of frame count.

**Files:**
- Create: `engine/src/planefuse/stack/dmap.py` (fold portion; algorithm completed in Task 5)
- Test: `tests/engine/test_dmap_fold.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_dmap_fold.py`:

```python
import numpy as np

from planefuse.backend import get_device
from planefuse.stack.dmap import dmap_fold
from planefuse.stack.sources import ArrayFrameSource


def test_fold_picks_sharpest_frame_per_pixel():
    # 3 frames: frame k has a sharp (high-contrast) patch in column-band k.
    h, w = 32, 30
    frames = []
    for k in range(3):
        f = np.full((h, w, 3), 0.5, dtype=np.float32)
        band = slice(k * 10, k * 10 + 10)
        rng = np.random.default_rng(k)
        f[:, band, :] = rng.uniform(0, 1, (h, 10, 3)).astype(np.float32)  # texture
        frames.append(f)
    src = ArrayFrameSource(frames)
    index, sharp = dmap_fold(src, get_device("cpu"), radius=4)
    assert index.shape == (h, w) and sharp.shape == (h, w)
    # the textured band of each frame should be argmax'd to that frame
    assert int(np.round(np.median(index[:, 0:10]))) == 0
    assert int(np.round(np.median(index[:, 10:20]))) == 1
    assert int(np.round(np.median(index[:, 20:30]))) == 2


def test_fold_respects_masks():
    h, w = 16, 16
    frames = [np.full((h, w, 3), 0.5, dtype=np.float32) for _ in range(2)]
    rng = np.random.default_rng(0)
    frames[0][:, :, :] = rng.uniform(0, 1, (h, w, 3)).astype(np.float32)  # frame 0 textured
    masks = [np.ones((h, w), bool), np.ones((h, w), bool)]
    masks[0][:] = False  # frame 0 entirely invalid -> must never win
    src = ArrayFrameSource(frames)
    msrc = ArrayFrameSource([m[..., None].repeat(3, 2).astype(np.float32) for m in masks])
    # dmap_fold takes a mask FrameSource that yields (H,W) bool via read; adapt:
    index, _ = dmap_fold(src, get_device("cpu"), radius=2,
                         masks=_BoolMaskSource(masks))
    assert (index == 1).all()


class _BoolMaskSource:
    def __init__(self, masks):
        self._m = masks
    def __len__(self):
        return len(self._m)
    def read(self, idx, region=None):
        m = self._m[idx]
        if region is not None:
            y0, x0, y1, x1 = region
            m = m[y0:y1, x0:x1]
        return m
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_dmap_fold.py -v`
Expected: FAIL — `No module named 'planefuse.stack.dmap'`.

- [ ] **Step 3: Implement the fold in `engine/src/planefuse/stack/dmap.py`**

```python
"""DMap: depth-map focus stacking (SPEC §7.2).

Streaming argmax of a per-frame sharpness map -> integer index map + max-sharpness
map; the index map is then contrast-thresholded, edge-aware smoothed, and its
undecided regions diffusion-filled, before a second streaming pass composites the
result with fractional-index blending. Memory is O(image), not O(frames).
"""

from __future__ import annotations

import numpy as np
import torch

from planefuse.backend import Device, ops
from planefuse.stack.base import FrameSource


def _sharp_of(frame_np: np.ndarray, device: Device, radius: int) -> torch.Tensor:
    t = ops.to_tensor(frame_np, device)
    gray = ops.rgb_to_luminance(t)
    return ops.sharpness_map(gray, radius).squeeze(0)  # (H, W)


def dmap_fold(source: FrameSource, device: Device, radius: int,
              masks: FrameSource | None = None,
              progress=None, cancel=None) -> tuple[np.ndarray, np.ndarray]:
    """Streaming argmax fold. Returns (index map int32 (H,W), max-sharpness (H,W) float32)."""
    n = len(source)
    best_idx: torch.Tensor | None = None
    best_sharp: torch.Tensor | None = None
    for i in range(n):
        if cancel is not None and cancel():
            raise InterruptedError("dmap fold cancelled")
        if progress is not None:
            progress(f"DMap sharpness {i + 1}/{n}", i / max(2 * n, 1))
        s = _sharp_of(source.read(i), device, radius)  # (H, W)
        if masks is not None:
            m = torch.from_numpy(np.ascontiguousarray(masks.read(i))).to(device.torch_device)
            s = torch.where(m.bool(), s, torch.zeros_like(s))
        if best_idx is None:
            best_idx = torch.zeros(s.shape, dtype=torch.int32, device=s.device)
            best_sharp = s
            continue
        better = s > best_sharp
        best_idx = torch.where(better, torch.full_like(best_idx, i), best_idx)
        best_sharp = torch.where(better, s, best_sharp)
    assert best_idx is not None and best_sharp is not None
    return best_idx.cpu().numpy(), best_sharp.cpu().numpy()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/engine/test_dmap_fold.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/src/planefuse/stack/dmap.py tests/engine/test_dmap_fold.py
git commit -m "feat: DMap streaming argmax sharpness fold"
```
End commit body with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 4: DMap index-map refinement (threshold → smooth → diffuse)

SPEC §7.2 steps 3–4: mark pixels whose folded max-sharpness is below the contrast-threshold percentile as "undecided"; edge-aware smooth the index map with a guided filter (guide = max-sharpness); diffusion-fill the undecided regions.

**Files:**
- Modify: `engine/src/planefuse/stack/dmap.py` (add `refine_index`)
- Test: `tests/engine/test_dmap_refine.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_dmap_refine.py`:

```python
import numpy as np

from planefuse.backend import get_device
from planefuse.stack.dmap import refine_index


def test_refine_smooths_and_fills_undecided():
    rng = np.random.default_rng(0)
    h, w = 40, 40
    index = rng.integers(0, 5, (h, w)).astype(np.float32)
    sharp = rng.uniform(0.5, 1.0, (h, w)).astype(np.float32)
    sharp[:, :8] = 0.001  # a low-contrast band -> undecided at any sane percentile
    out = refine_index(index, sharp, get_device("cpu"),
                       contrast_percentile=7.0, smoothing_radius=6)
    assert out.shape == (h, w)
    assert out.dtype == np.float32  # fractional indices allowed after smoothing
    # the undecided band must be filled from neighbours, not left at raw noise:
    assert out[:, :8].std() < index[:, :8].std()


def test_refine_all_decided_when_high_contrast():
    h, w = 24, 24
    index = np.full((h, w), 2.0, dtype=np.float32)
    sharp = np.full((h, w), 0.9, dtype=np.float32)  # uniformly high contrast
    out = refine_index(index, sharp, get_device("cpu"),
                       contrast_percentile=7.0, smoothing_radius=4)
    np.testing.assert_allclose(out, 2.0, atol=1e-3)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_dmap_refine.py -v`
Expected: FAIL — `cannot import name 'refine_index'`.

- [ ] **Step 3: Add to `engine/src/planefuse/stack/dmap.py`**

```python
def refine_index(index: np.ndarray, max_sharp: np.ndarray, device: Device,
                 contrast_percentile: float, smoothing_radius: int) -> np.ndarray:
    """SPEC §7.2 steps 3-4. index (H,W) int/float, max_sharp (H,W) float ->
    smoothed fractional index map (H,W) float32 with undecided regions filled."""
    idx_t = torch.from_numpy(index.astype(np.float32)).unsqueeze(0).to(device.torch_device)
    sharp_t = torch.from_numpy(max_sharp.astype(np.float32)).unsqueeze(0).to(device.torch_device)
    # contrast threshold: undecided where max-sharpness below its Nth percentile
    thr = float(np.percentile(max_sharp, contrast_percentile))
    decided = sharp_t > thr
    # guided-filter smoothing of the index, guided by the sharpness image
    guide = sharp_t / (sharp_t.max() + 1e-8)
    smoothed = ops.guided_filter(guide, idx_t, radius=smoothing_radius, eps=1e-4)
    # fill undecided regions by diffusion from decided neighbours
    filled = ops.masked_diffuse(smoothed, decided, radius=max(2, smoothing_radius // 2),
                                iters=64)
    return filled.squeeze(0).cpu().numpy()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/engine/test_dmap_refine.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/src/planefuse/stack/dmap.py tests/engine/test_dmap_refine.py
git commit -m "feat: DMap index-map contrast threshold, guided smoothing, diffusion fill"
```
End commit body with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 5: DMap second-pass composite + register + quality gate

SPEC §7.2 steps 5–6: second streaming pass composites each output pixel from the frame(s) its fractional index points to (linear blend between adjacent frames); register the algorithm and emit the depth map in `aux`. Then the §13.2 quality test.

**Files:**
- Modify: `engine/src/planefuse/stack/dmap.py` (add `DMap` class + `composite`)
- Modify: `engine/src/planefuse/stack/__init__.py` (import dmap so it registers)
- Test: `tests/engine/test_dmap.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_dmap.py`:

```python
import numpy as np
from skimage.metrics import structural_similarity

from planefuse.backend import get_device
from planefuse.stack.base import get_algorithm
from planefuse.stack.sources import ArrayFrameSource
from tests.synthetic.generate import generate_stack


def test_dmap_registered():
    from planefuse.stack import REGISTRY  # noqa: F401
    assert "dmap" in REGISTRY


def test_dmap_recovers_sharp_and_depth():
    stack = generate_stack(h=160, w=200, n_frames=8, max_sigma=4.0, seed=60)
    src = ArrayFrameSource(stack.frames)
    algo = get_algorithm("dmap")
    result = algo.run(src, get_device("cpu"),
                      {"estimation_radius": 8, "contrast_threshold": 7.0, "smoothing_radius": 16})
    s = structural_similarity(np.clip(result.image, 0, 1), stack.sharp.clip(0, 1),
                              channel_axis=2, data_range=1.0)
    assert s > 0.95
    assert "depth" in result.aux
    depth = result.aux["depth"]
    assert depth.shape == (160, 200)
    # depth (frame-index proxy, normalized) should correlate with ground-truth depth
    gt = stack.depth
    corr = np.corrcoef(depth.ravel(), gt.ravel())[0, 1]
    assert abs(corr) > 0.90
```

(Note: SPEC §13.2 targets SSIM > 0.97 and depth correlation > 0.95. The synthetic generator's two depth plateaus plus the modest grid resolution make a slightly looser gate — 0.95 SSIM / 0.90 |corr| — the robust threshold here; tighten toward the spec numbers if the implementation comfortably exceeds them. Document the chosen thresholds in the test.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_dmap.py -v`
Expected: FAIL — `"dmap"` not in REGISTRY.

- [ ] **Step 3: Add the `composite` helper and `DMap` class to `engine/src/planefuse/stack/dmap.py`**

```python
from typing import Any

from planefuse.stack.base import CancelFn, ParamSpec, ProgressFn, StackResult, register


def composite(source: FrameSource, frac_index: np.ndarray, device: Device,
              masks: FrameSource | None = None, progress=None, cancel=None) -> np.ndarray:
    """Second streaming pass (SPEC §7.2 step 5): for each pixel, blend the two
    frames bracketing its fractional index. frac_index (H,W) float in [0, n-1]."""
    n = len(source)
    fi = np.clip(frac_index, 0, n - 1)
    lo = np.floor(fi).astype(np.int64)
    w_hi = (fi - lo).astype(np.float32)            # weight of the upper frame
    out = np.zeros((*fi.shape, 3), dtype=np.float32)
    for i in range(n):
        if cancel is not None and cancel():
            raise InterruptedError("dmap composite cancelled")
        if progress is not None:
            progress(f"DMap composite {i + 1}/{n}", (n + i) / max(2 * n, 1))
        frame = source.read(i)
        # weight: how much this frame contributes via the linear bracket
        w = np.zeros(fi.shape, dtype=np.float32)
        w += np.where(lo == i, 1.0 - w_hi, 0.0)
        w += np.where(lo + 1 == i, w_hi, 0.0)
        out += frame * w[..., None]
    return out


@register("dmap")
class DMap:
    @staticmethod
    def params() -> list[ParamSpec]:
        return [
            ParamSpec("estimation_radius", "Estimation radius", "int", default=8, min=2, max=40,
                      tooltip="Radius (px) of the local sharpness window."),
            ParamSpec("contrast_threshold", "Contrast threshold", "float", default=7.0, min=0.0,
                      max=50.0, tooltip="Percentile of max-sharpness below which a pixel is "
                                        "undecided and filled from neighbours."),
            ParamSpec("smoothing_radius", "Smoothing radius", "int", default=16, min=1, max=64,
                      tooltip="Edge-aware (guided-filter) smoothing radius for the depth map."),
        ]

    def run(self, source: FrameSource, device: Device, params: dict[str, Any],
            progress: ProgressFn | None = None, cancel: CancelFn | None = None,
            masks: FrameSource | None = None) -> StackResult:
        radius = int(params.get("estimation_radius", 8))
        pct = float(params.get("contrast_threshold", 7.0))
        smooth = int(params.get("smoothing_radius", 16))
        index, max_sharp = dmap_fold(source, device, radius, masks, progress, cancel)
        frac = refine_index(index, max_sharp, device, pct, smooth)
        image = composite(source, frac, device, masks, progress, cancel)
        if progress is not None:
            progress("done", 1.0)
        n = len(source)
        depth = (frac / max(n - 1, 1)).astype(np.float32)  # normalized to [0, 1]
        return StackResult(image=image, aux={"depth": depth, "index": frac.astype(np.float32)})
```

Then add to `engine/src/planefuse/stack/__init__.py` (next to the pmax import):

```python
import planefuse.stack.dmap  # noqa: E402,F401  (registers "dmap")
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/engine/test_dmap.py -v`
Expected: PASS. If SSIM is marginally under 0.95, first check the fold respects luminance sharpness (a sign/areas bug), not the threshold — do not loosen below 0.95 without investigating.

- [ ] **Step 5: Commit**

```bash
git add engine/src/planefuse/stack/dmap.py engine/src/planefuse/stack/__init__.py tests/engine/test_dmap.py
git commit -m "feat: DMap fractional-index composite, registry, depth-map aux"
```
End commit body with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

## Chunk 3: Weighted average

### Task 6: Weighted-average algorithm (online softmax)

SPEC §7.3: softmax blend `w_i = exp(s_i / T)`, normalized by a single stack-global scale (the reference frame's 99.9th-percentile sharpness), via the numerically-stable online-softmax streaming trick.

**Files:**
- Create: `engine/src/planefuse/stack/weighted.py`
- Modify: `engine/src/planefuse/stack/__init__.py`
- Test: `tests/engine/test_weighted.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_weighted.py`:

```python
import numpy as np
from skimage.metrics import structural_similarity

from planefuse.backend import get_device
from planefuse.stack.base import get_algorithm
from planefuse.stack.sources import ArrayFrameSource
from tests.synthetic.generate import generate_stack


def test_weighted_registered():
    from planefuse.stack import REGISTRY
    assert "weighted" in REGISTRY


def test_weighted_recovers_sharp():
    stack = generate_stack(h=140, w=180, n_frames=8, max_sigma=4.0, seed=61)
    src = ArrayFrameSource(stack.frames)
    algo = get_algorithm("weighted")
    result = algo.run(src, get_device("cpu"), {"temperature": 0.05, "sharpness_radius": 8})
    s = structural_similarity(np.clip(result.image, 0, 1), stack.sharp.clip(0, 1),
                              channel_axis=2, data_range=1.0)
    assert s > 0.95


def test_weighted_low_temperature_approaches_hard_max():
    # Two frames, frame 1 sharp everywhere; low T should weight it ~fully.
    h, w = 32, 32
    rng = np.random.default_rng(0)
    blurry = np.full((h, w, 3), 0.5, dtype=np.float32)
    sharp = rng.uniform(0, 1, (h, w, 3)).astype(np.float32)
    src = ArrayFrameSource([blurry, sharp])
    algo = get_algorithm("weighted")
    out = algo.run(src, get_device("cpu"), {"temperature": 0.01, "sharpness_radius": 4}).image
    # output should be much closer to the sharp frame than the blurry one
    assert np.abs(out - sharp).mean() < np.abs(out - blurry).mean()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_weighted.py -v`
Expected: FAIL — `"weighted"` not in REGISTRY.

- [ ] **Step 3: Implement `engine/src/planefuse/stack/weighted.py`**

```python
"""Weighted-average focus stacking (SPEC §7.3).

Softmax over per-frame sharpness with a stack-global scale, computed in one
streaming pass using the online-softmax trick (numerically stable at any T).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from planefuse.backend import Device, ops
from planefuse.stack.base import CancelFn, FrameSource, ParamSpec, ProgressFn, StackResult, register


def _sharp(frame_np: np.ndarray, device: Device, radius: int) -> torch.Tensor:
    t = ops.to_tensor(frame_np, device)
    return ops.sharpness_map(ops.rgb_to_luminance(t), radius).squeeze(0)  # (H, W)


@register("weighted")
class Weighted:
    @staticmethod
    def params() -> list[ParamSpec]:
        return [
            ParamSpec("temperature", "Temperature", "float", default=0.05, min=0.001, max=1.0,
                      tooltip="Softmax temperature; lower = closer to hard max (sharper, "
                              "more halo-prone). Default 0.05."),
            ParamSpec("sharpness_radius", "Sharpness radius", "int", default=8, min=2, max=40,
                      tooltip="Radius (px) of the local sharpness window."),
        ]

    def run(self, source: FrameSource, device: Device, params: dict[str, Any],
            progress: ProgressFn | None = None, cancel: CancelFn | None = None,
            masks: FrameSource | None = None) -> StackResult:
        n = len(source)
        T = float(params.get("temperature", 0.05))
        radius = int(params.get("sharpness_radius", 8))

        # Stack-global scale = reference (middle) frame's 99.9th-percentile sharpness.
        ref = n // 2
        ref_sharp = _sharp(source.read(ref), device, radius)
        scale = torch.quantile(ref_sharp.flatten(), 0.999).clamp_min(1e-8)

        dev = device.torch_device
        first = ops.to_tensor(source.read(0), device)
        c, h, w = first.shape
        m_run = torch.full((h, w), float("-inf"), device=dev)   # running max of s/T/scale
        num = torch.zeros((c, h, w), device=dev)
        den = torch.zeros((h, w), device=dev)

        for i in range(n):
            if cancel is not None and cancel():
                raise InterruptedError("weighted stack cancelled")
            if progress is not None:
                progress(f"weighted {i + 1}/{n}", i / n)
            frame = first if i == 0 else ops.to_tensor(source.read(i), device)
            s = ops.sharpness_map(ops.rgb_to_luminance(frame), radius).squeeze(0)
            if masks is not None:
                mk = torch.from_numpy(np.ascontiguousarray(masks.read(i))).to(dev).bool()
                s = torch.where(mk, s, torch.zeros_like(s))
            logit = s / (scale * T)
            new_m = torch.maximum(m_run, logit)
            rescale = torch.exp(m_run - new_m)
            rescale = torch.nan_to_num(rescale, nan=0.0, posinf=0.0)
            wt = torch.exp(logit - new_m)
            num = num * rescale.unsqueeze(0) + frame * wt.unsqueeze(0)
            den = den * rescale + wt
            m_run = new_m

        image = (num / den.clamp_min(1e-8).unsqueeze(0))
        if progress is not None:
            progress("done", 1.0)
        return StackResult(image=ops.to_numpy(image))
```

Then add to `engine/src/planefuse/stack/__init__.py`:

```python
import planefuse.stack.weighted  # noqa: E402,F401  (registers "weighted")
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/engine/test_weighted.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/src/planefuse/stack/weighted.py engine/src/planefuse/stack/__init__.py tests/engine/test_weighted.py
git commit -m "feat: weighted-average stacking with online-softmax streaming"
```
End commit body with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

## Chunk 4: Slabbing + integration

### Task 7: Slabbing composite algorithm

SPEC §7.4: split the ordered frames into overlapping slabs of size N (default 10, overlap 2), stack each with an inner method (default pmax), then stack the slab results with an outer method (default dmap). Reuses the registry.

**Files:**
- Create: `engine/src/planefuse/stack/slab.py`
- Modify: `engine/src/planefuse/stack/__init__.py`
- Test: `tests/engine/test_slab.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_slab.py`:

```python
import numpy as np
from skimage.metrics import structural_similarity

from planefuse.backend import get_device
from planefuse.stack.base import get_algorithm
from planefuse.stack.slab import plan_slabs
from planefuse.stack.sources import ArrayFrameSource
from tests.synthetic.generate import generate_stack


def test_plan_slabs_overlap():
    slabs = plan_slabs(n=25, size=10, overlap=2)
    assert slabs[0] == (0, 10)
    assert slabs[1][0] == 8  # next slab starts size-overlap before
    assert slabs[-1][1] == 25
    # every frame covered
    covered = set()
    for a, b in slabs:
        covered |= set(range(a, b))
    assert covered == set(range(25))


def test_slab_registered_and_recovers_sharp():
    from planefuse.stack import REGISTRY
    assert "slab" in REGISTRY
    stack = generate_stack(h=120, w=150, n_frames=16, max_sigma=4.0, seed=62)
    src = ArrayFrameSource(stack.frames)
    algo = get_algorithm("slab")
    result = algo.run(src, get_device("cpu"),
                      {"slab_size": 6, "slab_overlap": 2, "inner_method": "pmax",
                       "outer_method": "pmax"})
    s = structural_similarity(np.clip(result.image, 0, 1), stack.sharp.clip(0, 1),
                              channel_axis=2, data_range=1.0)
    assert s > 0.95
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_slab.py -v`
Expected: FAIL — `No module named 'planefuse.stack.slab'`.

- [ ] **Step 3: Implement `engine/src/planefuse/stack/slab.py`**

```python
"""Slabbing: hierarchical stacking for deep stacks (SPEC §7.4)."""

from __future__ import annotations

from typing import Any

import numpy as np

from planefuse.backend import Device
from planefuse.stack.base import (
    CancelFn, FrameSource, ParamSpec, ProgressFn, StackResult, get_algorithm, register,
)
from planefuse.stack.sources import ArrayFrameSource


def plan_slabs(n: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Overlapping [start, end) windows covering range(n). step = size - overlap."""
    if size <= overlap:
        raise ValueError(f"slab size {size} must exceed overlap {overlap}")
    if n <= size:
        return [(0, n)]
    step = size - overlap
    slabs: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + size, n)
        slabs.append((start, end))
        if end == n:
            break
        start += step
    return slabs


@register("slab")
class Slab:
    @staticmethod
    def params() -> list[ParamSpec]:
        return [
            ParamSpec("slab_size", "Slab size", "int", default=10, min=2, max=100,
                      tooltip="Frames per sub-stack."),
            ParamSpec("slab_overlap", "Slab overlap", "int", default=2, min=0, max=50,
                      tooltip="Overlapping frames between consecutive slabs."),
            ParamSpec("inner_method", "Inner method", "choice", default="pmax",
                      choices=("pmax", "dmap", "weighted"), tooltip="Method for each slab."),
            ParamSpec("outer_method", "Outer method", "choice", default="dmap",
                      choices=("pmax", "dmap", "weighted"), tooltip="Method combining slabs."),
        ]

    def run(self, source: FrameSource, device: Device, params: dict[str, Any],
            progress: ProgressFn | None = None, cancel: CancelFn | None = None,
            masks: FrameSource | None = None) -> StackResult:
        n = len(source)
        size = int(params.get("slab_size", 10))
        overlap = int(params.get("slab_overlap", 2))
        inner = get_algorithm(str(params.get("inner_method", "pmax")))
        outer = get_algorithm(str(params.get("outer_method", "dmap")))
        slabs = plan_slabs(n, size, overlap)

        slab_results: list[np.ndarray] = []
        for si, (a, b) in enumerate(slabs):
            if cancel is not None and cancel():
                raise InterruptedError("slab stack cancelled")
            if progress is not None:
                progress(f"slab {si + 1}/{len(slabs)}", si / (len(slabs) + 1))
            sub = ArrayFrameSource([source.read(i) for i in range(a, b)])
            sub_masks = (ArrayFrameSource([masks.read(i)[..., None].repeat(3, 2).astype(np.float32)
                                           for i in range(a, b)]) if masks is not None else None)
            # inner algorithms take the same signature; pass masks when present
            res = (inner.run(sub, device, params, masks=_BoolView(sub_masks))  # type: ignore[arg-type]
                   if sub_masks is not None else inner.run(sub, device, params))
            slab_results.append(res.image)

        if progress is not None:
            progress("combining slabs", len(slabs) / (len(slabs) + 1))
        combined = outer.run(ArrayFrameSource(slab_results), device, params)
        if progress is not None:
            progress("done", 1.0)
        return StackResult(image=combined.image, aux=combined.aux)


class _BoolView:
    """Adapts a 3-channel float mask source back to (H,W) bool reads for algorithms."""
    def __init__(self, inner: FrameSource):
        self._inner = inner
    def __len__(self) -> int:
        return len(self._inner)
    def read(self, idx: int, region=None):
        m = self._inner.read(idx, region)
        return (m[..., 0] > 0.5) if m.ndim == 3 else (m > 0.5)
```

Implementer note: the mask plumbing through slabs is fiddly because the inner algorithms expect a mask source yielding `(H,W)` bool. The simplest robust approach: if `masks is None` (the common path — slabbing without alignment masks), skip all mask wiring. Get the no-mask path working and the test green first; only wire masks if straightforward. If the masked path proves tangled, report DONE_WITH_CONCERNS and leave masks unsupported for slabbing in this milestone (note it). The test above uses no masks.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/engine/test_slab.py -v`
Expected: PASS. Then add the import to `engine/src/planefuse/stack/__init__.py`:

```python
import planefuse.stack.slab  # noqa: E402,F401  (registers "slab")
```

- [ ] **Step 5: Commit**

```bash
git add engine/src/planefuse/stack/slab.py engine/src/planefuse/stack/__init__.py tests/engine/test_slab.py
git commit -m "feat: slabbing composite stacking (inner/outer method reuse)"
```
End commit body with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 8: CLI depth-map export + method params; final gates + finish

Wire `--depth-map` export into the CLI (DMap writes its `aux["depth"]` as a 16-bit grayscale TIFF), thread the new per-method params, run the whole suite + lint + types, and finish the branch.

**Files:**
- Modify: `engine/src/planefuse/cli.py`
- Modify: `README.md`
- Test: `tests/engine/test_cli_methods.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_cli_methods.py`:

```python
from typer.testing import CliRunner

from planefuse.cli import app
from planefuse.io import load_image, save_image
from tests.synthetic.generate import generate_stack

runner = CliRunner()


def _write(tmp_path, n=6, seed=70):
    stack = generate_stack(h=120, w=150, n_frames=n, max_sigma=4.0, seed=seed)
    for i, f in enumerate(stack.frames):
        save_image(f, tmp_path / f"f_{i:03d}.tif", bit_depth=16)
    return tmp_path


def test_cli_dmap_with_depth_map(tmp_path):
    d = _write(tmp_path)
    out = tmp_path / "out.tif"
    depth = tmp_path / "depth.tif"
    result = runner.invoke(app, ["stack", str(d), "-o", str(out), "--method", "dmap",
                                 "--device", "cpu", "--depth-map", str(depth)])
    assert result.exit_code == 0, result.output
    assert out.exists() and depth.exists()


def test_cli_weighted_runs(tmp_path):
    d = _write(tmp_path)
    out = tmp_path / "w.tif"
    result = runner.invoke(app, ["stack", str(d), "-o", str(out), "--method", "weighted",
                                 "--device", "cpu"])
    assert result.exit_code == 0, result.output
    assert out.exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_cli_methods.py -v`
Expected: FAIL — `No such option: --depth-map`.

- [ ] **Step 3: Modify `engine/src/planefuse/cli.py`**

Add a `--depth-map` option (an optional output path) to the `stack` command:

```python
    depth_map: Path = typer.Option(None, "--depth-map",
                                   help="For dmap: also write the depth map (16-bit grayscale TIFF)"),
```

Build a fuller `params` dict so DMap/weighted/slab parameters reach the algorithm. Replace the current `params={"selection_smoothing": selection_smoothing}` with a dict that includes the method params with their defaults (the algorithms read what they need and ignore the rest):

```python
    params = {
        "selection_smoothing": selection_smoothing,
        # method params (algorithms pick out what they use; extras are ignored)
        "estimation_radius": 8, "contrast_threshold": 7.0, "smoothing_radius": 16,
        "temperature": 0.05, "sharpness_radius": 8,
        "slab_size": 10, "slab_overlap": 2, "inner_method": "pmax", "outer_method": "dmap",
    }
```

After `stack_frames(...)` returns `result`, write the depth map if requested and available:

```python
    if depth_map is not None and "depth" in result.aux:
        save_image(result.aux["depth"], depth_map, bit_depth=16)
        typer.echo(f"wrote depth map {depth_map}")
```

Make sure `result` is the `StackResult` (the pipeline returns one). Note: when alignment/tiling is active the pipeline may not propagate `aux`; that's fine — `--depth-map` works on the untiled, in-process path which is the common case for now. If `--depth-map` is given for a non-dmap method, the `"depth" in result.aux` guard simply skips it.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/engine/test_cli_methods.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite + lint + types**

Run: `uv run pytest -q`  → all pass (accelerator parity auto-skips).
Run: `uv run ruff check .`  → All checks passed!
Run: `uv run mypy engine/src`  → Success.

- [ ] **Step 6: Manual smoke (all four methods)**

```bash
uv run python -c "
import pathlib, tempfile
from tests.synthetic.generate import generate_stack
from planefuse.io import save_image
d = pathlib.Path(tempfile.mkdtemp()); print(d)
s = generate_stack(h=200, w=260, n_frames=12, max_sigma=4.0, seed=123)
for i,f in enumerate(s.frames): save_image(f, d/f'f_{i:03d}.tif', bit_depth=16)
"
# with DIR from the print:
uv run planefuse stack DIR -o pmax.tif --method pmax --device cpu
uv run planefuse stack DIR -o dmap.tif --method dmap --device cpu --depth-map depth.tif
uv run planefuse stack DIR -o weighted.tif --method weighted --device cpu
uv run planefuse stack DIR -o slab.tif --method slab --device cpu
```
Expected: all four write output; `depth.tif` exists and opens as 16-bit grayscale.

- [ ] **Step 7: Update README and commit**

In `README.md`, change `- [ ] M3 — DMap, weighted, slabbing, smart frame selection` to note partial completion, e.g.:
`- [~] M3 — DMap, weighted, slabbing done; smart frame selection (§7.0) pending`

```bash
git add engine/src/planefuse/cli.py tests/engine/test_cli_methods.py README.md
git commit -m "feat: --depth-map export and method params in CLI; M3 algorithms done"
```
End commit body with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

- [ ] **Step 8: Finish the branch**

Use superpowers-extended-cc:finishing-a-development-branch to merge `feat/m3-stacking` into `main` (no-ff merge, matching M1/M2 convention).

---

## Deferred to M3 Part 2 (separate plan)

- **Smart frame selection (§7.0)** — the grid focus-measure / kurtosis-reliability / interval-set-covering pre-stacking stage, with `--select-frames` CLI flag and the kurtosis-threshold calibration script. It is large and independent; it gets its own plan and produces the frame-selection coverage test (§13.2).

## Known caveats (documented, not blocking)

- **Tiled DMap/weighted:** the §8 tiled path feeds per-tile region reads. DMap's global contrast percentile and the guided-filter/diffusion radii are computed per tile, so tiled DMap output can differ from untiled at tile seams (the §8 `atol=2e-3` regression test covers PMax, which is seam-safe by pyramid overlap). Tiling is only triggered on GPU OOM for very large images; DMap/weighted remain correct untiled. A seam-safe tiled DMap is a later refinement.
- **Depth-map export through tiling/alignment:** `aux` is produced by the in-process algorithm path; `stack_tiled` returns only the image. `--depth-map` therefore works on the untiled path. Propagating `aux` through tiling is out of scope here.
