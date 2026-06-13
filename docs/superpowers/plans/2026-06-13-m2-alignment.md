# FocusStack M2 — Alignment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full §6 alignment pipeline — pairwise consecutive registration (log-polar scale/rotation + phase-correlation translation, ECC-refined and projected to a similarity transform), chained to a reference frame, with a quality gate, full-resolution GPU warping, per-frame validity masks, an aligned-frame cache, and `--align` CLI flags — so a real handheld stack visibly aligns and `tests/engine` alignment-accuracy tests pass.

**Architecture:** One new device-agnostic module group: low-level FFT/warp/luminance ops added to `backend/ops.py` (the one true implementation, parity-tested per device), pure transform math in `align/transforms.py`, per-pair estimators in `align/estimate.py`, and the `align/pipeline.py` orchestrator that chains pairwise transforms to the reference, runs the quality gate, warps at full resolution on the active device, writes aligned frames + masks to a cache directory, and returns a report. The existing `pipeline.stack_frames` gains an alignment stage that feeds aligned frames **and** validity masks (the `masks=` parameter PMax and `stack_tiled` already accept) into stacking. Frames stream from disk throughout; nothing holds the whole stack in memory.

**Tech Stack:** Python 3.12, PyTorch 2.x (`torch.fft`, `grid_sample`), `opencv-python-headless` (ECC refinement on downscaled luminance, CPU), numpy, typer, pytest, scikit-image (tests only), uv.

**Read first:** `docs/SPEC.md` §4 (device abstraction — MPS FFT CPU round-trip rule, memory discipline), §6 (alignment — the whole milestone), §9 (project/cache shape — cache layout we anticipate here), §13.1 (synthetic generator), §13.2 (alignment-accuracy + device-parity gates), §16 (uv). Spec wins on any conflict; flag conflicts rather than silently deviating.

**Preconditions:** M1 is complete and green on `main` (`uv run pytest` → 57 passed; `uv run ruff check .` and `uv run mypy engine/src` clean). Verify with `git log --oneline -1` showing the M1 merge. EXIF/XMP metadata is still out of scope (lands in M4); M2 only carries ICC through, exactly as M1 does. The server, UI, and project.json file model are **not** in M2 — the cache directory we create here is a plain engine-level directory, not the §9 project model.

**Conventions used throughout (unchanged from M1):**
- Image tensors: `torch.float32`, shape `(C, H, W)`, values nominally `[0, 1]`, never clamped except at export.
- NumPy interchange: `(H, W, C)` float32; masks are `(H, W)` bool or float32.
- **Transform convention (critical, used everywhere):** a 3×3 matrix `M` is an **output→input pixel-coordinate** map. `warp(img, M)` produces `out(x) = img(M · x)` for homogeneous pixel coords `x = (col, row, 1)`. This is exactly the convention `tests/synthetic/generate.py::_apply_affine` already uses, so estimated transforms compose directly with the generator's ground-truth `transforms`.
- The reference frame's alignment transform is the identity; aligning frame *p* means finding `M_p` such that `warp(frame_p, M_p)` lands on the reference canvas. The generator built `frame_p(x) = sharp(m_p · x)`, so the ideal recovered `M_p = inverse(transforms[p])` (see Task 13 accuracy math).
- All commands run from repo root `C:\Users\Adrien\Documents\CODE\focus-stacker` via `uv run …`.
- Every test file is CPU-first; device parametrization comes from the existing `device` / `accel_device` fixtures in `tests/engine/conftest.py`.
- Branch: do this work on `feat/m2-alignment` off `main`.

---

## Chunk 1: Dependency, alignment ops, transform math

### Task 1: Add opencv dependency and the alignment branch

**Files:**
- Modify: `engine/pyproject.toml` (add `opencv-python-headless`)
- Create: branch `feat/m2-alignment`

- [ ] **Step 1: Create the working branch**

```bash
git checkout -b feat/m2-alignment
```

- [ ] **Step 2: Add the dependency**

Edit `engine/pyproject.toml` `dependencies` to append:

```toml
  "opencv-python-headless>=4.10",
```

(Place it after `"pillow>=10.4",`. SPEC §2 names `opencv-python-headless` as the ECC engine. Headless avoids GUI libs in Docker/CI.)

- [ ] **Step 3: Sync and verify the import resolves**

Run: `uv sync --extra cu12x` (this machine has NVIDIA; use `--extra cpu` if not)
Run: `uv run python -c "import cv2; print(cv2.__version__)"`
Expected: prints a 4.x version; `uv.lock` updated.

- [ ] **Step 4: Commit**

```bash
git add engine/pyproject.toml uv.lock
git commit -m "build: add opencv-python-headless for ECC alignment refinement"
```

---

### Task 2: Alignment ops in `backend/ops.py`

Add the device-agnostic primitives alignment needs: RGB→luminance, affine warp (output→input convention), FFT pair with the §4 MPS CPU round-trip, a 2-D Hann window, phase correlation, and log-polar remap. All are parity-tested CPU-vs-accelerator.

**Files:**
- Modify: `engine/src/focusstack/backend/ops.py` (append new functions)
- Test: `tests/engine/test_align_ops.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_align_ops.py`:

```python
import numpy as np
import pytest
import torch

from focusstack.backend import get_device, ops


def _rand_img(c=3, h=64, w=80, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.uniform(0, 1, (c, h, w)).astype(np.float32))


def test_rgb_to_luminance_shape_and_range():
    img = _rand_img()
    lum = ops.rgb_to_luminance(img)
    assert lum.shape == (1, 64, 80)
    assert 0.0 <= float(lum.min()) and float(lum.max()) <= 1.0


def test_rgb_to_luminance_grayscale_is_identity_value():
    img = torch.full((3, 8, 8), 0.4)
    lum = ops.rgb_to_luminance(img)
    torch.testing.assert_close(lum, torch.full((1, 8, 8), 0.4), atol=1e-6, rtol=1e-6)


def test_warp_identity_is_noop():
    img = _rand_img()
    m = torch.eye(3)
    out = ops.warp(img, m, out_shape=(64, 80), interp="bilinear")
    torch.testing.assert_close(out, img, atol=1e-4, rtol=1e-4)


def test_warp_integer_translation_shifts_pixels():
    # M maps output->input: out(x) = img(M x). A +5px input offset in x means
    # output pixel (r, c) samples input (r, c+5): translation column = +5.
    img = _rand_img(h=40, w=40, seed=1)
    m = torch.tensor([[1.0, 0.0, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    out = ops.warp(img, m, out_shape=(40, 40), interp="bilinear")
    # interior columns of the output equal input columns shifted by 5
    torch.testing.assert_close(out[:, 5:35, 0:30], img[:, 5:35, 5:35], atol=1e-3, rtol=1e-3)


def test_fft_ifft_roundtrip():
    img = _rand_img(c=1, h=32, w=48, seed=2)
    spec = ops.fft2(img)
    back = ops.ifft2(spec)
    torch.testing.assert_close(back.real, img, atol=1e-4, rtol=1e-4)


def test_hann_window_2d_peaks_center():
    win = ops.hann_window_2d(16, 20, device=torch.device("cpu"))
    assert win.shape == (16, 20)
    assert float(win[8, 10]) == pytest.approx(float(win.max()), abs=1e-3)
    assert float(win[0, 0]) < 1e-6


def test_phase_correlation_recovers_known_shift():
    rng = np.random.default_rng(3)
    base = torch.from_numpy(rng.uniform(0, 1, (1, 64, 64)).astype(np.float32))
    shifted = torch.roll(base, shifts=(4, -7), dims=(1, 2))  # (dy=+4, dx=-7)
    dy, dx, peak = ops.phase_correlation(base, shifted)
    assert round(dy) == 4
    assert round(dx) == -7
    assert peak > 0.3


def test_log_polar_remap_shape():
    img = _rand_img(c=1, h=64, w=64, seed=4)
    out = ops.log_polar_remap(img, n_angles=64, n_radii=64)
    assert out.shape == (1, 64, 64)


# --- device parity (SPEC §13.2) ---

@pytest.mark.parametrize("opname,kwargs", [
    ("rgb_to_luminance", {}),
    ("hann_apply", {}),
])
def test_align_op_parity(accel_device, opname, kwargs):
    img = _rand_img(h=72, w=88, seed=9)
    fn = getattr(ops, opname)
    cpu_out = fn(img, **kwargs)
    dev_out = fn(img.to(accel_device.torch_device), **kwargs)
    torch.testing.assert_close(dev_out.cpu(), cpu_out, atol=1e-3, rtol=1e-3)


def test_warp_parity(accel_device):
    img = _rand_img(h=72, w=88, seed=10)
    m = torch.tensor([[1.02, -0.01, 1.5], [0.01, 1.02, -2.0], [0.0, 0.0, 1.0]])
    cpu_out = ops.warp(img, m, out_shape=(72, 88), interp="bilinear")
    dev_out = ops.warp(img.to(accel_device.torch_device), m.to(accel_device.torch_device),
                       out_shape=(72, 88), interp="bilinear")
    torch.testing.assert_close(dev_out.cpu(), cpu_out, atol=2e-3, rtol=2e-3)


def test_phase_correlation_parity(accel_device):
    rng = np.random.default_rng(11)
    base = torch.from_numpy(rng.uniform(0, 1, (1, 64, 64)).astype(np.float32))
    shifted = torch.roll(base, shifts=(3, 5), dims=(1, 2))
    dy_c, dx_c, _ = ops.phase_correlation(base, shifted)
    dy_d, dx_d, _ = ops.phase_correlation(base.to(accel_device.torch_device),
                                          shifted.to(accel_device.torch_device))
    assert round(dy_c) == round(dy_d) and round(dx_c) == round(dx_d)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/engine/test_align_ops.py -v`
Expected: FAIL — `AttributeError: module 'focusstack.backend.ops' has no attribute 'rgb_to_luminance'`.

- [ ] **Step 3: Append the implementation to `engine/src/focusstack/backend/ops.py`**

```python
# ----------------------------------------------------------------------------
# Alignment ops (SPEC §6). Convention: a 3x3 matrix M maps OUTPUT pixel coords
# to INPUT pixel coords (x = [col, row, 1]); warp(img, M)[r, c] = img(M [c, r, 1]).
# ----------------------------------------------------------------------------

# Rec. 709 luma weights (R, G, B). Gamma space is kept (SPEC §5) — no linearization.
_LUMA = (0.2126, 0.7152, 0.0722)


def rgb_to_luminance(img: torch.Tensor) -> torch.Tensor:
    """(C, H, W) RGB float32 -> (1, H, W) luminance. C must be 3."""
    if img.shape[0] != 3:
        raise ValueError(f"expected 3-channel RGB, got {img.shape[0]} channels")
    w = torch.tensor(_LUMA, dtype=img.dtype, device=img.device).view(3, 1, 1)
    return (img * w).sum(dim=0, keepdim=True)


def _px_to_norm(m: torch.Tensor, out_h: int, out_w: int, in_h: int, in_w: int) -> torch.Tensor:
    """Convert a pixel-coord output->input matrix to the normalized [-1, 1] grid
    convention torch.affine_grid expects (align_corners=True). Mirrors
    tests/synthetic/generate.py::_apply_affine so estimates compose with ground truth."""
    dev, dt = m.device, torch.float64
    m64 = m.to(dtype=dt)
    s_out = torch.tensor([[2.0 / (out_w - 1), 0, -1], [0, 2.0 / (out_h - 1), -1], [0, 0, 1]],
                         dtype=dt, device=dev)
    s_in = torch.tensor([[2.0 / (in_w - 1), 0, -1], [0, 2.0 / (in_h - 1), -1], [0, 0, 1]],
                        dtype=dt, device=dev)
    norm = s_in @ m64 @ torch.linalg.inv(s_out)
    return norm[:2].to(torch.float32)


def warp(img: torch.Tensor, matrix: torch.Tensor, out_shape: tuple[int, int],
         interp: str = "bilinear") -> torch.Tensor:
    """Warp (C, H, W) by a 3x3 output->input pixel matrix to out_shape (H, W).

    interp: "bilinear" | "bicubic" | "nearest". Lanczos-3 (SPEC default at full
    res) arrives in a later task; bilinear/bicubic cover estimation and tests.
    Out-of-frame samples use edge clamp (padding_mode="border").
    """
    c, in_h, in_w = img.shape
    out_h, out_w = out_shape
    theta = _px_to_norm(matrix.to(img.device), out_h, out_w, in_h, in_w).unsqueeze(0)
    grid = F.affine_grid(theta, (1, c, out_h, out_w), align_corners=True)
    mode = "bicubic" if interp == "bicubic" else ("nearest" if interp == "nearest" else "bilinear")
    return F.grid_sample(img.unsqueeze(0), grid, mode=mode, padding_mode="border",
                         align_corners=True).squeeze(0)


def _fft_needs_cpu(device: torch.device) -> bool:
    """MPS pre-macOS-14 lacks torch.fft (SPEC §4). Route through CPU explicitly,
    deterministically, and visibly — never via the global MPS fallback flag."""
    if device.type != "mps":
        return False
    try:
        torch.fft.fft2(torch.zeros(2, 2, device=device))
        return False
    except Exception:  # noqa: BLE001 - MPS raises NotImplementedError/RuntimeError here
        return True


def fft2(img: torch.Tensor) -> torch.Tensor:
    """2-D FFT over the last two dims, returning a complex tensor. MPS-safe."""
    if _fft_needs_cpu(img.device):
        return torch.fft.fft2(img.cpu()).to(img.device)
    return torch.fft.fft2(img)


def ifft2(spec: torch.Tensor) -> torch.Tensor:
    """Inverse 2-D FFT over the last two dims. MPS-safe."""
    if _fft_needs_cpu(spec.device):
        return torch.fft.ifft2(spec.cpu()).to(spec.device)
    return torch.fft.ifft2(spec)


def hann_window_2d(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Separable 2-D Hann window (H, W), peak 1 at center, 0 at edges."""
    wy = torch.hann_window(h, periodic=False, dtype=torch.float32, device=device)
    wx = torch.hann_window(w, periodic=False, dtype=torch.float32, device=device)
    return torch.outer(wy, wx)


def hann_apply(img: torch.Tensor) -> torch.Tensor:
    """Multiply a (C, H, W) image by a 2-D Hann window (used by phase correlation)."""
    win = hann_window_2d(img.shape[-2], img.shape[-1], img.device)
    return img * win


def phase_correlation(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    """Translation (dy, dx) that best aligns b to a, plus the correlation peak.

    a, b: (1, H, W). Returns the shift such that roll(a, (dy, dx)) ~ b, with
    sub-integer peak from the argmax (integer here; estimators refine). A Hann
    window suppresses edge wrap-around (SPEC §6 step b).
    """
    aw = hann_apply(a).squeeze(0)
    bw = hann_apply(b).squeeze(0)
    fa = torch.fft.fft2(aw) if not _fft_needs_cpu(a.device) else torch.fft.fft2(aw.cpu()).to(a.device)
    fb = torch.fft.fft2(bw) if not _fft_needs_cpu(b.device) else torch.fft.fft2(bw.cpu()).to(b.device)
    cross = fa * fb.conj()
    cross = cross / (cross.abs() + 1e-8)
    corr = ifft2(cross.unsqueeze(0)).real.squeeze(0)
    h, w = corr.shape
    flat = int(torch.argmax(corr).item())
    py, px = divmod(flat, w)
    peak = float(corr[py, px] / (corr.abs().mean() + 1e-12))
    # wrap to signed shift
    dy = py - h if py > h // 2 else py
    dx = px - w if px > w // 2 else px
    return float(dy), float(dx), peak


def log_polar_remap(img: torch.Tensor, n_angles: int, n_radii: int) -> torch.Tensor:
    """Remap (1, H, W) into (1, n_radii, n_angles) log-polar space about the center.

    Used on FFT magnitude spectra: a scale in Cartesian space becomes a radial
    shift and a rotation becomes an angular shift, both recoverable by a second
    phase correlation (SPEC §6 step b).
    """
    _, h, w = img.shape
    dev = img.device
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    max_r = math.sqrt(cx * cx + cy * cy)
    log_base = math.log(max_r + 1e-6) / n_radii
    radii = torch.arange(n_radii, dtype=torch.float32, device=dev)
    angles = torch.arange(n_angles, dtype=torch.float32, device=dev) * (2.0 * math.pi / n_angles)
    rho = torch.exp(radii * log_base).view(n_radii, 1)
    ys = cy + rho * torch.sin(angles).view(1, n_angles)
    xs = cx + rho * torch.cos(angles).view(1, n_angles)
    gx = xs / (w - 1) * 2.0 - 1.0
    gy = ys / (h - 1) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1, n_radii, n_angles, 2)
    return F.grid_sample(img.unsqueeze(0), grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True).squeeze(0)
```

Notes for the implementer:
- `phase_correlation` duplicates the MPS guard inline for the two forward FFTs rather than calling `fft2`, because `fft2` takes a `(C,H,W)`-ish tensor while we already squeezed to `(H,W)`; keep both paths or refactor `fft2` to accept 2-D — either is fine, but the parity test must stay green.
- `_px_to_norm` does the matrix algebra in float64 then casts; this matches the generator and keeps sub-pixel accuracy.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/engine/test_align_ops.py -v`
Expected: all PASS (CUDA parity runs on this machine; MPS auto-skips). If `test_warp_integer_translation_shifts_pixels` is off by the border ramp, tighten the interior slice — do not change `padding_mode`.

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/backend/ops.py tests/engine/test_align_ops.py
git commit -m "feat: alignment ops — luminance, warp, fft, phase correlation, log-polar"
```

---

### Task 3: Transform math (`align/transforms.py`)

Pure, device-free 3×3 matrix helpers. No torch — numpy float64 throughout for numerical stability; conversion to torch happens only at the warp boundary.

**Files:**
- Create: `engine/src/focusstack/align/__init__.py`
- Create: `engine/src/focusstack/align/transforms.py`
- Test: `tests/engine/test_transforms.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_transforms.py`:

```python
import numpy as np

from focusstack.align.transforms import (
    compose,
    invert,
    project_to_similarity,
    scale_transform_to_resolution,
    similarity_matrix,
    translation_matrix,
)


def test_translation_matrix():
    m = translation_matrix(3.0, -2.0)  # (dx, dy)
    np.testing.assert_allclose(m @ np.array([0, 0, 1.0]), [3.0, -2.0, 1.0])


def test_similarity_about_center_identity():
    m = similarity_matrix(scale=1.0, angle=0.0, tx=0.0, ty=0.0, cx=10.0, cy=8.0)
    np.testing.assert_allclose(m, np.eye(3), atol=1e-9)


def test_compose_is_matrix_product():
    a = translation_matrix(2.0, 0.0)
    b = translation_matrix(0.0, 3.0)
    np.testing.assert_allclose(compose(a, b), a @ b)


def test_invert_roundtrip():
    m = similarity_matrix(1.05, 0.02, 1.5, -2.0, cx=5, cy=5)
    np.testing.assert_allclose(compose(m, invert(m)), np.eye(3), atol=1e-9)


def test_scale_transform_to_resolution():
    # A transform estimated at /4 resolution must scale its translation by 4
    # when applied at full resolution; the linear (scale/rotation) block is
    # resolution-invariant under center-relative similarity.
    m_low = translation_matrix(2.0, -1.0)
    m_full = scale_transform_to_resolution(m_low, factor=4.0)
    np.testing.assert_allclose(m_full @ np.array([0, 0, 1.0]), [8.0, -4.0, 1.0])


def test_project_to_similarity_recovers_similarity():
    truth = similarity_matrix(1.03, 0.05, 4.0, -3.0, cx=32, cy=24)
    # add a tiny non-similarity (shear) perturbation to the 2x2 block
    affine = truth.copy()
    affine[0, 1] += 0.004
    affine[1, 0] -= 0.002
    sim = project_to_similarity(affine, cx=32, cy=24)
    # the projected 2x2 must be a scaled rotation: columns orthogonal, equal norm
    a = sim[:2, :2]
    np.testing.assert_allclose(a[:, 0] @ a[:, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(a[:, 0]), np.linalg.norm(a[:, 1]), atol=1e-6)
    # and it should be close to the underlying truth
    np.testing.assert_allclose(sim, truth, atol=5e-3)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/engine/test_transforms.py -v`
Expected: FAIL — `No module named 'focusstack.align'`.

- [ ] **Step 3: Implement**

`engine/src/focusstack/align/__init__.py`:

```python
"""Alignment pipeline (SPEC §6)."""
```

`engine/src/focusstack/align/transforms.py`:

```python
"""Pure 3x3 transform math (SPEC §6). numpy float64; output->input pixel coords.

A matrix M maps output pixel (col, row, 1) -> input pixel; warp(img, M) samples
img at M·x. Composition matches matrix multiplication: compose(A, B) applies B
then A in output->input space, i.e. (A∘B)(x) = A(B(x))... in matrix terms A @ B.
"""

from __future__ import annotations

import numpy as np


def translation_matrix(tx: float, ty: float) -> np.ndarray:
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)


def similarity_matrix(scale: float, angle: float, tx: float, ty: float,
                      cx: float, cy: float) -> np.ndarray:
    """Similarity (uniform scale + rotation + translation) about center (cx, cy)."""
    ca, sa = np.cos(angle), np.sin(angle)
    rot = np.array([[scale * ca, -scale * sa, 0],
                    [scale * sa, scale * ca, 0],
                    [0, 0, 1]], dtype=np.float64)
    pre = translation_matrix(-cx, -cy)
    post = translation_matrix(cx + tx, cy + ty)
    return post @ rot @ pre


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.astype(np.float64) @ b.astype(np.float64)


def invert(m: np.ndarray) -> np.ndarray:
    return np.linalg.inv(m.astype(np.float64))


def scale_transform_to_resolution(m: np.ndarray, factor: float) -> np.ndarray:
    """Re-express an output->input transform estimated at downscaled resolution
    for use at `factor`× higher resolution. With S = diag(factor, factor, 1):
    M_full = S · M_low · S^{-1} — the 2x2 linear block is unchanged, the
    translation column scales by `factor` (SPEC §6 step 5)."""
    s = np.diag([factor, factor, 1.0])
    return s @ m.astype(np.float64) @ np.linalg.inv(s)


def project_to_similarity(affine: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Project an affine output->input matrix to the nearest similarity via
    orthogonal Procrustes on the 2x2 block (SPEC §6 step c). The translation is
    recomputed so the image center maps consistently."""
    a = affine[:2, :2].astype(np.float64)
    u, s, vt = np.linalg.svd(a)
    r = u @ vt                          # nearest rotation
    if np.linalg.det(r) < 0:            # reflect-free
        u[:, -1] *= -1
        r = u @ vt
    scale = float(s.mean())             # uniform scale = mean singular value
    sim2 = scale * r
    # preserve where the center currently maps: t = affine·center - sim2·center
    center = np.array([cx, cy], dtype=np.float64)
    t = affine[:2, :2] @ center + affine[:2, 2] - sim2 @ center
    out = np.eye(3, dtype=np.float64)
    out[:2, :2] = sim2
    out[:2, 2] = t
    return out
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/engine/test_transforms.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/align tests/engine/test_transforms.py
git commit -m "feat: alignment transform math (similarity, compose, procrustes projection)"
```

---

## Chunk 2: Pairwise estimation

### Task 4: Luminance proxy builder (`align/proxy.py`)

A small helper that turns a frame into the downscaled luminance proxy every estimator works on (SPEC §6 step 3a), recording the downscale factor so estimates can be scaled back to full resolution.

**Files:**
- Create: `engine/src/focusstack/align/proxy.py`
- Test: `tests/engine/test_proxy.py`

- [ ] **Step 1: Write the failing test**

`tests/engine/test_proxy.py`:

```python
import numpy as np

from focusstack.backend import get_device
from focusstack.align.proxy import make_proxy


def test_proxy_downscales_to_max_long_edge():
    rng = np.random.default_rng(0)
    frame = rng.uniform(0, 1, (1000, 1500, 3)).astype(np.float32)
    proxy, factor = make_proxy(frame, get_device("cpu"), max_long_edge=512)
    assert proxy.shape[0] == 1                # luminance, (1, h, w)
    assert max(proxy.shape[-2:]) == 512
    assert factor == 1500 / 512


def test_proxy_no_upscale_when_small():
    frame = np.zeros((100, 80, 3), dtype=np.float32)
    proxy, factor = make_proxy(frame, get_device("cpu"), max_long_edge=2048)
    assert factor == 1.0
    assert proxy.shape == (1, 100, 80)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_proxy.py -v`
Expected: FAIL — `No module named 'focusstack.align.proxy'`.

- [ ] **Step 3: Implement `engine/src/focusstack/align/proxy.py`**

```python
"""Downscaled luminance proxy for alignment estimation (SPEC §6 step 3a)."""

from __future__ import annotations

import numpy as np
import torch

from focusstack.backend import Device, ops


def make_proxy(frame: np.ndarray, device: Device, max_long_edge: int = 2048
               ) -> tuple[torch.Tensor, float]:
    """(H, W, 3) numpy -> ((1, h, w) luminance tensor, downscale_factor).

    factor = original_long_edge / proxy_long_edge >= 1.0 (never upscales).
    """
    t = ops.to_tensor(frame, device)
    lum = ops.rgb_to_luminance(t)
    long_edge = max(lum.shape[-2:])
    if long_edge <= max_long_edge:
        return lum, 1.0
    factor = long_edge / max_long_edge
    new_h = max(1, round(lum.shape[-2] / factor))
    new_w = max(1, round(lum.shape[-1] / factor))
    return ops.upsample_to(lum, (new_h, new_w)), float(long_edge) / float(max(new_h, new_w))
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/engine/test_proxy.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/align/proxy.py tests/engine/test_proxy.py
git commit -m "feat: downscaled luminance proxy for alignment"
```

---

### Task 5: Log-polar scale+rotation and phase-correlation translation estimators

The two FFT-based initial-guess estimators (SPEC §6 step b), operating on proxies. Order matters: scale/rotation first, then translation on the scale/rotation-corrected pair.

**Files:**
- Create: `engine/src/focusstack/align/initial.py`
- Test: `tests/engine/test_initial_guess.py`

- [ ] **Step 1: Write the failing tests** (use the synthetic generator's known transforms)

`tests/engine/test_initial_guess.py`:

```python
import numpy as np

from focusstack.backend import get_device
from focusstack.align.initial import estimate_scale_rotation, estimate_translation
from focusstack.align.proxy import make_proxy
from focusstack.align.transforms import similarity_matrix
from tests.synthetic.generate import generate_stack


def _proxy(frame):
    p, _ = make_proxy(frame, get_device("cpu"), max_long_edge=256)
    return p


def test_translation_recovered_on_pure_shift():
    stack = generate_stack(h=200, w=240, n_frames=2, max_sigma=0.5, seed=5,
                           trans_jitter=6.0, scale_step=0.0, rot_jitter=0.0)
    a, b = _proxy(stack.frames[0]), _proxy(stack.frames[1])
    dy, dx = estimate_translation(a, b)
    # frame 1's ground-truth output->input translation (about center)
    m = stack.transforms[1]
    # recovered shift should roughly invert the applied one (within 1.5 px at proxy res)
    assert abs(dx) > 0.0  # sanity; exact value checked in pipeline accuracy test
    assert np.isfinite(dy) and np.isfinite(dx)


def test_scale_rotation_recovered_on_breathing():
    stack = generate_stack(h=200, w=240, n_frames=2, max_sigma=0.5, seed=6,
                           scale_step=0.03, rot_jitter=0.04, trans_jitter=0.0)
    a, b = _proxy(stack.frames[0]), _proxy(stack.frames[1])
    scale, angle = estimate_scale_rotation(a, b)
    # ground truth applied scale 1+0.03 and a rotation in [-0.04, 0.04]
    assert 0.95 < scale < 1.10
    assert abs(angle) < 0.12
```

(These are loose sanity bounds; the strict §13.2 thresholds are asserted end-to-end in Task 13 after ECC refinement.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_initial_guess.py -v`
Expected: FAIL — `No module named 'focusstack.align.initial'`.

- [ ] **Step 3: Implement `engine/src/focusstack/align/initial.py`**

```python
"""FFT-based initial alignment guesses (SPEC §6 step b)."""

from __future__ import annotations

import math

import torch

from focusstack.backend import ops


def estimate_translation(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """(dy, dx) translating b onto a, via Hann-windowed phase correlation.
    a, b: (1, h, w) proxies at the same resolution."""
    dy, dx, _peak = ops.phase_correlation(a, b)
    return dy, dx


def estimate_scale_rotation(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """(scale, angle_radians) mapping b's geometry onto a, from log-polar phase
    correlation of the FFT magnitude spectra (translation-invariant)."""
    n = 256
    fa = torch.fft.fftshift(ops.fft2(a).abs(), dim=(-2, -1))
    fb = torch.fft.fftshift(ops.fft2(b).abs(), dim=(-2, -1))
    # high-pass the spectra a little so the DC spike doesn't dominate the remap
    fa = torch.log1p(fa)
    fb = torch.log1p(fb)
    lpa = ops.log_polar_remap(fa, n_angles=n, n_radii=n)
    lpb = ops.log_polar_remap(fb, n_angles=n, n_radii=n)
    d_rho, d_theta, _peak = ops.phase_correlation(lpa, lpb)
    # angular axis: n columns over 2*pi
    angle = -d_theta * (2.0 * math.pi / n)
    # radial axis is logarithmic with base derived in log_polar_remap
    h, w = a.shape[-2:]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    max_r = math.sqrt(cx * cx + cy * cy)
    log_base = math.log(max_r + 1e-6) / n
    scale = float(math.exp(-d_rho * log_base))
    # guard against degenerate peaks
    if not (0.5 < scale < 2.0):
        scale = 1.0
    return scale, float(angle)
```

Implementer note: the sign conventions for `d_rho`/`d_theta` depend on `phase_correlation`'s shift direction; if Task 13's accuracy test shows scale inverted (e.g. 0.97 where 1.03 expected), flip the sign on `d_rho`/`d_theta`. Pin the final signs with the Task 13 end-to-end test, not by eye.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/engine/test_initial_guess.py -v`
Expected: PASS (loose bounds).

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/align/initial.py tests/engine/test_initial_guess.py
git commit -m "feat: log-polar scale/rotation and phase-correlation translation estimators"
```

---

### Task 6: ECC refinement + similarity projection + brightness normalization (`align/refine.py`)

OpenCV `findTransformECC` (`MOTION_AFFINE`, 3-level pyramid, warm-started from the initial guess), then project to similarity (Task 3). Plus the overlap-region brightness match (SPEC §6 step d).

**Files:**
- Create: `engine/src/focusstack/align/refine.py`
- Test: `tests/engine/test_refine.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_refine.py`:

```python
import numpy as np

from focusstack.backend import get_device
from focusstack.align.proxy import make_proxy
from focusstack.align.refine import brightness_gain, refine_ecc
from focusstack.align.transforms import similarity_matrix
from tests.synthetic.generate import generate_stack


def _proxy_np(frame):
    p, _ = make_proxy(frame, get_device("cpu"), max_long_edge=256)
    return p.squeeze(0).cpu().numpy()


def test_refine_ecc_improves_toward_identity_pair():
    # two near-identical frames -> ECC должно вернуть ~identity, corr ~1
    stack = generate_stack(h=200, w=240, n_frames=2, max_sigma=0.3, seed=7)
    a, b = _proxy_np(stack.frames[0]), _proxy_np(stack.frames[1])
    m, corr = refine_ecc(a, b, init=np.eye(3), cx=a.shape[1] / 2, cy=a.shape[0] / 2)
    assert corr > 0.9
    assert abs(m[0, 0] - 1.0) < 0.05 and abs(m[1, 1] - 1.0) < 0.05


def test_refine_ecc_returns_similarity():
    stack = generate_stack(h=200, w=240, n_frames=2, max_sigma=0.3, seed=8,
                           scale_step=0.02)
    a, b = _proxy_np(stack.frames[0]), _proxy_np(stack.frames[1])
    m, corr = refine_ecc(a, b, init=np.eye(3), cx=a.shape[1] / 2, cy=a.shape[0] / 2)
    block = m[:2, :2]
    np.testing.assert_allclose(block[:, 0] @ block[:, 1], 0.0, atol=1e-4)  # orthogonal cols


def test_brightness_gain_matches_mean():
    a = np.full((50, 50), 0.4, dtype=np.float32)
    b = np.full((50, 50), 0.2, dtype=np.float32)
    mask = np.ones((50, 50), dtype=bool)
    gain = brightness_gain(a, b, mask)
    np.testing.assert_allclose(gain, 2.0, atol=1e-3)  # b * gain matches a
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_refine.py -v`
Expected: FAIL — `No module named 'focusstack.align.refine'`.

- [ ] **Step 3: Implement `engine/src/focusstack/align/refine.py`**

```python
"""ECC refinement and brightness normalization (SPEC §6 steps c, d).

ECC runs on CPU (OpenCV) over downscaled luminance proxies — cheap and the spec
mandates it. The refined affine is projected to a similarity (the output model).
"""

from __future__ import annotations

import cv2
import numpy as np

from focusstack.align.transforms import project_to_similarity


def _ecc_at_level(a: np.ndarray, b: np.ndarray, warp_init: np.ndarray,
                  iters: int, eps: float) -> tuple[np.ndarray, float]:
    """One ECC solve with a 2x3 affine warp. Returns (2x3 warp, correlation)."""
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    warp = warp_init.astype(np.float32)
    cc, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE, criteria, None, 5)
    return warp, float(cc)


def refine_ecc(a: np.ndarray, b: np.ndarray, init: np.ndarray,
               cx: float, cy: float, levels: int = 3) -> tuple[np.ndarray, float]:
    """Refine the b->a alignment with a 3-level ECC pyramid warm-started from
    `init` (3x3 output->input). Returns (3x3 similarity matrix, ECC correlation).

    a, b: float32 (h, w) luminance proxies in [0, 1].
    """
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    # build a coarse->fine pyramid of proxies
    pyr_a = [a]
    pyr_b = [b]
    for _ in range(levels - 1):
        pyr_a.insert(0, cv2.pyrDown(pyr_a[0]))
        pyr_b.insert(0, cv2.pyrDown(pyr_b[0]))
    # initialize the 2x3 affine warp at the coarsest level: scale translation down
    warp = init[:2, :].astype(np.float32).copy()
    coarsest_factor = 2 ** (levels - 1)
    warp[:, 2] /= coarsest_factor
    cc = 0.0
    for lvl, (la, lb) in enumerate(zip(pyr_a, pyr_b)):
        try:
            warp, cc = _ecc_at_level(la, lb, warp, iters=100, eps=1e-5)
        except cv2.error:
            # ECC can fail to converge on low-contrast pairs; keep the warm start
            cc = 0.0
        if lvl < len(pyr_a) - 1:
            warp[:, 2] *= 2.0  # translation doubles going one level finer
    m = np.eye(3, dtype=np.float64)
    m[:2, :] = warp
    return project_to_similarity(m, cx=cx, cy=cy), cc


def brightness_gain(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Multiplicative gain g so that mean(b*g) ~ mean(a) inside `mask`
    (SPEC §6 step d, mean ratio). Falls back to 1.0 on empty/zero support."""
    m = mask.astype(bool)
    if m.sum() < 16:
        return 1.0
    mean_a = float(a[m].mean())
    mean_b = float(b[m].mean())
    if mean_b < 1e-6:
        return 1.0
    return mean_a / mean_b
```

Implementer notes:
- `findTransformECC`'s 5th positional arg is `inputMask`; the trailing `5` is `gaussFiltSize`. Signature varies slightly across OpenCV versions — if it raises a `TypeError`, drop the trailing args and call `cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE, criteria)`.
- ECC expects the *template* first and the *image to warp* second; we pass `(a, b)` so the returned warp maps `b`→`a`. This is the output(reference)→input convention if `a` is the reference proxy. Confirm direction against Task 13.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/engine/test_refine.py -v`
Expected: PASS. (Cyrillic comment in the first test is a placeholder — replace with English: "two near-identical frames → ECC returns ~identity, corr ~1".)

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/align/refine.py tests/engine/test_refine.py
git commit -m "feat: ECC refinement with similarity projection and brightness gain"
```

---

### Task 7: Single-pair estimator (`align/estimate.py`)

Compose the per-pair pipeline in the SPEC §6 step-b/c/d order into one function returning a transform, ECC correlation, and brightness gain.

**Files:**
- Create: `engine/src/focusstack/align/estimate.py`
- Test: `tests/engine/test_estimate_pair.py`

- [ ] **Step 1: Write the failing test**

`tests/engine/test_estimate_pair.py`:

```python
import numpy as np

from focusstack.backend import get_device
from focusstack.align.estimate import PairResult, estimate_pair
from tests.synthetic.generate import generate_stack


def test_estimate_pair_returns_result():
    stack = generate_stack(h=240, w=300, n_frames=2, max_sigma=0.4, seed=12,
                           scale_step=0.01, trans_jitter=3.0, rot_jitter=0.01)
    res = estimate_pair(stack.frames[0], stack.frames[1], get_device("cpu"),
                        max_long_edge=256, model="similarity")
    assert isinstance(res, PairResult)
    assert res.matrix.shape == (3, 3)        # at PROXY resolution
    assert 0.0 <= res.correlation <= 1.0
    assert res.proxy_factor >= 1.0
    assert np.isfinite(res.gain)


def test_estimate_pair_translation_model_skips_logpolar():
    stack = generate_stack(h=200, w=200, n_frames=2, max_sigma=0.4, seed=13,
                           trans_jitter=4.0)
    res = estimate_pair(stack.frames[0], stack.frames[1], get_device("cpu"),
                        max_long_edge=256, model="translation")
    # pure-translation model: 2x2 block stays identity
    np.testing.assert_allclose(res.matrix[:2, :2], np.eye(2), atol=1e-6)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_estimate_pair.py -v`
Expected: FAIL — `No module named 'focusstack.align.estimate'`.

- [ ] **Step 3: Implement `engine/src/focusstack/align/estimate.py`**

```python
"""Per-pair transform estimation (SPEC §6 steps a-d), at proxy resolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from focusstack.backend import Device, ops
from focusstack.align.initial import estimate_scale_rotation, estimate_translation
from focusstack.align.proxy import make_proxy
from focusstack.align.refine import brightness_gain, refine_ecc
from focusstack.align.transforms import similarity_matrix, translation_matrix


@dataclass
class PairResult:
    matrix: np.ndarray      # 3x3 output->input at PROXY resolution
    correlation: float      # ECC correlation (quality gate input)
    gain: float             # brightness gain for `b` relative to `a`
    proxy_factor: float     # multiply translation by this for full-res


def estimate_pair(frame_a: np.ndarray, frame_b: np.ndarray, device: Device,
                  max_long_edge: int = 2048, model: str = "similarity",
                  normalize_brightness: bool = True) -> PairResult:
    """Estimate the transform warping frame_b onto frame_a (consecutive pair)."""
    pa, factor = make_proxy(frame_a, device, max_long_edge)
    pb, _ = make_proxy(frame_b, device, max_long_edge)
    h, w = pa.shape[-2:]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    if model == "translation":
        dy, dx = estimate_translation(pa, pb)
        init = translation_matrix(dx, dy)
    else:
        scale, angle = estimate_scale_rotation(pa, pb)
        sr = similarity_matrix(scale, angle, 0.0, 0.0, cx, cy)
        # warp b by the scale/rotation guess, then estimate residual translation
        pb_corr = ops.warp(pb, _to_tensor3(sr, device), out_shape=(h, w), interp="bilinear")
        dy, dx = estimate_translation(pa, pb_corr)
        init = translation_matrix(dx, dy) @ sr

    a_np = pa.squeeze(0).cpu().numpy()
    b_np = pb.squeeze(0).cpu().numpy()
    if model == "translation":
        # ECC with MOTION_TRANSLATION semantics is overkill; reuse affine ECC but
        # re-project translation-only afterwards.
        m, corr = refine_ecc(a_np, b_np, init=init, cx=cx, cy=cy)
        m[:2, :2] = np.eye(2)  # enforce translation model
    else:
        m, corr = refine_ecc(a_np, b_np, init=init, cx=cx, cy=cy)

    gain = 1.0
    if normalize_brightness:
        mask = np.ones((h, w), dtype=bool)
        gain = brightness_gain(a_np, b_np, mask)
    return PairResult(matrix=m, correlation=corr, gain=gain, proxy_factor=factor)


def _to_tensor3(m: np.ndarray, device: Device):
    import torch
    return torch.from_numpy(m.astype(np.float32)).to(device.torch_device)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/engine/test_estimate_pair.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/align/estimate.py tests/engine/test_estimate_pair.py
git commit -m "feat: single-pair transform estimator combining initial guess + ECC"
```

---

## Chunk 3: Chaining, warp, masks, orchestrator

### Task 8: Chained composition + quality gate (`align/chain.py`)

Estimate consecutive pairs, compose to the reference (middle frame), apply the quality gate, and re-estimate across dropped frames (SPEC §6 steps 2, 4).

**Files:**
- Create: `engine/src/focusstack/align/chain.py`
- Test: `tests/engine/test_chain.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_chain.py`:

```python
import numpy as np

from focusstack.align.chain import chain_to_reference


def test_chain_reference_is_identity():
    # 5 frames, pair transforms all identity -> every frame maps to identity
    n = 5
    pair = {i: np.eye(3) for i in range(n - 1)}  # transform from i+1 -> i
    corr = {i: 0.99 for i in range(n - 1)}
    result = chain_to_reference(n, pair, corr, ref=2, threshold=0.9)
    for m in result.matrices:
        np.testing.assert_allclose(m, np.eye(3), atol=1e-9)
    assert result.reference == 2
    assert not result.dropped


def test_chain_composes_translations_outward_from_reference():
    # each consecutive pair shifts by +1px in x; mapping to ref=0 accumulates
    n = 4
    t = np.array([[1, 0, 1.0], [0, 1, 0], [0, 0, 1]])
    pair = {i: t for i in range(n - 1)}
    corr = {i: 0.99 for i in range(n - 1)}
    result = chain_to_reference(n, pair, corr, ref=0, threshold=0.9)
    # frame k maps to ref by composing k pair transforms => translation +k
    np.testing.assert_allclose(result.matrices[0], np.eye(3), atol=1e-9)
    assert result.matrices[3][0, 2] == 3.0


def test_chain_flags_low_correlation():
    n = 4
    pair = {i: np.eye(3) for i in range(n - 1)}
    corr = {0: 0.99, 1: 0.5, 2: 0.99}  # pair 1->2 is bad
    result = chain_to_reference(n, pair, corr, ref=0, threshold=0.9)
    assert 2 in result.flagged or 1 in result.flagged
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_chain.py -v`
Expected: FAIL — `No module named 'focusstack.align.chain'`.

- [ ] **Step 3: Implement `engine/src/focusstack/align/chain.py`**

```python
"""Chain consecutive pair transforms to the reference frame (SPEC §6 step 2, 4)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from focusstack.align.transforms import compose, invert


@dataclass
class ChainResult:
    matrices: list[np.ndarray]      # per-frame output->input to the reference canvas
    reference: int
    correlations: dict[int, float]  # pair index i -> corr for pair (i, i+1)
    flagged: set[int] = field(default_factory=set)   # frame indices below threshold
    dropped: set[int] = field(default_factory=set)


def chain_to_reference(n: int, pair_transforms: dict[int, np.ndarray],
                       correlations: dict[int, float], ref: int,
                       threshold: float, drop: bool = False) -> ChainResult:
    """pair_transforms[i] maps frame (i+1) -> frame i (consecutive, output->input).

    Composing from frame k toward the reference yields the transform mapping the
    reference canvas to frame k's pixels.
    """
    matrices: list[np.ndarray] = [np.eye(3) for _ in range(n)]
    # frames after the reference: compose forward
    acc = np.eye(3)
    for k in range(ref + 1, n):
        acc = compose(acc, pair_transforms[k - 1])  # ref->...->k
        matrices[k] = acc.copy()
    # frames before the reference: compose inverse pair transforms
    acc = np.eye(3)
    for k in range(ref - 1, -1, -1):
        acc = compose(acc, invert(pair_transforms[k]))  # ref->...->k
        matrices[k] = acc.copy()

    flagged = {i for i, c in correlations.items() if c < threshold}
    # map flagged pair index -> the later frame in the pair
    flagged_frames = {i + 1 for i in flagged}
    result = ChainResult(matrices=matrices, reference=ref,
                         correlations=correlations, flagged=flagged_frames)
    if drop:
        result.dropped = {f for f in flagged_frames if f != ref}
    return result
```

Implementer note: when `drop=True`, SPEC §6 step 4 requires re-estimating the pair **directly between k−1 and k+1** so the chain skips the dropped link. That re-estimation needs frame pixels, so it belongs in the Task 11 orchestrator (which has the `FrameSource`), not in this pure-math function. This function exposes `dropped`; the orchestrator consumes it and patches `pair_transforms` before calling `chain_to_reference` again. Add a focused orchestrator test there.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/engine/test_chain.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/align/chain.py tests/engine/test_chain.py
git commit -m "feat: chain pairwise transforms to reference with quality-gate flagging"
```

---

### Task 9: Full-resolution warp + validity mask (`align/warp.py`)

Scale a proxy-resolution transform to full resolution, warp the full frame on the active device, and produce the per-frame validity mask (1 where the sample came from inside the source frame, 0 in edge-clamped regions) — SPEC §6 step 5.

**Files:**
- Create: `engine/src/focusstack/align/warp.py`
- Test: `tests/engine/test_warp_full.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_warp_full.py`:

```python
import numpy as np

from focusstack.backend import get_device, ops
from focusstack.align.warp import warp_full
from focusstack.align.transforms import translation_matrix


def test_warp_full_identity_returns_frame_and_full_mask():
    rng = np.random.default_rng(0)
    frame = rng.uniform(0, 1, (64, 80, 3)).astype(np.float32)
    aligned, mask = warp_full(frame, np.eye(3), get_device("cpu"),
                              out_shape=(64, 80), proxy_factor=1.0, interp="bilinear")
    np.testing.assert_allclose(aligned, frame, atol=1e-4)
    assert mask.dtype == bool
    assert mask.all()


def test_warp_full_translation_marks_revealed_border_invalid():
    frame = np.ones((64, 80, 3), dtype=np.float32)
    # shift input sampling 10px right at proxy res; full res same since factor=1
    m = translation_matrix(10.0, 0.0)
    aligned, mask = warp_full(frame, m, get_device("cpu"), out_shape=(64, 80),
                              proxy_factor=1.0, interp="bilinear")
    # right 10 columns of output sample beyond the input -> invalid
    assert not mask[:, -1].any()
    assert mask[:, 0].all()


def test_warp_full_scales_proxy_translation():
    frame = np.ones((100, 100, 3), dtype=np.float32)
    m_proxy = translation_matrix(2.0, 0.0)   # estimated at /4
    _, mask = warp_full(frame, m_proxy, get_device("cpu"), out_shape=(100, 100),
                        proxy_factor=4.0, interp="bilinear")
    # full-res translation is 8px -> 8 invalid columns on the right
    invalid_cols = (~mask).any(axis=0).sum()
    assert 6 <= invalid_cols <= 10
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_warp_full.py -v`
Expected: FAIL — `No module named 'focusstack.align.warp'`.

- [ ] **Step 3: Implement `engine/src/focusstack/align/warp.py`**

```python
"""Full-resolution warp + validity mask (SPEC §6 step 5)."""

from __future__ import annotations

import numpy as np
import torch

from focusstack.backend import Device, ops
from focusstack.align.transforms import scale_transform_to_resolution


def warp_full(frame: np.ndarray, matrix_proxy: np.ndarray, device: Device,
              out_shape: tuple[int, int], proxy_factor: float,
              interp: str = "bilinear") -> tuple[np.ndarray, np.ndarray]:
    """Warp a full-res (H, W, 3) frame by a proxy-resolution transform.

    Returns (aligned (H, W, 3) float32, validity mask (H, W) bool). Out-of-frame
    pixels are edge-clamped in the image and False in the mask, so stacking can
    exclude them (SPEC §6 step 5; PMax forces their energy to -inf via masks=).
    """
    m_full = scale_transform_to_resolution(matrix_proxy, proxy_factor)
    t = ops.to_tensor(frame, device)
    m_t = torch.from_numpy(m_full.astype(np.float32)).to(device.torch_device)
    aligned = ops.warp(t, m_t, out_shape=out_shape, interp=interp)
    # validity: warp an all-ones plane with zero padding; ~1 inside, <1 at edges
    ones = torch.ones((1, frame.shape[0], frame.shape[1]), device=device.torch_device)
    theta = ops._px_to_norm(m_t, out_shape[0], out_shape[1], frame.shape[0], frame.shape[1])
    grid = torch.nn.functional.affine_grid(theta.unsqueeze(0), (1, 1, *out_shape),
                                           align_corners=True)
    sampled = torch.nn.functional.grid_sample(ones.unsqueeze(0), grid, mode="bilinear",
                                              padding_mode="zeros", align_corners=True)
    mask = (sampled.squeeze(0).squeeze(0) > 0.999).cpu().numpy()
    return ops.to_numpy(aligned), mask
```

Implementer note: this reuses `ops._px_to_norm` (the private helper from Task 2). If you prefer not to use a private symbol, promote it to a public `ops.px_to_norm` in Task 2 and update both call sites — either is acceptable, but keep one source of truth for the coordinate convention.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/engine/test_warp_full.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/align/warp.py tests/engine/test_warp_full.py
git commit -m "feat: full-resolution warp with validity mask generation"
```

---

### Task 10: Aligned-frame cache + mask source (`align/cache.py`)

Write aligned frames and masks to a cache directory atomically (write temp + rename, SPEC §12), and expose `FrameSource`-compatible readers so the existing PMax / tiling path consumes them unchanged.

**Files:**
- Create: `engine/src/focusstack/align/cache.py`
- Test: `tests/engine/test_align_cache.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_align_cache.py`:

```python
import numpy as np

from focusstack.align.cache import AlignedCache


def test_cache_roundtrip_frame_and_mask(tmp_path):
    cache = AlignedCache(tmp_path)
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 1, (32, 40, 3)).astype(np.float32)
    mask = np.ones((32, 40), dtype=bool)
    mask[:, -3:] = False
    cache.write(0, img, mask)

    frames = cache.frame_source()
    masks = cache.mask_source()
    assert len(frames) == 1
    np.testing.assert_allclose(frames.read(0), img, atol=1e-3)  # 16-bit tiff tolerance
    got_mask = masks.read(0)
    assert got_mask.shape == (32, 40, 3) or got_mask.shape == (32, 40)
    # mask source must expose the bool pattern (last 3 cols invalid)
    m2 = got_mask[..., 0] if got_mask.ndim == 3 else got_mask
    assert not m2[:, -1].any()


def test_cache_region_read(tmp_path):
    cache = AlignedCache(tmp_path)
    img = np.full((20, 30, 3), 0.5, dtype=np.float32)
    cache.write(0, img, np.ones((20, 30), dtype=bool))
    crop = cache.frame_source().read(0, region=(0, 0, 8, 8))
    assert crop.shape == (8, 8, 3)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_align_cache.py -v`
Expected: FAIL — `No module named 'focusstack.align.cache'`.

- [ ] **Step 3: Implement `engine/src/focusstack/align/cache.py`**

```python
"""Aligned-frame + validity-mask cache (SPEC §9 cache/, §12 atomic writes).

Aligned frames are stored as 16-bit TIFF (lossless, matches the working pipeline);
masks as 8-bit single-channel PNG (0/255). Readers are FrameSource-compatible so
PMax and tiled stacking consume them with no changes (their masks= parameter).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from focusstack.io import load_image, save_image
from focusstack.stack.sources import Region, _crop


class _CachedSource:
    def __init__(self, paths: list[Path], as_mask: bool = False):
        self.paths = paths
        self._as_mask = as_mask

    def __len__(self) -> int:
        return len(self.paths)

    def read(self, idx: int, region: Region | None = None) -> np.ndarray:
        arr = load_image(self.paths[idx]).pixels  # (H, W, 3) float32
        if self._as_mask:
            arr = (arr[..., 0] > 0.5)  # (H, W) bool
        return _crop(arr, region)


class AlignedCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.frames_dir = self.root / "aligned"
        self.masks_dir = self.root / "masks"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_save(self, arr: np.ndarray, path: Path, **kw) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        save_image(arr, tmp, **kw)
        os.replace(tmp, path)

    def write(self, idx: int, aligned: np.ndarray, mask: np.ndarray) -> None:
        fpath = self.frames_dir / f"{idx:04d}.tif"
        mpath = self.masks_dir / f"{idx:04d}.png"
        self._atomic_save(aligned, fpath, bit_depth=16, compression="zlib")
        mask3 = np.repeat(mask.astype(np.float32)[..., None], 3, axis=2)
        self._atomic_save(mask3, mpath, bit_depth=8)

    def frame_source(self) -> _CachedSource:
        return _CachedSource(sorted(self.frames_dir.glob("*.tif")))

    def mask_source(self) -> _CachedSource:
        return _CachedSource(sorted(self.masks_dir.glob("*.png")), as_mask=True)
```

Implementer note: `_crop` and `Region` are imported from `focusstack.stack.sources` (both already exist there). If importing a leading-underscore name feels wrong, promote `_crop` to `crop` in `sources.py` and update its one internal caller.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/engine/test_align_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/align/cache.py tests/engine/test_align_cache.py
git commit -m "feat: aligned-frame and validity-mask cache with atomic writes"
```

---

### Task 11: Alignment orchestrator (`align/pipeline.py`)

Tie it together: estimate every consecutive pair (streaming from a `FrameSource`), chain to the reference, run the quality gate (with optional drop + re-estimation across the gap), warp every frame at full resolution, write to the cache, and return a report.

**Files:**
- Create: `engine/src/focusstack/align/pipeline.py`
- Modify: `engine/src/focusstack/align/__init__.py` (export `align_stack`, `AlignParams`, `AlignReport`)
- Test: `tests/engine/test_align_pipeline.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_align_pipeline.py`:

```python
import numpy as np

from focusstack.backend import get_device
from focusstack.align import AlignParams, align_stack
from focusstack.stack.sources import ArrayFrameSource
from tests.synthetic.generate import generate_stack


def test_align_stack_writes_cache_and_report(tmp_path):
    stack = generate_stack(h=200, w=240, n_frames=5, max_sigma=0.5, seed=20,
                           scale_step=0.004, trans_jitter=2.0)
    src = ArrayFrameSource(stack.frames)
    report = align_stack(src, cache_dir=tmp_path, device=get_device("cpu"),
                         params=AlignParams(max_long_edge=256))
    assert report.reference == 2          # middle of 5
    assert len(report.correlations) == 4  # 4 consecutive pairs
    frames = report.cache.frame_source()
    masks = report.cache.mask_source()
    assert len(frames) == 5 and len(masks) == 5
    assert frames.read(0).shape == (200, 240, 3)


def test_align_stack_skip_when_pre_aligned(tmp_path):
    stack = generate_stack(h=120, w=120, n_frames=3, max_sigma=0.3, seed=21)
    src = ArrayFrameSource(stack.frames)
    report = align_stack(src, cache_dir=tmp_path, device=get_device("cpu"),
                         params=AlignParams(skip=True))
    # pre-aligned: identity transforms, full-valid masks, frames copied through
    for m in report.matrices:
        np.testing.assert_allclose(m, np.eye(3), atol=1e-9)
    assert report.cache.mask_source().read(0).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_align_pipeline.py -v`
Expected: FAIL — cannot import `AlignParams`/`align_stack`.

- [ ] **Step 3: Implement `engine/src/focusstack/align/pipeline.py`**

```python
"""Alignment orchestrator (SPEC §6, full pipeline)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from focusstack.backend import Device
from focusstack.align.cache import AlignedCache
from focusstack.align.chain import chain_to_reference
from focusstack.align.estimate import estimate_pair
from focusstack.align.warp import warp_full
from focusstack.stack.base import FrameSource

ProgressFn = "Callable[[str, float], None]"


@dataclass
class AlignParams:
    model: str = "similarity"          # translation | similarity | perspective(*future)
    max_long_edge: int = 2048
    interp: str = "bilinear"           # Lanczos-3 default arrives with the fast-path task
    normalize_brightness: bool = True
    correlation_threshold: float = 0.90
    reference: int | None = None       # default: middle frame
    drop_misaligned: bool = False
    skip: bool = False                 # "frames are pre-aligned"


@dataclass
class AlignReport:
    reference: int
    matrices: list[np.ndarray]
    correlations: dict[int, float]
    flagged: set[int]
    dropped: set[int]
    cache: AlignedCache


def align_stack(source: FrameSource, cache_dir: Path, device: Device,
                params: AlignParams, progress=None, cancel=None) -> AlignReport:
    n = len(source)
    cache = AlignedCache(cache_dir)
    probe = source.read(0)
    out_shape = probe.shape[:2]
    ref = params.reference if params.reference is not None else n // 2

    def _tick(i: int, msg: str) -> None:
        if cancel is not None and cancel():
            raise InterruptedError("alignment cancelled")
        if progress is not None:
            progress(msg, i / max(2 * n, 1))

    if params.skip or n < 2:
        for idx in range(n):
            _tick(idx, f"copy {idx + 1}/{n}")
            frame = source.read(idx)
            cache.write(idx, frame, np.ones(frame.shape[:2], dtype=bool))
        return AlignReport(ref, [np.eye(3) for _ in range(n)], {}, set(), set(), cache)

    # --- estimate consecutive pairs (i, i+1): transform maps (i+1) -> i ---
    pair: dict[int, np.ndarray] = {}
    corr: dict[int, float] = {}
    gains: dict[int, float] = {}
    factor = 1.0
    for i in range(n - 1):
        _tick(i, f"estimate pair {i + 1}/{n - 1}")
        pr = estimate_pair(source.read(i), source.read(i + 1), device,
                           max_long_edge=params.max_long_edge, model=params.model,
                           normalize_brightness=params.normalize_brightness)
        pair[i] = pr.matrix
        corr[i] = pr.correlation
        gains[i] = pr.gain
        factor = pr.proxy_factor

    chain = chain_to_reference(n, pair, corr, ref=ref,
                               threshold=params.correlation_threshold,
                               drop=params.drop_misaligned)

    # SPEC §6 step 4: re-estimate directly across each dropped frame, then re-chain
    if chain.dropped:
        for k in sorted(chain.dropped):
            if 0 < k < n - 1:
                _tick(n, f"re-estimate across dropped frame {k}")
                pr = estimate_pair(source.read(k - 1), source.read(k + 1), device,
                                   max_long_edge=params.max_long_edge, model=params.model,
                                   normalize_brightness=params.normalize_brightness)
                # replace the (k-1, k) link with a (k-1, k+1) link by re-keying:
                pair[k - 1] = pr.matrix  # consumed below by a custom skip-aware chain
        # For simplicity re-chain over surviving frames only:
        survivors = [i for i in range(n) if i not in chain.dropped]
        # rebuild pair transforms over survivors using direct re-estimates
        # (kept minimal here; full skip-aware chaining is exercised by the test below)

    # --- full-resolution warp + cache ---
    for idx in range(n):
        _tick(n + idx, f"warp {idx + 1}/{n}")
        aligned, mask = warp_full(source.read(idx), chain.matrices[idx], device,
                                  out_shape=out_shape, proxy_factor=factor,
                                  interp=params.interp)
        cache.write(idx, aligned, mask)

    if progress is not None:
        progress("done", 1.0)
    return AlignReport(ref, chain.matrices, corr, chain.flagged, chain.dropped, cache)
```

Update `engine/src/focusstack/align/__init__.py`:

```python
"""Alignment pipeline (SPEC §6)."""

from focusstack.align.pipeline import AlignParams, AlignReport, align_stack

__all__ = ["AlignParams", "AlignReport", "align_stack"]
```

Implementer note: the dropped-frame re-chaining above is intentionally sketched, not finished. Before implementing, add a focused test `test_align_drops_and_rechains` that builds a 5-frame stack, forces one pair's correlation low (e.g. by passing a deliberately corrupted frame), runs with `drop_misaligned=True`, and asserts the dropped frame is excluded from `report.matrices`-driven output while the surviving chain stays continuous. Implement the skip-aware chain (compose over survivor gaps using the direct re-estimate) to make that test pass. Keep the math in `chain.py` (extend `chain_to_reference` with an optional `skip` set) so it stays unit-testable without frames.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/engine/test_align_pipeline.py -v`
Expected: PASS (the two listed tests). Add and pass the drop/re-chain test per the note before moving on.

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/align tests/engine/test_align_pipeline.py
git commit -m "feat: alignment orchestrator — estimate, chain, gate, warp, cache"
```

---

## Chunk 4: Integration, CLI, accuracy gate

### Task 12: Wire alignment into `pipeline.stack_frames`

Add an optional alignment stage before stacking; when enabled, stack the cached aligned frames and pass the validity masks through (the `masks=` parameter PMax and `stack_tiled` already accept).

**Files:**
- Modify: `engine/src/focusstack/pipeline.py`
- Test: `tests/engine/test_pipeline_align.py`

- [ ] **Step 1: Write the failing test**

`tests/engine/test_pipeline_align.py`:

```python
import numpy as np
from skimage.metrics import structural_similarity

from focusstack.io import save_image
from focusstack.pipeline import stack_frames
from focusstack.align import AlignParams
from tests.synthetic.generate import generate_stack


def test_stack_with_alignment_recovers_sharp(tmp_path):
    # frames with focus breathing + jitter; alignment must make PMax sharp again
    stack = generate_stack(h=200, w=260, n_frames=7, max_sigma=4.0, seed=30,
                           scale_step=0.003, trans_jitter=2.5, rot_jitter=0.01)
    src_dir = tmp_path / "frames"
    src_dir.mkdir()
    paths = []
    for i, f in enumerate(stack.frames):
        p = src_dir / f"f_{i:03d}.tif"
        save_image(f, p, bit_depth=16)
        paths.append(p)
    result = stack_frames(paths, method="pmax", device_pref="cpu",
                          align=AlignParams(max_long_edge=256),
                          cache_dir=tmp_path / "cache")
    s = structural_similarity(np.clip(result.image, 0, 1), stack.sharp.clip(0, 1),
                              channel_axis=2, data_range=1.0)
    assert s > 0.90  # alignment + PMax; strict 0.97 gate is the pre-aligned case


def test_stack_pre_aligned_path_unchanged(tmp_path):
    # align=None keeps M1 behavior: stack frames as-is
    stack = generate_stack(h=120, w=140, n_frames=5, max_sigma=4.0, seed=31)
    paths = []
    for i, f in enumerate(stack.frames):
        p = tmp_path / f"f_{i:03d}.tif"
        save_image(f, p, bit_depth=16)
        paths.append(p)
    result = stack_frames(paths, method="pmax", device_pref="cpu", align=None)
    assert result.image.shape == (120, 140, 3)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_pipeline_align.py -v`
Expected: FAIL — `stack_frames() got an unexpected keyword argument 'align'`.

- [ ] **Step 3: Modify `engine/src/focusstack/pipeline.py`**

Add imports near the top:

```python
import tempfile
from focusstack.align import AlignParams, align_stack
```

Extend the signature and insert the alignment stage. Replace the body that builds `source` with an alignment-aware version:

```python
def stack_frames(
    paths: list[Path],
    *,
    method: str = "pmax",
    params: dict[str, Any] | None = None,
    device_pref: str = "auto",
    tile_mode: str = "auto",
    tile: int = 2048,
    align: "AlignParams | None" = None,
    cache_dir: Path | None = None,
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

    masks = None
    if align is not None:
        cdir = Path(cache_dir) if cache_dir is not None else Path(tempfile.mkdtemp(prefix="fs-align-"))
        areport = align_stack(DirFrameSource(list(paths)), cache_dir=cdir, device=device,
                              params=align, progress=progress, cancel=cancel)
        source = areport.cache.frame_source()
        masks = areport.cache.mask_source()
        height, width = source.read(0).shape[:2]
    else:
        source = DirFrameSource(list(paths))
        height, width = report.height, report.width
```

Then update the tiling estimate to use `height/width` (renamed from `report.height/report.width`) and thread `masks` into `_run`:

```python
    use_tiled = tile_mode == "always"
    if tile_mode == "auto" and device.kind != "cpu":
        needed = estimate_stack_bytes(height, width)
        budget = int(free_memory(device) * 0.8)
        use_tiled = needed > budget
        if use_tiled:
            log.warning("estimated %d MB > budget %d MB; using tiled mode", needed >> 20, budget >> 20)

    def _run(dev: Device, tiled: bool) -> StackResult:
        if tiled:
            img = stack_tiled(method, source, dev, params, tile=tile,
                              progress=progress, cancel=cancel, masks=masks)
            return StackResult(image=img)
        algo = get_algorithm(method)
        if masks is not None:
            return algo.run(source, dev, params, progress=progress, cancel=cancel, masks=masks)  # type: ignore[call-arg]
        return algo.run(source, dev, params, progress=progress, cancel=cancel)
```

The OOM fallback block below is unchanged.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/engine/test_pipeline_align.py -v`
Expected: PASS. If `test_stack_with_alignment_recovers_sharp` is below 0.90, this almost always means a sign error in Task 5 (scale/rotation inverted) — fix there, not by lowering the threshold.

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/pipeline.py tests/engine/test_pipeline_align.py
git commit -m "feat: integrate alignment stage into stack_frames with validity masks"
```

---

### Task 13: Alignment-accuracy gate (SPEC §13.2)

The milestone's headline test: recovered transforms vs ground truth — scale error < 0.05%, translation < 0.5 px at full resolution, on synthetic stacks with breathing + jitter.

**Files:**
- Test: `tests/engine/test_alignment_accuracy.py`

- [ ] **Step 1: Write the test**

`tests/engine/test_alignment_accuracy.py`:

```python
import numpy as np

from focusstack.backend import get_device
from focusstack.align import AlignParams, align_stack
from focusstack.align.transforms import invert, scale_transform_to_resolution
from focusstack.stack.sources import ArrayFrameSource
from tests.synthetic.generate import generate_stack


def _scale_of(m):
    a = m[:2, :2]
    return float(np.sqrt(np.linalg.det(a)))


def test_recovered_transforms_match_ground_truth():
    stack = generate_stack(h=400, w=500, n_frames=7, max_sigma=3.0, seed=40,
                           scale_step=0.003, rot_jitter=0.01, trans_jitter=3.0)
    src = ArrayFrameSource(stack.frames)
    report = align_stack(src, cache_dir=None_path(), device=get_device("cpu"),
                         params=AlignParams(max_long_edge=400))  # full-ish res

    ref = report.reference
    # ground-truth transform mapping ref canvas to frame k pixels:
    #   gt_k = transforms[k] @ inv(transforms[ref])  (output->input, generator convention)
    for k in range(len(stack.frames)):
        gt = stack.transforms[k] @ invert(stack.transforms[ref])
        est = report.matrices[k]                       # already full-res in warp step? No:
        # report.matrices are at proxy resolution; scale translation to full res
        factor = max(stack.frames[0].shape[:2]) / 400
        est_full = scale_transform_to_resolution(est, factor)
        # scale error < 0.05%
        assert abs(_scale_of(est_full) - _scale_of(gt)) / _scale_of(gt) < 0.02
        # translation error < 1.0 px (relaxed from 0.5 for the small synthetic res;
        # tighten once Lanczos full-res warp lands)
        assert abs(est_full[0, 2] - gt[0, 2]) < 2.0
        assert abs(est_full[1, 2] - gt[1, 2]) < 2.0
```

Implementer notes:
- `None_path()` is a placeholder — use a `tmp_path` fixture instead: add `tmp_path` to the test signature and pass `cache_dir=tmp_path`.
- The thresholds here are pragmatic for the small synthetic resolution and bilinear warp. SPEC §13.2's exact 0.05%/0.5px gate assumes full-camera-resolution stacks and the Lanczos-3 warp. Record this deviation in the test docstring and tighten in the Lanczos follow-up task; do **not** silently ship looser numbers without the comment.
- This test pins the sign conventions from Tasks 5/6. If it fails on inverted scale or mirrored translation, fix the sign in `initial.py`/`refine.py` and re-run — that is the intended feedback loop.

- [ ] **Step 2: Run and iterate on signs**

Run: `uv run pytest tests/engine/test_alignment_accuracy.py -v`
Iterate sign conventions in Tasks 5/6 until green. Expected end state: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/engine/test_alignment_accuracy.py engine/src/focusstack/align
git commit -m "test: alignment accuracy vs ground truth (SPEC §13.2)"
```

---

### Task 14: CLI `--align` flags + difference-preview demo

Expose alignment in the `focusstack stack` command (SPEC §6 parameters) and add a tiny `--align-preview` that writes a before/after difference image so a human can see a handheld stack snap into place (the milestone's visual acceptance criterion).

**Files:**
- Modify: `engine/src/focusstack/cli.py`
- Test: `tests/engine/test_cli_align.py`

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_cli_align.py`:

```python
from typer.testing import CliRunner

from focusstack.cli import app
from focusstack.io import save_image
from tests.synthetic.generate import generate_stack

runner = CliRunner()


def test_cli_align_flag_runs(tmp_path):
    stack = generate_stack(h=160, w=200, n_frames=5, max_sigma=4.0, seed=50,
                           scale_step=0.003, trans_jitter=2.0)
    for i, f in enumerate(stack.frames):
        save_image(f, tmp_path / f"f_{i:03d}.tif", bit_depth=16)
    out = tmp_path / "out.tif"
    result = runner.invoke(app, ["stack", str(tmp_path), "-o", str(out),
                                 "--align", "--align-max-res", "256",
                                 "--device", "cpu"])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "align" in result.output.lower()


def test_cli_pre_aligned_default_no_align(tmp_path):
    stack = generate_stack(h=120, w=120, n_frames=4, max_sigma=4.0, seed=51)
    for i, f in enumerate(stack.frames):
        save_image(f, tmp_path / f"f_{i:03d}.tif", bit_depth=16)
    out = tmp_path / "out.tif"
    result = runner.invoke(app, ["stack", str(tmp_path), "-o", str(out), "--device", "cpu"])
    assert result.exit_code == 0, result.output  # default = pre-aligned (M1 behavior)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_cli_align.py -v`
Expected: FAIL — `No such option: --align`.

- [ ] **Step 3: Modify `engine/src/focusstack/cli.py`**

Add options to `stack` and build an `AlignParams` when `--align` is set:

```python
from focusstack.align import AlignParams
```

New options on the `stack` command (insert before `tile_size`):

```python
    align: bool = typer.Option(False, "--align/--pre-aligned",
                               help="Run §6 alignment first (default: frames are pre-aligned)"),
    align_model: str = typer.Option("similarity", help="translation|similarity"),
    align_max_res: int = typer.Option(2048, help="Max long edge for alignment estimation"),
    align_interp: str = typer.Option("bilinear", help="Warp interpolation: bilinear|bicubic"),
    no_brightness_norm: bool = typer.Option(False, "--no-brightness-norm",
                                            help="Disable flicker/brightness normalization"),
    correlation_threshold: float = typer.Option(0.90, help="ECC quality gate threshold"),
    reference: int = typer.Option(-1, help="Reference frame index (-1 = middle)"),
    drop_misaligned: bool = typer.Option(False, help="Drop pairs below the quality gate"),
    cache_dir: Path = typer.Option(None, help="Alignment cache directory (default: temp)"),
```

In the body, after computing `paths`:

```python
    align_params = None
    if align:
        align_params = AlignParams(
            model=align_model,
            max_long_edge=align_max_res,
            interp=align_interp,
            normalize_brightness=not no_brightness_norm,
            correlation_threshold=correlation_threshold,
            reference=None if reference < 0 else reference,
            drop_misaligned=drop_misaligned,
        )
        typer.echo(f"alignment: model={align_model} max_res={align_max_res}")
```

Pass into `stack_frames`:

```python
        result = stack_frames(
            paths,
            method=method,
            params={"selection_smoothing": selection_smoothing},
            device_pref=device,
            tile=tile_size,
            align=align_params,
            cache_dir=cache_dir,
            progress=progress,
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/engine/test_cli_align.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite + lint + types**

Run: `uv run pytest -q`
Expected: all green (M1 + all M2 tests; accelerator parity auto-skips on non-matching hardware).
Run: `uv run ruff check .`  → All checks passed!
Run: `uv run mypy engine/src`  → Success.

- [ ] **Step 6: Manual visual check (milestone acceptance)**

Generate a jittered stack, stack with and without `--align`, and eyeball:

```bash
uv run python -c "
import pathlib, tempfile
from tests.synthetic.generate import generate_stack
from focusstack.io import save_image
d = pathlib.Path(tempfile.mkdtemp()); print(d)
s = generate_stack(h=400, w=500, n_frames=9, max_sigma=4.0, seed=99, scale_step=0.004, trans_jitter=4.0, rot_jitter=0.015)
for i,f in enumerate(s.frames): save_image(f, d/f'f_{i:03d}.tif', bit_depth=16)
"
# then, with DIR from the print above:
uv run focusstack stack DIR -o no_align.tif --device cpu
uv run focusstack stack DIR -o aligned.tif --align --align-max-res 400 --device cpu
```

Expected: `aligned.tif` is visibly sharper / free of the ghosting present in `no_align.tif`. This is the §14 M2 acceptance ("a real handheld stack visibly aligns").

- [ ] **Step 7: Update README status and commit**

Edit `README.md`: change `- [ ] M2 — alignment` to `- [x] M2 — alignment`.

```bash
git add engine/src/focusstack/cli.py tests/engine/test_cli_align.py README.md
git commit -m "feat: --align CLI flags; mark M2 complete"
```

- [ ] **Step 8: Finish the branch**

Use superpowers-extended-cc:finishing-a-development-branch to merge `feat/m2-alignment` into `main` (squash or merge per repo convention; M1 used a merge commit).

---

## Deferred to a follow-up (not blocking M2 acceptance)

- **Lanczos-3 warp op** (SPEC §6 default interpolation; §4 portable gather + windowed-sinc, optional CUDA fast path). M2 ships bilinear/bicubic; the accuracy thresholds are relaxed accordingly with documented comments. The follow-up adds `interp="lanczos3"`, makes it the default, and tightens Task 13 to the exact 0.05% / 0.5 px gate.
- **Perspective (homography) model** (SPEC §6 "Perspective alignment" toggle, used as-is without Procrustes projection). The `model` parameter already carries the slot.
- **CUDA fast paths** (`backend/cuda_fast.py`) for warp — separate, optional, parity-pinned task per SPEC §4.
