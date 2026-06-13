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
    return F.interpolate(img.unsqueeze(0), size=shape, mode="bilinear", align_corners=False).squeeze(0)


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
    grid = F.affine_grid(theta, [1, c, out_h, out_w], align_corners=True)
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
    integer peak from the argmax. A Hann window suppresses edge wrap-around.
    """
    aw = hann_apply(a).squeeze(0)
    bw = hann_apply(b).squeeze(0)
    if _fft_needs_cpu(a.device):
        fa = torch.fft.fft2(aw.cpu()).to(a.device)
    else:
        fa = torch.fft.fft2(aw)
    if _fft_needs_cpu(b.device):
        fb = torch.fft.fft2(bw.cpu()).to(b.device)
    else:
        fb = torch.fft.fft2(bw)
    cross = fb * fa.conj()
    cross = cross / (cross.abs() + 1e-8)
    corr = ifft2(cross.unsqueeze(0)).real.squeeze(0)
    h, w = corr.shape
    flat = int(torch.argmax(corr).item())
    py, px = divmod(flat, w)
    peak = float(corr[py, px] / (corr.abs().mean() + 1e-12))
    dy = py - h if py > h // 2 else py
    dx = px - w if px > w // 2 else px
    return float(dy), float(dx), peak


def log_polar_remap(img: torch.Tensor, n_angles: int, n_radii: int) -> torch.Tensor:
    """Remap (1, H, W) into (1, n_radii, n_angles) log-polar space about the center.

    Used on FFT magnitude spectra: scale -> radial shift, rotation -> angular shift,
    both recoverable by a second phase correlation (SPEC §6 step b).
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
