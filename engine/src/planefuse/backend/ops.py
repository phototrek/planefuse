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

from planefuse.backend.device import Device


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
    dev = m.device
    dt = m.dtype if m.is_floating_point() else torch.float32
    work = m.to(dtype=dt)
    s_out = torch.tensor([[2.0 / (out_w - 1), 0, -1], [0, 2.0 / (out_h - 1), -1], [0, 0, 1]],
                         dtype=dt, device=dev)
    s_in = torch.tensor([[2.0 / (in_w - 1), 0, -1], [0, 2.0 / (in_h - 1), -1], [0, 0, 1]],
                        dtype=dt, device=dev)
    norm = s_in @ work @ torch.linalg.inv(s_out)
    return norm[:2].to(torch.float32)


def _pixel_grid(matrix: torch.Tensor, out_shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    out_h, out_w = out_shape
    device = matrix.device
    work = matrix.to(dtype=torch.float32)
    rows, cols = torch.meshgrid(
        torch.arange(out_h, dtype=torch.float32, device=device),
        torch.arange(out_w, dtype=torch.float32, device=device),
        indexing="ij",
    )
    homogeneous = torch.stack((cols, rows, torch.ones_like(cols)), dim=-1)
    mapped = homogeneous @ work.T
    denominator = mapped[..., 2]
    denominator = torch.where(
        denominator.abs() < 1e-8,
        torch.full_like(denominator, 1e-8),
        denominator,
    )
    return mapped[..., 0] / denominator, mapped[..., 1] / denominator


def _normalized_grid(
    matrix: torch.Tensor,
    out_shape: tuple[int, int],
    in_shape: tuple[int, int],
) -> torch.Tensor:
    source_x, source_y = _pixel_grid(matrix, out_shape)
    in_h, in_w = in_shape
    norm_x = source_x * (2.0 / max(in_w - 1, 1)) - 1.0
    norm_y = source_y * (2.0 / max(in_h - 1, 1)) - 1.0
    return torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)


def _lanczos3_kernel(distance: torch.Tensor) -> torch.Tensor:
    return torch.where(
        distance.abs() < 3.0,
        torch.sinc(distance) * torch.sinc(distance / 3.0),
        torch.zeros_like(distance),
    )


def _warp_lanczos3(img: torch.Tensor, matrix: torch.Tensor, out_shape: tuple[int, int]) -> torch.Tensor:
    channels, in_h, in_w = img.shape
    out_h, out_w = out_shape
    source_x, source_y = _pixel_grid(matrix.to(img.device), out_shape)
    offsets = torch.arange(-2, 4, dtype=torch.float32, device=img.device)
    flat = img.reshape(channels, -1)
    output = torch.empty((channels, out_h, out_w), dtype=torch.float32, device=img.device)

    # Keep the temporary 6x6 gather bounded for large full-resolution frames.
    rows_per_chunk = max(1, 65_536 // max(out_w, 1))
    for start in range(0, out_h, rows_per_chunk):
        stop = min(out_h, start + rows_per_chunk)
        x = source_x[start:stop]
        y = source_y[start:stop]
        x_indices = torch.floor(x).unsqueeze(-1) + offsets
        y_indices = torch.floor(y).unsqueeze(-1) + offsets
        x_weights = _lanczos3_kernel(x.unsqueeze(-1) - x_indices)
        y_weights = _lanczos3_kernel(y.unsqueeze(-1) - y_indices)
        x_weights = x_weights / x_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        y_weights = y_weights / y_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        x_indices = x_indices.to(torch.long).clamp_(0, in_w - 1)
        y_indices = y_indices.to(torch.long).clamp_(0, in_h - 1)
        indices = y_indices.unsqueeze(-1) * in_w + x_indices.unsqueeze(-2)
        samples = flat[:, indices.reshape(-1)].reshape(
            channels, stop - start, out_w, 6, 6
        )
        weights = y_weights.unsqueeze(-1) * x_weights.unsqueeze(-2)
        output[:, start:stop] = (samples * weights.unsqueeze(0)).sum(dim=(-1, -2))
    return output.to(dtype=img.dtype)


def warp(img: torch.Tensor, matrix: torch.Tensor, out_shape: tuple[int, int],
         interp: str = "bilinear") -> torch.Tensor:
    """Warp (C, H, W) by a 3x3 output->input pixel matrix to out_shape (H, W).

    interp: "bilinear" | "bicubic" | "nearest" | "lanczos3".
    Out-of-frame samples use edge clamp (padding_mode="border").
    """
    if interp == "lanczos3":
        return _warp_lanczos3(img, matrix, out_shape)
    c, in_h, in_w = img.shape
    out_h, out_w = out_shape
    work = matrix.to(img.device)
    affine_last_row = torch.tensor([0.0, 0.0, 1.0], dtype=work.dtype, device=work.device)
    if torch.allclose(work[2], affine_last_row, atol=1e-7, rtol=1e-7):
        theta = _px_to_norm(work, out_h, out_w, in_h, in_w).unsqueeze(0)
        grid = F.affine_grid(theta, [1, c, out_h, out_w], align_corners=True)
    else:
        grid = _normalized_grid(work, out_shape, (in_h, in_w))
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


# ----------------------------------------------------------------------------
# Sharpness + edge-aware filtering ops (SPEC §4, §7.2/§7.3).
# ----------------------------------------------------------------------------

def box_filter(img: torch.Tensor, radius: int) -> torch.Tensor:
    """(2r+1)x(2r+1) mean filter, reflect-padded. img: (C, H, W). radius 0 = identity."""
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
    """Windowed local Laplacian energy (SPEC §4, §7.2). gray: (1, H, W) -> (1, H, W)."""
    if gray.shape[0] != 1:
        raise ValueError(f"sharpness_map expects (1, H, W), got {gray.shape[0]} channels")
    k = _LAPLACIAN.to(device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)
    x = F.pad(gray.unsqueeze(0), (1, 1, 1, 1), mode="reflect")
    lap = F.conv2d(x, k).squeeze(0)
    return box_filter(lap * lap, radius)


def guided_filter(guide: torch.Tensor, src: torch.Tensor, radius: int, eps: float) -> torch.Tensor:
    """Edge-aware smoothing of `src` guided by `guide` (He et al. 2010). Both (1, H, W)."""
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
    `values` from known neighbours by iterated masked box blur. values (1,H,W) float;
    known (1,H,W) bool. Known pixels preserved."""
    cov = known.to(values.dtype)
    out = torch.where(known, values, torch.zeros_like(values))
    for _ in range(iters):
        # Push-pull diffusion: a normalized box blur weighted by coverage. Coverage
        # is propagated alongside the values so that wide unknown regions fill from
        # their boundaries inward instead of dividing by a vanishing denominator
        # (which would otherwise blow up to +/-inf).
        num = box_filter(out * cov, radius)
        den = box_filter(cov, radius)
        reached = den > 1e-6
        diffused = num / den.clamp_min(1e-8)
        new = torch.where(known | ~reached, out, diffused)
        new_cov = torch.where(known, cov, reached.to(values.dtype))
        delta = float((new - out).abs().max()) if out.numel() else 0.0
        out, cov = new, new_cov
        if bool(known.logical_or(reached).all()) and delta < tol:
            break
    return out
