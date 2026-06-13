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
    correlation of the FFT magnitude spectra (translation-invariant).

    NOTE: the sign that maps (d_rho, d_theta) -> (scale, angle) is provisional;
    it is pinned by the end-to-end alignment accuracy test (Task 13). If that
    test shows scale/angle inverted, flip the sign on d_rho / d_theta here.
    """
    n = 256
    fa = torch.fft.fftshift(ops.fft2(a).abs(), dim=(-2, -1))
    fb = torch.fft.fftshift(ops.fft2(b).abs(), dim=(-2, -1))
    fa = torch.log1p(fa)
    fb = torch.log1p(fb)
    lpa = ops.log_polar_remap(fa, n_angles=n, n_radii=n)
    lpb = ops.log_polar_remap(fb, n_angles=n, n_radii=n)
    d_rho, d_theta, _peak = ops.phase_correlation(lpa, lpb)
    angle = -d_theta * (2.0 * math.pi / n)
    h, w = a.shape[-2:]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    max_r = math.sqrt(cx * cx + cy * cy)
    log_base = math.log(max_r + 1e-6) / n
    scale = float(math.exp(-d_rho * log_base))
    if not (0.5 < scale < 2.0):
        scale = 1.0
    return scale, float(angle)
