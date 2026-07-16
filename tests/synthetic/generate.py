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

from planefuse.backend import get_device, ops


@dataclass
class SyntheticStack:
    frames: list[np.ndarray]  # float32 (H, W, 3)
    sharp: np.ndarray  # ground-truth all-in-focus (H, W, 3)
    depth: np.ndarray  # ground-truth depth (H, W) in [0, 1]
    transforms: list[np.ndarray]  # 3x3 ground-truth output->input pixel transforms


def make_scene(
    h: int, w: int, seed: int = 0, flat_region: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-octave textured image + depth map (ramp with two plateaus).

    When ``flat_region`` is True, the top-left quarter of the image (rows
    ``0:h//4``, cols ``0:w//4``) is set to a constant 0.5 (textureless) so the
    kurtosis reliability classifier can be calibrated/tested against it.
    """
    rng = np.random.default_rng(seed)
    dev = get_device("cpu")
    t = torch.zeros(3, h, w)
    for amplitude, sigma in [(1.0, 1.0), (0.5, 4.0), (0.25, 12.0)]:
        noise = torch.from_numpy(rng.uniform(0, 1, (h, w, 3)).astype(np.float32))
        t = t + amplitude * ops.gaussian_blur(ops.to_tensor(noise.numpy(), dev), sigma)
    t = (t - t.min()) / (t.max() - t.min()) * 0.8 + 0.1
    depth = np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1))
    depth[h // 6 : h // 3, w // 6 : w // 3] = 0.15  # near plateau
    depth[h // 2 : 5 * h // 6, w // 2 : 5 * w // 6] = 0.85  # far plateau
    img = ops.to_numpy(t)
    if flat_region:
        img[0 : h // 4, 0 : w // 4, :] = 0.5  # textureless corner
    return img, depth


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
    scale_step: float = 0.0,
    rot_jitter: float = 0.0,
    trans_jitter: float = 0.0,
    flicker: float = 0.0,
    noise: float = 0.0,
    flat_region: bool = False,
) -> SyntheticStack:
    rng = np.random.default_rng(seed + 1000)
    sharp_np, depth = make_scene(h, w, seed, flat_region=flat_region)
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
                h,
                w,
            )
            frame = _apply_affine(frame, m)
        if flicker:
            frame = frame * (1.0 + rng.uniform(-flicker, flicker))
        if noise:
            frame = frame + torch.from_numpy(rng.normal(0, noise, (3, h, w)).astype(np.float32))
        frames.append(ops.to_numpy(frame))
        transforms.append(m.astype(np.float64))
    return SyntheticStack(frames=frames, sharp=sharp_np, depth=depth, transforms=transforms)
