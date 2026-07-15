"""Display-only preview tonemapping for scene-linear camera-RGB frames.

The processing pipeline works on the untouched scene-linear data; these
transforms exist purely so the viewer can show white-balanced, exposure-
normalized, gamma-encoded previews. Every step degrades gracefully when
camera metadata is missing, so the worst case is still a brightened,
sRGB-encoded image instead of a flat linear one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from focusstack.io.metadata import ImageMetadata

SRGB_TO_XYZ_D65 = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

_REC709_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_MAX_EXPOSURE_SAMPLES = 1_000_000

# Bump when the tonemap math changes so cached display tile pyramids rebuild.
DISPLAY_TONEMAP_REVISION = 3


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    """Piecewise sRGB OETF over values already clipped to [0, 1]."""
    linear = np.asarray(linear, dtype=np.float32)
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def camera_to_srgb(color_matrix1: Sequence[float] | None) -> np.ndarray | None:
    """Camera-RGB -> sRGB matrix from a DNG ColorMatrix1 (XYZ -> cameraRGB).

    Uses the dcraw/LibRaw construction: rows of camera-from-sRGB are
    normalized to sum to 1 so sRGB white maps to camera (1,1,1) — the space
    produced by per-channel white balance — absorbing illuminant adaptation.
    """
    if color_matrix1 is None or len(color_matrix1) != 9:
        return None
    xyz_to_cam = np.asarray(color_matrix1, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(xyz_to_cam)):
        return None
    cam_from_srgb = xyz_to_cam @ SRGB_TO_XYZ_D65
    row_sums = cam_from_srgb.sum(axis=1, keepdims=True)
    if np.any(np.abs(row_sums) < 1e-10):
        return None
    cam_from_srgb = cam_from_srgb / row_sums
    if abs(float(np.linalg.det(cam_from_srgb))) < 1e-10:
        return None
    return np.linalg.inv(cam_from_srgb)


def auto_exposure_gain(
    rgb: np.ndarray, *, target: float = 0.9, percentile: float = 99.5
) -> float:
    """Deterministic exposure gain mapping the luma percentile to `target`."""
    luma = rgb[..., 0] * _REC709_LUMA[0] + rgb[..., 1] * _REC709_LUMA[1] + rgb[..., 2] * _REC709_LUMA[2]
    flat = luma.reshape(-1)
    stride = max(1, flat.size // _MAX_EXPOSURE_SAMPLES)
    sampled = flat[::stride]
    lit = sampled[sampled > 0.0]
    if lit.size == 0:
        return 1.0
    level = float(np.percentile(lit, percentile))
    return float(np.clip(target / max(level, 1e-6), 0.25, 64.0))


def _white_balance(pixels: np.ndarray, neutral: Sequence[float] | None) -> tuple[np.ndarray, float]:
    """Return balanced pixels plus the level where the first channel saturates.

    Sensor-clipped pixels read (1,1,1) before balancing, so unequal channel
    gains would tint them magenta. Clipping the result at the common
    saturation level (the smallest per-channel gain) keeps blown highlights
    neutral instead.
    """
    if neutral is not None and len(neutral) == 3:
        scale = np.asarray(neutral, dtype=np.float32)
        if np.all(np.isfinite(scale)) and np.all(scale > 0):
            return pixels / scale.reshape(1, 1, 3), float(1.0 / scale.max())
    # Gray-world fallback: match R and B means to G's mean.
    means = pixels.reshape(-1, 3).mean(axis=0)
    if not np.all(np.isfinite(means)) or np.any(means <= 1e-8):
        return pixels.copy(), 1.0
    gains = (means[1] / means).astype(np.float32)
    return pixels * gains.reshape(1, 1, 3), float(gains.min())


def tonemap_preview(pixels: np.ndarray, metadata: ImageMetadata | None) -> np.ndarray:
    """White balance + camera->sRGB + auto exposure + sRGB encode.

    Input is float32 (H, W, 3) scene-linear camera RGB and is never mutated;
    output is float32 in [0, 1].
    """
    rgb, saturation = _white_balance(
        np.asarray(pixels, dtype=np.float32),
        metadata.as_shot_neutral if metadata is not None else None,
    )
    rgb = np.minimum(rgb, saturation)
    matrix = camera_to_srgb(metadata.color_matrix1 if metadata is not None else None)
    if matrix is not None:
        rgb = rgb @ matrix.T.astype(np.float32)
    rgb *= auto_exposure_gain(rgb)
    return srgb_encode(np.clip(rgb, 0.0, 1.0))
