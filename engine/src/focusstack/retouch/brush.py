"""Deterministic brush stroke rasterization."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EMPTY_BBOX = (0, 0, 0, 0)


@dataclass
class Stroke:
    source_id: str
    points: list[tuple[float, float, float]]
    radius: float
    hardness: float = 0.5
    opacity: float = 1.0
    mode: str = "normal"


def _points(points) -> list[tuple[float, float, float]]:
    return [
        (float(point[0]), float(point[1]), float(point[2]) if len(point) >= 3 else 1.0)
        for point in points
    ]


def _dabs(points: list[tuple[float, float, float]], radius: float):
    if not points:
        return []
    spacing = max(1, round(radius / 4))
    dabs = [points[0]]
    for (x0, y0, p0), (x1, y1, p1) in zip(points, points[1:]):
        count = max(1, int(math.hypot(x1 - x0, y1 - y0) / spacing))
        for i in range(1, count + 1):
            t = i / count
            dabs.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, p0 + (p1 - p0) * t))
    return dabs


def stroke_mask(h: int, w: int, stroke: Stroke):
    """Return a cropped float32 mask and its image-space bbox."""
    if round(stroke.radius) < 1:
        return np.zeros((0, 0), np.float32), EMPTY_BBOX

    dabs = _dabs(_points(stroke.points), stroke.radius)
    if not dabs:
        return np.zeros((0, 0), np.float32), EMPTY_BBOX

    reach = math.ceil(stroke.radius)
    x0 = max(0, math.floor(min(x for x, _, _ in dabs) - reach))
    y0 = max(0, math.floor(min(y for _, y, _ in dabs) - reach))
    x1 = min(w, math.ceil(max(x for x, _, _ in dabs) + reach) + 1)
    y1 = min(h, math.ceil(max(y for _, y, _ in dabs) + reach) + 1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0), np.float32), EMPTY_BBOX

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    mask = np.zeros((y1 - y0, x1 - x0), np.float32)
    sigma = stroke.radius * (1.0 - stroke.hardness) + 1e-3
    inv = 1.0 / (2.0 * sigma * sigma)
    for cx, cy, pressure in dabs:
        dab = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) * inv).astype(np.float32)
        np.maximum(mask, dab * np.float32(pressure), out=mask)
    np.clip(mask, 0.0, 1.0, out=mask)
    return mask, (y0, x0, y1, x1)
