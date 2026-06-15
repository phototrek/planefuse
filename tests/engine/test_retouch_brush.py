import numpy as np

from focusstack.retouch.brush import Stroke, stroke_mask


def _stroke(**kw):
    base = {
        "source_id": "s",
        "points": [(50.0, 40.0, 1.0)],
        "radius": 10.0,
        "hardness": 0.5,
        "opacity": 1.0,
        "mode": "normal",
    }
    base.update(kw)
    return Stroke(**base)


def test_mask_shape_matches_bbox():
    mask, (y0, x0, y1, x1) = stroke_mask(100, 120, _stroke())
    assert mask.shape == (y1 - y0, x1 - x0)
    assert mask.dtype == np.float32
    assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0


def test_mask_peaks_at_center():
    mask, (y0, x0, _y1, _x1) = stroke_mask(100, 120, _stroke())
    assert mask[40 - y0, 50 - x0] > 0.9


def test_bbox_is_smaller_than_image():
    mask, (y0, x0, y1, x1) = stroke_mask(100, 120, _stroke(radius=8.0, hardness=0.9))
    assert mask.shape == (y1 - y0, x1 - x0)
    assert (x1 - x0) < 120 and (y1 - y0) < 100


def test_deterministic():
    a, bbox_a = stroke_mask(100, 120, _stroke())
    b, bbox_b = stroke_mask(100, 120, _stroke())
    assert bbox_a == bbox_b
    assert np.array_equal(a, b)


def test_offscreen_stroke_is_empty():
    mask, bbox = stroke_mask(100, 120, _stroke(points=[(-50.0, -50.0, 1.0)], radius=5.0))
    assert bbox == (0, 0, 0, 0)
    assert mask.size == 0


def test_subpixel_radius_is_empty():
    mask, bbox = stroke_mask(100, 120, _stroke(radius=0.4))
    assert bbox == (0, 0, 0, 0)
    assert mask.size == 0


def test_pressure_scales_strength():
    full, _ = stroke_mask(100, 120, _stroke(points=[(50.0, 40.0, 1.0)]))
    half, _ = stroke_mask(100, 120, _stroke(points=[(50.0, 40.0, 0.5)]))
    assert float(half.max()) < float(full.max())


def test_hardness_changes_falloff():
    soft, bbox = stroke_mask(100, 120, _stroke(hardness=0.0))
    hard, hard_bbox = stroke_mask(100, 120, _stroke(hardness=0.9))
    assert hard_bbox == bbox
    y0, x0, _y1, _x1 = bbox
    assert soft[40 - y0, 55 - x0] > hard[40 - y0, 55 - x0]
