from pathlib import Path

import numpy as np

from planefuse.io import srgb_encode, tonemap_preview
from planefuse.io.display import SRGB_TO_XYZ_D65, auto_exposure_gain, camera_to_srgb
from planefuse.io.metadata import ImageMetadata

IDENTITY9 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def test_srgb_encode_endpoints_and_monotonic():
    values = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    encoded = srgb_encode(values)
    assert encoded[0] == 0.0
    np.testing.assert_allclose(encoded[-1], 1.0, atol=1e-6)
    assert np.all(np.diff(encoded) > 0)
    np.testing.assert_allclose(srgb_encode(np.float32(0.001)), 0.001 * 12.92, rtol=1e-6)


def test_camera_to_srgb_rejects_degenerate_matrices():
    assert camera_to_srgb(None) is None
    assert camera_to_srgb((1.0, 2.0, 3.0)) is None
    assert camera_to_srgb((0.0,) * 9) is None
    singular = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    assert camera_to_srgb(singular) is None


def test_camera_to_srgb_inverts_row_normalized_camera_matrix():
    matrix = camera_to_srgb(IDENTITY9)
    assert matrix is not None
    cam_from_srgb = SRGB_TO_XYZ_D65 / SRGB_TO_XYZ_D65.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(matrix @ cam_from_srgb, np.eye(3), atol=1e-10)
    # White is preserved: camera (1,1,1) maps to sRGB (1,1,1).
    np.testing.assert_allclose(matrix @ np.ones(3), np.ones(3), atol=1e-10)


def _metadata(**overrides) -> ImageMetadata:
    values = {
        "source_path": Path("frame.dng"),
        "color_matrix1": IDENTITY9,
        "as_shot_neutral": (0.5, 1.0, 0.4),
    }
    values.update(overrides)
    return ImageMetadata(**values)


def test_tonemap_preview_neutralizes_as_shot_neutral_pixels():
    metadata = _metadata()
    pixels = np.empty((8, 9, 3), dtype=np.float32)
    pixels[:] = np.asarray(metadata.as_shot_neutral, dtype=np.float32) * 0.02
    before = pixels.copy()
    out = tonemap_preview(pixels, metadata)
    np.testing.assert_array_equal(pixels, before)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 1.0))
    np.testing.assert_allclose(out[..., 0], out[..., 1], atol=1e-4)
    np.testing.assert_allclose(out[..., 2], out[..., 1], atol=1e-4)


def test_tonemap_preview_renders_blown_highlights_neutral_not_magenta():
    # Sensor-clipped pixels read (1,1,1); unequal WB gains must not tint them.
    metadata = _metadata()
    pixels = np.empty((8, 8, 3), dtype=np.float32)
    pixels[:] = np.asarray(metadata.as_shot_neutral, dtype=np.float32) * 0.05
    pixels[:4, :4] = 1.0
    out = tonemap_preview(pixels, metadata)
    highlight = out[0, 0]
    assert float(highlight.max() - highlight.min()) < 0.02
    assert float(highlight.min()) > 0.8


def test_tonemap_preview_brightens_dark_frames_without_metadata():
    pixels = np.full((6, 6, 3), 0.01, dtype=np.float32)
    out = tonemap_preview(pixels, None)
    assert float(out.mean()) > float(srgb_encode(pixels).mean())
    np.testing.assert_array_equal(out, tonemap_preview(pixels, None))


def test_tonemap_preview_is_monotonic_in_exposure():
    ramp = np.linspace(0.001, 0.05, 24, dtype=np.float32).reshape(4, 6)
    pixels = np.stack([ramp, ramp, ramp], axis=-1)
    out = tonemap_preview(pixels, None)
    luma = out.mean(axis=-1).reshape(-1)
    assert np.all(np.diff(luma) >= 0)


def test_tonemap_preview_recovers_scene_hues_through_camera_model():
    # Simulate a camera (Canon 5D3 ColorMatrix1, warm as-shot cast) capturing
    # known sRGB patches; the preview transform must recover their hues.
    canon_cm = (0.6722, -0.0635, -0.0963, -0.4287, 1.246, 0.2028, -0.0908, 0.2162, 0.5668)
    cam_from_srgb = np.asarray(canon_cm).reshape(3, 3) @ SRGB_TO_XYZ_D65
    cam_from_srgb /= cam_from_srgb.sum(axis=1, keepdims=True)
    neutral = (0.45, 1.0, 0.62)
    scene = np.array([
        [0.4, 0.02, 0.02],  # red
        [0.02, 0.4, 0.02],  # green
        [0.02, 0.02, 0.4],  # blue
        [0.2, 0.2, 0.2],    # gray
    ])
    cam = (scene @ cam_from_srgb.T) * np.asarray(neutral)
    pixels = np.tile(cam.astype(np.float32).reshape(4, 1, 3), (1, 8, 1))
    metadata = _metadata(color_matrix1=canon_cm, as_shot_neutral=neutral)
    out = tonemap_preview(pixels, metadata)
    red, green, blue, gray = (out[row, 0] for row in range(4))
    assert int(red.argmax()) == 0
    assert int(green.argmax()) == 1
    assert int(blue.argmax()) == 2
    np.testing.assert_allclose(gray, float(gray.mean()), atol=0.02)


def test_auto_exposure_gain_is_clamped():
    assert auto_exposure_gain(np.full((4, 4, 3), 1e-9, dtype=np.float32)) == 64.0
    assert auto_exposure_gain(np.full((4, 4, 3), 100.0, dtype=np.float32)) == 0.25
    assert auto_exposure_gain(np.zeros((4, 4, 3), dtype=np.float32)) == 1.0
