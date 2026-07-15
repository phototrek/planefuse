from pathlib import Path

import numpy as np

from focusstack.io import ProcessingDomain, load_raw, save_image, save_linear_dng
from focusstack.pipeline import stack_frames
from tests.engine.test_raw_loader import write_test_raw


def test_rendered_weighted_golden_has_analytical_result(tmp_path: Path):
    paths = []
    for index, value in enumerate((0.2, 0.6)):
        path = tmp_path / f"rendered-{index}.tif"
        save_image(np.full((16, 20, 3), value, dtype=np.float32), path, bit_depth=16)
        paths.append(path)
    result = stack_frames(
        paths,
        method="weighted",
        params={"temperature": 0.05, "sharpness_radius": 2},
        device_pref="cpu",
    )
    assert result.domain is ProcessingDomain.RENDERED_RGB
    np.testing.assert_allclose(result.image, 0.4, atol=2.0 / 65535.0)


def test_raw_golden_preserves_domain_recipe_pixels_and_dng_contract(tmp_path: Path):
    paths = [write_test_raw(tmp_path / f"raw-{index}.dng") for index in range(2)]
    decoded = load_raw(paths[0])
    result = stack_frames(
        paths,
        method="weighted",
        params={"temperature": 0.05, "sharpness_radius": 2},
        device_pref="cpu",
    )
    assert result.domain is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB
    assert result.metadata is not None
    assert result.metadata.decoder["gamma"] == [1.0, 1.0]
    assert result.metadata.decoder["auto_brightness"] is False
    np.testing.assert_allclose(result.image, decoded.pixels, atol=2e-6)
    report = save_linear_dng(
        result.image,
        tmp_path / "golden-stack.dng",
        result.metadata,
        result.provenance,
    )
    assert report.rawpy_validated
    assert report.max_code_error <= 1
