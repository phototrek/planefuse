import numpy as np

from planefuse.align.proxy import make_proxy
from planefuse.backend import get_device
from planefuse.io import ProcessingDomain
from planefuse.pipeline import stack_frames
from planefuse.select.focus_measure import compute_focus_measures
from planefuse.stack.sources import ArrayFrameSource
from tests.engine.test_raw_loader import write_test_raw


def test_raw_pipeline_retains_scene_linear_domain_and_reference_metadata(tmp_path):
    paths = [write_test_raw(tmp_path / f"frame-{index}.dng") for index in range(2)]
    result = stack_frames(
        paths,
        method="weighted",
        params={"temperature": 0.05, "sharpness_radius": 2},
        device_pref="cpu",
    )
    assert result.domain is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB
    assert result.metadata is not None
    assert result.metadata.unique_camera_model == "PlaneFuse Camera Co SameCam Pro"
    assert result.metadata.decoder["white_balance"] == [1.0, 1.0, 1.0, 1.0]
    assert result.image.max() < 1.0


def test_scene_linear_analysis_proxy_is_neutral_and_exposure_scale_invariant():
    rng = np.random.default_rng(14)
    frame = rng.uniform(0.03, 0.8, (40, 52, 3)).astype(np.float32)
    device = get_device("cpu")
    first, _ = make_proxy(frame, device, normalize_scene_linear=True)
    second, _ = make_proxy(frame * 3.7, device, normalize_scene_linear=True)
    np.testing.assert_allclose(first.numpy(), second.numpy(), atol=2e-6)
    expected_neutral = frame.mean(axis=2)
    lo, hi = np.percentile(expected_neutral, (0.5, 99.5))
    expected = np.clip((expected_neutral - lo) / (hi - lo), 0.0, 1.0)
    np.testing.assert_allclose(first.numpy()[0], expected, atol=2e-6)


def test_scene_linear_focus_measure_is_exposure_scale_invariant():
    rng = np.random.default_rng(15)
    frames = [rng.uniform(0.02, 0.9, (24, 30, 3)).astype(np.float32) for _ in range(2)]
    first = ArrayFrameSource(frames, domain=ProcessingDomain.SCENE_LINEAR_CAMERA_RGB)
    second = ArrayFrameSource(
        [frame * 5.0 for frame in frames],
        domain=ProcessingDomain.SCENE_LINEAR_CAMERA_RGB,
    )
    phi_a, _ = compute_focus_measures(first, get_device("cpu"), grid_rows=3, grid_cols=5)
    phi_b, _ = compute_focus_measures(second, get_device("cpu"), grid_rows=3, grid_cols=5)
    np.testing.assert_allclose(phi_a, phi_b, atol=2e-6)
