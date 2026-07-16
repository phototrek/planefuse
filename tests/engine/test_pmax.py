import numpy as np
import pytest
from skimage.metrics import structural_similarity

from planefuse.backend import get_device
from planefuse.stack import ArrayFrameSource, get_algorithm
from tests.synthetic.generate import generate_stack


@pytest.fixture(scope="module")
def synth():
    return generate_stack(h=192, w=256, n_frames=10, max_sigma=5.0, seed=7)


def _ssim(a, b):
    return structural_similarity(a.clip(0, 1), b.clip(0, 1), channel_axis=2, data_range=1.0)


def test_pmax_registered_with_params():
    algo = get_algorithm("pmax")
    names = [p.name for p in algo.params()]
    assert "selection_smoothing" in names


def test_pmax_beats_every_input_frame(synth):
    algo = get_algorithm("pmax")
    res = algo.run(ArrayFrameSource(synth.frames), get_device("cpu"), {"selection_smoothing": 1})
    result_ssim = _ssim(res.image, synth.sharp)
    best_frame_ssim = max(_ssim(f, synth.sharp) for f in synth.frames)
    assert result_ssim > best_frame_ssim


def test_pmax_quality_gate(synth):
    # SPEC §13.2: SSIM > 0.97 vs ground-truth sharp image
    algo = get_algorithm("pmax")
    res = algo.run(ArrayFrameSource(synth.frames), get_device("cpu"), {"selection_smoothing": 1})
    assert _ssim(res.image, synth.sharp) > 0.97


def test_pmax_progress_and_cancel(synth):
    algo = get_algorithm("pmax")
    calls = []
    algo.run(
        ArrayFrameSource(synth.frames),
        get_device("cpu"),
        {"selection_smoothing": 0},
        progress=lambda msg, frac: calls.append(frac),
    )
    assert calls and calls[-1] == pytest.approx(1.0)
    with pytest.raises(InterruptedError):
        algo.run(
            ArrayFrameSource(synth.frames),
            get_device("cpu"),
            {"selection_smoothing": 0},
            cancel=lambda: True,
        )


def test_pmax_validity_mask_excludes_region(synth):
    """A frame whose mask invalidates a region must never win there."""
    frames = [f.copy() for f in synth.frames]
    frames[3][:, :, :] = 5.0  # absurd hot frame; would dominate energy everywhere
    masks = [np.ones(f.shape[:2], dtype=bool) for f in frames]
    masks[3][:, :] = False  # ...but it is fully invalid
    algo = get_algorithm("pmax")
    res = algo.run(
        ArrayFrameSource(frames),
        get_device("cpu"),
        {"selection_smoothing": 0},
        masks=ArrayMaskSource(masks),
    )
    assert res.image.max() < 2.0  # the hot frame never selected


def test_pmax_smoothing_changes_output(synth):
    algo = get_algorithm("pmax")
    src = ArrayFrameSource(synth.frames)
    a = algo.run(src, get_device("cpu"), {"selection_smoothing": 0}).image
    b = algo.run(src, get_device("cpu"), {"selection_smoothing": 2}).image
    assert not np.allclose(a, b)
    assert _ssim(b, synth.sharp) > 0.97  # smoothing must not wreck quality


def test_pmax_device_parity(accel_device, synth):
    algo = get_algorithm("pmax")
    src = ArrayFrameSource(synth.frames)
    cpu = algo.run(src, get_device("cpu"), {"selection_smoothing": 1}).image
    dev = algo.run(src, accel_device, {"selection_smoothing": 1}).image
    np.testing.assert_allclose(dev, cpu, atol=1e-3)


class ArrayMaskSource:
    def __init__(self, masks):
        self._masks = masks

    def __len__(self):
        return len(self._masks)

    def read(self, idx, region=None):
        m = self._masks[idx]
        if region is None:
            return m
        y0, x0, y1, x1 = region
        return m[y0:y1, x0:x1]
