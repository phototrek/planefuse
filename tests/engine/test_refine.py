import numpy as np

from planefuse.backend import get_device
from planefuse.align.proxy import make_proxy
from planefuse.align import refine as refine_module
from planefuse.align.refine import brightness_gain, refine_ecc
from tests.synthetic.generate import generate_stack


def _proxy_np(frame):
    p, _ = make_proxy(frame, get_device("cpu"), max_long_edge=256)
    return p.squeeze(0).cpu().numpy()


def test_refine_ecc_improves_toward_identity_pair():
    # two near-identical frames -> ECC returns ~identity, corr ~1
    stack = generate_stack(h=200, w=240, n_frames=2, max_sigma=0.3, seed=7)
    a, b = _proxy_np(stack.frames[0]), _proxy_np(stack.frames[1])
    m, corr = refine_ecc(a, b, init=np.eye(3), cx=a.shape[1] / 2, cy=a.shape[0] / 2)
    assert corr > 0.9
    assert abs(m[0, 0] - 1.0) < 0.05 and abs(m[1, 1] - 1.0) < 0.05


def test_refine_ecc_returns_similarity():
    stack = generate_stack(h=200, w=240, n_frames=2, max_sigma=0.3, seed=8,
                           scale_step=0.02)
    a, b = _proxy_np(stack.frames[0]), _proxy_np(stack.frames[1])
    m, corr = refine_ecc(a, b, init=np.eye(3), cx=a.shape[1] / 2, cy=a.shape[0] / 2)
    block = m[:2, :2]
    np.testing.assert_allclose(block[:, 0] @ block[:, 1], 0.0, atol=1e-4)  # orthogonal cols


def test_refine_ecc_rejects_unknown_model():
    a = np.zeros((32, 32), dtype=np.float32)
    with np.testing.assert_raises_regex(ValueError, "model"):
        refine_ecc(a, a, init=np.eye(3), cx=16, cy=16, model="elastic")


def test_refine_ecc_retries_reverse_after_false_scale_basin(monkeypatch):
    image = np.zeros((64, 80), dtype=np.float32)
    false_forward = np.array(
        [[0.90, 0.0, 4.0], [0.0, 0.90, 3.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    expected = np.array(
        [[0.997, 0.0, 1.25], [0.0, 0.997, -0.75], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    results = iter([(false_forward, 0.99), (np.linalg.inv(expected), 0.96)])

    def fake_run_ecc_pyramid(*_args):
        return next(results)

    monkeypatch.setattr(refine_module, "_run_ecc_pyramid", fake_run_ecc_pyramid)
    recovered, correlation = refine_ecc(
        image,
        image,
        init=np.eye(3),
        cx=image.shape[1] / 2,
        cy=image.shape[0] / 2,
    )

    np.testing.assert_allclose(recovered, expected, atol=1e-9)
    assert correlation == 0.96


def test_refine_ecc_retries_without_pyramid_when_coarse_levels_fail(monkeypatch):
    image = np.zeros((64, 80), dtype=np.float32)
    false_forward = np.array(
        [[0.90, 0.0, 4.0], [0.0, 0.90, 3.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    false_reverse = np.array(
        [[1.20, 0.0, -8.0], [0.0, 1.20, -6.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    expected = np.array(
        [[0.997, 0.0, 1.25], [0.0, 0.997, -0.75], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    results = iter([
        (false_forward, 0.99),
        (false_reverse, 0.10),
        (expected, 0.97),
    ])

    def fake_run_ecc_pyramid(*_args):
        return next(results)

    monkeypatch.setattr(refine_module, "_run_ecc_pyramid", fake_run_ecc_pyramid)
    recovered, correlation = refine_ecc(
        image,
        image,
        init=np.eye(3),
        cx=image.shape[1] / 2,
        cy=image.shape[0] / 2,
    )

    np.testing.assert_allclose(recovered, expected, atol=1e-9)
    assert correlation == 0.97


def test_brightness_gain_matches_mean():
    a = np.full((50, 50), 0.4, dtype=np.float32)
    b = np.full((50, 50), 0.2, dtype=np.float32)
    mask = np.ones((50, 50), dtype=bool)
    gain = brightness_gain(a, b, mask)
    np.testing.assert_allclose(gain, 2.0, atol=1e-3)  # b * gain matches a
