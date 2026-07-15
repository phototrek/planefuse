import numpy as np
import pytest
import torch

from focusstack.backend import ops


def _rand_img(c=3, h=64, w=80, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.uniform(0, 1, (c, h, w)).astype(np.float32))


def test_rgb_to_luminance_shape_and_range():
    img = _rand_img()
    lum = ops.rgb_to_luminance(img)
    assert lum.shape == (1, 64, 80)
    assert 0.0 <= float(lum.min()) and float(lum.max()) <= 1.0


def test_rgb_to_luminance_grayscale_is_identity_value():
    img = torch.full((3, 8, 8), 0.4)
    lum = ops.rgb_to_luminance(img)
    torch.testing.assert_close(lum, torch.full((1, 8, 8), 0.4), atol=1e-6, rtol=1e-6)


def test_warp_identity_is_noop():
    img = _rand_img()
    m = torch.eye(3)
    out = ops.warp(img, m, out_shape=(64, 80), interp="bilinear")
    torch.testing.assert_close(out, img, atol=1e-4, rtol=1e-4)


def test_px_to_norm_preserves_float32():
    matrix = torch.eye(3, dtype=torch.float32)
    theta = ops._px_to_norm(matrix, 64, 80, 64, 80)
    assert theta.dtype == torch.float32


def test_warp_integer_translation_shifts_pixels():
    img = _rand_img(h=40, w=40, seed=1)
    m = torch.tensor([[1.0, 0.0, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    out = ops.warp(img, m, out_shape=(40, 40), interp="bilinear")
    torch.testing.assert_close(out[:, 5:35, 0:30], img[:, 5:35, 5:35], atol=1e-3, rtol=1e-3)


def _lanczos3(value: float) -> float:
    if abs(value) >= 3.0:
        return 0.0
    return float(np.sinc(value) * np.sinc(value / 3.0))


def _reference_lanczos3(img: np.ndarray, matrix: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    channels, height, width = img.shape
    out = np.empty((channels, *out_shape), dtype=np.float32)
    for row in range(out_shape[0]):
        for col in range(out_shape[1]):
            source = matrix @ np.array([col, row, 1.0])
            x, y = source[0] / source[2], source[1] / source[2]
            x_base, y_base = int(np.floor(x)), int(np.floor(y))
            x_indices = np.arange(x_base - 2, x_base + 4)
            y_indices = np.arange(y_base - 2, y_base + 4)
            x_weights = np.array([_lanczos3(x - ix) for ix in x_indices])
            y_weights = np.array([_lanczos3(y - iy) for iy in y_indices])
            x_weights /= x_weights.sum()
            y_weights /= y_weights.sum()
            x_indices = np.clip(x_indices, 0, width - 1)
            y_indices = np.clip(y_indices, 0, height - 1)
            patch = img[:, y_indices[:, None], x_indices[None, :]]
            out[:, row, col] = np.sum(patch * y_weights[:, None] * x_weights[None, :], axis=(1, 2))
    return out


def test_lanczos3_matches_six_tap_reference_on_fractional_shift():
    img = torch.zeros((1, 11, 13), dtype=torch.float32)
    img[0, 5, 6] = 1.0
    matrix = torch.tensor([[1.0, 0.0, 0.35], [0.0, 1.0, -0.4], [0.0, 0.0, 1.0]])
    expected = _reference_lanczos3(img.numpy(), matrix.numpy(), (11, 13))
    actual = ops.warp(img, matrix, out_shape=(11, 13), interp="lanczos3")
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-6, rtol=2e-6)


def test_lanczos3_edge_extension_preserves_constant_image():
    img = torch.full((3, 17, 19), 0.37)
    matrix = torch.tensor([[1.0, 0.0, 4.4], [0.0, 1.0, -3.7], [0.0, 0.0, 1.0]])
    actual = ops.warp(img, matrix, out_shape=(17, 19), interp="lanczos3")
    torch.testing.assert_close(actual, img, atol=2e-6, rtol=2e-6)


def test_fft_ifft_roundtrip():
    img = _rand_img(c=1, h=32, w=48, seed=2)
    spec = ops.fft2(img)
    back = ops.ifft2(spec)
    torch.testing.assert_close(back.real, img, atol=1e-4, rtol=1e-4)


def test_hann_window_2d_peaks_center():
    win = ops.hann_window_2d(16, 20, device=torch.device("cpu"))
    assert win.shape == (16, 20)
    assert float(win[8, 10]) == pytest.approx(float(win.max()), abs=1e-3)
    assert float(win[0, 0]) < 1e-6


def test_phase_correlation_recovers_known_shift():
    rng = np.random.default_rng(3)
    base = torch.from_numpy(rng.uniform(0, 1, (1, 64, 64)).astype(np.float32))
    shifted = torch.roll(base, shifts=(4, -7), dims=(1, 2))  # (dy=+4, dx=-7)
    dy, dx, peak = ops.phase_correlation(base, shifted)
    assert round(dy) == 4
    assert round(dx) == -7
    assert peak > 0.3


def test_log_polar_remap_shape():
    img = _rand_img(c=1, h=64, w=64, seed=4)
    out = ops.log_polar_remap(img, n_angles=64, n_radii=64)
    assert out.shape == (1, 64, 64)


@pytest.mark.parametrize("opname,kwargs", [
    ("rgb_to_luminance", {}),
    ("hann_apply", {}),
])
def test_align_op_parity(accel_device, opname, kwargs):
    img = _rand_img(h=72, w=88, seed=9)
    fn = getattr(ops, opname)
    cpu_out = fn(img, **kwargs)
    dev_out = fn(img.to(accel_device.torch_device), **kwargs)
    torch.testing.assert_close(dev_out.cpu(), cpu_out, atol=1e-3, rtol=1e-3)


def test_warp_parity(accel_device):
    img = _rand_img(h=72, w=88, seed=10)
    m = torch.tensor([[1.02, -0.01, 1.5], [0.01, 1.02, -2.0], [0.0, 0.0, 1.0]])
    cpu_out = ops.warp(img, m, out_shape=(72, 88), interp="bilinear")
    dev_out = ops.warp(img.to(accel_device.torch_device), m.to(accel_device.torch_device),
                       out_shape=(72, 88), interp="bilinear")
    torch.testing.assert_close(dev_out.cpu(), cpu_out, atol=2e-3, rtol=2e-3)


def test_lanczos3_warp_parity(accel_device):
    img = _rand_img(h=36, w=44, seed=12)
    matrix = torch.tensor([[1.01, -0.015, 0.75], [0.015, 1.01, -1.2], [0.0, 0.0, 1.0]])
    cpu_out = ops.warp(img, matrix, out_shape=(36, 44), interp="lanczos3")
    dev_out = ops.warp(
        img.to(accel_device.torch_device),
        matrix.to(accel_device.torch_device),
        out_shape=(36, 44),
        interp="lanczos3",
    )
    torch.testing.assert_close(dev_out.cpu(), cpu_out, atol=3e-4, rtol=3e-4)


def test_phase_correlation_parity(accel_device):
    rng = np.random.default_rng(11)
    base = torch.from_numpy(rng.uniform(0, 1, (1, 64, 64)).astype(np.float32))
    shifted = torch.roll(base, shifts=(3, 5), dims=(1, 2))
    dy_c, dx_c, _ = ops.phase_correlation(base, shifted)
    dy_d, dx_d, _ = ops.phase_correlation(base.to(accel_device.torch_device),
                                          shifted.to(accel_device.torch_device))
    assert round(dy_c) == round(dy_d) and round(dx_c) == round(dx_d)
