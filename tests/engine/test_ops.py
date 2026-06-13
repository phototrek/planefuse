import numpy as np
import pytest
import torch

from focusstack.backend import get_device, ops


def _rand_img(c=3, h=64, w=80, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.uniform(0, 1, (c, h, w)).astype(np.float32))


def test_to_tensor_roundtrip():
    arr = np.random.default_rng(0).uniform(0, 1, (40, 50, 3)).astype(np.float32)
    t = ops.to_tensor(arr, get_device("cpu"))
    assert t.shape == (3, 40, 50)
    assert t.dtype == torch.float32
    back = ops.to_numpy(t)
    assert back.shape == (40, 50, 3)
    np.testing.assert_allclose(back, arr, atol=1e-7)


def test_gaussian_blur_preserves_mean_and_shape():
    img = _rand_img()
    out = ops.gaussian_blur(img, sigma=2.0)
    assert out.shape == img.shape
    assert abs(out.mean().item() - img.mean().item()) < 1e-2


def test_gaussian_blur_reduces_variance():
    img = _rand_img()
    out = ops.gaussian_blur(img, sigma=2.0)
    assert out.var().item() < img.var().item()


def test_downsample2_halves_dims():
    img = _rand_img(h=64, w=80)
    out = ops.downsample2(img)
    assert out.shape == (3, 32, 40)


def test_upsample_to_exact_shape():
    img = _rand_img(h=32, w=40)
    out = ops.upsample_to(img, (63, 81))
    assert out.shape == (3, 63, 81)


def test_box_filter3_constant_invariant():
    img = torch.full((3, 16, 16), 0.5)
    out = ops.box_filter3(img)
    assert torch.allclose(out, img, atol=1e-6)


def test_median_filter2d_removes_speckle():
    x = torch.zeros(21, 21, dtype=torch.int32)
    x[10, 10] = 7  # lone speckle
    out = ops.median_filter2d(x, radius=1)
    assert out.dtype == torch.int32
    assert out[10, 10] == 0


def test_median_filter2d_preserves_constant_regions():
    x = torch.full((16, 16), 3, dtype=torch.int32)
    out = ops.median_filter2d(x, radius=2)
    assert torch.equal(out, x)


# --- device parity (SPEC §13.2): compares each available accelerator to CPU ---


@pytest.mark.parametrize(
    "opname,kwargs",
    [
        ("gaussian_blur", {"sigma": 1.5}),
        ("downsample2", {}),
        ("box_filter3", {}),
    ],
)
def test_op_parity(accel_device, opname, kwargs):
    img = _rand_img(h=96, w=112, seed=42)
    fn = getattr(ops, opname)
    cpu_out = fn(img, **kwargs)
    dev_out = fn(img.to(accel_device.torch_device), **kwargs)
    torch.testing.assert_close(dev_out.cpu(), cpu_out, atol=1e-3, rtol=1e-3)


def test_upsample_parity(accel_device):
    img = _rand_img(h=48, w=56, seed=7)
    cpu_out = ops.upsample_to(img, (96, 112))
    dev_out = ops.upsample_to(img.to(accel_device.torch_device), (96, 112))
    torch.testing.assert_close(dev_out.cpu(), cpu_out, atol=1e-3, rtol=1e-3)
