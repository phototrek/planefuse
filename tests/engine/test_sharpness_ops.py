import numpy as np
import pytest
import torch

from planefuse.backend import ops


def _rand(c=1, h=64, w=80, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.uniform(0, 1, (c, h, w)).astype(np.float32))


def test_box_filter_radius0_is_identity():
    img = _rand(c=3)
    torch.testing.assert_close(ops.box_filter(img, radius=0), img, atol=1e-6, rtol=1e-6)


def test_box_filter_constant_invariant():
    img = torch.full((1, 20, 20), 0.3)
    torch.testing.assert_close(ops.box_filter(img, radius=4), img, atol=1e-5, rtol=1e-5)


def test_sharpness_map_higher_on_texture():
    flat = torch.full((1, 48, 48), 0.5)
    rng = np.random.default_rng(1)
    textured = torch.from_numpy(rng.uniform(0, 1, (1, 48, 48)).astype(np.float32))
    s_flat = ops.sharpness_map(flat, radius=8)
    s_tex = ops.sharpness_map(textured, radius=8)
    assert float(s_tex.mean()) > 10 * float(s_flat.mean()) + 1e-6
    assert s_flat.shape == (1, 48, 48)


def test_guided_filter_smooths_but_follows_guide_edge():
    guide = torch.zeros(1, 32, 32)
    guide[:, :, 16:] = 1.0
    rng = np.random.default_rng(2)
    src = guide + torch.from_numpy(rng.normal(0, 0.1, (1, 32, 32)).astype(np.float32))
    out = ops.guided_filter(guide, src, radius=4, eps=1e-3)
    assert out.shape == src.shape
    assert out[:, :, :16].var() < src[:, :, :16].var()
    assert float(out[:, :, 15].mean()) < float(out[:, :, 16].mean())


def test_masked_diffuse_fills_unknown_from_known():
    val = torch.zeros(1, 16, 16)
    val[:, :, :8] = 2.0
    known = torch.zeros(1, 16, 16, dtype=torch.bool)
    known[:, :, :8] = True
    filled = ops.masked_diffuse(val, known, radius=2, iters=50)
    assert float(filled[:, :, 8:].min()) > 1.0
    torch.testing.assert_close(filled[:, :, :8], val[:, :, :8], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("opname,kwargs", [
    ("box_filter", {"radius": 3}),
    ("sharpness_map", {"radius": 6}),
])
def test_sharpness_op_parity(accel_device, opname, kwargs):
    img = _rand(c=1, h=72, w=88, seed=9)
    fn = getattr(ops, opname)
    cpu_out = fn(img, **kwargs)
    dev_out = fn(img.to(accel_device.torch_device), **kwargs)
    torch.testing.assert_close(dev_out.cpu(), cpu_out, atol=1e-3, rtol=1e-3)


def test_guided_filter_parity(accel_device):
    guide = _rand(c=1, h=64, w=72, seed=3)
    src = _rand(c=1, h=64, w=72, seed=4)
    cpu_out = ops.guided_filter(guide, src, radius=4, eps=1e-3)
    dev_out = ops.guided_filter(guide.to(accel_device.torch_device),
                                src.to(accel_device.torch_device), radius=4, eps=1e-3)
    torch.testing.assert_close(dev_out.cpu(), cpu_out, atol=2e-3, rtol=2e-3)
