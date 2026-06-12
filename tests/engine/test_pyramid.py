import numpy as np
import torch

from focusstack.backend import get_device, ops
from focusstack.stack.pyramid import build_laplacian, collapse_laplacian, pyramid_depth


def test_pyramid_depth_formula():
    # SPEC §7.1: depth = floor(log2(min(H,W))) - 5, coarsest level >= 32 px
    assert pyramid_depth(1024, 1536) == 5
    assert pyramid_depth(4000, 6000) == 6
    assert pyramid_depth(64, 64) == 1
    assert pyramid_depth(40, 40) == 1  # never below 1


def test_build_collapse_is_exact_identity():
    rng = np.random.default_rng(0)
    img = torch.from_numpy(rng.uniform(0, 1, (3, 96, 128)).astype(np.float32))
    depth = pyramid_depth(96, 128)
    lap, residual = build_laplacian(img, depth)
    assert len(lap) == depth
    rec = collapse_laplacian(lap, residual)
    torch.testing.assert_close(rec, img, atol=1e-5, rtol=1e-5)


def test_level_shapes_halve():
    img = torch.zeros(3, 96, 128)
    lap, residual = build_laplacian(img, 3)
    assert lap[0].shape == (3, 96, 128)
    assert lap[1].shape == (3, 48, 64)
    assert lap[2].shape == (3, 24, 32)
    assert residual.shape == (3, 12, 16)


def test_pyramid_parity(accel_device):
    rng = np.random.default_rng(5)
    img = torch.from_numpy(rng.uniform(0, 1, (3, 96, 128)).astype(np.float32))
    lap_c, res_c = build_laplacian(img, 3)
    lap_d, res_d = build_laplacian(img.to(accel_device.torch_device), 3)
    for lc, ld in zip(lap_c, lap_d):
        torch.testing.assert_close(ld.cpu(), lc, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(res_d.cpu(), res_c, atol=1e-3, rtol=1e-3)
