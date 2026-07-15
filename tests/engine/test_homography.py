import cv2
import numpy as np
import torch

from focusstack.backend import ops
from focusstack.align.refine import refine_ecc


def test_projective_warp_matches_opencv_inverse_map():
    rng = np.random.default_rng(123)
    image = rng.uniform(0, 1, (48, 64, 3)).astype(np.float32)
    matrix = np.array(
        [[1.01, -0.015, 1.2], [0.01, 0.99, -0.8], [0.00035, -0.0002, 1.0]],
        dtype=np.float32,
    )
    expected = cv2.warpPerspective(
        image,
        matrix,
        (64, 48),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    actual = ops.warp(
        torch.from_numpy(image).permute(2, 0, 1),
        torch.from_numpy(matrix),
        out_shape=(48, 64),
        interp="bilinear",
    ).permute(1, 2, 0).numpy()
    np.testing.assert_allclose(actual, expected, atol=2.5e-2, rtol=2.5e-2)


def test_perspective_ecc_retains_projective_terms():
    rng = np.random.default_rng(9)
    sharp = rng.uniform(0, 1, (160, 192)).astype(np.float32)
    reference = cv2.GaussianBlur(sharp, (0, 0), 1.2)
    expected = np.array(
        [[1.002, -0.004, 1.5], [0.003, 0.998, -1.1], [0.00012, -0.00009, 1.0]],
        dtype=np.float64,
    )
    moving = cv2.warpPerspective(
        reference,
        np.linalg.inv(expected),
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )

    recovered, correlation = refine_ecc(
        reference,
        moving,
        init=expected.copy(),
        cx=reference.shape[1] / 2,
        cy=reference.shape[0] / 2,
        model="perspective",
    )

    assert correlation > 0.98
    assert abs(recovered[2, 0]) > 1e-5
    assert abs(recovered[2, 1]) > 1e-5
    np.testing.assert_allclose(recovered / recovered[2, 2], expected, atol=2e-2, rtol=1e-2)
