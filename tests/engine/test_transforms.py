import numpy as np

from focusstack.align.transforms import (
    compose,
    invert,
    project_to_similarity,
    scale_transform_to_resolution,
    similarity_matrix,
    translation_matrix,
)


def test_translation_matrix():
    m = translation_matrix(3.0, -2.0)  # (dx, dy)
    np.testing.assert_allclose(m @ np.array([0, 0, 1.0]), [3.0, -2.0, 1.0])


def test_similarity_about_center_identity():
    m = similarity_matrix(scale=1.0, angle=0.0, tx=0.0, ty=0.0, cx=10.0, cy=8.0)
    np.testing.assert_allclose(m, np.eye(3), atol=1e-9)


def test_compose_is_matrix_product():
    a = translation_matrix(2.0, 0.0)
    b = translation_matrix(0.0, 3.0)
    np.testing.assert_allclose(compose(a, b), a @ b)


def test_invert_roundtrip():
    m = similarity_matrix(1.05, 0.02, 1.5, -2.0, cx=5, cy=5)
    np.testing.assert_allclose(compose(m, invert(m)), np.eye(3), atol=1e-9)


def test_scale_transform_to_resolution():
    m_low = translation_matrix(2.0, -1.0)
    m_full = scale_transform_to_resolution(m_low, factor=4.0)
    np.testing.assert_allclose(m_full @ np.array([0, 0, 1.0]), [8.0, -4.0, 1.0])


def test_project_to_similarity_recovers_similarity():
    truth = similarity_matrix(1.03, 0.05, 4.0, -3.0, cx=32, cy=24)
    affine = truth.copy()
    affine[0, 1] += 0.004
    affine[1, 0] -= 0.002
    sim = project_to_similarity(affine, cx=32, cy=24)
    a = sim[:2, :2]
    np.testing.assert_allclose(a[:, 0] @ a[:, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(a[:, 0]), np.linalg.norm(a[:, 1]), atol=1e-6)
    np.testing.assert_allclose(sim, truth, atol=4e-2)
