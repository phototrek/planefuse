import numpy as np

from planefuse.align.chain import chain_to_reference


def test_chain_reference_is_identity():
    n = 5
    pair = {i: np.eye(3) for i in range(n - 1)}  # transform from i+1 -> i
    corr = {i: 0.99 for i in range(n - 1)}
    result = chain_to_reference(n, pair, corr, ref=2, threshold=0.9)
    for m in result.matrices:
        np.testing.assert_allclose(m, np.eye(3), atol=1e-9)
    assert result.reference == 2
    assert not result.dropped


def test_chain_composes_translations_outward_from_reference():
    n = 4
    t = np.array([[1, 0, 1.0], [0, 1, 0], [0, 0, 1]])
    pair = {i: t for i in range(n - 1)}
    corr = {i: 0.99 for i in range(n - 1)}
    result = chain_to_reference(n, pair, corr, ref=0, threshold=0.9)
    np.testing.assert_allclose(result.matrices[0], np.eye(3), atol=1e-9)
    assert result.matrices[3][0, 2] == 3.0


def test_chain_composition_order_is_apply_closest_link_first():
    # Non-commutative links: pair[0]=translate, pair[1]=scale-about-origin.
    # frame 2 aligns to ref=0 by applying pair[1] (closest to frame 2) first,
    # then pair[0]; as matrices that is pair[1] @ pair[0]. The reversed order
    # would give a different result, so this pins the composition direction
    # (pure-translation tests cannot, since translations commute).
    a = np.array([[1, 0, 5.0], [0, 1, 0], [0, 0, 1]])   # pair[0]: translate x+5
    b = np.array([[2, 0, 0.0], [0, 2, 0], [0, 0, 1]])   # pair[1]: scale x2 about origin
    pair = {0: a, 1: b}
    corr = {0: 0.99, 1: 0.99}
    result = chain_to_reference(3, pair, corr, ref=0, threshold=0.9)
    np.testing.assert_allclose(result.matrices[1], a, atol=1e-9)
    np.testing.assert_allclose(result.matrices[2], b @ a, atol=1e-9)        # correct order
    assert not np.allclose(result.matrices[2], a @ b)                       # not the reverse


def test_chain_flags_low_correlation():
    n = 4
    pair = {i: np.eye(3) for i in range(n - 1)}
    corr = {0: 0.99, 1: 0.5, 2: 0.99}  # pair 1->2 is bad
    result = chain_to_reference(n, pair, corr, ref=0, threshold=0.9)
    assert 2 in result.flagged or 1 in result.flagged
