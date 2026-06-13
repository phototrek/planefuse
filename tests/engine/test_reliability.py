import numpy as np

from focusstack.select.reliability import excess_kurtosis, smooth_curves, classify_reliable


def test_smooth_curves_sums_neighbours():
    phi = np.zeros((2, 3, 3), dtype=np.float32)
    phi[:, 1, 1] = 1.0
    out = smooth_curves(phi)
    assert np.allclose(out[:, 1, 1], 1.0)
    assert np.allclose(out[:, 0, 0], 1.0)
    assert np.allclose(out[:, 0, 1], 1.0)


def test_excess_kurtosis_peaky_vs_flat():
    n = 21
    flat = np.ones(n, dtype=np.float64)
    peaky = np.zeros(n, dtype=np.float64)
    peaky[n // 2] = 1.0
    assert excess_kurtosis(peaky) > excess_kurtosis(flat)
    assert excess_kurtosis(flat) < 0.0


def test_classify_reliable_thresholds_and_respects_cell_mask():
    n = 15
    phi = np.zeros((n, 1, 2), dtype=np.float32)
    phi[n // 2, 0, 0] = 1.0          # peaky
    phi[:, 0, 1] = 1.0               # flat
    cell_reliable = np.ones((1, 2), dtype=bool)
    rel = classify_reliable(phi, cell_reliable, kurtosis_threshold=1.0)
    assert rel[0, 0] and not rel[0, 1]


def test_classify_respects_incoming_cell_mask():
    n = 15
    phi = np.zeros((n, 1, 1), dtype=np.float32)
    phi[n // 2, 0, 0] = 1.0
    rel = classify_reliable(phi, np.zeros((1, 1), bool), kurtosis_threshold=1.0)
    assert not rel[0, 0]
