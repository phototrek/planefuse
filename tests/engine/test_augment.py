import numpy as np

from planefuse.select.intervals import build_rows


def test_build_rows_one_per_reliable_cell():
    n = 5
    phi = np.zeros((n, 1, 2), dtype=np.float32)
    phi[1, 0, 0] = 1.0   # cell0 peaks at frame 1
    phi[3, 0, 1] = 1.0   # cell1 peaks at frame 3
    reliable = np.ones((1, 2), dtype=bool)
    rows = build_rows(phi, reliable, focus_tolerance=0.85, n_frames=n)
    intervals = [iv for iv, _ in rows]
    assert (1, 1) in intervals and (3, 3) in intervals


def test_peak_set_augmentation_adds_boundary_rows():
    n = 6
    phi = np.zeros((n, 1, 2), dtype=np.float32)
    phi[2, 0, 0] = 1.0
    phi[3, 0, 1] = 1.0
    reliable = np.ones((1, 2), dtype=bool)
    rows = build_rows(phi, reliable, focus_tolerance=0.85, n_frames=n)
    synthetic = [iv for iv, kind in rows if kind == "synthetic"]
    flat = set()
    for lo, hi in synthetic:
        flat |= set(range(lo, hi + 1))
    assert 1 in flat and 4 in flat


def test_build_rows_no_reliable_cells_returns_empty():
    n = 5
    phi = np.zeros((n, 1, 1), dtype=np.float32)
    phi[2, 0, 0] = 1.0
    reliable = np.zeros((1, 1), dtype=bool)  # nothing reliable
    rows = build_rows(phi, reliable, focus_tolerance=0.85, n_frames=n)
    assert rows == []
