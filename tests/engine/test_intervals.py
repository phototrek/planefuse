import numpy as np

from planefuse.select.intervals import in_focus_interval, peak_index


def test_peak_index_is_argmax():
    curve = np.array([0.1, 0.3, 0.9, 0.4, 0.2])
    assert peak_index(curve) == 2


def test_in_focus_interval_relative_tolerance_run():
    curve = np.array([0.2, 0.9, 1.0, 0.86, 0.5])
    lo, hi = in_focus_interval(curve, focus_tolerance=0.85)
    assert (lo, hi) == (1, 3)


def test_in_focus_interval_single_frame_when_sharp_isolated():
    curve = np.array([0.1, 0.1, 1.0, 0.1, 0.1])
    lo, hi = in_focus_interval(curve, focus_tolerance=0.85)
    assert (lo, hi) == (2, 2)


def test_in_focus_interval_stops_at_gap():
    curve = np.array([0.95, 0.2, 1.0, 0.2, 0.9])
    lo, hi = in_focus_interval(curve, focus_tolerance=0.85)
    assert (lo, hi) == (2, 2)
