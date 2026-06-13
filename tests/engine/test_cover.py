from focusstack.select.cover import min_stab_cover, select_indices


def test_min_stab_cover_picks_right_endpoints():
    intervals = [(0, 2), (1, 4), (5, 6)]
    pts = min_stab_cover(intervals)
    assert pts == [2, 6]


def test_min_stab_cover_single_point_when_all_overlap():
    intervals = [(0, 5), (2, 4), (3, 6)]
    pts = min_stab_cover(intervals)
    assert len(pts) == 1
    assert 3 <= pts[0] <= 4


def test_select_indices_returns_kept_and_redundant():
    n = 7
    intervals = [(0, 1), (2, 3), (5, 6)]
    kept, redundant, warning = select_indices(intervals, n_frames=n)
    assert warning is None
    assert set(kept) | set(redundant) == set(range(n))
    assert set(kept).isdisjoint(redundant)
    for lo, hi in intervals:
        assert any(lo <= k <= hi for k in kept)


def test_degenerate_floor_keeps_all_when_no_rows():
    kept, redundant, warning = select_indices([], n_frames=10)
    assert kept == list(range(10))
    assert redundant == []
    assert warning is not None


def test_degenerate_floor_keeps_all_when_fewer_than_three():
    kept, redundant, warning = select_indices([(0, 9)], n_frames=10)
    assert kept == list(range(10))
    assert warning is not None
