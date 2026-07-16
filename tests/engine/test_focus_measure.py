import numpy as np

from planefuse.backend import get_device
from planefuse.select.focus_measure import compute_focus_measures
from planefuse.stack.sources import ArrayFrameSource


def test_focus_measures_shape_and_sharper_frame_scores_higher():
    h, w = 64, 96
    rng = np.random.default_rng(0)
    blurry = np.full((h, w, 3), 0.5, dtype=np.float32)
    textured = rng.uniform(0, 1, (h, w, 3)).astype(np.float32)
    src = ArrayFrameSource([blurry, textured])
    phi, cell_reliable = compute_focus_measures(src, get_device("cpu"),
                                                grid_rows=8, grid_cols=12)
    assert phi.shape == (2, 8, 12)
    assert cell_reliable.shape == (8, 12)
    assert float(phi[1].mean()) > 10 * float(phi[0].mean()) + 1e-9


def test_focus_measures_mask_marks_cells_unreliable():
    h, w = 32, 32
    rng = np.random.default_rng(1)
    frames = [rng.uniform(0, 1, (h, w, 3)).astype(np.float32) for _ in range(2)]
    src = ArrayFrameSource(frames)

    class _Masks:
        def __len__(self): return 2
        def read(self, idx, region=None):
            m = np.ones((h, w), bool)
            if idx == 0:
                m[:, :16] = False  # left half of frame 0 invalid
            return m

    phi, cell_reliable = compute_focus_measures(src, get_device("cpu"),
                                                grid_rows=4, grid_cols=4, masks=_Masks())
    assert not cell_reliable[:, :2].any()
    assert cell_reliable[:, 2:].all()
