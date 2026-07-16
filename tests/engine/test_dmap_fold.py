import numpy as np

from planefuse.backend import get_device
from planefuse.stack.dmap import dmap_fold
from planefuse.stack.sources import ArrayFrameSource


def test_fold_picks_sharpest_frame_per_pixel():
    h, w = 32, 30
    frames = []
    for k in range(3):
        f = np.full((h, w, 3), 0.5, dtype=np.float32)
        band = slice(k * 10, k * 10 + 10)
        rng = np.random.default_rng(k)
        f[:, band, :] = rng.uniform(0, 1, (h, 10, 3)).astype(np.float32)
        frames.append(f)
    src = ArrayFrameSource(frames)
    index, sharp = dmap_fold(src, get_device("cpu"), radius=4)
    assert index.shape == (h, w) and sharp.shape == (h, w)
    assert int(np.round(np.median(index[:, 0:10]))) == 0
    assert int(np.round(np.median(index[:, 10:20]))) == 1
    assert int(np.round(np.median(index[:, 20:30]))) == 2


class _BoolMaskSource:
    def __init__(self, masks):
        self._m = masks

    def __len__(self):
        return len(self._m)

    def read(self, idx, region=None):
        m = self._m[idx]
        if region is not None:
            y0, x0, y1, x1 = region
            m = m[y0:y1, x0:x1]
        return m


def test_fold_respects_masks():
    h, w = 16, 16
    frames = [np.full((h, w, 3), 0.5, dtype=np.float32) for _ in range(2)]
    rng = np.random.default_rng(0)
    frames[0][:, :, :] = rng.uniform(0, 1, (h, w, 3)).astype(np.float32)  # frame 0 textured
    masks = [np.ones((h, w), bool), np.ones((h, w), bool)]
    masks[0][:] = False  # frame 0 entirely invalid -> must never win
    src = ArrayFrameSource(frames)
    index, _ = dmap_fold(src, get_device("cpu"), radius=2, masks=_BoolMaskSource(masks))
    assert (index == 1).all()
