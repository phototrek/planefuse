import numpy as np

from focusstack.backend import get_device
from focusstack.align import AlignParams, align_stack
from focusstack.stack.sources import ArrayFrameSource
from tests.synthetic.generate import generate_stack


def test_align_stack_writes_cache_and_report(tmp_path):
    stack = generate_stack(h=200, w=240, n_frames=5, max_sigma=0.5, seed=20,
                           scale_step=0.004, trans_jitter=2.0)
    src = ArrayFrameSource(stack.frames)
    report = align_stack(src, cache_dir=tmp_path, device=get_device("cpu"),
                         params=AlignParams(max_long_edge=256))
    assert report.reference == 2          # middle of 5
    assert len(report.correlations) == 4  # 4 consecutive pairs
    frames = report.cache.frame_source()
    masks = report.cache.mask_source()
    assert len(frames) == 5 and len(masks) == 5
    assert frames.read(0).shape == (200, 240, 3)


def test_align_stack_skip_when_pre_aligned(tmp_path):
    stack = generate_stack(h=120, w=120, n_frames=3, max_sigma=0.3, seed=21)
    src = ArrayFrameSource(stack.frames)
    report = align_stack(src, cache_dir=tmp_path, device=get_device("cpu"),
                         params=AlignParams(skip=True))
    for m in report.matrices:
        np.testing.assert_allclose(m, np.eye(3), atol=1e-9)
    assert report.cache.mask_source().read(0).all()


def test_align_stack_drop_misaligned_excludes_bad_frame(tmp_path):
    # Build a 5-frame stack, then corrupt frame 2 so its pair correlations are low.
    stack = generate_stack(h=160, w=200, n_frames=5, max_sigma=0.5, seed=22,
                           trans_jitter=1.0)
    frames = list(stack.frames)
    rng = np.random.default_rng(0)
    frames[2] = rng.uniform(0, 1, frames[2].shape).astype(np.float32)  # noise = unalignable
    src = ArrayFrameSource(frames)
    report = align_stack(src, cache_dir=tmp_path, device=get_device("cpu"),
                         params=AlignParams(max_long_edge=256, correlation_threshold=0.90,
                                            drop_misaligned=True))
    # frame 2 should be dropped; its cached mask must be entirely invalid so the
    # stacker ignores it, and surviving frames' transforms stay finite.
    assert 2 in report.dropped
    dropped_mask = report.cache.mask_source().read(2)
    assert not dropped_mask.any()
    for i in range(5):
        assert np.all(np.isfinite(report.matrices[i]))
