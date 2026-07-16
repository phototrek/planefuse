import numpy as np

from planefuse.align.estimate import PairResult
from planefuse.backend import get_device
from planefuse.align import AlignParams, align_stack
from planefuse.stack.sources import ArrayFrameSource
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


def test_flagged_pair_recovers_via_direct_reestimate(tmp_path, monkeypatch):
    # Frame 0's consecutive link to frame 1 is bad, but frame 0 aligns fine
    # directly against the reference (frame 1) — the direct re-estimate should
    # recover it rather than leaving it flagged.
    frames = [np.full((24, 24, 3), value, dtype=np.float32) for value in (0.0, 0.5, 1.0)]
    src = ArrayFrameSource(frames)

    def fake_estimate(frame_a, frame_b, _device, **_kwargs):
        pair_values = (float(frame_a[0, 0, 0]), float(frame_b[0, 0, 0]))
        # Only the consecutive (0 -> 1) direction is bad; the direct
        # recovery re-estimate goes (ref=1 -> 0), the reverse order.
        correlation = 0.2 if pair_values == (0.0, 0.5) else 0.99
        return PairResult(np.eye(3), correlation, 1.0, 1.0)

    monkeypatch.setattr("planefuse.align.pipeline.estimate_pair", fake_estimate)

    report = align_stack(
        src,
        cache_dir=tmp_path,
        device=get_device("cpu"),
        params=AlignParams(reference=1, correlation_threshold=0.9),
    )
    assert report.recovered == {0}
    assert report.flagged == set()
    assert not report.dropped


def test_unrecoverable_direct_alignment_stays_flagged_until_drop(tmp_path, monkeypatch):
    frames = [np.full((24, 24, 3), value, dtype=np.float32) for value in (0.0, 0.5, 1.0)]
    src = ArrayFrameSource(frames)

    def fake_estimate(frame_a, frame_b, _device, **_kwargs):
        pair_values = (float(frame_a[0, 0, 0]), float(frame_b[0, 0, 0]))
        correlation = 0.2 if pair_values in {(0.0, 0.5), (0.5, 0.0)} else 0.99
        return PairResult(np.eye(3), correlation, 1.0, 1.0)

    monkeypatch.setattr("planefuse.align.pipeline.estimate_pair", fake_estimate)

    kept = align_stack(
        src,
        cache_dir=tmp_path / "kept",
        device=get_device("cpu"),
        params=AlignParams(reference=1, correlation_threshold=0.9),
    )
    assert kept.flagged == {0}
    assert not kept.dropped
    assert not kept.recovered

    dropped = align_stack(
        src,
        cache_dir=tmp_path / "dropped",
        device=get_device("cpu"),
        params=AlignParams(reference=1, correlation_threshold=0.9, drop_misaligned=True),
    )
    assert dropped.flagged == {0}
    assert dropped.dropped == {0}
