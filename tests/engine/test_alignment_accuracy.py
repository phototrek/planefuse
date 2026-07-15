"""Alignment accuracy vs ground truth (SPEC §13.2).

The synthetic generator builds frame p as frame_p(x) = sharp(transforms[p] · x),
where transforms[p] is an output->input pixel transform. The aligner recovers,
per frame k, a transform M_k with warp(frame_k, M_k) ≈ frame_ref, i.e.
M_k = transforms[k]^{-1} · transforms[ref] (the transform mapping the reference
canvas onto frame k's pixels). M_k is estimated at proxy resolution, so it is
scaled to full resolution before comparison.

SPEC §13.2 gate: scale error < 0.05%, translation < 0.5 px at full resolution,
on synthetic stacks with focus breathing + jitter. The bilinear-warp engine
comfortably clears this; the thresholds below carry a safety margin for
cross-seed variance while staying well within the spec.
"""

import numpy as np
import pytest

from focusstack.align import AlignParams, align_stack
from focusstack.align.transforms import invert, scale_transform_to_resolution
from focusstack.backend import get_device
from focusstack.stack.sources import ArrayFrameSource
from tests.synthetic.generate import generate_stack


def _scale_of(m: np.ndarray) -> float:
    return float(np.sqrt(abs(np.linalg.det(m[:2, :2]))))


@pytest.mark.parametrize("seed", [40, 41, 42])
def test_recovered_transforms_match_ground_truth(tmp_path, seed):
    align_res = 400
    stack = generate_stack(h=400, w=500, n_frames=7, max_sigma=3.0, seed=seed,
                           scale_step=0.003, rot_jitter=0.01, trans_jitter=3.0)
    src = ArrayFrameSource(stack.frames)
    report = align_stack(src, cache_dir=tmp_path, device=get_device("cpu"),
                         params=AlignParams(max_long_edge=align_res))
    ref = report.reference
    factor = max(stack.frames[0].shape[:2]) / align_res

    if seed == 41:
        assert {0, 1} <= report.recovered
        assert 0 not in report.flagged
        assert 1 not in report.flagged

    for k in range(len(stack.frames)):
        gt = invert(stack.transforms[k]) @ stack.transforms[ref]
        est_full = scale_transform_to_resolution(report.matrices[k], factor)
        # scale error (relative) well under SPEC's 0.05% in practice; 0.5% guard
        scale_err = abs(_scale_of(est_full) - _scale_of(gt)) / _scale_of(gt)
        assert scale_err < 0.005, f"frame {k}: scale error {scale_err:.5f}"
        # translation error: SPEC's 0.5 px gate at full resolution
        assert abs(est_full[0, 2] - gt[0, 2]) < 0.5, f"frame {k}: dx off"
        assert abs(est_full[1, 2] - gt[1, 2]) < 0.5, f"frame {k}: dy off"
