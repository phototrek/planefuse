"""Alignment orchestrator (SPEC §6, full pipeline)."""

from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from planefuse.align.cache import AlignedCache, cache_equivalent
from planefuse.align.chain import chain_to_reference
from planefuse.align.estimate import estimate_pair
from planefuse.align.proxy import make_proxy, scene_linear_proxy_luminance
from planefuse.align.warp import warp_full
from planefuse.backend import Device, ops
from planefuse.io.metadata import ProcessingDomain
from planefuse.stack.base import FrameSource
from planefuse.stack.sources import ArrayFrameSource, MemoryFrameSource, TensorFrameSource

ProgressFn = Callable[[str, float], None]


@dataclass
class AlignParams:
    model: str = "similarity"
    max_long_edge: int = 2048
    interp: str = "lanczos3"
    normalize_brightness: bool = True
    correlation_threshold: float = 0.90
    reference: int | None = None
    drop_misaligned: bool = False
    skip: bool = False
    refine_full_res: bool = False  # warm-started ECC polish at native resolution


@dataclass
class AlignReport:
    reference: int
    matrices: list[np.ndarray]
    correlations: dict[int, float]
    flagged: set[int]
    recovered: set[int]
    dropped: set[int]
    cache: AlignedCache
    proxy_factor: float  # resolution of `matrices`: 1.0 when full-res-refined
    # Set when align_stack held the aligned frames as tensors (store="device"
    # or "host") instead of writing them to `cache`.
    frames: TensorFrameSource | None = None
    masks: TensorFrameSource | None = None

    def frame_source(self) -> FrameSource:
        return self.frames if self.frames is not None else self.cache.frame_source()

    def mask_source(self) -> FrameSource:
        return self.masks if self.masks is not None else self.cache.mask_source()


def _reads_are_free(source: FrameSource) -> bool:
    """Whether read() hands back a held array (no decode, no disk)."""
    return isinstance(source, (ArrayFrameSource, MemoryFrameSource))


def _build_proxies(source: FrameSource, device: Device, params: AlignParams,
                   scene_linear: bool) -> list[tuple[torch.Tensor, float]]:
    """make_proxy for every frame, once. The host-only scene-linear luminance
    runs on worker threads; the device upload/resample stays on this thread."""
    n = len(source)
    if not scene_linear:
        return [make_proxy(source.read(i), device, params.max_long_edge) for i in range(n)]
    limit = source.workers if isinstance(source, MemoryFrameSource) else (os.cpu_count() or 1)
    workers = max(1, min(n, limit))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="planefuse-proxy") as pool:
        neutrals = pool.map(lambda i: scene_linear_proxy_luminance(source.read(i)), range(n))
        return [
            make_proxy(source.read(i), device, params.max_long_edge,
                       normalize_scene_linear=True, neutral=neutral)
            for i, neutral in enumerate(neutrals)
        ]


def align_stack(source: FrameSource, cache_dir: Path, device: Device,
                params: AlignParams, progress: ProgressFn | None = None,
                cancel: Callable[[], bool] | None = None,
                store: str = "cache") -> AlignReport:
    """Estimate, chain and warp the stack onto its reference.

    store: where the aligned frames and masks go. "cache" writes them to the
    on-disk AlignedCache (report.cache); "device" keeps them as tensors on
    `device`, "host" as tensors in host memory (report.frames / report.masks),
    holding the same values the cache would hand back.

    A MemoryFrameSource is consumed: each frame is released once it is warped.
    """
    if store not in ("cache", "device", "host"):
        raise ValueError(f"unknown aligned-frame store {store!r}")
    n = len(source)
    cache = AlignedCache(
        cache_dir,
        domain=source.domain,
        metadata=source.metadata,
        source_hash=source.source_hash,
    )
    probe = source.read(0)
    out_shape = probe.shape[:2]
    ref = params.reference if params.reference is not None else n // 2

    def _tick(i: int, msg: str) -> None:
        if cancel is not None and cancel():
            raise InterruptedError("alignment cancelled")
        if progress is not None:
            progress(msg, i / max(2 * n, 1))

    held_frames: list[torch.Tensor] = []
    held_masks: list[torch.Tensor] = []

    def _hold(aligned: torch.Tensor, mask: torch.Tensor) -> None:
        aligned = cache_equivalent(aligned, source.domain)
        if store == "host":
            aligned, mask = aligned.cpu(), mask.cpu()
        held_frames.append(aligned)
        held_masks.append(mask)

    def _held_report(**kwargs) -> AlignReport:
        report = AlignReport(**kwargs)
        if store != "cache":
            report.frames = TensorFrameSource(held_frames, domain=cache.domain,
                                              metadata=cache.metadata, source_hash=cache.source_hash)
            report.masks = TensorFrameSource(held_masks, as_mask=True,
                                             domain=ProcessingDomain.RENDERED_RGB,
                                             source_hash=cache.source_hash)
        return report

    if params.skip or n < 2:
        for idx in range(n):
            _tick(idx, f"copy {idx + 1}/{n}")
            frame = source.read(idx)
            if store == "cache":
                cache.write(idx, frame, np.ones(frame.shape[:2], dtype=bool))
            else:
                _hold(ops.to_tensor(frame, device),
                      torch.ones(frame.shape[:2], dtype=torch.bool, device=device.torch_device))
        return _held_report(reference=ref, matrices=[np.eye(3) for _ in range(n)], correlations={},
                            flagged=set(), recovered=set(), dropped=set(), cache=cache,
                            proxy_factor=1.0)

    # estimate consecutive pairs (i, i+1): transform maps (i+1) -> i
    pair: dict[int, np.ndarray] = {}
    corr: dict[int, float] = {}
    factor = 1.0
    scene_linear = source.domain is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB
    if _reads_are_free(source):
        # Each frame's proxy is built once and shared by its two pairs; the
        # pairs are then estimated concurrently. Each ECC solve stays
        # single-threaded and the device work is serialized, so a pair's
        # result must not depend on scheduling.
        _tick(0, f"estimate pair 1/{n - 1}")
        proxies = _build_proxies(source, device, params, scene_linear)
        workers = max(1, min(n - 1, os.cpu_count() or 1))
        device_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="planefuse-pair") as pool:
            futures: list[Future] = [
                pool.submit(estimate_pair, source.read(i), source.read(i + 1), device,
                            max_long_edge=params.max_long_edge, model=params.model,
                            normalize_brightness=params.normalize_brightness,
                            normalize_scene_linear=scene_linear,
                            refine_full_res=params.refine_full_res,
                            proxy_a=proxies[i], proxy_b=proxies[i + 1],
                            device_lock=device_lock)
                for i in range(n - 1)
            ]
            try:
                for i, future in enumerate(futures):
                    if i > 0:
                        _tick(i, f"estimate pair {i + 1}/{n - 1}")
                    pr = future.result()
                    pair[i] = pr.matrix
                    corr[i] = pr.correlation
                    factor = pr.proxy_factor
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        del proxies
    else:
        for i in range(n - 1):
            _tick(i, f"estimate pair {i + 1}/{n - 1}")
            pr = estimate_pair(source.read(i), source.read(i + 1), device,
                               max_long_edge=params.max_long_edge, model=params.model,
                               normalize_brightness=params.normalize_brightness,
                               normalize_scene_linear=scene_linear,
                               refine_full_res=params.refine_full_res)
            pair[i] = pr.matrix
            corr[i] = pr.correlation
            factor = pr.proxy_factor

    # ---- dropped-frame handling (SPEC §6 step 4) ----
    # Attribute blame to the frame, not the link: a bad frame corrupts BOTH its
    # adjacent pairs. corr[i] gates pair (i, i+1). An interior frame k is the
    # culprit when both its incoming pair (k-1) and outgoing pair (k) fall below
    # threshold; an endpoint is the culprit when its single adjacent pair is bad.
    # If the chosen reference is itself a culprit we re-select the nearest good
    # frame as reference (the reference must never be dropped). We then re-chain
    # over the surviving frames so the chain bridges each gap with a DIRECT
    # re-estimate (k-1, k+1) instead of composing through the unreliable link.
    # Dropped frames keep an identity matrix (finite) and get an all-False mask
    # so the stacker ignores them.
    dropped: set[int] = set()
    culprits: set[int] = set()
    if params.drop_misaligned:
        thr = params.correlation_threshold
        bad_pair = {i for i, c in corr.items() if c < thr}
        for k in range(n):
            incoming_bad = (k - 1) in bad_pair  # pair (k-1, k)
            outgoing_bad = k in bad_pair        # pair (k, k+1)
            if k == 0:
                culprit = outgoing_bad
            elif k == n - 1:
                culprit = incoming_bad
            else:
                culprit = incoming_bad and outgoing_bad
            if culprit:
                culprits.add(k)
        # The reference must be a good frame: if it is a culprit, move it to the
        # nearest non-culprit (preferring lower index on a tie).
        if ref in culprits:
            candidates = [k for k in range(n) if k not in culprits]
            if candidates:
                ref = min(candidates, key=lambda k: (abs(k - ref), k))

    chain = chain_to_reference(n, pair, corr, ref=ref,
                               threshold=params.correlation_threshold,
                               drop=False)
    matrices = chain.matrices

    # A single unreliable consecutive estimate contaminates every chained
    # transform whose path crosses that link. Re-estimate those frames directly
    # against the chosen reference and accept only correlations that clear the
    # same quality threshold. This prevents a bad link from silently poisoning
    # otherwise alignable frames on the far side of the chain.
    bad_pair = {i for i, c in corr.items() if c < params.correlation_threshold}
    affected: set[int] = set()
    for frame_idx in range(n):
        if frame_idx < ref:
            path = range(frame_idx, ref)
        elif frame_idx > ref:
            path = range(ref, frame_idx)
        else:
            continue
        if any(link in bad_pair for link in path):
            affected.add(frame_idx)

    recovered: set[int] = set()
    for frame_idx in sorted(affected):
        direct = estimate_pair(
            source.read(ref),
            source.read(frame_idx),
            device,
            max_long_edge=params.max_long_edge,
            model=params.model,
            normalize_brightness=params.normalize_brightness,
            normalize_scene_linear=(
                source.domain is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB
            ),
            refine_full_res=params.refine_full_res,
        )
        if direct.correlation >= params.correlation_threshold:
            matrices[frame_idx] = direct.matrix
            recovered.add(frame_idx)

    flagged = affected - recovered
    if params.drop_misaligned:
        dropped = {frame_idx for frame_idx in culprits if frame_idx != ref and frame_idx in flagged}

    if dropped:
        survivors = [i for i in range(n) if i not in dropped]
        # ref always survives (it is never a member of `dropped`).
        ref_pos = survivors.index(ref)

        # Effective consecutive transform over the survivor sequence: maps
        # survivors[j+1] -> survivors[j] (output->input). Reuse the original
        # pair when adjacent; re-estimate directly across a gap.
        eff_pair: dict[int, np.ndarray] = {}
        for j in range(len(survivors) - 1):
            s_prev, s_next = survivors[j], survivors[j + 1]
            if s_next - s_prev == 1:
                eff_pair[j] = pair[s_prev]
            else:
                pr = estimate_pair(source.read(s_prev), source.read(s_next), device,
                                   max_long_edge=params.max_long_edge,
                                   model=params.model,
                                   normalize_brightness=params.normalize_brightness,
                                   normalize_scene_linear=(
                                       source.domain
                                       is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB
                                   ),
                                   refine_full_res=params.refine_full_res)
                eff_pair[j] = pr.matrix

        sub = chain_to_reference(len(survivors), eff_pair,
                                 {j: 1.0 for j in range(len(survivors) - 1)},
                                 ref=ref_pos, threshold=0.0, drop=False)

        matrices = [np.eye(3) for _ in range(n)]
        for pos, frame_idx in enumerate(survivors):
            matrices[frame_idx] = sub.matrices[pos]

    # full-resolution warp + cache (or held tensors)
    for idx in range(n):
        _tick(n + idx, f"warp {idx + 1}/{n}")
        if store == "cache":
            aligned, mask = warp_full(source.read(idx), matrices[idx], device,
                                      out_shape=out_shape, proxy_factor=factor,
                                      interp=params.interp)
            if idx in dropped:
                mask = np.zeros(out_shape, dtype=bool)
            cache.write(idx, aligned, mask)
        else:
            aligned_t, mask_t = warp_full(source.read(idx), matrices[idx], device,
                                          out_shape=out_shape, proxy_factor=factor,
                                          interp=params.interp, keep_on_device=True)
            if idx in dropped:
                mask_t = torch.zeros(out_shape, dtype=torch.bool, device=mask_t.device)
            _hold(aligned_t, mask_t)
        if isinstance(source, MemoryFrameSource):
            source.release_frame(idx)  # every estimate above has already read it

    if progress is not None:
        progress("done", 1.0)
    return _held_report(reference=ref, matrices=matrices, correlations=corr, flagged=flagged,
                        recovered=recovered, dropped=dropped, cache=cache, proxy_factor=factor)
