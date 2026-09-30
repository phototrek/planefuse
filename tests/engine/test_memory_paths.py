"""The in-memory fast paths must reproduce the streaming / disk-cache paths exactly."""

import hashlib
import threading

import numpy as np
import pytest
import torch

import planefuse.align.pipeline as align_pipeline
import planefuse.stack.sources as sources

from planefuse.align import AlignParams, align_stack
from planefuse.align.estimate import estimate_pair
from planefuse.align.proxy import make_proxy
from planefuse.backend import get_device
from planefuse.io import load_raw, save_image, validate_stack
from planefuse.pipeline import stack_frames
from planefuse.stack import DirFrameSource, MemoryFrameSource
from planefuse.stack.sources import ArrayFrameSource, decode_frames
from tests.engine.test_raw_loader import write_test_raw
from tests.synthetic.generate import generate_stack


def _write_stack(tmp_path, n=5, h=160, w=200, seed=40):
    stack = generate_stack(h=h, w=w, n_frames=n, max_sigma=3.0, seed=seed,
                           scale_step=0.003, trans_jitter=2.0, rot_jitter=0.01)
    paths = []
    for i, frame in enumerate(stack.frames):
        path = tmp_path / f"f_{i:03d}.tif"
        save_image(frame, path, bit_depth=16)
        paths.append(path)
    return stack, paths


def test_memory_source_matches_dir_source(tmp_path):
    _stack, paths = _write_stack(tmp_path)
    disk = DirFrameSource(paths, reference_index=2)
    memory = MemoryFrameSource(paths, decode_frames(paths, workers=3), reference_index=2, workers=3)
    assert memory.source_hash == disk.source_hash
    assert memory.source_hashes == disk.source_hashes
    assert memory.domain is disk.domain
    for i in range(len(paths)):
        np.testing.assert_array_equal(memory.read(i), disk.read(i))
        np.testing.assert_array_equal(memory.read(i, (10, 20, 50, 90)), disk.read(i, (10, 20, 50, 90)))


@pytest.mark.parametrize("second_options", [{}, {"model": "Different Camera"}, {"width": 42}])
def test_validate_from_loaded_frames_matches_decoding(tmp_path, second_options):
    first = write_test_raw(tmp_path / "first.dng")
    second = write_test_raw(tmp_path / "second.dng", **second_options)
    broken = tmp_path / "broken.dng"
    broken.write_bytes(b"not a raw file")
    paths = [first, second, broken]
    plain = validate_stack(paths)
    loaded = validate_stack(paths, decode_frames(paths, workers=3))
    assert [(s.status, s.message) for s in loaded.files] == [(s.status, s.message) for s in plain.files]
    assert (loaded.width, loaded.height, loaded.domain, loaded.decoder) == (
        plain.width, plain.height, plain.domain, plain.decoder)


def test_estimate_pair_with_prebuilt_proxies_is_identical():
    stack = generate_stack(h=240, w=300, n_frames=2, max_sigma=0.4, seed=12,
                           scale_step=0.01, trans_jitter=3.0, rot_jitter=0.01)
    device = get_device("cpu")
    a, b = stack.frames
    plain = estimate_pair(a, b, device, max_long_edge=256)
    reused = estimate_pair(a, b, device, max_long_edge=256,
                           proxy_a=make_proxy(a, device, 256), proxy_b=make_proxy(b, device, 256))
    np.testing.assert_array_equal(plain.matrix, reused.matrix)
    assert plain.correlation == reused.correlation and plain.gain == reused.gain


@pytest.mark.parametrize("store", ["device", "host"])
def test_align_tensor_store_matches_cache(tmp_path, device, store):
    stack = generate_stack(h=200, w=240, n_frames=5, max_sigma=0.5, seed=20,
                           scale_step=0.004, trans_jitter=2.0)
    frames = list(stack.frames)
    frames[3] = np.random.default_rng(0).uniform(0, 1, frames[3].shape).astype(np.float32)
    params = AlignParams(max_long_edge=256, interp="lanczos3", drop_misaligned=True)
    cached = align_stack(ArrayFrameSource(frames), tmp_path / "c", device, params)
    held = align_stack(ArrayFrameSource(frames), tmp_path / "h", device, params, store=store)
    assert held.frames is not None and held.masks is not None
    assert held.dropped == cached.dropped
    for m_held, m_cached in zip(held.matrices, cached.matrices):
        np.testing.assert_array_equal(m_held, m_cached)
    for i in range(len(frames)):
        np.testing.assert_array_equal(held.frame_source().read(i), cached.frame_source().read(i))
        np.testing.assert_array_equal(held.mask_source().read(i), cached.mask_source().read(i))


@pytest.mark.parametrize("tile_mode", ["never", "always"])
def test_stack_frames_memory_path_matches_streaming(tmp_path, monkeypatch, tile_mode):
    _stack, paths = _write_stack(tmp_path, n=6, h=300, w=380, seed=41)
    kwargs = dict(method="pmax", device_pref="cpu", tile_mode=tile_mode, tile=256,
                  align=AlignParams(max_long_edge=256))
    monkeypatch.delenv("PLANEFUSE_MEMORY_BUDGET_MB", raising=False)
    loads = {"n": 0}
    real_load = sources.load_image

    def counting_load(path):
        loads["n"] += 1
        return real_load(path)

    proxy_builds = {"n": 0}
    real_build = align_pipeline._build_proxies

    def counting_build(*args, **kw):
        proxy_builds["n"] += 1
        return real_build(*args, **kw)

    monkeypatch.setattr(sources, "load_image", counting_load)
    monkeypatch.setattr(align_pipeline, "_build_proxies", counting_build)
    fast = stack_frames(paths, cache_dir=tmp_path / "fast", **kwargs)
    assert loads["n"] == len(paths)  # each frame decoded exactly once
    assert proxy_builds["n"] == 1  # the pooled pair path ran
    loads["n"] = proxy_builds["n"] = 0
    monkeypatch.setenv("PLANEFUSE_MEMORY_BUDGET_MB", "0")
    slow = stack_frames(paths, cache_dir=tmp_path / "slow", **kwargs)
    assert loads["n"] > len(paths)
    assert proxy_builds["n"] == 0
    np.testing.assert_array_equal(fast.image, slow.image)
    assert fast.provenance["sources"] == slow.provenance["sources"]
    assert fast.provenance["alignment"] == slow.provenance["alignment"]
    assert list((tmp_path / "slow" / "aligned").glob("*.tif"))  # the fallback used the disk cache
    assert not list((tmp_path / "fast" / "aligned").glob("*.tif"))


@pytest.mark.parametrize("align", [None, AlignParams(max_long_edge=256)])
def test_memory_frames_are_not_mutated_on_cpu(tmp_path, monkeypatch, align):
    # On the cpu device torch tensors can share memory with the held numpy
    # frames, so any in-place op downstream would corrupt them.
    _stack, paths = _write_stack(tmp_path, n=5, seed=42)
    captured: list[np.ndarray] = []
    real_init = MemoryFrameSource.__init__

    def capturing_init(self, *args, **kw):
        real_init(self, *args, **kw)
        captured.extend(self._frames)

    monkeypatch.delenv("PLANEFUSE_MEMORY_BUDGET_MB", raising=False)
    monkeypatch.setattr(MemoryFrameSource, "__init__", capturing_init)
    before: list[str] = []

    real_read = MemoryFrameSource.read

    def first_read_checksums(self, idx, region=None):
        if not before:
            before.extend(hashlib.sha256(f.tobytes()).hexdigest() for f in captured)
        return real_read(self, idx, region)

    monkeypatch.setattr(MemoryFrameSource, "read", first_read_checksums)
    stack_frames(paths, method="pmax", device_pref="cpu", align=align, cache_dir=tmp_path / "c")
    assert len(captured) == len(paths) and before
    assert [hashlib.sha256(f.tobytes()).hexdigest() for f in captured] == before


def test_concurrent_raw_decodes_equal_serial(tmp_path):
    path = write_test_raw(tmp_path / "frame.dng", width=64, height=48)
    serial = load_raw(path).pixels
    barrier = threading.Barrier(6)
    results: list[np.ndarray | BaseException] = [None] * 6  # type: ignore[list-item]

    def decode(slot: int) -> None:
        barrier.wait()
        try:
            results[slot] = load_raw(path).pixels
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
            results[slot] = exc

    threads = [threading.Thread(target=decode, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for result in results:
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, serial)


def test_device_oom_while_holding_aligned_frames_retries_with_cache(tmp_path, monkeypatch):
    _stack, paths = _write_stack(tmp_path, n=5, seed=43)
    kwargs = dict(method="pmax", device_pref="cpu", align=AlignParams(max_long_edge=256))
    monkeypatch.delenv("PLANEFUSE_MEMORY_BUDGET_MB", raising=False)
    expected = stack_frames(paths, cache_dir=tmp_path / "plain", **kwargs)

    real_warp = align_pipeline.warp_full
    calls = {"held": 0}

    def warp_then_oom(*args, keep_on_device=False, **kw):
        if keep_on_device:
            calls["held"] += 1
            if calls["held"] == 3:  # after two input frames were already released
                raise torch.cuda.OutOfMemoryError("simulated device OOM")
        return real_warp(*args, keep_on_device=keep_on_device, **kw)

    monkeypatch.setattr(align_pipeline, "warp_full", warp_then_oom)
    retried = stack_frames(paths, cache_dir=tmp_path / "retry", **kwargs)
    assert calls["held"] == 3
    assert list((tmp_path / "retry" / "aligned").glob("*.tif"))
    np.testing.assert_array_equal(retried.image, expected.image)
    assert retried.provenance["sources"] == expected.provenance["sources"]
