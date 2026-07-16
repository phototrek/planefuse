import numpy as np
import pytest
import torch

from planefuse.backend import Device
from planefuse.errors import ValidationError
from planefuse.io import ProcessingDomain, save_image
from planefuse.pipeline import stack_frames
from tests.synthetic.generate import generate_stack


@pytest.fixture
def stack_dir(tmp_path):
    synth = generate_stack(h=96, w=128, n_frames=6, max_sigma=4.0, seed=2)
    for i, f in enumerate(synth.frames):
        save_image(f, tmp_path / f"frame_{i:03d}.tif", bit_depth=16)
    return tmp_path, synth


def test_stack_frames_from_dir(stack_dir):
    d, synth = stack_dir
    result = stack_frames(sorted(d.glob("*.tif")), method="pmax", device_pref="cpu")
    assert result.image.shape == synth.sharp.shape
    assert result.domain is ProcessingDomain.RENDERED_RGB


def test_validation_failure_raises_with_filenames(stack_dir):
    d, _ = stack_dir
    bad = np.zeros((10, 10, 3), dtype=np.float32)
    save_image(bad, d / "zz_wrong_size.tif")
    with pytest.raises(ValidationError, match="zz_wrong_size"):
        stack_frames(sorted(d.glob("*.tif")), method="pmax", device_pref="cpu")


def test_forced_tiled_mode_matches_direct(stack_dir):
    d, _ = stack_dir
    paths = sorted(d.glob("*.tif"))
    direct = stack_frames(paths, method="pmax", device_pref="cpu", tile_mode="never")
    tiled = stack_frames(paths, method="pmax", device_pref="cpu", tile_mode="always", tile=64)
    np.testing.assert_allclose(tiled.image, direct.image, atol=2e-3)


def test_cuda_like_oom_retries_tiled(stack_dir, monkeypatch):
    directory, synth = stack_dir
    device = Device("cuda", torch.device("cpu"), "simulated CUDA")
    calls: list[tuple[str, str]] = []

    class FailingAlgorithm:
        def run(self, *_args, **_kwargs):
            calls.append(("direct", "cuda"))
            raise torch.cuda.OutOfMemoryError("simulated CUDA out of memory")

    def tiled(_method, _source, actual_device, _params, **_kwargs):
        calls.append(("tiled", actual_device.kind))
        return synth.sharp.astype(np.float32)

    monkeypatch.setattr("planefuse.pipeline.get_device", lambda _prefer: device)
    monkeypatch.setattr("planefuse.pipeline.get_algorithm", lambda _method: FailingAlgorithm())
    monkeypatch.setattr("planefuse.pipeline.stack_tiled", tiled)
    monkeypatch.setattr("planefuse.pipeline.empty_cache", lambda _device: None)
    result = stack_frames(
        sorted(directory.glob("*.tif")),
        method="pmax",
        device_pref="cuda",
        tile_mode="never",
    )
    assert result.image.shape == synth.sharp.shape
    assert calls == [("direct", "cuda"), ("tiled", "cuda")]


def test_mps_like_oom_falls_back_from_direct_to_tiled_to_cpu(stack_dir, monkeypatch):
    directory, synth = stack_dir
    mps = Device("mps", torch.device("cpu"), "simulated MPS")
    cpu = Device("cpu", torch.device("cpu"), "CPU")
    calls: list[tuple[str, str]] = []

    class FailingAlgorithm:
        def run(self, *_args, **_kwargs):
            calls.append(("direct", "mps"))
            raise RuntimeError("MPS backend out of memory")

    def tiled(_method, _source, actual_device, _params, **_kwargs):
        calls.append(("tiled", actual_device.kind))
        if actual_device.kind == "mps":
            raise RuntimeError("MPS backend out of memory")
        return synth.sharp.astype(np.float32)

    monkeypatch.setattr(
        "planefuse.pipeline.get_device", lambda prefer: cpu if prefer == "cpu" else mps
    )
    monkeypatch.setattr("planefuse.pipeline.get_algorithm", lambda _method: FailingAlgorithm())
    monkeypatch.setattr("planefuse.pipeline.stack_tiled", tiled)
    monkeypatch.setattr("planefuse.pipeline.empty_cache", lambda _device: None)
    result = stack_frames(
        sorted(directory.glob("*.tif")),
        method="pmax",
        device_pref="mps",
        tile_mode="never",
    )
    assert result.image.shape == synth.sharp.shape
    assert calls == [("direct", "mps"), ("tiled", "mps"), ("tiled", "cpu")]
