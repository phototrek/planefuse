from typer.testing import CliRunner

from planefuse.cli import app
from planefuse.io import save_image
from tests.synthetic.generate import generate_stack

runner = CliRunner()


def _write(tmp_path, n=6, seed=70):
    stack = generate_stack(h=120, w=150, n_frames=n, max_sigma=4.0, seed=seed)
    for i, f in enumerate(stack.frames):
        save_image(f, tmp_path / f"f_{i:03d}.tif", bit_depth=16)
    return tmp_path


def test_cli_dmap_with_depth_map(tmp_path):
    d = _write(tmp_path)
    out = tmp_path / "out.tif"
    depth = tmp_path / "depth.tif"
    result = runner.invoke(app, ["stack", str(d), "-o", str(out), "--method", "dmap",
                                 "--device", "cpu", "--depth-map", str(depth)])
    assert result.exit_code == 0, result.output
    assert out.exists() and depth.exists()


def test_cli_weighted_runs(tmp_path):
    d = _write(tmp_path)
    out = tmp_path / "w.tif"
    result = runner.invoke(app, ["stack", str(d), "-o", str(out), "--method", "weighted",
                                 "--device", "cpu"])
    assert result.exit_code == 0, result.output
    assert out.exists()


def _captured_params(monkeypatch, tmp_path, extra):
    import planefuse.cli as cli

    seen = {}

    def fake_stack_frames(paths, method, params, **kwargs):
        seen.update(params)
        raise RuntimeError("captured")

    monkeypatch.setattr(cli, "stack_frames", fake_stack_frames)
    d = _write(tmp_path, n=3)
    runner.invoke(app, ["stack", str(d), "-o", str(tmp_path / "o.tif"), "--method", "dmap",
                        "--device", "cpu", *extra])
    return seen


def test_cli_dmap_radii_default_to_the_tuned_values(monkeypatch, tmp_path):
    params = _captured_params(monkeypatch, tmp_path, [])
    assert params["estimation_radius"] == 8
    assert params["smoothing_radius"] == 16


def test_cli_dmap_radii_reach_the_stacker(monkeypatch, tmp_path):
    params = _captured_params(monkeypatch, tmp_path,
                              ["--estimation-radius", "2", "--smoothing-radius", "32"])
    assert params["estimation_radius"] == 2
    assert params["smoothing_radius"] == 32
