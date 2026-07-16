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
