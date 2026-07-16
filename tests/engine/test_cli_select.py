from typer.testing import CliRunner

from planefuse.cli import app
from planefuse.io import save_image
from tests.synthetic.generate import generate_stack

runner = CliRunner()


def test_cli_select_frames_runs(tmp_path):
    stack = generate_stack(h=120, w=150, n_frames=20, max_sigma=5.0, seed=90)
    for i, f in enumerate(stack.frames):
        save_image(f, tmp_path / f"f_{i:03d}.tif", bit_depth=16)
    out = tmp_path / "out.tif"
    result = runner.invoke(app, ["stack", str(tmp_path), "-o", str(out),
                                 "--method", "pmax", "--device", "cpu",
                                 "--select-frames", "--focus-tolerance", "0.85"])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "select" in result.output.lower()
