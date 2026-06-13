from typer.testing import CliRunner

from focusstack.cli import app
from focusstack.io import save_image
from tests.synthetic.generate import generate_stack

runner = CliRunner()


def test_cli_align_flag_runs(tmp_path):
    stack = generate_stack(h=160, w=200, n_frames=5, max_sigma=4.0, seed=50,
                           scale_step=0.003, trans_jitter=2.0)
    for i, f in enumerate(stack.frames):
        save_image(f, tmp_path / f"f_{i:03d}.tif", bit_depth=16)
    out = tmp_path / "out.tif"
    result = runner.invoke(app, ["stack", str(tmp_path), "-o", str(out),
                                 "--align", "--align-max-res", "256",
                                 "--device", "cpu"])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "align" in result.output.lower()


def test_cli_pre_aligned_default_no_align(tmp_path):
    stack = generate_stack(h=120, w=120, n_frames=4, max_sigma=4.0, seed=51)
    for i, f in enumerate(stack.frames):
        save_image(f, tmp_path / f"f_{i:03d}.tif", bit_depth=16)
    out = tmp_path / "out.tif"
    result = runner.invoke(app, ["stack", str(tmp_path), "-o", str(out), "--device", "cpu"])
    assert result.exit_code == 0, result.output
