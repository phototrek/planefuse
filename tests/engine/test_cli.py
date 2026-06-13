import numpy as np
from skimage.metrics import structural_similarity
from typer.testing import CliRunner

from focusstack.cli import app
from focusstack.io import load_image, save_image
from tests.synthetic.generate import generate_stack

runner = CliRunner()


def test_stack_command_end_to_end(tmp_path):
    synth = generate_stack(h=96, w=128, n_frames=6, max_sigma=4.0, seed=4)
    for i, f in enumerate(synth.frames):
        save_image(f, tmp_path / f"f_{i:03d}.tif", bit_depth=16)
    out = tmp_path / "stacked.tif"
    result = runner.invoke(
        app, ["stack", str(tmp_path), "-o", str(out), "--method", "pmax", "--device", "cpu"]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    img = load_image(out)
    s = structural_similarity(img.pixels, synth.sharp.clip(0, 1), channel_axis=2, data_range=1.0)
    assert s > 0.95
    assert "cpu" in result.output.lower()


def _err_text(result) -> str:
    # click >= 8.2 captures stderr separately (result.stderr); older click
    # mixed it into result.output. Handle both so the suite isn't pinned.
    try:
        return result.output + result.stderr
    except ValueError:
        return result.output


def test_stack_command_no_images(tmp_path):
    result = runner.invoke(app, ["stack", str(tmp_path), "-o", str(tmp_path / "x.tif")])
    assert result.exit_code != 0
    assert "no input images" in _err_text(result).lower()


def test_stack_command_validation_error(tmp_path):
    save_image(np.zeros((10, 10, 3), dtype=np.float32), tmp_path / "a.tif")
    save_image(np.zeros((20, 20, 3), dtype=np.float32), tmp_path / "b.tif")
    result = runner.invoke(app, ["stack", str(tmp_path), "-o", str(tmp_path / "x.tif")])
    assert result.exit_code != 0
    assert "wrong_size" in _err_text(result)


def test_serve_hint_without_server_package():
    result = runner.invoke(app, ["serve"])
    assert result.exit_code != 0
    assert "focusstack-server" in _err_text(result)
