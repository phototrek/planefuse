import numpy as np
from skimage.metrics import structural_similarity
from typer.testing import CliRunner

from planefuse.cli import app
from planefuse.io import ProcessingDomain, load_image, save_image, validate_linear_dng
from tests.engine.test_raw_loader import write_test_raw
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


def test_stack_command_discovers_raw_and_exports_linear_dng(tmp_path):
    for index in range(2):
        write_test_raw(tmp_path / f"raw_{index:03d}.dng")
    out = tmp_path / "stacked.dng"
    companion = tmp_path / "stacked-float.tif"
    result = runner.invoke(
        app,
        [
            "stack",
            str(tmp_path),
            "-o",
            str(out),
            "--method",
            "weighted",
            "--device",
            "cpu",
            "--float-tiff",
            str(companion),
        ],
    )
    assert result.exit_code == 0, result.output
    assert validate_linear_dng(out).rawpy_validated
    assert companion.exists()
    assert "scene-linear" in result.output.lower()


def test_stack_command_rejects_dng_for_rendered_input(tmp_path):
    for index in range(2):
        save_image(np.zeros((12, 14, 3), dtype=np.float32), tmp_path / f"f{index}.tif")
    result = runner.invoke(
        app,
        ["stack", str(tmp_path), "-o", str(tmp_path / "invalid.dng"), "--device", "cpu"],
    )
    assert result.exit_code != 0
    assert ProcessingDomain.SCENE_LINEAR_CAMERA_RGB.value in _err_text(result)


def test_serve_hint_without_server_package(monkeypatch):
    # The server package is now installed (M4), so simulate its absence to
    # exercise the graceful hint. Without this, `serve` would launch uvicorn
    # and block forever.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("planefuse_server"):
            raise ImportError("simulated missing package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = runner.invoke(app, ["serve"])
    assert result.exit_code != 0
    assert "planefuse-server" in _err_text(result)
