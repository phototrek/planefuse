"""FocusStack CLI (SPEC §14 M1). `focusstack stack DIR -o out.tif --method pmax`."""

from __future__ import annotations

import time
from pathlib import Path

import typer

from focusstack.errors import FocusStackError
from focusstack.io import load_image, save_image
from focusstack.pipeline import stack_frames
from focusstack.stack import REGISTRY

app = typer.Typer(no_args_is_help=True, add_completion=False)

EXTENSIONS = ("*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png")


@app.command()
def stack(
    input_dir: Path = typer.Argument(..., help="Directory of aligned stack frames"),
    output: Path = typer.Option(..., "-o", "--output", help="Output image (.tif/.jpg/.png)"),
    method: str = typer.Option("pmax", help=f"Stacking method: {sorted(REGISTRY)}"),
    device: str = typer.Option("auto", help="Compute device: auto|cuda|mps|cpu"),
    selection_smoothing: int = typer.Option(1, min=0, max=3, help="PMax halo control (0=off)"),
    tile_size: int = typer.Option(2048, help="Tile size when tiling is needed"),
    jpeg_quality: int = typer.Option(95, min=1, max=100),
):
    """Stack pre-aligned frames into a single all-in-focus image."""
    paths: list[Path] = []
    for pattern in EXTENSIONS:
        paths.extend(input_dir.glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        typer.echo(f"error: no input images found in {input_dir}", err=True)
        raise typer.Exit(1)
    typer.echo(f"{len(paths)} frames, method={method}, device={device}")

    last = {"msg": ""}

    def progress(msg: str, frac: float) -> None:
        if msg != last["msg"]:
            typer.echo(f"  [{frac * 100:5.1f}%] {msg}")
            last["msg"] = msg

    t0 = time.perf_counter()
    try:
        result = stack_frames(
            paths,
            method=method,
            params={"selection_smoothing": selection_smoothing},
            device_pref=device,
            tile=tile_size,
            progress=progress,
        )
    except FocusStackError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from None
    # ICC from the reference (middle) frame, SPEC §5
    icc = load_image(paths[len(paths) // 2]).icc
    depth_out = 16 if output.suffix.lower() in (".tif", ".tiff") else 8
    save_image(result.image, output, bit_depth=depth_out, icc=icc, jpeg_quality=jpeg_quality)
    typer.echo(f"wrote {output} in {time.perf_counter() - t0:.1f}s")


@app.command()
def serve():
    """Launch the FocusStack server + web UI (requires the focusstack-server package)."""
    try:
        from focusstack_server.main import run  # type: ignore[import-not-found]
    except ImportError:
        typer.echo(
            "error: the web UI is not installed. Install the focusstack-server package "
            "(coming in milestone M4): uv sync (workspace) or pip install focusstack-server",
            err=True,
        )
        raise typer.Exit(1) from None
    run()
