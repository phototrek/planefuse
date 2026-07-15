"""FocusStack CLI (SPEC §14 M1). `focusstack stack DIR -o out.tif --method pmax`."""

from __future__ import annotations

import time
from pathlib import Path

import typer

from focusstack.align import AlignParams
from focusstack.errors import FocusStackError
from focusstack.io import (
    RAW_EXTENSIONS,
    ProcessingDomain,
    save_float_tiff,
    save_image,
    save_linear_dng,
)
from focusstack.pipeline import stack_frames
from focusstack.stack import REGISTRY

app = typer.Typer(no_args_is_help=True, add_completion=False)

EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"} | RAW_EXTENSIONS


@app.command()
def stack(
    input_dir: Path = typer.Argument(..., help="Directory of aligned stack frames"),
    output: Path = typer.Option(..., "-o", "--output", help="Output image (.dng/.tif/.jpg/.png)"),
    method: str = typer.Option("pmax", help=f"Stacking method: {sorted(REGISTRY)}"),
    device: str = typer.Option("auto", help="Compute device: auto|cuda|mps|cpu"),
    selection_smoothing: int = typer.Option(1, min=0, max=3, help="PMax halo control (0=off)"),
    align: bool = typer.Option(False, "--align/--pre-aligned",
                               help="Run §6 alignment first (default: frames are pre-aligned)"),
    align_model: str = typer.Option(
        "similarity", help="Alignment model: translation|similarity|perspective"
    ),
    align_max_res: int = typer.Option(2048, help="Max long edge for alignment estimation"),
    align_interp: str = typer.Option(
        "lanczos3", help="Full-resolution warp: lanczos3|bilinear|bicubic"
    ),
    no_brightness_norm: bool = typer.Option(False, "--no-brightness-norm",
                                            help="Disable flicker/brightness normalization"),
    correlation_threshold: float = typer.Option(0.90, help="ECC quality-gate threshold"),
    reference: int = typer.Option(-1, help="Reference frame index (-1 = middle)"),
    drop_misaligned: bool = typer.Option(False, help="Drop pairs below the quality gate"),
    cache_dir: Path = typer.Option(None, help="Alignment cache directory (default: temp)"),
    select_frames_flag: bool = typer.Option(False, "--select-frames",
                                            help="Thin the stack via §7.0 smart frame selection"),
    focus_tolerance: float = typer.Option(0.85, "--focus-tolerance",
                                          help="Relative in-focus tolerance for selection"),
    tile_size: int = typer.Option(2048, help="Tile size when tiling is needed"),
    jpeg_quality: int = typer.Option(95, min=1, max=100),
    depth_map: Path = typer.Option(None, "--depth-map",
                                   help="For dmap: also write the depth map (16-bit grayscale TIFF)"),
    float_tiff: Path = typer.Option(
        None,
        "--float-tiff",
        help="Also preserve unclamped working values in a 32-bit float TIFF",
    ),
):
    """Stack pre-aligned frames into a single all-in-focus image."""
    paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    if not paths:
        typer.echo(f"error: no input images found in {input_dir}", err=True)
        raise typer.Exit(1)
    typer.echo(f"{len(paths)} frames, method={method}, device={device}")

    align_params = None
    if align:
        align_params = AlignParams(
            model=align_model,
            max_long_edge=align_max_res,
            interp=align_interp,
            normalize_brightness=not no_brightness_norm,
            correlation_threshold=correlation_threshold,
            reference=None if reference < 0 else reference,
            drop_misaligned=drop_misaligned,
        )
        typer.echo(f"alignment: model={align_model} max_res={align_max_res}")

    select_params = None
    if select_frames_flag:
        from focusstack.select import SelectParams
        select_params = SelectParams(focus_tolerance=focus_tolerance)
        typer.echo("frame selection: enabled")

    last = {"msg": ""}

    def progress(msg: str, frac: float) -> None:
        if msg != last["msg"]:
            typer.echo(f"  [{frac * 100:5.1f}%] {msg}")
            last["msg"] = msg

    params = {
        "selection_smoothing": selection_smoothing,
        "estimation_radius": 8, "contrast_threshold": 7.0, "smoothing_radius": 16,
        "temperature": 0.05, "sharpness_radius": 8,
        "slab_size": 10, "slab_overlap": 2, "inner_method": "pmax", "outer_method": "dmap",
    }

    t0 = time.perf_counter()
    try:
        result = stack_frames(
            paths,
            method=method,
            params=params,
            device_pref=device,
            tile=tile_size,
            progress=progress,
            align=align_params,
            cache_dir=cache_dir,
            select=select_params,
        )
    except FocusStackError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from None
    try:
        if output.suffix.lower() == ".dng":
            if result.domain is not ProcessingDomain.SCENE_LINEAR_CAMERA_RGB:
                raise FocusStackError(
                    "Linear DNG output requires processing domain "
                    f"{ProcessingDomain.SCENE_LINEAR_CAMERA_RGB.value}"
                )
            if result.metadata is None:
                raise FocusStackError("Linear DNG output requires reference camera metadata")
            validation = save_linear_dng(
                result.image,
                output,
                result.metadata,
                result.provenance,
            )
            typer.echo(
                "scene-linear no-bake DNG validated with LibRaw "
                f"(max code error {validation.max_code_error})"
            )
        else:
            depth_out = 16 if output.suffix.lower() in (".tif", ".tiff") else 8
            save_image(
                result.image,
                output,
                bit_depth=depth_out,
                metadata=result.metadata,
                provenance=result.provenance,
                jpeg_quality=jpeg_quality,
            )
        if float_tiff is not None:
            save_float_tiff(
                result.image,
                float_tiff,
                metadata=result.metadata,
                provenance=result.provenance,
            )
            typer.echo(f"wrote float TIFF companion {float_tiff}")
    except (FocusStackError, OSError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"wrote {output} in {time.perf_counter() - t0:.1f}s")
    if depth_map is not None and "depth" in result.aux:
        save_image(result.aux["depth"], depth_map, bit_depth=16)
        typer.echo(f"wrote depth map {depth_map}")


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
