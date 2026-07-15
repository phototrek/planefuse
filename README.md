# FocusStack

FocusStack is a local, professional focus-stacking workspace with CUDA, Apple MPS,
and CPU execution. It supports developed RGB images and a same-camera, no-bake
camera-RAW workflow that exports a validated, lossless 16-bit Linear DNG for
Capture One Pro.

![FocusStack RAW workspace](docs/assets/raw-workspace.png)

## What is implemented

- Streaming PMax, DMap, weighted, and slab stacking with tiled/CPU OOM recovery.
- Similarity, translation, and perspective alignment with portable Lanczos-3 warping.
- 8/16-bit TIFF, JPEG, and true 16-bit RGB PNG, including ICC/EXIF/XMP handling.
- Same-camera RAW decoding through LibRaw with AHD demosaic and no baked white
  balance, gamma, auto brightness, tone curve, denoise, sharpening, or output-color conversion.
- Atomic, lossless Linear DNG export with camera calibration, source hashes,
  transforms, exclusions, parameters, decoder recipe, and reversible float-to-16-bit mapping.
- Local Svelte workspace with validation, smart-selection review, queue/history,
  deep zoom, synchronized compare, histogram/clipping, retouching, and pro export controls.
- CLI, localhost-only server, CPU/CUDA containers, and cross-platform launchers.

## Quick start

Requirements: Python 3.12, [uv 0.11.28+](https://docs.astral.sh/uv/), and
Node.js 22+.

```bash
uv sync --frozen --extra cpu --extra raw
cd ui && npm ci --no-fund && npm run build && cd ..
uv run --frozen --extra cpu --extra raw focusstack serve
```

On Apple silicon the macOS Torch wheel uses MPS automatically. NVIDIA users on
Windows/Linux select `--extra cu12x` instead of `--extra cpu`.

The launchers perform the same locked setup on first run:

```text
scripts/start-macos.sh
scripts\start-windows-cpu.bat
scripts\start-windows-gpu.bat
```

## CLI

Rendered stack:

```bash
uv run --frozen --extra cpu focusstack stack ./tiffs -o result.tif --method pmax --align
```

No-bake RAW stack and maximum-information Capture One output:

```bash
uv run --frozen --extra cpu --extra raw focusstack stack ./raw-stack \
  -o result.dng --method pmax --align --float-tiff result-scene-linear-float.tif
```

RAW frames must come from the same camera and sensor mode. The DNG is a demosaiced
LinearRaw file, not a re-created sensor mosaic; that is the format that retains the
stacked scene-linear RGB data without inventing mosaic samples.

## Documentation

- [Installation](docs/INSTALL.md)
- [User guide](docs/USER_GUIDE.md)
- [RAW and Capture One workflow](docs/RAW_DNG_CAPTURE_ONE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [API](docs/API.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Performance](docs/PERFORMANCE.md)
- [Release checklist](docs/RELEASE_CHECKLIST.md)
- [Implementation specification](docs/SPEC.md)

## Verification status

As of 2026-07-15: the complete Python suite, static checks, production UI build,
CPU browser workflows, and a targeted MPS RAW→DNG browser workflow pass. The DNG
is reopened by tifffile and LibRaw and pixel-checked before atomic publication.
CUDA hardware and manual import in the current Capture One Pro 16.7.5 remain
explicit release-machine gates; see the [release checklist](docs/RELEASE_CHECKLIST.md).

## Security boundary

The app has no authentication because it binds to `127.0.0.1` only. Docker also
publishes `127.0.0.1:8425:8425`; do not change this to a LAN-wide binding. Source
photos are used in place and are never deleted by project removal.
