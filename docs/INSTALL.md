# Installation

## Requirements

- Python 3.12
- uv 0.11.28 or newer
- Node.js 22 or newer and npm
- Current GPU driver for CUDA; macOS 14+ recommended for MPS FFT support

Install the pinned uv release using the official instructions:

```bash
curl -LsSf https://astral.sh/uv/0.11.28/install.sh | sh
```

On Windows, `winget install --id=astral-sh.uv -e` is also supported by uv.

## Native install

Apple silicon or CPU:

```bash
uv sync --frozen --extra cpu --extra raw
cd ui
npm ci --no-fund
npm run build
cd ..
uv run --frozen --extra cpu --extra raw focusstack serve
```

Windows/Linux with NVIDIA CUDA 12.8:

```bash
uv sync --frozen --extra cu12x --extra raw
cd ui
npm ci --no-fund
npm run build
cd ..
uv run --frozen --extra cu12x --extra raw focusstack serve
```

`cpu` and `cu12x` are mutually exclusive. `raw` adds rawpy/LibRaw and ExifRead.
The richer pyexiv2 adapter is an optional `metadata` extra; core metadata and RAW
workflows do not require Homebrew or a native Exiv2 installation.

## One-click launchers

- On macOS, double-click `Launch FocusStack.command` in Finder.
- On Windows, double-click `Launch FocusStack.bat` in Explorer. A working NVIDIA
  driver selects the GPU launcher by default; otherwise it falls back to CPU.

The root launchers delegate to the scripts in [scripts](../scripts), which check
required commands, use `npm ci`, build the UI only when absent, and launch from
the frozen lockfile. Their process exit code is propagated to the terminal.

For explicit device selection or troubleshooting, run
`scripts/start-windows-gpu.bat`, `scripts/start-windows-cpu.bat`, or
`scripts/start-macos.sh` directly.

## Docker

CPU:

```bash
docker build --target cpu -t focusstack:cpu .
docker run --rm -p 127.0.0.1:8425:8425 -v focusstack-data:/data focusstack:cpu
```

NVIDIA:

```bash
FOCUSSTACK_PHOTOS=/absolute/path/to/photos docker compose up --build
```

The host port stays localhost-only and the photo mount is read-only. Docker on
macOS cannot expose MPS; use the native install for Apple GPU acceleration.
