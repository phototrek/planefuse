# Installation

## Recommended: guided setup

Open the
[visual PlaneFuse installation assistant](https://adrienlf.github.io/planefuse/).
It detects the computer and keeps progress through four short steps. The page
is also available as [`installer/index.html`](../installer/index.html) in the
repository.

The downloadable ready folder contains PlaneFuse's source, its already-built
web interface, and the official uv 0.11.28 executable for that platform. On
first launch, uv downloads and manages the required Python version and locked
processing libraries automatically. Photographers do not install Node.js, uv,
or Python themselves. The assistant points to the `continuous` release, which
CI refreshes only after the current `main` commit passes the complete Python,
UI, Docker, macOS, Windows, and first-launch test matrix.

### Apple-silicon macOS 14 or newer

1. Download `PlaneFuse-macOS-Apple-Silicon.zip` from the assistant.
2. Double-click the ZIP and move the resulting **PlaneFuse** folder somewhere
   convenient.
3. Inside the folder, right-click `Launch PlaneFuse.command` and choose
   **Open**.
4. Keep the Terminal window open while the first-time setup finishes.

The launcher uses Apple MPS acceleration when available and otherwise falls
back to CPU.

### Windows 11 x64

1. Download `PlaneFuse-Windows-x64.zip` from the assistant.
2. Right-click the ZIP, choose **Extract All**, and open the resulting
   **PlaneFuse** folder.
3. Double-click `Launch PlaneFuse.bat`.
4. Keep the setup window open while the first-time setup finishes.

The Windows launcher selects a working NVIDIA GPU automatically and falls back
to CPU.

The first launch downloads the private environment and can take 5–20 minutes;
NVIDIA libraries may take longer. Later launches reuse it and normally start
in seconds. PlaneFuse serves its interface only on `127.0.0.1`, opens the
browser automatically, and selects another local port if 8425 is busy.
Each folder contains `RELEASE.json` identifying its exact source commit.
`SHA256SUMS` is published beside the ZIPs for download verification.

## Source installation

Source installation is for development, Linux, and anyone who prefers a normal
repository checkout.

### Requirements

- Python 3.12
- uv 0.11.28 or newer
- Node.js 22 or newer and npm
- A current GPU driver for CUDA; macOS 14+ for MPS

Install uv using its official instructions:

```bash
curl -LsSf https://astral.sh/uv/0.11.28/install.sh | sh
```

On Windows, `winget install --id=astral-sh.uv -e` is also supported.

Apple silicon or CPU:

```bash
uv sync --frozen --extra cpu --extra raw
cd ui
npm ci --no-fund
npm run build
cd ..
uv run --frozen --extra cpu --extra raw planefuse serve
```

Windows/Linux with NVIDIA CUDA 12.8:

```bash
uv sync --frozen --extra cu12x --extra raw
cd ui
npm ci --no-fund
npm run build
cd ..
uv run --frozen --extra cu12x --extra raw planefuse serve
```

`cpu` and `cu12x` are mutually exclusive. `raw` adds rawpy/LibRaw and
ExifRead. The richer pyexiv2 adapter is optional; core metadata and RAW
workflows do not require Homebrew or a native Exiv2 installation.

## Source launchers

- On macOS, double-click `Launch PlaneFuse.command` in Finder.
- On Windows, double-click `Launch PlaneFuse.bat` in Explorer. A working NVIDIA
  driver selects the GPU launcher; otherwise it falls back to CPU.

The source launchers check required commands, use `npm ci`, build the UI only
when absent, and launch from the frozen lockfile. A release ready folder skips
the Node.js checks because the UI is already present and uses its bundled uv
executable. For explicit device selection or troubleshooting, run
`scripts/start-windows-gpu.bat`,
`scripts/start-windows-cpu.bat`, or `scripts/start-macos.sh` directly.

## Docker

CPU:

```bash
docker build --target cpu -t planefuse:cpu .
docker run --rm -p 127.0.0.1:8425:8425 -v planefuse-data:/data planefuse:cpu
```

NVIDIA:

```bash
PLANEFUSE_PHOTOS=/absolute/path/to/photos docker compose up --build
```

The host port stays localhost-only and the photo mount is read-only. Docker on
macOS cannot expose MPS; use the ready-folder setup for Apple GPU acceleration.
