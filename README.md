# FocusStack

Professional GPU-accelerated focus stacking. CUDA (NVIDIA), MPS (Apple silicon), CPU fallback.
See `docs/SPEC.md` for the full specification.

## Quick start

    uv sync --extra cu12x        # NVIDIA (Windows/Linux); --extra cpu without a GPU; plain uv sync on Apple silicon
    uv run focusstack stack ./my_stack -o result.tif --method pmax

## Status

- [x] M1 — engine core, PMax, CLI
- [x] M2 — alignment
- [x] M3 — DMap, weighted, slabbing, smart frame selection
- [x] M4 — server + web UI (import → stack → view → export)
- [x] M5 — retouch engine, server, UI, and browser workflow complete
- [~] M6 — Docker + compose + CI done; UI polish (histogram, compare, export templates, validation UX) pending

## Web UI

The Svelte UI is a single workspace screen: add inputs (folders or individual
files) → choose an algorithm → **Run** → results land in the render drawer →
view with deep-zoom → export. Retouch launches from a finished result.

It lives in `ui/` and builds into the server's static dir, so one process
serves UI + API:

    cd ui && npm install && npm run build
    uv run --extra cu12x focusstack serve   # NVIDIA/CUDA GPU; opens http://127.0.0.1:8425
    # CPU-only machine: uv run --extra cpu focusstack serve

The `cpu` and `cu12x` extras are mutually exclusive and have no default — pick
one explicitly. A plain `uv run …` reverts the env to the CPU torch build, so
always pass `--extra cu12x` to use the GPU (check it worked: `/api/system`
reports `"device": "cuda"`).

For UI development with hot reload, run the server and the Vite dev server side
by side (Vite proxies `/api` and `/ws` to the server, so open **5173**, not 8425):

    uv run --extra cu12x focusstack serve   # terminal 1 (GPU)
    cd ui && npm run dev                     # terminal 2 → http://localhost:5173

## Docker

Two image targets share one `Dockerfile`:

    docker compose up --build            # CUDA target, needs nvidia-container-toolkit
    docker build --target cpu -t focusstack:cpu .   # CPU-only, smaller, no GPU

Compose binds the port to `127.0.0.1` only (the server is auth-less and localhost-only
by design), mounts your photo library read-only at `/photos`, and keeps projects/cache
in the `fs-data` volume. Set `FOCUSSTACK_PHOTOS=/path/to/photos` to point at your library.

**Apple silicon:** Docker on macOS has no GPU passthrough, so MPS is unreachable from
containers. Run natively with `uv sync && uv run focusstack serve` for MPS acceleration;
the `cpu` image works but is the slow path.

## Development

    uv run pytest          # CPU + whatever accelerator this machine has
    uv run ruff check .
    uv run mypy engine/src
    cd ui && npm run check         # svelte-check
    cd ui && npm run test:e2e      # Playwright smoke
