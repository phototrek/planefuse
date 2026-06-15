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
- [~] M5 — retouch engine + server done; retouch UI pending
- [ ] M6 — polish, Docker, CI

## Web UI

The Svelte UI lives in `ui/` and builds into the server's static dir, so one
process serves UI + API:

    cd ui && npm install && npm run build
    uv run focusstack serve        # opens http://127.0.0.1:8425

For UI development with hot reload, run the server and the Vite dev server side
by side (Vite proxies `/api` and `/ws` to the server):

    uv run focusstack serve        # terminal 1
    cd ui && npm run dev           # terminal 2 → http://localhost:5173

## Development

    uv run pytest          # CPU + whatever accelerator this machine has
    uv run ruff check .
    uv run mypy engine/src
    cd ui && npm run check         # svelte-check
    cd ui && npm run test:e2e      # Playwright smoke
