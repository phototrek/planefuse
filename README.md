# FocusStack

Professional GPU-accelerated focus stacking. CUDA (NVIDIA), MPS (Apple silicon), CPU fallback.
See `docs/SPEC.md` for the full specification.

## Quick start

    uv sync --extra cu12x        # NVIDIA (Windows/Linux); --extra cpu without a GPU; plain uv sync on Apple silicon
    uv run focusstack stack ./my_stack -o result.tif --method pmax

## Status

- [x] M1 — engine core, PMax, CLI
- [x] M2 — alignment
- [ ] M3 — DMap, weighted, slabbing, smart frame selection
- [ ] M4 — server + web UI
- [ ] M5 — retouching
- [ ] M6 — polish, Docker, CI

## Development

    uv run pytest          # CPU + whatever accelerator this machine has
    uv run ruff check .
    uv run mypy engine/src
