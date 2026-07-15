# syntax=docker/dockerfile:1
# Multi-stage build, two final targets (SPEC §16): `cpu` (default in CI / non-NVIDIA
# hosts) and `cuda` (run with --gpus all). MPS is not reachable from containers;
# Apple-silicon users run natively via uv.

# --- UI build: produces the static SPA into the server package's static dir ---
FROM node:22.23.1-slim AS ui
WORKDIR /app/ui
COPY ui/package.json ui/package-lock.json ./
# --fetch-retries guards against npm silently skipping an optional native dep
# (e.g. @rollup/rollup-linux-x64-gnu) when a download stalls (npm bug #4828).
RUN npm ci --no-audit --no-fund --fetch-retries=5 --fetch-retry-maxtimeout=120000
COPY ui/ ./
# adapter-static emits to ../server/src/focusstack_server/static (svelte.config.js)
RUN npm run build

# --- CPU target: small image on python:3.12-slim ---
FROM python:3.12.11-slim-bookworm AS cpu
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_HTTP_TIMEOUT=300
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY engine/ engine/
COPY server/ server/
RUN uv sync --frozen --no-dev --extra cpu --extra raw
COPY --from=ui /app/server/src/focusstack_server/static server/src/focusstack_server/static
ENV FOCUSSTACK_HOST=0.0.0.0 FOCUSSTACK_PORT=8425 FOCUSSTACK_DATA_DIR=/data
EXPOSE 8425
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8425/api/system').status==200 else 1)"
CMD ["uv", "run", "--frozen", "--no-dev", "--extra", "cpu", "--extra", "raw", "focusstack", "serve"]

# --- CUDA target: runtime on nvidia/cuda; uv manages Python 3.12 ---
# CUDA 12.8 runtime to match the cu128 torch wheels pinned in pyproject.toml.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS cuda
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_HTTP_TIMEOUT=300 UV_PYTHON_INSTALL_DIR=/opt/uv-python
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY engine/ engine/
COPY server/ server/
RUN uv python install 3.12.11 && uv sync --frozen --no-dev --extra cu12x --extra raw
COPY --from=ui /app/server/src/focusstack_server/static server/src/focusstack_server/static
ENV FOCUSSTACK_HOST=0.0.0.0 FOCUSSTACK_PORT=8425 FOCUSSTACK_DATA_DIR=/data
EXPOSE 8425
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8425/api/system').status==200 else 1)"
CMD ["uv", "run", "--frozen", "--no-dev", "--extra", "cu12x", "--extra", "raw", "focusstack", "serve"]
