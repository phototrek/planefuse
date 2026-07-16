#!/usr/bin/env bash
# Start PlaneFuse on macOS. The macOS torch wheel uses MPS when available.
# Serves UI + API on localhost only: http://127.0.0.1:8425
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x tools/uv ]; then
  UV_BIN="$PWD/tools/uv"
elif command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
else
  echo "This source folder needs uv 0.11.28+." >&2
  echo "For the guided setup, visit https://adrienlf.github.io/planefuse/." >&2
  exit 1
fi

if [ ! -f server/src/planefuse_server/static/index.html ]; then
  command -v node >/dev/null 2>&1 || { echo "Building from source needs Node.js 22+." >&2; exit 1; }
  command -v npm >/dev/null 2>&1 || { echo "Building from source needs npm." >&2; exit 1; }
  node -e 'const major=Number(process.versions.node.split(".")[0]); if (major < 22) { console.error("Building from source needs Node.js 22+."); process.exit(1); }'
  echo "Building web UI (first run)..."
  (cd ui && npm ci --no-fund && npm run build)
fi

echo
echo "Preparing PlaneFuse (the first launch downloads its private Python environment)."
echo "Starting on macOS with MPS when available, otherwise CPU."
echo "RAW support and validated Linear DNG export are enabled."
echo "Your browser opens automatically at http://127.0.0.1:8425."
echo
exec "$UV_BIN" run --frozen --no-dev --extra cpu --extra raw planefuse serve
