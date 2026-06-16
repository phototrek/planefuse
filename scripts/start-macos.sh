#!/usr/bin/env bash
# Start FocusStack on macOS. The "cpu" extra pulls the macOS torch wheel, which
# uses Apple Silicon's MPS GPU automatically (falls back to CPU on Intel Macs).
# There is no CUDA on macOS. Serves UI + API on http://127.0.0.1:8425
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f server/src/focusstack_server/static/index.html ]; then
  echo "Building web UI (first run)..."
  (cd ui && npm install && npm run build)
fi

echo
echo "Starting FocusStack (macOS / MPS). Your browser opens automatically (default http://127.0.0.1:8425)."
echo
exec uv run --extra cpu focusstack serve
