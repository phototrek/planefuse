#!/usr/bin/env bash
# Start PlaneFuse on macOS. The macOS torch wheel uses MPS when available.
# Serves UI + API on localhost only: http://127.0.0.1:8425
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null 2>&1 || { echo "PlaneFuse needs uv 0.11.28+ (https://docs.astral.sh/uv/)." >&2; exit 1; }
command -v node >/dev/null 2>&1 || { echo "PlaneFuse needs Node.js 22+." >&2; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "PlaneFuse needs npm." >&2; exit 1; }
node -e 'const major=Number(process.versions.node.split(".")[0]); if (major < 22) { console.error("PlaneFuse needs Node.js 22+."); process.exit(1); }'

if [ ! -f server/src/planefuse_server/static/index.html ]; then
  echo "Building web UI (first run)..."
  (cd ui && npm ci --no-fund && npm run build)
fi

echo
echo "Starting PlaneFuse (macOS: MPS when available, otherwise CPU)."
echo "RAW support and validated Linear DNG export are enabled."
echo "Your browser opens automatically at http://127.0.0.1:8425."
echo
exec uv run --frozen --extra cpu --extra raw planefuse serve
