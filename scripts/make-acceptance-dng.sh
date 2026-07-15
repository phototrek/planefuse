#!/usr/bin/env bash
# Build a redistributable synthetic Linear DNG for manual importer acceptance.
set -euo pipefail
cd "$(dirname "$0")/.."
work="$(mktemp -d -t focusstack-acceptance.XXXXXX)"
trap 'rm -rf "$work"' EXIT
uv run --frozen --extra cpu --extra raw python ui/tests/fixtures/make_raw_stack.py "$work/frames"
uv run --frozen --extra cpu --extra raw focusstack stack "$work/frames" \
  --output docs/assets/focusstack-linear-dng-acceptance.dng \
  --method weighted \
  --device cpu
