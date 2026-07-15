#!/usr/bin/env bash
# Rebuild the UI and capture deterministic screenshots from the generated RAW fixture.
set -euo pipefail
cd "$(dirname "$0")/../ui"
FOCUSSTACK_DOCS_CAPTURE=1 npm run test:e2e -- docs.spec.ts
