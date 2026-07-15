#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

status=0
scripts/start-macos.sh || status=$?
if [ "$status" -ne 0 ]; then
  echo
  echo "FocusStack could not start (exit code $status)."
  read -r -p "Press Return to close this window…" _
fi
exit "$status"
