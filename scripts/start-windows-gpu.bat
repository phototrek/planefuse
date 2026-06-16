@echo off
REM Start FocusStack on an NVIDIA GPU (CUDA 12.x torch wheel).
REM Double-click me, or run from a terminal. Serves UI + API on http://127.0.0.1:8425
cd /d "%~dp0.."

if not exist "server\src\focusstack_server\static\index.html" (
  echo Building web UI ^(first run^)...
  pushd ui && call npm install && call npm run build && popd
)

echo.
echo Starting FocusStack ^(GPU / CUDA^). Your browser opens automatically ^(default http://127.0.0.1:8425^).
echo.
uv run --extra cu12x focusstack serve
