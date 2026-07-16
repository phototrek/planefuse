@echo off
REM Start PlaneFuse on CPU only (no NVIDIA GPU). Slower, but works anywhere.
REM Double-click me, or run from a terminal. Serves UI + API on http://127.0.0.1:8425
cd /d "%~dp0.."

set "UV_EXE="
if exist "tools\uv.exe" set "UV_EXE=%CD%\tools\uv.exe"
if not defined UV_EXE (
  where uv >nul 2>nul || (echo This source folder needs uv 0.11.28+. Visit https://adrienlf.github.io/planefuse/ for guided setup. & exit /b 1)
  set "UV_EXE=uv"
)

if not exist "server\src\planefuse_server\static\index.html" (
  where node >nul 2>nul || (echo Building from source needs Node.js 22+. & exit /b 1)
  where npm >nul 2>nul || (echo Building from source needs npm. & exit /b 1)
  node -e "if (Number(process.versions.node.split('.')[0]) ^< 22) process.exit(1)" || (echo Building from source needs Node.js 22+. & exit /b 1)
  echo Building web UI ^(first run^)...
  pushd ui
  call npm ci --no-fund || (popd & exit /b 1)
  call npm run build || (popd & exit /b 1)
  popd
)

echo.
echo Preparing PlaneFuse ^(the first launch downloads its private Python environment^).
echo Starting on CPU. Your browser opens automatically ^(default http://127.0.0.1:8425^).
echo.
"%UV_EXE%" run --frozen --no-dev --extra cpu --extra raw planefuse serve
exit /b %errorlevel%
