@echo off
REM Start PlaneFuse on an NVIDIA GPU (CUDA 12.x torch wheel).
REM Double-click me, or run from a terminal. Serves UI + API on http://127.0.0.1:8425
cd /d "%~dp0.."

where uv >nul 2>nul || (echo PlaneFuse needs uv 0.11.28+ ^(https://docs.astral.sh/uv/^). & exit /b 1)
where node >nul 2>nul || (echo PlaneFuse needs Node.js 22+. & exit /b 1)
where npm >nul 2>nul || (echo PlaneFuse needs npm. & exit /b 1)
where nvidia-smi >nul 2>nul || (echo No NVIDIA driver was found. Use start-windows-cpu.bat instead. & exit /b 1)
node -e "if (Number(process.versions.node.split('.')[0]) ^< 22) process.exit(1)" || (echo PlaneFuse needs Node.js 22+. & exit /b 1)

if not exist "server\src\planefuse_server\static\index.html" (
  echo Building web UI ^(first run^)...
  pushd ui
  call npm ci --no-fund || (popd & exit /b 1)
  call npm run build || (popd & exit /b 1)
  popd
)

echo.
echo Starting PlaneFuse ^(GPU / CUDA^). Your browser opens automatically ^(default http://127.0.0.1:8425^).
echo.
uv run --frozen --extra cu12x --extra raw planefuse serve
exit /b %errorlevel%
