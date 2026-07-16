@echo off
setlocal
cd /d "%~dp0"

where nvidia-smi >nul 2>nul
if errorlevel 1 goto cpu
nvidia-smi >nul 2>nul
if errorlevel 1 goto cpu

echo Starting PlaneFuse with the NVIDIA GPU...
call "scripts\start-windows-gpu.bat"
goto done

:cpu
echo No working NVIDIA GPU was detected. Starting PlaneFuse on CPU...
call "scripts\start-windows-cpu.bat"

:done
set "planefuse_exit=%errorlevel%"
if not "%planefuse_exit%"=="0" (
  echo.
  echo PlaneFuse could not start ^(exit code %planefuse_exit%^).
  pause
)
exit /b %planefuse_exit%
