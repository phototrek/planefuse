@echo off
setlocal
cd /d "%~dp0"

where nvidia-smi >nul 2>nul
if errorlevel 1 goto cpu
nvidia-smi >nul 2>nul
if errorlevel 1 goto cpu

echo Starting FocusStack with the NVIDIA GPU...
call "scripts\start-windows-gpu.bat"
goto done

:cpu
echo No working NVIDIA GPU was detected. Starting FocusStack on CPU...
call "scripts\start-windows-cpu.bat"

:done
set "focusstack_exit=%errorlevel%"
if not "%focusstack_exit%"=="0" (
  echo.
  echo FocusStack could not start ^(exit code %focusstack_exit%^).
  pause
)
exit /b %focusstack_exit%
