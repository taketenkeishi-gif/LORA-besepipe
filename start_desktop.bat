@echo off
setlocal
set "ROOT=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\start_desktop.ps1"
if errorlevel 1 (
  echo.
  echo [ERROR] Desktop launch failed.
  pause
  exit /b 1
)
echo.
echo Desktop mode started.
pause
