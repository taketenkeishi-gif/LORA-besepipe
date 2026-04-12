@echo off
setlocal
set "ROOT=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\start_web.ps1"
if errorlevel 1 (
  echo.
  echo [ERROR] Web launch failed.
  pause
  exit /b 1
)
echo.
echo Web mode started.
pause
