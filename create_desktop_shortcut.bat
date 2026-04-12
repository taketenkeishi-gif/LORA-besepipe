@echo off
setlocal
set "ROOT=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\create_desktop_shortcut.ps1"
if errorlevel 1 (
  echo Failed to create shortcut.
  pause
  exit /b 1
)
echo Shortcut created on Desktop.
pause
