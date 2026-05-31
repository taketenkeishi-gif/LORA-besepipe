@echo off
setlocal
set "ROOT=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\stop_all.ps1"
echo.
echo All LoRA Workbench related processes were requested to stop.
pause
