@echo off
setlocal

set "ROOT=%~dp0"

echo [LoRA Workbench] Starting Web mode...

start "LORA Backend" cmd /k "cd /d ""%ROOT%backend"" && if not exist .venv python -m venv .venv && call .venv\Scripts\activate.bat && pip install -r requirements.txt && python -m app.main"
start "LORA Frontend" cmd /k "cd /d ""%ROOT%frontend"" && npm install && npm run dev -- --host 127.0.0.1 --port 5173"

echo.
echo Backend:  http://127.0.0.1:8000/docs
echo Frontend: http://127.0.0.1:5173
echo.
echo Press any key to close this launcher.
pause >nul
