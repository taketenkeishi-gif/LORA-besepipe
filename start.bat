@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion
set "ROOT=%~dp0"

:MENU
cls
echo.
echo  ============================================
echo    LoRA Basepipe  ^|  起動メニュー
echo  ============================================
echo.
echo    [1]  Web モード       ^(ブラウザで開く^)
echo    [2]  Desktop モード   ^(Electron ウィンドウ^)
echo    [3]  全プロセス停止
echo    [4]  デスクトップショートカット作成
echo    [0]  終了
echo.
choice /c 12340 /n /m "  番号を選択してください: "
set CHOICE=%ERRORLEVEL%

if %CHOICE%==1 goto WEB
if %CHOICE%==2 goto DESKTOP
if %CHOICE%==3 goto STOP
if %CHOICE%==4 goto SHORTCUT
if %CHOICE%==5 goto END
goto MENU

:WEB
echo.
echo  [起動中] Web モード (Backend:8000 + Frontend:5173)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\start_web.ps1"
if errorlevel 1 (
  echo.
  echo  [ERROR] 起動に失敗しました。ログを確認してください。
  echo          .runtime\logs\backend.err.log
)
echo.
pause
goto MENU

:DESKTOP
echo.
echo  [起動中] Desktop モード (Electron)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\start_desktop.ps1"
if errorlevel 1 (
  echo.
  echo  [ERROR] 起動に失敗しました。ログを確認してください。
)
echo.
pause
goto MENU

:STOP
echo.
echo  [停止中] 全プロセスを停止します...
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\stop_all.ps1"
echo.
echo  停止完了 (Port 8000 / 5173)
pause
goto MENU

:SHORTCUT
echo.
echo  [作成中] デスクトップにショートカットを作成します...
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\create_desktop_shortcut.ps1"
if errorlevel 1 (
  echo  [ERROR] ショートカット作成に失敗しました。
) else (
  echo  デスクトップに LoRA_Workbench.lnk を作成しました。
)
pause
goto MENU

:END
exit /b 0
