# LoRA Basepipe - 起動メニュー
$ROOT = Split-Path $PSScriptRoot -Parent

function Show-Menu {
    Clear-Host
    Write-Host ""
    Write-Host "  ============================================" -ForegroundColor DarkCyan
    Write-Host "    LoRA Basepipe  |  起動メニュー" -ForegroundColor Cyan
    Write-Host "  ============================================" -ForegroundColor DarkCyan
    Write-Host ""
    Write-Host "    [1]  Web モード       " -NoNewline -ForegroundColor Green
    Write-Host "(ブラウザ: http://127.0.0.1:5173)"
    Write-Host "    [2]  Desktop モード   " -NoNewline -ForegroundColor Blue
    Write-Host "(Electron 起動)"
    Write-Host "    [3]  停止する" -ForegroundColor Yellow
    Write-Host "    [4]  デスクトップショートカット作成" -ForegroundColor Gray
    Write-Host "    [0]  終了" -ForegroundColor DarkGray
    Write-Host ""
}

while ($true) {
    Show-Menu
    $choice = Read-Host "  選択してください"

    switch ($choice) {
        "1" {
            Write-Host ""
            Write-Host "  [起動] Web モード (Backend:8000 + Frontend:5173)..." -ForegroundColor Green
            & powershell -NoProfile -ExecutionPolicy Bypass -File "$ROOT\scripts\start_web.ps1"
            if ($LASTEXITCODE -ne 0) {
                Write-Host ""
                Write-Host "  [ERROR] バックエンドの起動に失敗しました" -ForegroundColor Red
                Write-Host "          ログ: $ROOT\.runtime\logs\backend.err.log" -ForegroundColor DarkRed
            }
            Write-Host ""
            Read-Host "  Enterキーでメニューに戻る"
        }
        "2" {
            Write-Host ""
            Write-Host "  [起動] Desktop モード (Electron)..." -ForegroundColor Blue
            & powershell -NoProfile -ExecutionPolicy Bypass -File "$ROOT\scripts\start_desktop.ps1"
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  [ERROR] デスクトップモードの起動に失敗しました" -ForegroundColor Red
            }
            Write-Host ""
            Read-Host "  Enterキーでメニューに戻る"
        }
        "3" {
            Write-Host ""
            Write-Host "  [停止] プロセスを停止中..." -ForegroundColor Yellow
            & powershell -NoProfile -ExecutionPolicy Bypass -File "$ROOT\scripts\stop_all.ps1"
            Write-Host "  停止完了 (Port 8000 / 5173)" -ForegroundColor Green
            Write-Host ""
            Read-Host "  Enterキーでメニューに戻る"
        }
        "4" {
            Write-Host ""
            Write-Host "  [作成] デスクトップショートカットを作成中..." -ForegroundColor Gray
            & powershell -NoProfile -ExecutionPolicy Bypass -File "$ROOT\scripts\create_desktop_shortcut.ps1"
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  [ERROR] ショートカット作成に失敗しました" -ForegroundColor Red
            } else {
                Write-Host "  作成完了: LoRA_Workbench.lnk をデスクトップに配置しました" -ForegroundColor Green
            }
            Write-Host ""
            Read-Host "  Enterキーでメニューに戻る"
        }
        "0" {
            Write-Host ""
            Write-Host "  終了します。" -ForegroundColor DarkGray
            exit 0
        }
        default {
            Write-Host "  無効な入力です。1-4 または 0 を入力してください。" -ForegroundColor DarkYellow
            Start-Sleep -Seconds 1
        }
    }
}