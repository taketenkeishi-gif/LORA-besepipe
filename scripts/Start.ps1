param([switch]$Open)
$ErrorActionPreference = 'Stop'
$appRoot = Split-Path -Parent $PSScriptRoot
$url = 'http://127.0.0.1:5175'
$ready = $false
try {
    $instance = Invoke-RestMethod "$url/instance" -TimeoutSec 2
    if ($instance.application -ne 'LoRA Studio Next' -or $instance.root -ne $appRoot) { throw 'Port belongs to another application' }
    if ($instance.python_prefix -ne (Join-Path $appRoot 'backend/.venv')) { throw '専用Python環境への切り替えが未反映です。起動中の新サーバーの再起動が必要です。' }
    $ready = $true
} catch {
    if (Get-NetTCPConnection -State Listen -LocalPort 5175 -ErrorAction SilentlyContinue) {
        throw '5175番ポートは別のアプリが使用しています。停止せず中止しました。'
    }
}
if (-not $ready) {
    $python = Join-Path $appRoot 'backend/.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $python)) { throw 'このアプリ専用のPython環境が見つかりません。READMEを参照してください。' }
    $process = Start-Process -FilePath $python -ArgumentList '-m','uvicorn','desktop_server:app','--host','127.0.0.1','--port','5175','--log-level','warning' -WorkingDirectory (Join-Path $appRoot 'backend') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $appRoot '.runtime/server.out.log') -RedirectStandardError (Join-Path $appRoot '.runtime/server.err.log') -PassThru
    $process.Id | Set-Content (Join-Path $appRoot '.runtime/launcher.pid')
    $deadline = (Get-Date).AddSeconds(45)
    do {
        try { $ready = (Invoke-RestMethod "$url/instance" -TimeoutSec 2).application -eq 'LoRA Studio Next' } catch { }
        if (-not $ready) { Start-Sleep -Milliseconds 500 }
    } until ($ready -or (Get-Date) -ge $deadline)
    if (-not $ready) { throw '起動確認が45秒以内に完了しませんでした。.runtime/server.err.logを確認してください。' }
}
if ($Open) { Start-Process $url }
Write-Output $url
