$ErrorActionPreference = 'Stop'
$appRoot = Split-Path -Parent $PSScriptRoot
$desktopRoot = Join-Path $appRoot 'desktop'
$exe = Join-Path $desktopRoot 'runtime/LoRA Studio.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw 'LoRA Studioの実行ファイルが見つかりません。' }
$index = Join-Path $appRoot 'frontend/dist/index.html'
$newestSource = Get-ChildItem -LiteralPath (Join-Path $appRoot 'frontend/src') -Recurse -File | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
if (-not (Test-Path -LiteralPath $index) -or $newestSource.LastWriteTimeUtc -gt (Get-Item -LiteralPath $index).LastWriteTimeUtc) {
    Push-Location (Join-Path $appRoot 'frontend')
    try { & npm.cmd run build; if ($LASTEXITCODE -ne 0) { throw '画面のビルドに失敗しました。' } } finally { Pop-Location }
}
$packagedRoot = Join-Path $desktopRoot 'runtime/resources/app'
New-Item -ItemType Directory -Path $packagedRoot -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $desktopRoot 'main.cjs'),(Join-Path $desktopRoot 'preload.cjs'),(Join-Path $desktopRoot 'package.json') -Destination $packagedRoot -Force
$env:LORA_STUDIO_ROOT = $appRoot
Start-Process -FilePath $exe -WorkingDirectory $appRoot
