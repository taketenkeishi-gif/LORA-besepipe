param()

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$runtimeDir = Join-Path $root ".runtime"
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

& (Join-Path $root "scripts\start_web.ps1") -Restart

function Ensure-DesktopDeps($rootPath) {
  $desktop = Join-Path $rootPath "desktop"
  if (-not (Test-Path (Join-Path $desktop "node_modules"))) {
    Write-Host "[setup] install desktop dependencies"
    Push-Location $desktop
    npm install
    Pop-Location
  }
}

function Start-Desktop($rootPath) {
  $desktop = Join-Path $rootPath "desktop"
  $electronCmd = Join-Path $desktop "node_modules\.bin\electron.cmd"
  if (-not (Test-Path $electronCmd)) {
    throw "electron.cmd not found"
  }
  $p = Start-Process -FilePath $electronCmd -ArgumentList "." -WorkingDirectory $desktop -PassThru
  Set-Content -Path (Join-Path $runtimeDir "desktop.pid") -Value $p.Id
  Write-Host "[ok] desktop launched"
}

Ensure-DesktopDeps $root
Start-Desktop $root

