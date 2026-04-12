param()

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

# Web stack startup
& (Join-Path $root "scripts\start_web.ps1")

function Ensure-DesktopDeps($rootPath) {
  $desktop = Join-Path $rootPath "desktop"
  if (-not (Test-Path (Join-Path $desktop "node_modules"))) {
    Write-Host "[setup] Installing desktop dependencies..."
    Push-Location $desktop
    npm install
    Pop-Location
  }
}

function Start-Desktop($rootPath) {
  $desktop = Join-Path $rootPath "desktop"
  $cmd = "cd /d `"$desktop`" && npm run start"
  Start-Process -FilePath "cmd.exe" -ArgumentList "/k", $cmd -WindowStyle Normal | Out-Null
  Write-Host "[ok] Desktop app launch command sent."
}

Ensure-DesktopDeps $root
Start-Desktop $root
