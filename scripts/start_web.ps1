param(
  [switch]$Restart
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$runtimeDir = Join-Path $root ".runtime"
$logDir = Join-Path $runtimeDir "logs"
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if ($Restart) {
  Write-Host "[restart] stopping existing backend/frontend"
  & (Join-Path $root "scripts\stop_all.ps1")
  Start-Sleep -Milliseconds 400
}

function Test-Endpoint($url) {
  try {
    $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
    return $r.StatusCode -ge 200 -and $r.StatusCode -lt 500
  } catch {
    return $false
  }
}

function Wait-Endpoint($url, $seconds) {
  $deadline = (Get-Date).AddSeconds($seconds)
  while ((Get-Date) -lt $deadline) {
    if (Test-Endpoint $url) { return $true }
    Start-Sleep -Milliseconds 500
  }
  return $false
}

function Ensure-BackendDeps($rootPath) {
  $backend = Join-Path $rootPath "backend"
  $venvPy = Join-Path $backend ".venv\Scripts\python.exe"
  if (-not (Test-Path $venvPy)) {
    Write-Host "[setup] create backend venv"
    & python -m venv (Join-Path $backend ".venv")
  }
  if (-not (Test-Path $venvPy)) { throw "backend venv python not found" }
  $depsMarker = Join-Path $backend ".venv\deps.ok"
  if (-not (Test-Path $depsMarker)) {
    Write-Host "[setup] install backend dependencies"
    & $venvPy -m pip install -r (Join-Path $backend "requirements.txt")
    New-Item -ItemType File -Path $depsMarker -Force | Out-Null
  }
}

function Ensure-FrontendDeps($rootPath) {
  $frontend = Join-Path $rootPath "frontend"
  if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Write-Host "[setup] install frontend dependencies"
    Push-Location $frontend
    npm install
    Pop-Location
  }
}

function Start-Backend($rootPath) {
  if (Test-Endpoint "http://127.0.0.1:8000/health") {
    Write-Host "[skip] backend already running"
    return
  }
  $backend = Join-Path $rootPath "backend"
  $cmd = "cd /d `"$backend`" && call .venv\Scripts\activate.bat && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
  $out = Join-Path $logDir "backend.out.log"
  $err = Join-Path $logDir "backend.err.log"
  $p = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $cmd -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
  Set-Content -Path (Join-Path $runtimeDir "backend.pid") -Value $p.Id
  if (-not (Wait-Endpoint "http://127.0.0.1:8000/health" 45)) {
    throw "backend failed to start. logs: $out / $err"
  }
  Write-Host "[ok] backend ready"
}

function Start-Frontend($rootPath) {
  if (Test-Endpoint "http://127.0.0.1:5173") {
    Write-Host "[skip] frontend already running"
    return
  }
  $frontend = Join-Path $rootPath "frontend"
  $cmd = "cd /d `"$frontend`" && npm run dev -- --host 127.0.0.1 --port 5173 --strictPort"
  $out = Join-Path $logDir "frontend.out.log"
  $err = Join-Path $logDir "frontend.err.log"
  $p = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $cmd -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
  Set-Content -Path (Join-Path $runtimeDir "frontend.pid") -Value $p.Id
  if (-not (Wait-Endpoint "http://127.0.0.1:5173" 45)) {
    throw "frontend failed to start. logs: $out / $err"
  }
  Write-Host "[ok] frontend ready"
}

Ensure-BackendDeps $root
Ensure-FrontendDeps $root
Start-Backend $root
Start-Frontend $root

Write-Host ""
Write-Host "Started:"
Write-Host "  Backend : http://127.0.0.1:8000/docs"
Write-Host "  Frontend: http://127.0.0.1:5173"
Write-Host "  Logs    : $logDir"
