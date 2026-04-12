param()

$ErrorActionPreference = "Stop"

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

function Ensure-BackendDeps($root) {
  $backend = Join-Path $root "backend"
  $venvPy = Join-Path $backend ".venv\Scripts\python.exe"
  if (-not (Test-Path $venvPy)) {
    Write-Host "[setup] Creating backend venv..."
    & python -m venv (Join-Path $backend ".venv")
  }
  if (-not (Test-Path $venvPy)) {
    throw "Backend venv python not found."
  }
  if (-not (Test-Path (Join-Path $backend ".venv\deps.ok"))) {
    Write-Host "[setup] Installing backend dependencies..."
    & $venvPy -m pip install -r (Join-Path $backend "requirements.txt")
    New-Item -ItemType File -Path (Join-Path $backend ".venv\deps.ok") -Force | Out-Null
  }
}

function Ensure-FrontendDeps($root) {
  $frontend = Join-Path $root "frontend"
  if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Write-Host "[setup] Installing frontend dependencies..."
    Push-Location $frontend
    npm install
    Pop-Location
  }
}

function Start-Backend($root) {
  if (Test-Endpoint "http://127.0.0.1:8000/health") {
    Write-Host "[skip] Backend already running."
    return
  }
  $backend = Join-Path $root "backend"
  $cmd = "cd /d `"$backend`" && call .venv\Scripts\activate.bat && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
  Start-Process -FilePath "cmd.exe" -ArgumentList "/k", $cmd -WindowStyle Normal | Out-Null
  if (-not (Wait-Endpoint "http://127.0.0.1:8000/health" 40)) {
    throw "Backend failed to start on 127.0.0.1:8000"
  }
  Write-Host "[ok] Backend is ready."
}

function Start-Frontend($root) {
  if (Test-Endpoint "http://127.0.0.1:5173") {
    Write-Host "[skip] Frontend already running."
    return
  }
  $frontend = Join-Path $root "frontend"
  $cmd = "cd /d `"$frontend`" && npm run dev -- --host 127.0.0.1 --port 5173 --strictPort"
  Start-Process -FilePath "cmd.exe" -ArgumentList "/k", $cmd -WindowStyle Normal | Out-Null
  if (-not (Wait-Endpoint "http://127.0.0.1:5173" 40)) {
    throw "Frontend failed to start on 127.0.0.1:5173"
  }
  Write-Host "[ok] Frontend is ready."
}

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Ensure-BackendDeps $root
Ensure-FrontendDeps $root
Start-Backend $root
Start-Frontend $root

Write-Host ""
Write-Host "Backend : http://127.0.0.1:8000/docs"
Write-Host "Frontend: http://127.0.0.1:5173"
