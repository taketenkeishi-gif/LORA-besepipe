param()

$ErrorActionPreference = "SilentlyContinue"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$runtimeDir = Join-Path $root ".runtime"

function Kill-Tree($targetProcId) {
  if ($targetProcId -and $targetProcId -match "^\d+$") {
    $exists = Get-Process -Id ([int]$targetProcId) -ErrorAction SilentlyContinue
    if ($exists) {
      cmd /c "taskkill /PID $targetProcId /T /F" 1>$null 2>$null
    }
  }
}

function Stop-ByPidFile($name) {
  $file = Join-Path $runtimeDir "$name.pid"
  if (Test-Path $file) {
    $procPid = (Get-Content $file -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    Kill-Tree $procPid
    Remove-Item $file -Force -ErrorAction SilentlyContinue
  }
}

function Stop-PortOwner($port) {
  $raw = cmd /c "netstat -ano | findstr :$port"
  if (-not $raw) { return }
  $lines = ($raw -split "`r?`n") | Where-Object { $_ -and $_.Trim().Length -gt 0 }
  foreach ($line in $lines) {
    $cols = ($line -replace "\s+", " ").Trim().Split(" ")
    if ($cols.Length -lt 5) { continue }
    $targetPid = $cols[$cols.Length - 1]
    Kill-Tree $targetPid
  }
}

Stop-ByPidFile "desktop"
Stop-ByPidFile "frontend"
Stop-ByPidFile "backend"

# PIDファイルがない/壊れた場合のフォールバック
Stop-PortOwner 5173
Stop-PortOwner 8000

Write-Host "[ok] stop_all complete."

