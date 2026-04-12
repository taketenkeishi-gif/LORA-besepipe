param()

$ErrorActionPreference = "SilentlyContinue"

function Stop-PortOwner($port) {
  $lines = cmd /c "netstat -ano | findstr :$port"
  if (-not $lines) { return }
  foreach ($line in $lines) {
    $cols = ($line -replace "\s+", " ").Trim().Split(" ")
    if ($cols.Length -lt 5) { continue }
    $targetPid = $cols[$cols.Length - 1]
    if ($targetPid -match "^\d+$") {
      cmd /c "taskkill /PID $targetPid /T /F" | Out-Null
    }
  }
}

# Kill listeners first (backend / frontend ports)
Stop-PortOwner 8000
Stop-PortOwner 5173

# Kill common desktop-related residual processes
$patterns = @(
  "npm run start",
  "electron .",
  "desktop\\main.cjs",
  "lora-workbench-desktop"
)

$procs = Get-CimInstance Win32_Process | Where-Object {
  $cmd = $_.CommandLine
  if (-not $cmd) { return $false }
  foreach ($p in $patterns) {
    if ($cmd -like "*$p*") { return $true }
  }
  return $false
}

foreach ($p in $procs) {
  cmd /c "taskkill /PID $($p.ProcessId) /T /F" | Out-Null
}

Write-Host "[ok] stop_all complete."
