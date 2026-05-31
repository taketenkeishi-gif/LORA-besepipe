param()

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopPath "LoRA_Workbench.lnk"
$target = Join-Path $root "launch_desktop_hidden.vbs"

$wsh = New-Object -ComObject WScript.Shell
$shortcut = $wsh.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $target
$shortcut.WorkingDirectory = $root
$shortcut.WindowStyle = 7

$iconPath = Join-Path $root "desktop\icon.ico"
if (Test-Path $iconPath) {
  $shortcut.IconLocation = $iconPath
} else {
  $electronExe = Join-Path $root "desktop\node_modules\electron\dist\electron.exe"
  if (Test-Path $electronExe) {
    $shortcut.IconLocation = "$electronExe,0"
  } else {
    $shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,220"
  }
}
$shortcut.Description = "LoRA制作ワークベンチ (Desktop)"
$shortcut.Save()

Write-Host "[ok] desktop shortcut created: $shortcutPath"

