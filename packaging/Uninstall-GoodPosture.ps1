[CmdletBinding()]
param(
    [string]$InstallDirectory = (Join-Path $env:LOCALAPPDATA "Programs\GoodPosture"),
    [string]$StartMenuDirectory = (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"),
    [string]$UserDataDirectory = (Join-Path $env:LOCALAPPDATA "GoodPosture\GoodPosture"),
    [switch]$RemoveUserData,
    [switch]$SkipRegistry
)

$ErrorActionPreference = "Stop"

# Remove a legacy/current opt-in startup entry even when the application files are gone.
if (-not $SkipRegistry) {
    $runKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
    Remove-ItemProperty -Path $runKey -Name "GoodPosture" -ErrorAction SilentlyContinue
    Remove-Item -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\GoodPosture" `
        -Recurse -Force -ErrorAction SilentlyContinue
}

Remove-Item -LiteralPath (Join-Path $StartMenuDirectory "GoodPosture.lnk") `
    -Force -ErrorAction SilentlyContinue

if ($RemoveUserData) {
    Remove-Item -LiteralPath $UserDataDirectory -Recurse -Force -ErrorAction SilentlyContinue
}

if (Test-Path -LiteralPath $InstallDirectory) {
    $deferredRemoval = Join-Path $env:TEMP ("GoodPosture-uninstall-" + [guid]::NewGuid() + ".ps1")
    @"
Start-Sleep -Milliseconds 750
Remove-Item -LiteralPath '$($InstallDirectory.Replace("'", "''"))' -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath `$PSCommandPath -Force -ErrorAction SilentlyContinue
"@ | Set-Content -LiteralPath $deferredRemoval -Encoding UTF8
    Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$deferredRemoval`""
    )
}

if ($RemoveUserData) {
    Write-Host "GoodPosture and its local user data are being removed."
} else {
    Write-Host "GoodPosture is being removed; local user data was preserved by choice."
}
