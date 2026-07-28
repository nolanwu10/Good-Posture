[CmdletBinding()]
param(
    [string]$InstallDirectory = (Join-Path $env:LOCALAPPDATA "Programs\GoodPosture"),
    [string]$StartMenuDirectory = (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"),
    [switch]$SkipRegistry
)

$ErrorActionPreference = "Stop"
$sourceDirectory = Join-Path $PSScriptRoot "App"
$sourceExecutable = Join-Path $sourceDirectory "GoodPosture.exe"
$manifestPath = Join-Path $PSScriptRoot "release-manifest.json"
if (-not (Test-Path -LiteralPath $sourceExecutable -PathType Leaf)) {
    throw "Run this installer from the extracted GoodPosture release directory."
}
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "The release manifest is missing. Extract the complete GoodPosture archive and try again."
}

$releaseManifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if (
    -not $releaseManifest.version -or
    $releaseManifest.appExecutableSha256 -notmatch "^[A-Fa-f0-9]{64}$"
) {
    throw "The release manifest is invalid. Download or extract the package again."
}
$expectedExecutableHash = $releaseManifest.appExecutableSha256.ToUpperInvariant()
$sourceExecutableHash = (
    Get-FileHash -LiteralPath $sourceExecutable -Algorithm SHA256
).Hash
if ($sourceExecutableHash -ne $expectedExecutableHash) {
    throw "The packaged executable does not match its release manifest. Download or extract the package again."
}

$fullInstallDirectory = [System.IO.Path]::GetFullPath($InstallDirectory)
$installRoot = [System.IO.Path]::GetPathRoot($fullInstallDirectory)
if (
    [string]::IsNullOrWhiteSpace($installRoot) -or
    $fullInstallDirectory.TrimEnd("\") -eq $installRoot.TrimEnd("\")
) {
    throw "The install directory must be a dedicated GoodPosture folder, not a drive root."
}
if (
    (Test-Path -LiteralPath $fullInstallDirectory) -and
    ((Get-Item -LiteralPath $fullInstallDirectory).Attributes -band
        [System.IO.FileAttributes]::ReparsePoint)
) {
    throw "The install directory cannot be a symbolic link or junction."
}
$targetExecutable = Join-Path $fullInstallDirectory "GoodPosture.exe"
$runningTarget = Get-Process -Name "GoodPosture" -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Path -and
        [System.IO.Path]::GetFullPath($_.Path).Equals(
            $targetExecutable,
            [System.StringComparison]::OrdinalIgnoreCase
        )
    }
if ($runningTarget) {
    throw "GoodPosture is still running. Exit GoodPosture from the system tray, then run this installer again. No files were changed."
}

$installParent = Split-Path -Parent $fullInstallDirectory
$installLeaf = Split-Path -Leaf $fullInstallDirectory
New-Item -ItemType Directory -Path $installParent -Force | Out-Null
$stagingDirectory = Join-Path $installParent (
    $installLeaf + ".installing-" + [guid]::NewGuid()
)
$backupDirectory = Join-Path $installParent (
    $installLeaf + ".previous-" + [guid]::NewGuid()
)
$movedPreviousInstall = $false

try {
    New-Item -ItemType Directory -Path $stagingDirectory | Out-Null
    Copy-Item -Path (Join-Path $sourceDirectory "*") `
        -Destination $stagingDirectory -Recurse
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "Uninstall-GoodPosture.ps1") `
        -Destination $stagingDirectory
    Copy-Item -LiteralPath $manifestPath -Destination $stagingDirectory

    $stagedExecutable = Join-Path $stagingDirectory "GoodPosture.exe"
    $stagedHash = (
        Get-FileHash -LiteralPath $stagedExecutable -Algorithm SHA256
    ).Hash
    if ($stagedHash -ne $expectedExecutableHash) {
        throw "The staged executable failed verification. The existing installation was not changed."
    }

    if (Test-Path -LiteralPath $fullInstallDirectory) {
        Move-Item -LiteralPath $fullInstallDirectory -Destination $backupDirectory
        $movedPreviousInstall = $true
    }
    Move-Item -LiteralPath $stagingDirectory -Destination $fullInstallDirectory

    $installedHash = (
        Get-FileHash -LiteralPath $targetExecutable -Algorithm SHA256
    ).Hash
    if ($installedHash -ne $expectedExecutableHash) {
        throw "The installed executable failed verification."
    }
} catch {
    if (Test-Path -LiteralPath $fullInstallDirectory) {
        Remove-Item -LiteralPath $fullInstallDirectory -Recurse -Force
    }
    if ($movedPreviousInstall) {
        if (Test-Path -LiteralPath $backupDirectory) {
            Move-Item -LiteralPath $backupDirectory -Destination $fullInstallDirectory
        }
    }
    throw
} finally {
    if (Test-Path -LiteralPath $stagingDirectory) {
        Remove-Item -LiteralPath $stagingDirectory -Recurse -Force
    }
}

if (Test-Path -LiteralPath $backupDirectory) {
    Remove-Item -LiteralPath $backupDirectory -Recurse -Force
}

New-Item -ItemType Directory -Path $StartMenuDirectory -Force | Out-Null
$shortcutPath = Join-Path $StartMenuDirectory "GoodPosture.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $targetExecutable
$shortcut.Arguments = "desktop"
$shortcut.WorkingDirectory = $fullInstallDirectory
$shortcut.Description = "Local-first posture-awareness coach"
$shortcut.Save()

if (-not $SkipRegistry) {
    $uninstallKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\GoodPosture"
    New-Item -Path $uninstallKey -Force | Out-Null
    Set-ItemProperty -Path $uninstallKey -Name DisplayName -Value "GoodPosture"
    Set-ItemProperty -Path $uninstallKey -Name DisplayVersion -Value $releaseManifest.version
    Set-ItemProperty -Path $uninstallKey -Name Publisher -Value "GoodPosture"
    Set-ItemProperty -Path $uninstallKey -Name InstallLocation -Value $fullInstallDirectory
    Set-ItemProperty -Path $uninstallKey -Name NoModify -Type DWord -Value 1
    Set-ItemProperty -Path $uninstallKey -Name NoRepair -Type DWord -Value 1
    Set-ItemProperty -Path $uninstallKey -Name UninstallString -Value (
        "powershell.exe -ExecutionPolicy Bypass -File `"" +
        (Join-Path $fullInstallDirectory "Uninstall-GoodPosture.ps1") + "`""
    )
}

Write-Host (
    "GoodPosture " + $releaseManifest.version +
    " installed and verified for the current user. " +
    "Launch it from the GoodPosture Start-menu shortcut. " +
    "It will not start with Windows unless you enable that setting in the app."
)
