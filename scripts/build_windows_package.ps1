[CmdletBinding()]
param(
    [string]$Python = ".venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modelPath = Join-Path $projectRoot "models\pose_landmarker_lite.task"
$assetsPath = Join-Path $projectRoot "src\goodposture\assets"
$expectedModelSha256 = "59929E1D1EE95287735DDD833B19CF4AC46D29BC7AFDDBBF6753C459690D574A"

if (-not (Test-Path -LiteralPath $modelPath -PathType Leaf)) {
    throw "The pinned local model is missing. Run scripts\download_model.ps1 first."
}
if ((Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash -ne $expectedModelSha256) {
    throw "The local model checksum does not match the pinned release asset."
}

$pythonPath = (Resolve-Path (Join-Path $projectRoot $Python)).Path
$pysidePath = (& $pythonPath -c "from pathlib import Path; import PySide6; print(Path(PySide6.__file__).parent)")
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $pysidePath)) {
    throw "The pinned PySide6 runtime could not be located."
}
$buildPath = Join-Path $projectRoot "build"
$distPath = Join-Path $projectRoot "dist"
$bundlePath = Join-Path $distPath "GoodPosture"
$releaseVersion = "0.1.6"
$releasePath = Join-Path $distPath "GoodPosture-$releaseVersion-windows-x64"

$originalPath = $env:PATH
try {
    # Prevent unrelated developer tools on PATH from supplying incompatible
    # native DLLs to PyInstaller's dependency scan.
    $env:PATH = @(
        (Split-Path -Parent $pythonPath),
        (Join-Path $env:SystemRoot "System32"),
        $env:SystemRoot
    ) -join ";"
    & $pythonPath -m PyInstaller `
        --noconfirm `
        --clean `
        --onedir `
        --windowed `
        --name GoodPosture `
        --contents-directory . `
        --paths (Join-Path $projectRoot "src") `
        --collect-all mediapipe `
        --add-data "$modelPath;models" `
        --add-data "$assetsPath;goodposture\assets" `
        --workpath $buildPath `
        --distpath $distPath `
        (Join-Path $projectRoot "src\goodposture\__main__.py")
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed."
    }
} finally {
    $env:PATH = $originalPath
}

$smokeProcess = Start-Process `
    -FilePath (Join-Path $bundlePath "GoodPosture.exe") `
    -ArgumentList "package-smoke" `
    -PassThru `
    -Wait
if ($smokeProcess.ExitCode -ne 0) {
    throw "The packaged offline smoke test failed."
}

if (Test-Path -LiteralPath $releasePath) {
    Remove-Item -LiteralPath $releasePath -Recurse -Force
}
New-Item -ItemType Directory -Path $releasePath | Out-Null
Copy-Item -LiteralPath $bundlePath -Destination (Join-Path $releasePath "App") -Recurse
$releaseManifest = [ordered]@{
    version = $releaseVersion
    appExecutableSha256 = (
        Get-FileHash -LiteralPath (Join-Path $bundlePath "GoodPosture.exe") `
            -Algorithm SHA256
    ).Hash
}
$releaseManifest |
    ConvertTo-Json |
    Set-Content -LiteralPath (Join-Path $releasePath "release-manifest.json") `
        -Encoding UTF8
Copy-Item -LiteralPath (Join-Path $projectRoot "packaging\Install-GoodPosture.ps1") -Destination $releasePath
Copy-Item -LiteralPath (Join-Path $projectRoot "packaging\Uninstall-GoodPosture.ps1") -Destination $releasePath
Copy-Item -LiteralPath (Join-Path $projectRoot "packaging\README.txt") -Destination $releasePath
Copy-Item -LiteralPath (Join-Path $projectRoot "RELEASE_NOTES.md") -Destination $releasePath
Copy-Item -LiteralPath (Join-Path $projectRoot "packaging\THIRD_PARTY_NOTICES.md") -Destination $releasePath

$archivePath = "$releasePath.zip"
if (Test-Path -LiteralPath $archivePath) {
    Remove-Item -LiteralPath $archivePath -Force
}
Compress-Archive -LiteralPath $releasePath -DestinationPath $archivePath
Write-Host "Package created: $archivePath"
