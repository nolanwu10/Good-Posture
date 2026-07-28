param(
    [string]$Destination = (
        Join-Path $PSScriptRoot '..\models\pose_landmarker_lite.task'
    )
)

$ErrorActionPreference = 'Stop'

$modelUrl = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
$expectedSha256 = '59929E1D1EE95287735DDD833B19CF4AC46D29BC7AFDDBBF6753C459690D574A'
$destinationPath = [System.IO.Path]::GetFullPath($Destination)
$destinationDirectory = Split-Path -Parent $destinationPath

New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null

if (Test-Path -LiteralPath $destinationPath) {
    $existingHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $destinationPath).Hash
    if ($existingHash -eq $expectedSha256) {
        Write-Output "Verified existing model: $destinationPath"
        exit 0
    }
}

$temporaryPath = Join-Path $destinationDirectory 'pose_landmarker_lite.download'
try {
    Invoke-WebRequest -Uri $modelUrl -OutFile $temporaryPath
    $downloadedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $temporaryPath).Hash
    if ($downloadedHash -ne $expectedSha256) {
        throw "Model checksum mismatch. Expected $expectedSha256, got $downloadedHash."
    }
    Move-Item -LiteralPath $temporaryPath -Destination $destinationPath -Force
    Write-Output "Downloaded and verified model: $destinationPath"
}
finally {
    if (Test-Path -LiteralPath $temporaryPath) {
        Remove-Item -LiteralPath $temporaryPath -Force
    }
}
