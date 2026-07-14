param(
    [string]$Python = "python",
    [string]$OutputDir = "dist"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$OutputPath = Join-Path $ProjectRoot $OutputDir

New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null

Push-Location $ProjectRoot
try {
    & $Python -m pip wheel . --no-deps --no-build-isolation --wheel-dir $OutputPath
    if ($LASTEXITCODE -ne 0) {
        throw "Wheel build failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$Wheel = Get-ChildItem -Path $OutputPath -Filter "torch_like-*.whl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($null -eq $Wheel) {
    throw "No torch_like wheel was produced in $OutputPath"
}

Write-Host "Built package: $($Wheel.FullName)"
