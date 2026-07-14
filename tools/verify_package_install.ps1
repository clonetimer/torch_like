param(
    [string]$Python = "python",
    [string]$Environment = ".packaging-venv",
    [string]$WheelDirectory = "dist"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvironmentPath = Join-Path $ProjectRoot $Environment
$WheelPath = Join-Path $ProjectRoot $WheelDirectory
$Wheel = Get-ChildItem -Path $WheelPath -Filter "torch_like-*.whl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($null -eq $Wheel) {
    throw "No wheel found in $WheelPath. Run tools/build_package.ps1 first."
}

if (-not (Test-Path $EnvironmentPath)) {
    & $Python -m venv --system-site-packages $EnvironmentPath
    if ($LASTEXITCODE -ne 0) {
        throw "Virtual environment creation failed with exit code $LASTEXITCODE"
    }
}

$EnvironmentPython = Join-Path $EnvironmentPath "Scripts\python.exe"

# Reuse the selected interpreter's heavy dependencies (not its torch_like source)
# so an offline wheel smoke test does not download another PyTorch installation.
$SiteQuery = "import site; print(next(path for path in site.getsitepackages() if path.lower().endswith('site-packages')))"
$DependencySite = (& $Python -c $SiteQuery).Trim()
$EnvironmentSite = (& $EnvironmentPython -c $SiteQuery).Trim()
if (-not (Test-Path $DependencySite)) {
    throw "Dependency site-packages directory does not exist: $DependencySite"
}
Set-Content -Path (Join-Path $EnvironmentSite "_torch_like_build_dependencies.pth") -Value $DependencySite -Encoding ASCII

& $EnvironmentPython -m pip install --force-reinstall --no-deps $Wheel.FullName
if ($LASTEXITCODE -ne 0) {
    throw "Wheel installation failed with exit code $LASTEXITCODE"
}

Push-Location ([System.IO.Path]::GetTempPath())
try {
    & $EnvironmentPython -I -c "import importlib.metadata as m; import torch_like as tl; assert m.version('torch_like') == tl.__version__; assert tl.list_domains(); print({'version': tl.__version__, 'module': tl.__file__, 'domains': len(tl.list_domains())})"
    if ($LASTEXITCODE -ne 0) {
        throw "Installed-package smoke test failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
