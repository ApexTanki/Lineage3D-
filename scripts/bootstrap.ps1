# Clones vcpkg at the commit pinned in vcpkg.json ("builtin-baseline") into external/vcpkg and builds vcpkg.exe.
# Everything stays inside the project folder: vcpkg's downloads and build trees live in external/vcpkg,
# its binary cache in .cache/vcpkg-binary and the installed packages in .cache/vcpkg_installed.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$vcpkg = Join-Path $root 'external\vcpkg'
$baseline = (Get-Content (Join-Path $root 'vcpkg.json') -Raw | ConvertFrom-Json).'builtin-baseline'

if (-not (Test-Path (Join-Path $vcpkg '.git'))) {
    git clone https://github.com/microsoft/vcpkg.git $vcpkg
    if ($LASTEXITCODE -ne 0) { throw 'git clone of vcpkg failed' }
}
git -C $vcpkg cat-file -e "$baseline^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) {
    git -C $vcpkg fetch --quiet origin
    if ($LASTEXITCODE -ne 0) { throw 'git fetch of vcpkg failed' }
}
git -C $vcpkg -c advice.detachedHead=false checkout --quiet $baseline
if ($LASTEXITCODE -ne 0) { throw "checkout of vcpkg commit $baseline failed" }

if (-not (Test-Path (Join-Path $vcpkg 'vcpkg.exe'))) {
    & (Join-Path $vcpkg 'bootstrap-vcpkg.bat') -disableMetrics
    if ($LASTEXITCODE -ne 0) { throw 'bootstrap-vcpkg.bat failed' }
}
New-Item -ItemType Directory -Force -Path (Join-Path $root '.cache\vcpkg-binary') | Out-Null
Write-Host "vcpkg is ready: $vcpkg (commit $baseline)"
