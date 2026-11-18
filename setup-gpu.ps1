$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$projectDir = $PSScriptRoot
$toolsDir = Join-Path $projectDir '.tools'
$environmentDir = Join-Path $projectDir '.venv'
$mambaExe = Join-Path $toolsDir 'micromamba.exe'
$env:MAMBA_ROOT_PREFIX = Join-Path $toolsDir 'mamba'
$env:CONDA_PKGS_DIRS = Join-Path $toolsDir 'packages'

New-Item -ItemType Directory -Path $toolsDir -Force | Out-Null
if (-not (Test-Path -LiteralPath $mambaExe)) {
    Write-Host 'Downloading the portable micromamba environment manager...'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64.exe' -OutFile $mambaExe
}

if (-not (Test-Path -LiteralPath (Join-Path $environmentDir 'python.exe'))) {
    & $mambaExe create --yes --prefix $environmentDir --override-channels --channel conda-forge 'python=3.10' pip 'cudatoolkit=11.2' 'cudnn=8.1.0'
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the GPU environment.' }
}

& $mambaExe run --prefix $environmentDir python -m pip install -r (Join-Path $projectDir 'requirements-gpu.txt')
if ($LASTEXITCODE -ne 0) { throw 'Could not install the Python packages.' }

& $mambaExe run --prefix $environmentDir python (Join-Path $projectDir 'check_gpu.py')
if ($LASTEXITCODE -ne 0) { throw 'GPU verification failed. See the output above.' }

Write-Host 'Ready. Run .\run-training.cmd to train on the GPU.'
