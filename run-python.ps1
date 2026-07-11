[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$CacheDir = "",
    [switch]$InstallPythonDeps,
    [switch]$ReuseCache,
    [switch]$RegenerateModel,
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"
chcp 65001 *> $null
$utf8 = [System.Text.UTF8Encoding]::new()
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

if ($null -ne (Get-Variable PSStyle -ErrorAction SilentlyContinue) -and $null -ne $PSStyle) {
    $PSStyle.OutputRendering = "Ansi"
}

function Resolve-Python {
    param([string]$Requested)
    $candidates = @()
    if ($Requested) { $candidates += $Requested }
    if ($env:RYZEN_AI_PYTHON) { $candidates += $env:RYZEN_AI_PYTHON }
    $candidates += "C:\ProgramData\miniforge3\envs\ryzen-ai-1.8.0-beta\python.exe"
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $candidates += $cmd.Source }

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) { return (Resolve-Path $candidate).Path }
    }
    throw "No Python found. Pass -Python <path-to-Ryzen-AI-python.exe> or set RYZEN_AI_PYTHON."
}

$pythonExe = Resolve-Python $Python
$script = Join-Path $PSScriptRoot "repro.py"
$requirements = Join-Path $PSScriptRoot "requirements.txt"
$model = Join-Path $PSScriptRoot "model.onnx"
if (-not $CacheDir) { $CacheDir = Join-Path ([IO.Path]::GetTempPath()) "amd-vaiml-window-partition-python-cache" }

if ($InstallPythonDeps) {
    & $pythonExe -m pip install --upgrade -r $requirements
    if ($LASTEXITCODE -ne 0) { throw "pip install failed with exit code $LASTEXITCODE" }
}

Write-Host "Python : $pythonExe"
Write-Host "Script : $script"
Write-Host "Model  : $model (generated locally; not shipped)"

$common = @($script)
if ($RegenerateModel -or -not (Test-Path $model)) { $common += "--regenerate-model" }
$common += @("--cache-dir", $CacheDir)

Write-Host "`n=== Python CPU control (expected: exit 0, finite [64,64,96]) ==="
& $pythonExe @common cpu
if ($LASTEXITCODE -ne 0) {
    throw "CPU control failed with exit code $LASTEXITCODE"
}

if ($CpuOnly) {
    exit 0
}

if (-not $ReuseCache -and (Test-Path $CacheDir)) { Remove-Item $CacheDir -Recurse -Force }

Write-Host "`n=== Python VitisAI (expected on affected runtime: first inference crashes with 0xC0000005) ==="
$arguments = @($common + "vitisai")
if ($ReuseCache) { $arguments += "--reuse-cache" }
& $pythonExe @arguments
$code = $LASTEXITCODE
$unsigned = [BitConverter]::ToUInt32([BitConverter]::GetBytes([int]$code), 0)
Write-Host ("python repro exit code: {0} (0x{1:X8})" -f $code, $unsigned)
exit $code
