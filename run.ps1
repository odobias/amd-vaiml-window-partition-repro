[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$CacheDir = "",
    [switch]$Bootstrap,
    [switch]$InstallPythonDeps,
    [switch]$ReuseCache,
    [switch]$RegenerateModel
)

$ErrorActionPreference = "Stop"
chcp 65001 *> $null
$utf8 = [System.Text.UTF8Encoding]::new()
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
if ($null -ne (Get-Variable PSStyle -ErrorAction SilentlyContinue) -and $null -ne $PSStyle.OutputRendering) {
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

if ($Bootstrap) {
    $bootstrapScript = Join-Path $PSScriptRoot "bootstrap.ps1"
    $bootstrapArgs = @()
    if ($Python) { $bootstrapArgs += @("-Python", $Python) }
    if ($InstallPythonDeps) { $bootstrapArgs += "-InstallPythonDeps" }
    & $bootstrapScript @bootstrapArgs
    if ($LASTEXITCODE -ne 0) { throw "bootstrap failed with exit code $LASTEXITCODE" }
}

$pythonExe = Resolve-Python $Python
$script = Join-Path $PSScriptRoot "repro.py"
$model = Join-Path $PSScriptRoot "model.onnx"

Write-Host "Python : $pythonExe"
Write-Host "Script : $script"
Write-Host "Model  : $model (generated locally; not shipped)"

$common = @($script)
if ($RegenerateModel -or -not (Test-Path $model)) { $common += "--regenerate-model" }
if ($CacheDir) { $common += @("--cache-dir", $CacheDir) }

Write-Host "`n=== CPU control (expected: exit 0, finite [64,64,96]) ==="
& $pythonExe @common cpu
if ($LASTEXITCODE -ne 0) {
    throw "CPU control failed with exit code $LASTEXITCODE"
}

Write-Host "`n=== VitisAI (expected on affected runtime: session succeeds, first inference crashes with 0xC0000005) ==="
$arguments = @($common + "vitisai")
if ($ReuseCache) { $arguments += "--reuse-cache" }
& $pythonExe @arguments
$code = $LASTEXITCODE
$unsigned = [BitConverter]::ToUInt32([BitConverter]::GetBytes([int]$code), 0)
Write-Host ("VitisAI process exit code: {0} (0x{1:X8})" -f $code, $unsigned)
exit $code