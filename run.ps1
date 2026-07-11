[CmdletBinding()]
param(
    [string]$OrtDll = "",
    [string]$CacheDir = "",
    [switch]$Bootstrap,
    [switch]$Build,
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
if ($null -ne (Get-Variable PSStyle -ErrorAction SilentlyContinue) -and $null -ne $PSStyle.OutputRendering) {
    $PSStyle.OutputRendering = "Ansi"
}

function Resolve-OrtDll {
    param([string]$Requested)
    $candidates = @()
    if ($Requested) { $candidates += $Requested }
    if ($env:ORT_DLL) { $candidates += $env:ORT_DLL }
    $candidates += "C:\ProgramData\miniforge3\envs\ryzen-ai-1.8.0-beta\Lib\site-packages\onnxruntime\capi\onnxruntime.dll"
    $candidates += "C:\Program Files\WindowsApps\WindowsWorkload.WinMLShared.5.0_1.2605.851.0_x64__8wekyb3d8bbwe\onnxruntime.dll"

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) { return (Resolve-Path $candidate).Path }
    }
    throw "No ONNX Runtime DLL found. Pass -OrtDll or set ORT_DLL."
}

if ($Bootstrap) {
    $bootstrapArgs = @()
    if ($OrtDll) { $bootstrapArgs += @("-OrtDll", $OrtDll) }
    & (Join-Path $PSScriptRoot "bootstrap.ps1") @bootstrapArgs
    if ($LASTEXITCODE -ne 0) { throw "bootstrap failed with exit code $LASTEXITCODE" }
}

$exe = Join-Path $PSScriptRoot "build\Release\amd_vaiml_repro.exe"
if ($Build -or -not (Test-Path $exe)) {
    $buildArgs = @()
    if ($Bootstrap) { $buildArgs += "-Bootstrap" }
    & (Join-Path $PSScriptRoot "build.ps1") @buildArgs
    if ($LASTEXITCODE -ne 0) { throw "build failed with exit code $LASTEXITCODE" }
}

$ortDllPath = Resolve-OrtDll $OrtDll
$model = Join-Path $PSScriptRoot "model.onnx"
if (-not $CacheDir) { $CacheDir = Join-Path ([IO.Path]::GetTempPath()) "amd-vaiml-window-partition-cpp-cache" }

Write-Host "Exe    : $exe"
Write-Host "ORT DLL: $ortDllPath"
Write-Host "Model  : $model (generated locally; not shipped)"

$arguments = @("--ort-dll", $ortDllPath, "--model", $model)
$arguments += @("--cache-dir", $CacheDir)
if ($RegenerateModel -or -not (Test-Path $model)) { $arguments += "--regenerate-model" }
if ($CpuOnly) { $arguments += "--cpu-only" }
if (-not $ReuseCache -and (Test-Path $CacheDir)) { Remove-Item $CacheDir -Recurse -Force }

& $exe @arguments
$code = $LASTEXITCODE
$unsigned = [BitConverter]::ToUInt32([BitConverter]::GetBytes([int]$code), 0)
Write-Host ("repro exit code: {0} (0x{1:X8})" -f $code, $unsigned)
exit $code