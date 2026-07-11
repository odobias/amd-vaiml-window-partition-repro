[CmdletBinding()]
param(
    [string]$Python = "",
    [switch]$InstallPythonDeps,
    [switch]$DownloadRyzenAI,
    [string]$DownloadDir = "",
    [switch]$AcceptAmdDownloadTerms
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

$sdkInstaller = @{
    Name = "ryzen-ai-lt-1.8.0-beta.exe"
    Url = "https://download.amd.com/opendownload/RyzenAI/1.8.0b0/ryzen-ai-lt-1.8.0-beta.exe"
    Sha256 = "44D566E4BB520375DA904F4D146EC021B22CC870C2288E7B9F3A2C18C1A57EF1"
}
$npuDriver = @{
    Name = "NPU_RAI_376_WHQL.zip"
    Url = "https://download.amd.com/opendownload/RyzenAI/1.8.0b0/NPU_RAI_376_WHQL.zip"
    Sha256 = "D09695C5833A2263A0A77484FA145013D9A570C0110CFBFA65CC2DD31899F7CF"
}

function Resolve-ReproPython {
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
    return ""
}

function Save-VerifiedDownload {
    param(
        [hashtable]$Item,
        [string]$Directory
    )
    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
    $out = Join-Path $Directory $Item.Name
    if (-not (Test-Path $out)) {
        Write-Host "Downloading $($Item.Name) ..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri $Item.Url -OutFile $out
    } else {
        Write-Host "Using existing $out" -ForegroundColor DarkGray
    }
    $actual = (Get-FileHash $out -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($actual -ne $Item.Sha256) {
        throw "SHA-256 mismatch for $($Item.Name): expected $($Item.Sha256), got $actual"
    }
    Write-Host "Verified $($Item.Name): $actual" -ForegroundColor Green
    return $out
}

if ($DownloadRyzenAI) {
    if (-not $AcceptAmdDownloadTerms) {
        throw "Pass -AcceptAmdDownloadTerms to download AMD installers from the official AMD URLs."
    }
    if (-not $DownloadDir) { $DownloadDir = Join-Path $PSScriptRoot ".downloads" }
    $sdk = Save-VerifiedDownload -Item $sdkInstaller -Directory $DownloadDir
    $driver = Save-VerifiedDownload -Item $npuDriver -Directory $DownloadDir
    Write-Host ""
    Write-Host "Downloaded AMD dependencies:" -ForegroundColor Green
    Write-Host "  SDK installer : $sdk"
    Write-Host "  NPU driver zip: $driver"
    Write-Host ""
    Write-Host "Run/install those AMD packages, then re-run this bootstrap without -DownloadRyzenAI."
    Write-Host "This script does not silently launch vendor installers; they may require admin rights and license prompts."
}

$pythonExe = Resolve-ReproPython $Python
if (-not $pythonExe) {
    throw "No Python found. Install Ryzen AI SDK, pass -Python <path>, set RYZEN_AI_PYTHON, or use -DownloadRyzenAI -AcceptAmdDownloadTerms."
}

Write-Host "Python: $pythonExe"
$requirements = Join-Path $PSScriptRoot "requirements.txt"
if ($InstallPythonDeps) {
    Write-Host "Installing Python graph-generation dependencies from $requirements ..."
    & $pythonExe -m pip install --upgrade -r $requirements
    if ($LASTEXITCODE -ne 0) { throw "pip install failed with exit code $LASTEXITCODE" }
}

$probe = @"
import importlib.util
import json
import sys

mods = ["numpy", "onnx", "onnxruntime"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print(json.dumps({"missing": missing}))
    sys.exit(2)

import onnxruntime as ort
providers = ort.get_available_providers()
print(json.dumps({"python": sys.executable, "onnxruntime": ort.__version__, "providers": providers}, indent=2))
if "VitisAIExecutionProvider" not in providers:
    sys.exit(3)
"@
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("amd-vaiml-repro-probe-" + [Guid]::NewGuid().ToString("N") + ".py")
[IO.File]::WriteAllText($tmp, $probe, $utf8)
try {
    & $pythonExe $tmp
    $code = $LASTEXITCODE
} finally {
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
}

if ($code -eq 2) {
    throw "Python dependencies are missing. Re-run with -InstallPythonDeps, using the Ryzen AI SDK Python."
}
if ($code -eq 3) {
    throw "VitisAIExecutionProvider is not available. Do not install vanilla pip onnxruntime; use the Ryzen AI SDK Python/runtime."
}
if ($code -ne 0) {
    throw "Provider probe failed with exit code $code"
}

Write-Host "Bootstrap validation passed." -ForegroundColor Green
