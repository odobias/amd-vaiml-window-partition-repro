[CmdletBinding()]
param(
    [string]$OrtIncludeDir = "",
    [string]$OrtDll = "",
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
    return ""
}

function Resolve-OrtHeader {
    param([string]$Requested, [string]$Name)
    $candidates = @()
    if ($Requested) { $candidates += (Join-Path $Requested $Name) }
    if ($env:ORT_INCLUDE_DIR) { $candidates += (Join-Path $env:ORT_INCLUDE_DIR $Name) }
    $candidates += (Join-Path $PSScriptRoot "third_party\onnxruntime\include\$Name")
    $candidates += "C:\Projects\upstream\onnxruntime\include\onnxruntime\core\session\$Name"

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

$ortDllPath = Resolve-OrtDll $OrtDll
if (-not $ortDllPath) {
    throw "No ONNX Runtime DLL found. Install Ryzen AI SDK, set ORT_DLL, pass -OrtDll, or use -DownloadRyzenAI -AcceptAmdDownloadTerms."
}

if (-not $OrtIncludeDir) { $OrtIncludeDir = Join-Path $PSScriptRoot "third_party\onnxruntime\include" }
New-Item -ItemType Directory -Path $OrtIncludeDir -Force | Out-Null
foreach ($name in @("onnxruntime_c_api.h", "onnxruntime_ep_c_api.h")) {
    $header = Resolve-OrtHeader $OrtIncludeDir $name
    $localHeader = Join-Path $OrtIncludeDir $name
    if (-not $header) {
        $url = "https://raw.githubusercontent.com/microsoft/onnxruntime/main/include/onnxruntime/core/session/$name"
        Write-Host "Downloading $name ..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri $url -OutFile $localHeader
        $header = $localHeader
    } else {
        if ((Resolve-Path $header).Path -ne (Resolve-Path $localHeader -ErrorAction SilentlyContinue).Path) {
            Copy-Item $header $localHeader -Force
            $header = $localHeader
        }
    }
}

$header = Join-Path $OrtIncludeDir "onnxruntime_c_api.h"
if (Test-Path $header) {
    New-Item -ItemType Directory -Path $OrtIncludeDir -Force | Out-Null
} else {
    throw "onnxruntime_c_api.h was not created."
}

Write-Host "ONNX Runtime DLL   : $ortDllPath"
Write-Host "ONNX Runtime header: $header"
Write-Host "Bootstrap validation passed. Build with .\build.ps1, run with .\run.ps1." -ForegroundColor Green
