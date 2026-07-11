[CmdletBinding()]
param(
    [string]$OrtIncludeDir = "",
    [string]$Configuration = "Release",
    [switch]$Bootstrap
)

$ErrorActionPreference = "Stop"
chcp 65001 *> $null
$utf8 = [System.Text.UTF8Encoding]::new()
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if ($Bootstrap) {
    $args = @()
    if ($OrtIncludeDir) { $args += @("-OrtIncludeDir", $OrtIncludeDir) }
    & (Join-Path $PSScriptRoot "bootstrap.ps1") @args
    if ($LASTEXITCODE -ne 0) { throw "bootstrap failed with exit code $LASTEXITCODE" }
}

if (-not $OrtIncludeDir) {
    $local = Join-Path $PSScriptRoot "third_party\onnxruntime\include"
    if (Test-Path (Join-Path $local "onnxruntime_c_api.h")) {
        $OrtIncludeDir = $local
    } elseif (Test-Path "C:\Projects\upstream\onnxruntime\include\onnxruntime\core\session\onnxruntime_c_api.h") {
        $OrtIncludeDir = "C:\Projects\upstream\onnxruntime\include\onnxruntime\core\session"
    } else {
        throw "onnxruntime_c_api.h not found. Re-run with -Bootstrap or pass -OrtIncludeDir."
    }
}

$build = Join-Path $PSScriptRoot "build"
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if ($cmake) {
    cmake -S $PSScriptRoot -B $build -A x64 -DORT_INCLUDE_DIR="$OrtIncludeDir"
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed with exit code $LASTEXITCODE" }
    cmake --build $build --config $Configuration
    if ($LASTEXITCODE -ne 0) { throw "cmake build failed with exit code $LASTEXITCODE" }
} else {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "cmake not found and vswhere.exe not found. Install CMake or Visual Studio Build Tools." }
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsPath) { throw "Visual Studio C++ Build Tools not found." }
    $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found: $vcvars" }
    $outDir = Join-Path $build $Configuration
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
    $src = Join-Path $PSScriptRoot "src\main.cpp"
    $exeOut = Join-Path $outDir "amd_vaiml_repro.exe"
    $cmdFile = Join-Path ([IO.Path]::GetTempPath()) ("amd-vaiml-repro-build-" + [Guid]::NewGuid().ToString("N") + ".cmd")
    $cmdText = @"
@echo off
call "$vcvars"
cl /nologo /EHsc /std:c++17 /W4 /DWIN32_LEAN_AND_MEAN /DNOMINMAX /I"$OrtIncludeDir" "$src" /Fe:"$exeOut"
"@
    [IO.File]::WriteAllText($cmdFile, $cmdText, $utf8)
    try {
        cmd.exe /c "`"$cmdFile`""
        if ($LASTEXITCODE -ne 0) { throw "cl build failed with exit code $LASTEXITCODE" }
    } finally {
        Remove-Item $cmdFile -Force -ErrorAction SilentlyContinue
    }
}

$exe = Join-Path $build "$Configuration\amd_vaiml_repro.exe"
if (-not (Test-Path $exe)) { throw "built exe not found: $exe" }
Write-Host "Built: $exe" -ForegroundColor Green
