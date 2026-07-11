# AMD VAIML window-partition first-inference crash

Minimal, source-only C++ reproducer for a native crash in AMD's Vitis AI
Execution Provider (VAIML). The executable generates a tiny synthetic ONNX graph
on the target machine, confirms ORT CPU can execute it, then runs the same graph
through `VitisAIExecutionProvider`.

On the affected stack, VAIML compiles the generated graph without errors and
creates an inference session, then `onnxruntime_vitisai_ep.dll` access-violates
on the first `Run()`.

## What ships

This directory is intentionally source-only:

- `bootstrap.ps1` - validates the ONNX Runtime / Ryzen AI runtime paths, fetches
  the ONNX Runtime C API header if needed, and can download AMD SDK/driver installers.
- `build.ps1` - builds the C++ repro with CMake or plain Visual Studio `cl.exe`.
- `run.ps1` - portable Windows runner for a new machine.
- `src/main.cpp` - writes the minimal ONNX protobuf, runs CPU control, then runs VitisAI.
- `.gitignore` - ignores generated model/build/download artifacts.

`model.onnx` is **not committed**. It is generated locally by the C++ executable.
The generated graph has no trained weights and no captured customer inputs; it
uses synthetic LayerNorm parameters (`ones` / `zeros`) and a deterministic random
input seed (`20260711`).

## Reproduced environment

- AMD Ryzen AI 9 HX 370 / XDNA 2 NPU
- Ryzen AI SDK 1.8.0-beta
- ONNX Runtime `1.25.1.dev20260617`
- C++ runner built with Visual Studio 2022 Build Tools
- Windows 11 build 26200

Both a cold compile and a load from the populated VAIML cache reproduce the same
process crash.

## Bootstrap on a new machine

The failing provider is AMD-specific. A vanilla ONNX Runtime DLL is **not**
enough and can actively confuse the repro because it does not include
`VitisAIExecutionProvider`. Use the Ryzen AI SDK runtime or a Windows ML AMD
VitisAI EP package.

The bootstrap script validates the selected ONNX Runtime DLL and C API header.
It downloads `onnxruntime_c_api.h` if needed:

```powershell
.\bootstrap.ps1
```

Override the ONNX Runtime DLL if needed:

```powershell
.\bootstrap.ps1 `
  -OrtDll C:\path\to\onnxruntime.dll
```

The runner tries these ONNX Runtime DLL locations, in order:

1. `-OrtDll <path>` argument
2. `$env:ORT_DLL`
3. `C:\ProgramData\miniforge3\envs\ryzen-ai-1.8.0-beta\Lib\site-packages\onnxruntime\capi\onnxruntime.dll`
4. `C:\Program Files\WindowsApps\WindowsWorkload.WinMLShared.5.0_1.2605.851.0_x64__8wekyb3d8bbwe\onnxruntime.dll`

If the AMD SDK is missing, bootstrap can download the official AMD installer and
NPU driver archive and verify their SHA-256 hashes:

```powershell
.\bootstrap.ps1 `
  -DownloadRyzenAI `
  -AcceptAmdDownloadTerms
```

It deliberately does **not** silently launch vendor installers. Those may require
administrator rights and license prompts, and pretending otherwise would be
bullshit.

## Run

From the repository root:

```powershell
.\run.ps1 -Bootstrap -Build -RegenerateModel
```

Build only:

```powershell
.\build.ps1 -Bootstrap
```

Override the ONNX Runtime DLL if needed:

```powershell
.\run.ps1 `
  -OrtDll C:\path\to\onnxruntime.dll
```

Force model regeneration:

```powershell
.\run.ps1 -RegenerateModel
```

Verify the cached VAIML path after one cold run:

```powershell
.\run.ps1 -ReuseCache
```

Use an explicit VAIML cache directory:

```powershell
.\run.ps1 `
  -CacheDir C:\temp\amd-vaiml-window-repro-cache
```

## Expected behavior on the affected stack

1. `amd_vaiml_repro.exe` generates `model.onnx` locally.
2. CPU returns a finite tensor with shape `[64, 64, 96]`.
3. `aiecompiler` prints `Compilation Complete` with `ERROR:0`.
4. The C++ runner creates a VitisAI session.
5. The first VitisAI inference terminates the process with access violation
   `0xC0000005` (signed exit code `-1073741819`).

## Generated graph

The generated `model.onnx` contains a `LayerNormalization` followed by the
dynamic shape/reshape/transpose sequence that partitions a `[1,4096,96]` tensor
into 64 windows of `[64,96]`.

Expected generated model properties:

```text
bytes: about 13 KB
```

The C++ runner writes the ONNX protobuf directly. It intentionally avoids Python,
the `onnx` package, and committed binary model files.

ORT graph rewrites are deliberately disabled. With normal optimization the
dynamic shape sequence can be folded or assigned to CPU, which prevents VAIML
from claiming the failing structure and hides the defect rather than fixing it.

## Observed native failure

The top frame is inside `onnxruntime_vitisai_ep.dll`:

```text
Compilation Complete
(WARNING:75, CRITICAL-WARNING:0, ERROR:0)
Session created; providers=['VitisAIExecutionProvider', 'CPUExecutionProvider']
Running first inference...
Exception Code: 0xC0000005
onnxruntime_vitisai_ep.dll + 0x3524D7B
onnxruntime_vitisai_ep.dll + 0x330F67C
onnxruntime_vitisai_ep.dll + 0x33FB8DF
...
onnxruntime_providers_vitisai.dll + 0x236C4
```

The DLL is stripped and the reported nearby `xir_deserialize_cif` symbol has
large offsets, so that symbol name should not be treated as a reliable source
location.

## Relationship to the original FakeAudio failure

This graph was reduced from the first HTSAT window-partition block while
investigating FakeAudio on VAIML. The complete isolated block exposes a second
runtime defect after successful compilation:

```text
FlexMLDispatcher: HSI input slot count 1 is inconsistent with ifms.size()=4
and deviceBatchSize=1
```

That larger graph contains model-derived structure and is not required for this
report. This directory is the clean vendor-facing reproducer: tiny, synthetic,
deterministic, source-only, and free of proprietary tensors.