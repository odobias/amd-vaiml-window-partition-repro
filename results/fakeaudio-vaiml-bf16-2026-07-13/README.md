# FakeAudio BF16 VAIML result

In the original Ryzen AI 1.8.0-beta run, AMD's BF16 configuration creates a
valid ONNX Runtime session but does not compile this model for the NPU. The main
partition fails L1 buffer placement at layer 224; the remaining partition is
rejected by the 2% GOP threshold. ORT then runs the entire graph on CPU.

## Follow-up: Ryzen AI 1.7.1 GA

The result above reproduces on Ryzen AI 1.8.0-beta, but it is not a general
limitation of this model. We reran the same ONNX file on 2026-07-14 with:

- Ryzen AI 1.7.1 GA (`ryzen-ai-1.7.1` conda environment)
- AMD NPU driver `32.0.203.280`
- ONNX Runtime VitisAI `1.23.3`
- the model-local copy of AMD's BF16 configuration

This time compilation and inference completed without provider fallback:

- the compiler created two partitions and reported **98.33% of original
  operations on AIE**, with 1.67% on CPU
- maximum probability difference from the CPU reference: `0.011589`
- fixture accuracy: 3/5 (60%), matching the CPU fixture baseline
- cold/hot session load: 513.87 s / 122.39 s
- mean inference: 70.18 ms

ORT profiling reports one fused VitisAI node and 20 CPU nodes. That does not
mean the workload is 95% CPU: the VitisAI node contains the compiled AIE
partition, while the profiler counts each small CPU-side preprocessing or
boundary operation separately. The compiler's original-operation placement is
the useful figure.

The compiler still emits a fail-safe notice, but no compilation error:

```text
Partition completed with 2 partitions
The fail-safe partitioner has detected operations that are not supported on AIE.
98.33% of operations will run on AIE, 1.67% of operations will run on CPU.
```

A second cold-cache run on the same stack completed successfully and reproduced
the important results exactly: `0.011589` maximum probability delta, 3/5 fixture
accuracy, one fused VitisAI node, 20 CPU nodes, and the compiler's 98.33%/1.67%
AIE/CPU split. It reported 356.84 s cold load and 109.77 ms mean inference.

That repeat also reported an anomalous 8,204.10 s hot load while the machine was
on battery. The run may have crossed host suspension, or the compiler may have
stalled; the available log cannot distinguish those cases. That hot-load value
is not a usable performance measurement, but it did not change placement,
probabilities, or the successful exit.

The ONNX file remains the artifact linked below, with SHA-256 `d4b2...e85c6`.
`NpuInferenceBench` also reports a package hash of `5491...dcbb`; that hash
includes the fixture package and is not the hash of `model.onnx`.

FakeAudio produced the same `0.011589` probability delta under the 1.7.1
runtime before the production driver was installed. Driver `32.0.203.280`
fixed a separate Whisper session-creation crash, but did not materially change
FakeAudio. The difference from this report's original result therefore tracks
the 1.8.0-beta versus 1.7.1 runtime/compiler path.

## Setup

- Date: 2026-07-13
- Hardware: AMD Ryzen AI 9 HX 370 / XDNA 2
- Ryzen AI SDK: 1.8.0-beta
- ONNX Runtime: 1.25.1
- OS: Windows 11 build 26200
- VAIML: BF16, optimization level 3, Peano compiler
- Model: [FakeAudio ONNX, opset 17](https://huggingface.co/odobias/fakeaudio-onnx-vaiml-repro)
- Direct download: [`model.onnx`](https://huggingface.co/odobias/fakeaudio-onnx-vaiml-repro/resolve/main/model.onnx?download=true)
- SHA-256: `d4b2dde8f4f95862ecb9f5e90821d853adea3774e7c48857f60ee6084d5e85c6`

The ONNX model was used as published, without optimization or opset conversion.

## Runs

AMD's helper was run against an empty cache:

```powershell
python amd\compile_npu.py <fakeaudio-model.onnx> `
  --vai-config amd\vitisai_config.json `
  --cache-dir <fresh-cache>
```

The same model and config were then run through `NpuInferenceBench` with five
labeled fixtures, a CPU reference, and ORT profiling enabled.

## Compiler result

```text
ERROR: [VAIML-BE-SKGEN 11005] Flexible L1 placement failure in layer 224 : L-224 - Failed to place the buffers.
ERROR: [VAIML-COMPILE 1000] vaiml_compile_v4 error: Flexible L1 placement failure in layer 224 : L-224 - Failed to place the buffers.
INFO: [VAIP-VAIML-PASS] Model compilation completed with errors.
INFO: [VAIP-VAIML-PASS] Failed to compile subgraph vaiml_par_1. It will fall back to CPU.
INFO: [VAIP-VAIML-PASS] Subgraph vaiml_par_0 GOPs% (0.956%) is less than the threshold (2%). It will fall back to CPU.
DEBUG2: [VAIP-VAIML-PASS] Total subgraphs: 0
```

`compile_npu.py` still ends with:

```text
=== Compilation Successful ===
```

The script prints that message after `ort.InferenceSession(...)` returns. It
does not check whether VitisAI claimed any nodes.

## Profile

- Cold session load: 271.788 s
- Hot session load: 221.095 s
- Mean inference: 112.992 ms
- VitisAI/NPU nodes: 0
- CPU nodes: 512
- CPU offload: 100%
- Maximum probability difference from CPU: 0.000000
- Fixture accuracy: 3/5 (60%)

The compiler reports 598 operators before ORT graph optimization; the runtime
profile reports 512 nodes afterward. No nodes were assigned to VitisAI.

The probabilities and fixture accuracy match the CPU reference. The latency is
therefore CPU performance, not NPU performance.

## Logs

- [`compile.log`](compile.log): AMD helper run with a fresh cache
- [`benchmark.log`](benchmark.log): cold/hot loads, profiling, and final JSON

Both logs contain VAIML compiler output; the benchmark log adds node assignment
and fixture results.

## Open questions

- Can `L-224` be made to fit with different tiling, spilling, or partition
  boundaries?
- Can the helper report partition status or claimed node count instead of
  treating session creation as compilation success?
