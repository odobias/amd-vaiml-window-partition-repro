# FakeAudio test with AMD's BF16 VAIML flow

Tested on 2026-07-13.

## Short version

We tried AMD's `compile_npu.py` and `vitisai_config.json` on the current,
unmodified FakeAudio model.

The model runs without crashing, but it does **not** run on the NPU:

- VAIML fails to compile the main partition at layer 224.
- The only other partition is too small to be accepted.
- ONNX Runtime silently runs the complete model on the CPU.
- Profiling confirms **0 NPU nodes, 512 CPU nodes, and 100% CPU offload**.

The helper still prints `Compilation Successful` because it checks whether an
ONNX Runtime session was created, not whether VAIML successfully placed any work
on the NPU.

## What we tested

We used:

- AMD Ryzen AI 9 HX 370 with an XDNA 2 NPU
- Ryzen AI SDK 1.8.0-beta
- ONNX Runtime 1.25.1
- Windows 11 build 26200
- the original FP32 FakeAudio model, opset 17
- AMD's BF16 config with optimization level 3 and the Peano compiler

We did not optimize, convert, or otherwise modify the ONNX model before the
test.

### Public model artifact

The exact model used for this test is publicly available from
[Hugging Face](https://huggingface.co/odobias/fakeaudio-onnx-vaiml-repro):

- [Download `model.onnx`](https://huggingface.co/odobias/fakeaudio-onnx-vaiml-repro/resolve/main/model.onnx?download=true)
- Size: 127,844,774 bytes
- SHA-256: `d4b2dde8f4f95862ecb9f5e90821d853adea3774e7c48857f60ee6084d5e85c6`

The Hugging Face model card also records the artifact metadata and a download
command. Verify the hash before running the reproduction.

First, we ran AMD's helper with an empty cache:

```powershell
python amd\compile_npu.py <fakeaudio-model.onnx> `
  --vai-config amd\vitisai_config.json `
  --cache-dir <fresh-cache>
```

We then ran the same model and configuration through `NpuInferenceBench`. This
second run used five labeled audio fixtures, compared the output with the CPU
reference, and enabled ONNX Runtime profiling so we could see where every node
actually executed.

## What failed

VAIML cannot place the buffers for layer 224:

```text
WARNING: [VAIML-BE-SKGEN 11007] 224 of layer L-224 : Failed to place the buffers.
ERROR: [VAIML-BE-SKGEN 11005] Flexible L1 placement failure in layer 224 : L-224 - Failed to place the buffers.
ERROR: [VAIML-COMPILE 1000] vaiml_compile_v4 error: Flexible L1 placement failure in layer 224 : L-224 - Failed to place the buffers.
INFO: [VAIP-VAIML-PASS] Model compilation completed with errors.
INFO: [VAIP-VAIML-PASS] Failed to compile subgraph vaiml_par_1. It will fall back to CPU.
INFO: [VAIP-VAIML-PASS] Subgraph vaiml_par_0 GOPs% (0.956%) is less than the threshold (2%). It will fall back to CPU.
DEBUG2: [VAIP-VAIML-PASS] Total subgraphs: 0
```

After that failure, VAIML falls back to CPU for the main partition. It also
rejects the remaining partition because it represents only 0.956% of the
model's work, below the 2% threshold. The final compiler summary shows zero NPU
subgraphs and all 598 pre-optimization operators assigned to CPU.

## Why the helper reports success

Despite the compiler errors, `compile_npu.py` ends with:

```text
=== Compilation Successful ===
```

That message means only that `ort.InferenceSession(...)` returned successfully.
ONNX Runtime can create a valid session after VitisAI rejects every NPU
partition because the CPU provider remains available.

In other words, the provider loaded successfully, but it did not execute the
model.

## What profiling showed

| Measurement | Result |
| --- | ---: |
| Cold session load | 271.788 s |
| Hot session load | 221.095 s |
| Mean inference | 112.992 ms |
| VitisAI/NPU nodes | 0 |
| CPU nodes | 512 |
| CPU offload | 100% |
| Maximum probability difference from CPU | 0.000000 |
| Fixture accuracy | 3/5 (60%) |

The compiler reports 598 operators before ONNX Runtime graph optimization; the
runtime profile reports 512 nodes afterward. The counts differ because they
refer to different graph stages, not because any nodes moved to the NPU.

The model's 60% fixture accuracy and probabilities exactly match the CPU
reference. That confirms the CPU fallback is numerically consistent, but it
does not demonstrate NPU correctness. The reported 113 ms latency is also
CPU-only and should not be treated as NPU performance.

## Full logs

- [`compile.log`](compile.log) contains the complete output from AMD's helper
  using a fresh cache.
- [`benchmark.log`](benchmark.log) contains the benchmark's cold and hot session
  loads, VAIML output, profiling results, and final JSON result.

The files overlap because both runs invoke the same VAIML compiler. They are
kept separately because `compile.log` captures the behavior of AMD's helper,
while `benchmark.log` adds independent node-assignment and correctness checks.

## Questions for AMD

1. Why does buffer placement fail for `L-224`, and can a different tiling,
   spilling, or partition boundary make it fit on XDNA 2?
2. Should `compile_npu.py` report failure when VAIML compilation fails or when
   the resulting session contains no NPU nodes?
3. Can the helper expose node assignment so a CPU-only session is not mistaken
   for a successful NPU compile?
