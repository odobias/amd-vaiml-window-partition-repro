# FakeAudio BF16 VAIML result (2026-07-13)

## Verdict

AMD's `compile_npu.py` and `vitisai_config.json` do not place the current
FakeAudio benchmark model on the NPU in the reproduced environment.

VAIML fails to compile the main partition with a flexible L1 buffer-placement
error at layer 224. It then rejects the remaining small partition because it is
below the 2% GOP threshold. ONNX Runtime still creates a session, but every
profiled inference node executes on the CPU.

The absence of an inference crash is therefore not evidence that this model
runs successfully on the NPU.

## Environment

- AMD Ryzen AI 9 HX 370 / XDNA 2 NPU
- Ryzen AI SDK 1.8.0-beta
- ONNX Runtime 1.25.1
- Windows 11 build 26200
- Original FP32 FakeAudio ONNX, opset 17
- AMD-provided BF16 configuration, optimization level 3, Peano compiler
- No ONNX graph optimization or opset conversion before compilation

## Procedure

First, the unmodified FakeAudio ONNX was passed to AMD's helper with a fresh
cache:

```powershell
python amd\compile_npu.py <fakeaudio-model.onnx> `
  --vai-config amd\vitisai_config.json `
  --cache-dir <fresh-cache>
```

The same model and AMD configuration were then run through
`NpuInferenceBench`, built against the Ryzen AI ONNX Runtime. The benchmark used
five labeled fixture inputs, compared probabilities with the CPU reference,
and enabled ONNX Runtime profiling to audit actual node assignment.

## Compilation failure

The compiler reports:

```text
WARNING: [VAIML-BE-SKGEN 11007] 224 of layer L-224 : Failed to place the buffers.
ERROR: [VAIML-BE-SKGEN 11005] Flexible L1 placement failure in layer 224 : L-224 - Failed to place the buffers.
ERROR: [VAIML-COMPILE 1000] vaiml_compile_v4 error: Flexible L1 placement failure in layer 224 : L-224 - Failed to place the buffers.
INFO: [VAIP-VAIML-PASS] Model compilation completed with errors.
INFO: [VAIP-VAIML-PASS] Failed to compile subgraph vaiml_par_1. It will fall back to CPU.
INFO: [VAIP-VAIML-PASS] Subgraph vaiml_par_0 GOPs% (0.956%) is less than the threshold (2%). It will fall back to CPU.
DEBUG2: [VAIP-VAIML-PASS] Total subgraphs: 0
```

The VAIP summary consequently assigns all 598 pre-optimization operators to
CPU. Cache generation also reports:

```text
Model signature file does not exist: <cache>/original-model-signature.txt
```

Despite those errors, `compile_npu.py` prints:

```text
=== Compilation Successful ===
```

That message is not a compiler-success signal. The helper prints it
unconditionally after `ort.InferenceSession(...)` returns. Session creation can
succeed after VitisAI rejects every NPU partition because ONNX Runtime retains
CPU execution.

## Profiled benchmark result

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

The 512-node count is from the optimized runtime graph, while the compiler's
598-operator count is from the earlier graph representation.

`VitisAIExecutionProvider` appears as the resolved provider and provider setup
returns success. This only proves that the EP loaded; it does not prove that it
claimed any graph nodes. The profile is decisive: `ep_nodes=0`,
`cpu_nodes=512`, and `cpu_offload_pct=100`.

The 60% fixture accuracy is identical to the CPU reference and is not attributed
to VAIML. Likewise, the reported latency is CPU-only and must not be presented
as NPU performance.

## Attached logs

- [`compile.log`](compile.log) — complete output from AMD's helper on a fresh
  cache.
- [`benchmark.log`](benchmark.log) — complete VAIML output plus the final
  structured benchmark result and node-assignment audit.

The key compile failure is near the end of each log. The final JSON object in
`benchmark.log` contains the exact metrics, per-sample outputs, and CPU operator
counts.

## Requested follow-up

1. Investigate the flexible L1 placement failure for `L-224`.
2. Determine whether this partition needs a different tiling, spill, or
   partition boundary strategy for XDNA 2.
3. Make `compile_npu.py` distinguish session creation from successful NPU
   compilation.
4. Report claimed NPU nodes (or fail when there are none) so CPU-only execution
   cannot be mistaken for a successful compile.
