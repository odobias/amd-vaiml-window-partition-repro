# Intel VPUX compiler abort on broadcast attention bias

## Summary

The Intel NPU compiler aborts while compiling the transformer backbone of a
FakeAudio detector based on MS-CLAP/HTSAT windowed attention. The source ONNX
graph contains a valid broadcast addition immediately before `Softmax`:

```text
attention scores: [nW, H, L, L]
attention bias:   [ 1, H, L, L]
result:           [nW, H, L, L]
```

During SDPA fusion, the VPUX compiler appears to flatten or otherwise transform
the operands inconsistently. The failing compiler IR was reported as an
`IE.Add` between:

```text
1x64x64x64 + 16x64x64
```

These transformed shapes are not broadcast-compatible, and compilation ends in
an LLVM abort rather than returning a recoverable compilation error.

Explicitly expanding the bias constant to the full score shape before the
addition makes the model compile and run correctly. This transformation is
numerically identical to the original ONNX broadcast and was required in seven
attention blocks.

## Impact

- A valid ONNX model cannot be compiled for the Intel NPU without modifying its
  attention subgraphs.
- The compiler abort can terminate the hosting process, so an application cannot
  catch the failure and fall back to another execution provider.
- The workaround creates a nominally Intel-specific model artifact, obstructing
  our goal of using identical model bytes across Intel, AMD, Qualcomm, CPU, and
  GPU executors.

## Environment where reproduced

| Component | Value |
| --- | --- |
| Host | Intel Core Ultra, Lunar Lake |
| Operating system | Windows 11 |
| Intel NPU driver | `10.0.26100.x` |
| Direct OpenVINO reproduction | OpenVINO `2026.2.1` |
| ORT/OpenVINO environment also tested | ONNX Runtime OpenVINO EP `1.24.1`, OpenVINO `2025.4.1` |
| Target device | `NPU` |
| Model family | FakeAudio detector, MS-CLAP/HTSAT backbone |
| Failing artifact | Unsurgered FP32 backbone ONNX |

The failure occurs in direct OpenVINO compilation, so ONNX Runtime is not
required to reproduce it.

## Minimal compile reproduction

Given the unsurgered backbone as `model.backbone-fp32.onnx`:

```python
import openvino as ov

core = ov.Core()
model = core.read_model("model.backbone-fp32.onnx")

# If the attached model has a dynamic input, reshape it to the exact shape of
# the attached frontend-output tensor before compilation.

# The compiler abort occurs here.
compiled = core.compile_model(model, "NPU")
```

The defect is associated with the pre-`Softmax` attention-bias additions, not
with model input reshaping.

## Expected behavior

The compiler should preserve standard ONNX multidirectional broadcasting:

```text
[nW, H, L, L] + [1, H, L, L] -> [nW, H, L, L]
```

It should either:

1. compile the valid graph, or
2. return a recoverable error describing an unsupported operation.

It should not abort the process.

## Actual behavior

The VPUX SDPA fusion path rewrites the attention operation such that the score
and bias operands reach `IE.Add` with incompatible ranks/shapes:

```text
IE.Add 1x64x64x64 + 16x64x64
```

Compilation then terminates with an LLVM error/abort. Because the process is
terminated, the full compiler diagnostic was not consistently capturable as a
Python exception.

## Semantics-preserving workaround

Before each `Softmax`, locate an `Add` where one operand is a constant attention
bias. Insert `Shape` and `Expand` so the constant has exactly the runtime score
shape:

```python
from collections import defaultdict

import onnx
from onnx import helper


def expand_attention_bias(source_path, output_path):
    model = onnx.load(source_path)
    graph = model.graph
    initializers = {initializer.name for initializer in graph.initializer}
    producer = {output: node for node in graph.node for output in node.output}
    prepend = defaultdict(list)
    patched = 0

    for softmax in graph.node:
        if softmax.op_type != "Softmax":
            continue

        add = producer.get(softmax.input[0])
        if add is None or add.op_type != "Add" or len(add.input) != 2:
            continue

        left, right = add.input
        constant = left if left in initializers else (
            right if right in initializers else None
        )
        if constant is None:
            continue

        scores = right if constant == left else left
        stem = add.name.replace("/", "_").strip("_")
        score_shape = f"{stem}__score_shape"
        expanded_bias = f"{stem}__expanded_bias"

        prepend[id(add)].append(
            helper.make_node("Shape", [scores], [score_shape])
        )
        prepend[id(add)].append(
            helper.make_node("Expand", [constant, score_shape], [expanded_bias])
        )
        add.input[:] = [
            expanded_bias if value == constant else value for value in add.input
        ]
        patched += 1

    rebuilt = []
    for node in graph.node:
        rebuilt.extend(prepend.get(id(node), []))
        rebuilt.append(node)
    del graph.node[:]
    graph.node.extend(rebuilt)
    onnx.save(model, output_path)
    return patched
```

Observed result:

- Seven attention blocks patched.
- The patched backbone compiles successfully on the Intel NPU.
- CPU FP32 frontend plus patched NPU backbone matches the full FP32 CPU model:
  maximum probability delta approximately `0.0001`, with no prediction flips on
  the five-sample validation fixture.
- End-to-end latency was approximately `30 ms`, with roughly 78–85% of the time
  spent in the NPU backbone.

## Why this appears to be a compiler defect

1. The original addition follows valid ONNX broadcast semantics.
2. The problematic shapes appear only after the compiler's attention/SDPA
   transformation.
3. Materializing the broadcast with `Expand` changes neither the mathematical
   operation nor validation results.
4. The materialized form compiles, indicating that the NPU supports the
   underlying computation.
5. An unsupported pattern should produce a recoverable diagnostic, not an LLVM
   process abort.

## Separate precision limitation

This compiler failure is independent of a second issue in the complete model:
the log-mel frontend must remain FP32 because FP16 underflows very small
mel-energy values before `Log`. Running the full graph at NPU FP16 therefore
produces incorrect predictions even after the attention workaround.

The recommended reproduction isolates the transformer backbone so that the
frontend precision limitation does not obscure the VPUX compiler defect.

## Requested investigation

Please investigate:

1. Why the SDPA fusion converts the valid four-dimensional broadcast addition
   into operands resembling `1x64x64x64` and `16x64x64`.
2. Whether rank information is lost while flattening the window or head
   dimensions.
3. Whether the compiler can preserve the original broadcast or internally
   materialize it.
4. Why this path reaches an LLVM abort instead of returning a recoverable
   compilation error.
5. Which OpenVINO/NPU compiler release contains, or is expected to contain, a
   fix.

## Suggested attachments

Send the following with the issue:

- `model.backbone-fp32.onnx` — smallest known failing artifact.
- `model.backbone.npu-intel.onnx` — working artifact with explicit bias
  expansion.
- A frontend output tensor used as the backbone input.
- Full OpenVINO/NPU compilation log with maximum compiler verbosity.
- `ov.Core().get_property("NPU", "FULL_DEVICE_NAME")` and available NPU driver
  version details.
- The exact OpenVINO package version and Python environment lock file.

Do not send only the complete FakeAudio graph: the isolated backbone gives Intel
a smaller reproduction and avoids conflating this compiler abort with the
separate FP16 frontend accuracy problem.
