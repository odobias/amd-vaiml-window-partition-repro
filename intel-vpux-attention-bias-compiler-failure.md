# Intel VPUX abort while fusing HTSAT attention bias

## Model

The failure was isolated from this public FakeAudio MS-CLAP/HTSAT model:

- [Hugging Face repository](https://huggingface.co/odobias/fakeaudio-onnx-vaiml-repro)
- [Download `model.onnx`](https://huggingface.co/odobias/fakeaudio-onnx-vaiml-repro/resolve/main/model.onnx?download=true)
- Size: 127,844,774 bytes
- SHA-256: `d4b2dde8f4f95862ecb9f5e90821d853adea3774e7c48857f60ee6084d5e85c6`

The smallest failing artifact is `model.backbone-fp32.onnx`, derived by
separating the log-mel frontend from that model. It should be attached to the
Intel issue together with the working transformed backbone.

## Environment

- Intel Core Ultra, Lunar Lake
- Windows 11
- Intel NPU driver `10.0.26100.x`
- OpenVINO `2026.2.1`
- Target: `NPU`

The abort occurs through direct `ov.Core().compile_model`; ONNX Runtime is not
involved.

## Failure

The backbone uses standard ONNX broadcasting before `Softmax`:

```text
scores [nW,H,L,L] + bias [1,H,L,L] -> [nW,H,L,L]
```

During VPUX SDPA fusion, the operands reach `IE.Add` as:

```text
1x64x64x64 + 16x64x64
```

The shapes are no longer broadcast-compatible. Compilation terminates in an
LLVM abort instead of returning an error to the caller.

Minimal reproduction:

```python
import openvino as ov

core = ov.Core()
model = core.read_model("model.backbone-fp32.onnx")
core.compile_model(model, "NPU")  # process aborts
```

If the attached backbone has a dynamic input, reshape it to the attached
frontend-output tensor's shape before compilation.

## Workaround

For each of seven `Add -> Softmax` attention blocks, explicitly materialize the
bias broadcast:

```text
shape = Shape(scores)
expanded_bias = Expand(bias, shape)
Add(scores, expanded_bias) -> Softmax
```

The resulting `model.backbone.npu-intel.onnx` compiles successfully. This is a
semantics-preserving rewrite; CPU FP32 frontend plus the transformed NPU
backbone showed:

- maximum probability delta versus full FP32 CPU: approximately `0.0001`
- prediction flips on five validation samples: `0`
- end-to-end latency: approximately `30 ms`
- NPU backbone share: approximately 78–85%

This points to a compiler fusion defect rather than unsupported NPU arithmetic:
the original broadcast is valid, and materializing it changes no results.

## Separate precision issue

The complete graph cannot run accurately at NPU FP16 because its log-mel
frontend underflows before `Log`. That is independent of this compiler abort.
The isolated backbone avoids conflating the two issues.

## Requested investigation

Why does SDPA fusion transform a valid rank-four broadcast into
`1x64x64x64 + 16x64x64`?

## Attachments needed

- `model.backbone-fp32.onnx` — failing backbone
- `model.backbone.npu-intel.onnx` — working explicit-broadcast backbone
- one frontend-output input tensor
- full verbose NPU compilation log
- exact `FULL_DEVICE_NAME`, driver version, and OpenVINO package version
