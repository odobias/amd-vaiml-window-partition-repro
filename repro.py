#!/usr/bin/env python3
"""Minimal synthetic VAIML first-inference crash reproducer.

This script generates a tiny ONNX graph locally, then runs it through either ORT
CPU or AMD's VitisAIExecutionProvider. The generated graph contains only a
LayerNormalization followed by the dynamic shape/reshape/transpose sequence used
to partition a [1,4096,96] tensor into 64 windows of [64,96]. There are no
trained model weights or captured customer inputs.
"""
import argparse
import hashlib
import json
import platform
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
MODEL = HERE / "model.onnx"
INPUT = "/embedder/base/htsat/patch_embed/norm/LayerNormalization_output_0"
OUTPUT = "/embedder/base/htsat/layers.0/blocks.0/Reshape_3_output_0"
# Updated by the generation code if the graph definition changes intentionally.
GENERATED_MODEL_SHA256 = "d4059b6e2f8beb7a2b36ee2217beb0f52d444be034d004906fc34b1b66ea2eac"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def const_i64(name: str, output: str, values) -> onnx.NodeProto:
    arr = np.asarray(values, dtype=np.int64)
    return helper.make_node(
        "Constant",
        [],
        [output],
        name=name,
        value=numpy_helper.from_array(arr),
    )


def gather_dim(nodes, source: str, dim: int, suffix: str) -> str:
    shape_out = f"/embedder/base/htsat/layers.0/blocks.0/Shape{suffix}_output_0"
    const_out = f"/embedder/base/htsat/layers.0/blocks.0/Constant{suffix}_output_0"
    gather_out = f"/embedder/base/htsat/layers.0/blocks.0/Gather{suffix}_output_0"
    nodes.append(helper.make_node("Shape", [source], [shape_out], name=f"/embedder/base/htsat/layers.0/blocks.0/Shape{suffix}"))
    nodes.append(const_i64(f"/embedder/base/htsat/layers.0/blocks.0/Constant{suffix}", const_out, dim))
    nodes.append(helper.make_node("Gather", [shape_out, const_out], [gather_out], name=f"/embedder/base/htsat/layers.0/blocks.0/Gather{suffix}", axis=0))
    return gather_out


def unsqueeze(nodes, value: str, out: str, name: str, axes_name: str) -> str:
    axes_out = f"onnx::Unsqueeze_{axes_name}"
    nodes.append(const_i64(f"Constant_{axes_name}", axes_out, [0]))
    nodes.append(helper.make_node("Unsqueeze", [value, axes_out], [out], name=name))
    return out


def build_model(path: Path) -> str:
    nodes = []

    # Original input shape is [batch, 4096, channels]. The dynamic shape nodes are
    # intentionally retained; disabling ORT graph optimizations keeps VAIML from
    # hiding the bug through constant folding or CPU fallback.
    batch = gather_dim(nodes, INPUT, 0, "")
    channels = gather_dim(nodes, INPUT, 2, "_1")

    weight = numpy_helper.from_array(np.ones((96,), dtype=np.float32), "embedder.base.htsat.layers.0.blocks.0.norm1.weight")
    bias = numpy_helper.from_array(np.zeros((96,), dtype=np.float32), "embedder.base.htsat.layers.0.blocks.0.norm1.bias")
    norm_out = "/embedder/base/htsat/layers.0/blocks.0/norm1/LayerNormalization_output_0"
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            [INPUT, weight.name, bias.name],
            [norm_out],
            name="/embedder/base/htsat/layers.0/blocks.0/norm1/LayerNormalization",
            axis=-1,
            epsilon=np.float32(1e-5).item(),
        )
    )

    batch_u = unsqueeze(nodes, batch, "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_output_0", "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze", "369")
    const64a = "/embedder/base/htsat/layers.0/blocks.0/Constant_2_output_0"
    const64b = "/embedder/base/htsat/layers.0/blocks.0/Constant_3_output_0"
    nodes.append(const_i64("/embedder/base/htsat/layers.0/blocks.0/Constant_2", const64a, [64]))
    nodes.append(const_i64("/embedder/base/htsat/layers.0/blocks.0/Constant_3", const64b, [64]))
    channels_u = unsqueeze(nodes, channels, "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_1_output_0", "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_1", "375")
    shape1 = "/embedder/base/htsat/layers.0/blocks.0/Concat_output_0"
    nodes.append(helper.make_node("Concat", [batch_u, const64a, const64b, channels_u], [shape1], name="/embedder/base/htsat/layers.0/blocks.0/Concat", axis=0))
    r0 = "/embedder/base/htsat/layers.0/blocks.0/Reshape_output_0"
    nodes.append(helper.make_node("Reshape", [norm_out, shape1], [r0], name="/embedder/base/htsat/layers.0/blocks.0/Reshape", allowzero=0))

    r0_dims = [gather_dim(nodes, r0, i, f"_r0_{i}") for i in range(4)]
    h_div = "/embedder/base/htsat/layers.0/blocks.0/Div_output_0"
    w_div = "/embedder/base/htsat/layers.0/blocks.0/Div_1_output_0"
    nodes.append(const_i64("/embedder/base/htsat/layers.0/blocks.0/Constant_8", "/embedder/base/htsat/layers.0/blocks.0/Constant_8_output_0", 8))
    nodes.append(helper.make_node("Div", [r0_dims[1], "/embedder/base/htsat/layers.0/blocks.0/Constant_8_output_0"], [h_div], name="/embedder/base/htsat/layers.0/blocks.0/Div"))
    h_cast0 = "/embedder/base/htsat/layers.0/blocks.0/Cast_output_0"
    h_cast1 = "/embedder/base/htsat/layers.0/blocks.0/Cast_1_output_0"
    nodes.append(helper.make_node("Cast", [h_div], [h_cast0], name="/embedder/base/htsat/layers.0/blocks.0/Cast", to=TensorProto.INT64))
    nodes.append(helper.make_node("Cast", [h_cast0], [h_cast1], name="/embedder/base/htsat/layers.0/blocks.0/Cast_1", to=TensorProto.INT64))
    nodes.append(const_i64("/embedder/base/htsat/layers.0/blocks.0/Constant_9", "/embedder/base/htsat/layers.0/blocks.0/Constant_9_output_0", 8))
    nodes.append(helper.make_node("Div", [r0_dims[2], "/embedder/base/htsat/layers.0/blocks.0/Constant_9_output_0"], [w_div], name="/embedder/base/htsat/layers.0/blocks.0/Div_1"))
    w_cast0 = "/embedder/base/htsat/layers.0/blocks.0/Cast_2_output_0"
    w_cast1 = "/embedder/base/htsat/layers.0/blocks.0/Cast_3_output_0"
    nodes.append(helper.make_node("Cast", [w_div], [w_cast0], name="/embedder/base/htsat/layers.0/blocks.0/Cast_2", to=TensorProto.INT64))
    nodes.append(helper.make_node("Cast", [w_cast0], [w_cast1], name="/embedder/base/htsat/layers.0/blocks.0/Cast_3", to=TensorProto.INT64))

    c8a = "/embedder/base/htsat/layers.0/blocks.0/Constant_10_output_0"
    c8b = "/embedder/base/htsat/layers.0/blocks.0/Constant_11_output_0"
    nodes.append(const_i64("/embedder/base/htsat/layers.0/blocks.0/Constant_10", c8a, [8]))
    nodes.append(const_i64("/embedder/base/htsat/layers.0/blocks.0/Constant_11", c8b, [8]))
    shape2_parts = [
        unsqueeze(nodes, r0_dims[0], "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_2_output_0", "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_2", "400"),
        unsqueeze(nodes, h_cast1, "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_3_output_0", "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_3", "402"),
        c8a,
        unsqueeze(nodes, w_cast1, "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_4_output_0", "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_4", "406"),
        c8b,
        unsqueeze(nodes, r0_dims[3], "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_5_output_0", "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_5", "410"),
    ]
    shape2 = "/embedder/base/htsat/layers.0/blocks.0/Concat_1_output_0"
    nodes.append(helper.make_node("Concat", shape2_parts, [shape2], name="/embedder/base/htsat/layers.0/blocks.0/Concat_1", axis=0))
    r1 = "/embedder/base/htsat/layers.0/blocks.0/Reshape_1_output_0"
    nodes.append(helper.make_node("Reshape", [r0, shape2], [r1], name="/embedder/base/htsat/layers.0/blocks.0/Reshape_1", allowzero=0))
    t0 = "/embedder/base/htsat/layers.0/blocks.0/Transpose_output_0"
    nodes.append(helper.make_node("Transpose", [r1], [t0], name="/embedder/base/htsat/layers.0/blocks.0/Transpose", perm=[0, 1, 3, 2, 4, 5]))

    cm1a = "/embedder/base/htsat/layers.0/blocks.0/Constant_12_output_0"
    c8c = "/embedder/base/htsat/layers.0/blocks.0/Constant_13_output_0"
    c8d = "/embedder/base/htsat/layers.0/blocks.0/Constant_14_output_0"
    nodes.append(const_i64("/embedder/base/htsat/layers.0/blocks.0/Constant_12", cm1a, [-1]))
    nodes.append(const_i64("/embedder/base/htsat/layers.0/blocks.0/Constant_13", c8c, [8]))
    nodes.append(const_i64("/embedder/base/htsat/layers.0/blocks.0/Constant_14", c8d, [8]))
    shape3 = "/embedder/base/htsat/layers.0/blocks.0/Concat_2_output_0"
    nodes.append(helper.make_node("Concat", [cm1a, c8c, c8d, unsqueeze(nodes, r0_dims[3], "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_6_output_0", "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_6", "421")], [shape3], name="/embedder/base/htsat/layers.0/blocks.0/Concat_2", axis=0))
    r2 = "/embedder/base/htsat/layers.0/blocks.0/Reshape_2_output_0"
    nodes.append(helper.make_node("Reshape", [t0, shape3], [r2], name="/embedder/base/htsat/layers.0/blocks.0/Reshape_2", allowzero=0))

    cm1b = "/embedder/base/htsat/layers.0/blocks.0/Constant_15_output_0"
    c64 = "/embedder/base/htsat/layers.0/blocks.0/Constant_16_output_0"
    nodes.append(const_i64("/embedder/base/htsat/layers.0/blocks.0/Constant_15", cm1b, [-1]))
    nodes.append(const_i64("/embedder/base/htsat/layers.0/blocks.0/Constant_16", c64, [64]))
    shape4 = "/embedder/base/htsat/layers.0/blocks.0/Concat_3_output_0"
    nodes.append(helper.make_node("Concat", [cm1b, c64, unsqueeze(nodes, channels, "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_7_output_0", "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_7", "429")], [shape4], name="/embedder/base/htsat/layers.0/blocks.0/Concat_3", axis=0))
    nodes.append(helper.make_node("Reshape", [r2, shape4], [OUTPUT], name="/embedder/base/htsat/layers.0/blocks.0/Reshape_3", allowzero=0))

    graph = helper.make_graph(
        nodes,
        "amd-vaiml-window-partition-repro",
        [helper.make_tensor_value_info(INPUT, TensorProto.FLOAT, [1, 4096, 96])],
        [helper.make_tensor_value_info(OUTPUT, TensorProto.FLOAT, ["unk__6", 64, 96])],
        [weight, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return file_sha256(path)


def ensure_model(path: Path) -> str:
    if not path.exists():
        print(f"Generating synthetic ONNX repro: {path}", flush=True)
        return build_model(path)
    return file_sha256(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=("cpu", "vitisai"))
    parser.add_argument("--model", default=str(MODEL), help="Path to generated ONNX repro model")
    parser.add_argument("--regenerate-model", action="store_true", help="Overwrite --model before running")
    parser.add_argument(
        "--cache-dir",
        default=str(Path(tempfile.gettempdir()) / "amd-vaiml-window-partition-repro"),
    )
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if args.regenerate_model and model_path.exists():
        model_path.unlink()
    actual_hash = ensure_model(model_path)
    if GENERATED_MODEL_SHA256 and actual_hash != GENERATED_MODEL_SHA256:
        raise RuntimeError(f"generated model SHA256 mismatch: {actual_hash}")

    available = ort.get_available_providers()
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "onnxruntime": ort.__version__,
        "available_providers": available,
        "model": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "model_sha256": actual_hash,
        "requested_provider": args.provider,
        "graph_optimizations": "disabled",
    }
    print(json.dumps(metadata, indent=2), flush=True)

    options = ort.SessionOptions()
    options.log_severity_level = 2
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    providers = ["CPUExecutionProvider"]
    if args.provider == "vitisai":
        if "VitisAIExecutionProvider" not in available:
            raise RuntimeError("VitisAIExecutionProvider is not available in this ONNX Runtime")
        cache_dir = Path(args.cache_dir).resolve()
        if cache_dir.exists() and not args.reuse_cache:
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        providers = [
            (
                "VitisAIExecutionProvider",
                {
                    "cache_dir": str(cache_dir),
                    "cache_key": f"window-partition-{actual_hash[:12]}",
                },
            ),
            "CPUExecutionProvider",
        ]

    print("Creating InferenceSession...", flush=True)
    session = ort.InferenceSession(str(model_path), options, providers=providers)
    print(f"Session created; providers={session.get_providers()}", flush=True)

    value = np.random.default_rng(20260711).standard_normal((1, 4096, 96), dtype=np.float32)
    print("Running first inference...", flush=True)
    result = session.run(None, {session.get_inputs()[0].name: value})[0]
    summary = {
        "shape": list(result.shape),
        "min": float(result.min()),
        "max": float(result.max()),
        "mean": float(result.mean()),
        "finite": bool(np.isfinite(result).all()),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
