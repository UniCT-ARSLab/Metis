import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import tensorflow as tf

from core.policy_artifact import (
    POLICY_MANIFEST_FILENAME,
    POLICY_MODEL_FILENAME,
    load_policy_manifest,
    resolve_policy_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a portable Metis Keras policy to TensorFlow Lite and/or ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--policy",
        default=f"checkpoints/generic/{POLICY_MODEL_FILENAME}",
        help="policy.keras file or Metis policy bundle directory.",
    )
    parser.add_argument("--format", choices=["all", "tflite", "onnx"], default="all")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination directory. Defaults to the policy.keras bundle directory.",
    )
    parser.add_argument(
        "--tflite-quantization",
        choices=["none", "dynamic", "float16", "float32"],
        default="none",
    )
    parser.add_argument("--onnx-opset", type=int, default=18)
    return parser.parse_args()


def convert_tflite(model, output_path, quantization="none"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "float32":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float32]
    elif quantization != "none":
        raise ValueError(f"Unsupported TFLite quantization: {quantization!r}")

    payload = converter.convert()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    validate_tflite(output_path)
    return output_path


def validate_tflite(path):
    interpreter = tf.lite.Interpreter(model_path=str(path))
    input_details = interpreter.get_input_details()
    if len(input_details) != 1:
        raise RuntimeError(f"Metis policies require one observation input, got {len(input_details)}")
    input_shape = list(input_details[0]["shape"])
    if input_shape and input_shape[0] <= 0:
        input_shape[0] = 1
        interpreter.resize_tensor_input(input_details[0]["index"], input_shape, strict=False)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    sample = np.zeros(input_details[0]["shape"], dtype=input_details[0]["dtype"])
    interpreter.set_tensor(input_details[0]["index"], sample)
    interpreter.invoke()
    if not interpreter.get_output_details():
        raise RuntimeError(f"TFLite policy has no outputs: {path}")


def convert_onnx_from_tflite(tflite_path, output_path, opset=18):
    try:
        import tf2onnx
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ONNX export requires the optional dependencies. Install them with: "
            "python -m pip install -r python/requirements-export.txt"
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_tflite(
        str(tflite_path),
        opset=int(opset),
        output_path=str(output_path),
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"ONNX converter did not create a valid file: {output_path}")
    return output_path


def write_export_manifest(source_manifest, output_dir, exports):
    payload = deepcopy(source_manifest)
    payload["source_model"] = POLICY_MODEL_FILENAME
    payload["exports"] = exports
    destination = Path(output_dir) / POLICY_MANIFEST_FILENAME
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def main():
    args = parse_args()
    if args.onnx_opset < 14 or args.onnx_opset > 18:
        raise ValueError("--onnx-opset must be between 14 and 18")

    policy_path = resolve_policy_path(args.policy)
    if not policy_path.is_file():
        raise RuntimeError(f"Keras policy not found: {policy_path}")
    manifest = load_policy_manifest(policy_path)
    if manifest is None:
        raise RuntimeError(f"Metis manifest not found next to {policy_path}: expected policy.json")

    output_dir = Path(args.output_dir) if args.output_dir else policy_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(policy_path, compile=False)
    print(
        f"Loaded Keras policy: {policy_path} algorithm={manifest.get('algorithm')} "
        f"obs_dim={manifest.get('observation', {}).get('size')}",
        flush=True,
    )

    exports = {}
    tflite_path = output_dir / "policy.tflite"
    temporary_directory = None
    try:
        if args.format in {"all", "tflite"}:
            convert_tflite(model, tflite_path, args.tflite_quantization)
            exports["tflite"] = {
                "file": tflite_path.name,
                "quantization": args.tflite_quantization,
            }
            print(f"Exported TFLite policy: {tflite_path}", flush=True)

        if args.format in {"all", "onnx"}:
            onnx_source = tflite_path
            if not onnx_source.is_file() or args.tflite_quantization != "none":
                temporary_directory = tempfile.TemporaryDirectory(prefix="metis-onnx-")
                onnx_source = Path(temporary_directory.name) / "policy-float32.tflite"
                convert_tflite(model, onnx_source, "none")
            onnx_path = output_dir / "policy.onnx"
            convert_onnx_from_tflite(onnx_source, onnx_path, args.onnx_opset)
            exports["onnx"] = {"file": onnx_path.name, "opset": args.onnx_opset}
            print(f"Exported ONNX policy: {onnx_path}", flush=True)
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()

    manifest_path = write_export_manifest(manifest, output_dir, exports)
    print(f"Export manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"Export failed: {exc}") from None
