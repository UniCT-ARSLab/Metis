"""M6.3: export the composite policy.keras to ONNX and verify numerical parity with Keras."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf  # noqa: E402

from core.policy_artifact import load_policy_model  # noqa: E402  (registers composite layer)

OUT = Path("checkpoints/openarm_reach_hold_m6_3_policy")
POLICY = OUT / "policy.keras"
ONNX = OUT / "policy.onnx"
M5V2 = "python/demos/openarm_reach_hold_m5_v2/train.npz"


def main():
    import tf2onnx
    import onnxruntime as ort

    model, _manifest, _kind, _path = load_policy_model(str(POLICY))
    obs = np.asarray(np.load(M5V2, allow_pickle=True)["obs"], np.float32)
    obs = obs[np.random.default_rng(2).choice(len(obs), size=1024, replace=False)]
    keras_out = model(tf.convert_to_tensor(obs), training=False).numpy()

    spec = (tf.TensorSpec((None, obs.shape[1]), tf.float32, name="obs"),)

    @tf.function(input_signature=spec)
    def fwd(x):
        return model(x, training=False)

    tf2onnx.convert.from_function(fwd, input_signature=spec, opset=15, output_path=str(ONNX))
    print(f"exported {ONNX}", flush=True)

    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {iname: obs})[0]
    diff = float(np.max(np.abs(onnx_out - keras_out)))
    finite = bool(np.all(np.isfinite(onnx_out)) and np.all(np.abs(onnx_out) <= 1.0 + 1e-6))
    res = {"onnx_parity_max_diff": diff, "finite_and_bounded": finite, "PASS": diff <= 1e-5 and finite}
    (OUT / "onnx_parity.json").write_text(json.dumps(res, indent=2))
    print("ONNX_PARITY " + json.dumps(res), flush=True)
    if not res["PASS"]:
        raise SystemExit("ONNX parity FAILED")


if __name__ == "__main__":
    main()
