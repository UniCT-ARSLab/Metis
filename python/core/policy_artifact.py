import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf


POLICY_FORMAT = "metis-policy"
POLICY_FORMAT_VERSION = 1
POLICY_MODEL_FILENAME = "policy.keras"
POLICY_MANIFEST_FILENAME = "policy.json"
DETERMINISTIC_POLICY_ALGORITHMS = {"ddpg", "ddpg_bc", "ddpgfd", "td3", "td3_bc"}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _decoder_for_algorithm(algorithm):
    if algorithm == "dqn":
        return "argmax_q_values"
    if algorithm == "sac":
        return "tanh_mean_then_scale"
    if algorithm == "ppo":
        return "ppo_deterministic_heads"
    return "tanh_then_scale"


def build_policy_metadata(algorithm, env):
    agent_spec = env._spec_for_agent(env.agent_id)
    action_metadata = {
        "type": str(env.action_type),
        "size": int(env.action_size),
        "names": list(env.action_names),
        "space": _jsonable(env.action_space_spec),
    }
    if env.action_type == "continuous":
        action_metadata["low"] = _jsonable(env.action_low)
        action_metadata["high"] = _jsonable(env.action_high)
    return {
        "format": POLICY_FORMAT,
        "format_version": POLICY_FORMAT_VERSION,
        "algorithm": str(algorithm),
        "model_file": POLICY_MODEL_FILENAME,
        "observation": {
            "dtype": "float32",
            "size": int(env.obs_dim),
            "names": list(agent_spec.get("observation_names", [])),
        },
        "action": action_metadata,
        "inference": {
            "decoder": _decoder_for_algorithm(str(algorithm)),
            "deterministic": True,
        },
        "parameter_sharing": True,
    }


def load_policy_manifest(policy_path):
    path = Path(policy_path)
    manifest_path = path / POLICY_MANIFEST_FILENAME if path.is_dir() else path.with_name(POLICY_MANIFEST_FILENAME)
    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != POLICY_FORMAT:
        raise ValueError(f"Unsupported policy manifest format in {manifest_path}")
    if int(payload.get("format_version", 0)) != POLICY_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported policy manifest version={payload.get('format_version')} in {manifest_path}"
        )
    return payload


def resolve_policy_path(path):
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / POLICY_MODEL_FILENAME
    return candidate


def policy_algorithms_are_compatible(expected, saved):
    if not expected or not saved:
        return True
    if expected == saved:
        return True
    return expected in DETERMINISTIC_POLICY_ALGORITHMS and saved in DETERMINISTIC_POLICY_ALGORITHMS


def validate_policy_algorithm(manifest, expected_algorithm, policy_path):
    if manifest is None or not expected_algorithm:
        return
    saved_algorithm = str(manifest.get("algorithm", ""))
    if not policy_algorithms_are_compatible(str(expected_algorithm), saved_algorithm):
        raise RuntimeError(
            f"Policy {policy_path} was saved by algorithm={saved_algorithm!r}, but "
            f"algorithm={expected_algorithm!r} was requested."
        )


def load_policy_model(policy_path):
    """Load a complete Keras policy, or classify a legacy HDF5 weights file."""
    path = resolve_policy_path(policy_path)
    if not path.is_file():
        raise RuntimeError(f"Policy file not found: {path}")

    manifest = load_policy_manifest(path)
    lower_name = path.name.lower()
    if lower_name.endswith(".weights.h5") or lower_name.endswith(".weights.hdf5"):
        return None, manifest, "weights", path

    try:
        model = tf.keras.models.load_model(path, compile=False)
        suffix_kind = "h5_model" if path.suffix.lower() in {".h5", ".hdf5"} else "keras_model"
        return model, manifest, suffix_kind, path
    except Exception as model_error:
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            raise RuntimeError(f"Could not load Keras policy model: {path}") from model_error

        # Historical Metis commands wrote weights with arbitrary .h5 names. A target
        # architecture is needed before Keras can distinguish and restore those files.
        return None, manifest, "weights", path


def load_policy_into_model(model, policy_path, expected_algorithm=None):
    """Warm-start an existing architecture from .keras, full .h5, or weights .h5."""
    loaded_model, manifest, source_kind, path = load_policy_model(policy_path)
    validate_policy_algorithm(manifest, expected_algorithm, path)

    try:
        if loaded_model is not None:
            model.set_weights(loaded_model.get_weights())
        else:
            model.load_weights(path)
    except Exception as exc:
        raise RuntimeError(
            f"Policy {path} is incompatible with the current model architecture "
            f"for algorithm={expected_algorithm or 'auto'}."
        ) from exc

    return {
        "path": path,
        "manifest": manifest,
        "source_kind": source_kind,
    }


class PolicyArtifactSaver:
    def __init__(self, model, directory, metadata):
        self.model = model
        self.directory = Path(directory)
        self.metadata = deepcopy(metadata)

    @property
    def model_path(self):
        return self.directory / POLICY_MODEL_FILENAME

    @property
    def manifest_path(self):
        return self.directory / POLICY_MANIFEST_FILENAME

    def save(self, episode):
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary_model = self.directory / ".policy.tmp.keras"
        temporary_manifest = self.directory / ".policy.tmp.json"
        temporary_model.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)

        self.model.save(temporary_model)
        os.replace(temporary_model, self.model_path)

        payload = deepcopy(self.metadata)
        payload.update({
            "episode": int(episode),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "tensorflow_version": tf.__version__,
            "keras_version": tf.keras.__version__,
            "model_inputs": [tensor.name for tensor in self.model.inputs],
            "model_outputs": list(self.model.output_names),
        })
        temporary_manifest.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, self.manifest_path)
        print(
            f"Saved Keras policy: {self.model_path} (episode={int(episode)})",
            flush=True,
        )
        return self.model_path
