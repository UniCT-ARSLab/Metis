# Exporting Metis policies

Metis keeps training state separate from the policy used for inference.

## Training artifacts

Every checkpoint refreshes these files under `--checkpoint-dir`:

```text
checkpoints/my_run/
|-- ckpt-100.*
|-- policy.keras
`-- policy.json
```

- `ckpt-*` contains optimizer, target-network, critic, temperature, and counter state
  needed for resume.
- `policy.keras` contains the architecture and weights used to select actions.
- `policy.json` records the observation contract, action components, algorithm, and
  output decoder.

The `.keras` file is the preferred portable format. Legacy `.weights.h5` files are
still supported, but code must reconstruct their architecture before loading them.

## Running a Keras policy

`run.py` finds the policy bundle inside a checkpoint directory:

```bash
python/.venv/bin/python python/run.py \
  --checkpoint-dir checkpoints/my_run \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/my_scenario.tscn
```

The algorithm is read from `policy.json`. To load a specific file:

```text
--load-from policy --policy-path exports/my_policy/policy.keras
```

`--policy-path` also accepts a full HDF5 model (`model.h5`) or legacy weights
(`model.weights.h5`). A standalone file without `policy.json` has no Metis contract.
`run.py` can still validate input size, but ambiguous architectures such as discrete
PPO versus DQN may require `--algorithm`.

## Extracting a policy from an exact checkpoint

The root `policy.keras` follows the most recently saved checkpoint, while the best
checkpoint may live under `best/`. To extract precisely that policy without restoring
its critic, replay buffer, optimizers, or entropy state:

```bash
python/.venv/bin/python python/run.py \
  --algorithm sac \
  --load-from checkpoint \
  --checkpoint-path checkpoints/my_run/best/ckpt-1200 \
  --export-policy-dir checkpoints/my_run/best/actor_policy \
  --export-policy-only \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/my_scenario.tscn \
  --headless
```

This writes `actor_policy/policy.keras` and its matching `policy.json`. The scenario
is opened only to validate the observation and action contract. The resulting bundle
can be run directly or passed to a new training job with `--policy-path`.

## Warm-starting a new training run

```bash
python/.venv/bin/python python/train.py \
  --algorithm dqn \
  --policy-path exports/my_policy/policy.keras \
  --checkpoint-dir checkpoints/new_run \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/my_scenario.tscn
```

This is a warm start. Episode counters, optimizers, replay, critics, target networks,
and exploration schedules start fresh. Use `--resume` or `--resume-checkpoint` to
continue the original training state. Metis rejects combining resume and
`--policy-path` because that would create an ambiguous state. Off-policy trainers
also honor `--critic-warmup-updates` after an actor-only warm start, keeping the
imported actor frozen while the new critic learns the new reward scale.

## TensorFlow Lite

TFLite is part of TensorFlow and needs no optional converter package:

```bash
python/.venv/bin/python python/export.py \
  --policy checkpoints/my_run \
  --format tflite
```

The command writes `policy.tflite` next to `policy.keras` and updates the manifest.

```text
--tflite-quantization none
--tflite-quantization dynamic
--tflite-quantization float16
```

`none` keeps float32 and is the most predictable default. `dynamic` mainly compresses
weights. `float16` can help on hardware that accelerates half precision. Full int8
quantization needs a representative dataset and is never enabled implicitly.

## ONNX

Install the optional exporter dependencies:

```bash
python/.venv/bin/python -m pip install -r python/requirements-export.txt
```

Then export:

```bash
python/.venv/bin/python python/export.py \
  --policy checkpoints/my_run \
  --format onnx
```

Use `--format all` to produce both TFLite and ONNX. ONNX export defaults to opset 18;
`--onnx-opset` accepts the converter-supported range 14 through 18.

## Deployment contract

The model is only the numerical policy. A deployment outside Metis must:

1. Build observations in the same order and with the same normalization.
2. Feed a float32 batch shaped `[N, observation.size]`.
3. Decode outputs according to `inference.decoder` in `policy.json`.
4. Apply component names, bounds, and structure from the `action` section.

Current decoders are:

- `argmax_q_values`: choose the largest DQN Q value;
- `tanh_then_scale`: deterministic DDPG/TD3-family actor output;
- `tanh_mean_then_scale`: deterministic SAC actor mean;
- `ppo_deterministic_heads`: argmax for discrete heads and mean for continuous heads.

Critics, replay, exploration noise, and the Godot sensor implementation are not part
of a production policy artifact.
