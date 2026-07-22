# Python reference

The Python runtime discovers the contract declared by a Godot scene and connects it
to the selected RL backend. Scenario-specific trainers are deliberately avoided.

## Public commands

### `python/train.py`

Parses shared arguments, probes the action space when necessary, and lazily loads a
module from `python/algorithms`.

`--algorithm auto` selects DQN for discrete actions, DDPG for continuous actions, and
PPO for hybrid actions. Other algorithms are selected by name.

`--policy-path` warm-starts only the policy from a Metis bundle, `.keras` model, full
`.h5` model, or `.weights.h5` file. It does not restore optimizer, critic, replay, or
episode state.

`--dashboard` starts the optional local metrics server on port `8770`. Change the port
with `--dashboard-port`. The Flask dependencies are deliberately kept out of the base
runtime and can be installed from `requirements-dashboard.txt`.

### `python/run.py`

Loads `policy.keras` and `policy.json` by default. It can also load an explicit policy
file, legacy H5 weights, or a training checkpoint.

Lockstep execution is used for reproducible evaluation. Real-time execution is useful
for watching a policy at the scene's natural pace.

- `--continue-after-success` asks compatible scenes to keep simulation and inference
  active after a successful goal.
- `--no-reset` starts a new logical episode at the current pose after terminal events.
- `--no-initial-reset` preserves the scene's initial state before first inference.
- `--infinite` removes the episode-count limit.
- `--no-time-limit` removes the run-side step limit.

Collision and failure behavior still belongs to the scenario. A scene may stop a body
at a terminal state even when no physical reset is requested.

### `python/export.py`

Exports a Keras policy bundle to TensorFlow Lite, ONNX, or both. TFLite uses the main
TensorFlow installation. ONNX requires the optional packages in
`requirements-export.txt`.

### `python/recorder.py`

Starts one manually controlled Godot agent and writes an `.npz` dataset containing
observations, applied actions, rewards, next observations, `terminated`, `truncated`,
agent IDs, episodes, and steps.

## Internal packages

### `python/algorithms/`

- `dqn.py`: value learning for discrete actions;
- `ppo.py`: on-policy actor-critic for discrete, continuous, and hybrid spaces;
- `sac.py`: entropy-regularized continuous actor-critic;
- `ddpg.py` and `td3.py`: deterministic actor-critic entry points;
- `ddpg_bc.py` and `td3_bc.py`: online learning with behavior cloning;
- `ddpgfd.py`: prioritized demonstration replay;
- `common.py`: shared DDPG/TD3-family implementation.

Each backend owns its parser additions, sync and async loop, checkpoint state, and
logs, and exposes `main()` to the dispatcher.

### `python/core/`

- `models.py`: Keras model factories and compiled inference functions;
- `policy_artifact.py`: atomic `policy.keras` and manifest output;
- `replay_buffer.py`: uniform/prioritized replay, protected demonstrations, snapshots;
- `opponent_pool.py`: historical policy snapshots and opponent sampling;
- `training.py`: async collection, parallel stepping, TensorFlow setup, best-policy
  evaluation, metrics, and shared utilities.

Core modules must not know about concrete scenes such as Cars, Tanks, or Breakout.

### `python/envs/`

- `scenario.py`: Gymnasium wrapper around the TCP protocol;
- `process_manager.py`: Godot startup, readiness checks, logs, and shutdown.

This is the I/O boundary. Algorithm modules do not open sockets or construct Godot
commands directly.

### `python/tools/`

Contains random rollout and bridge benchmarks. These are diagnostic utilities, not
training backends.

### `python/dashboard/`

Contains the optional local HTTP and WebSocket server and its self-contained browser
client. `core.training.print_episode_metrics()` forwards one flat metric row per
episode to the server when `--dashboard` is enabled. The package is imported lazily,
so Flask is not required for training without the flag.

### `python/tests/`

Tests cover replay semantics, targets and truncation, action packing, asynchronous
collection, opponent pools, checkpoints, and dispatch. A new algorithm should add
tests for its learning and transition semantics, not only an import test.

## Dependency direction

```text
CLI -> algorithms -> core
 |         |          |
 +---------+--------> envs
```

`core` does not import concrete algorithms. `envs` does not import TensorFlow. Godot
does not depend on a Python learner class. Keeping this direction prevents circular
imports and allows collector workers to run without owning the learner.

## Algorithm matrix

| Algorithm | Actions | Policy type | Replay | Demonstrations |
|---|---|---|---|---|
| DQN | discrete | off-policy | uniform/prioritized | optional prefill |
| PPO | discrete/continuous/hybrid | on-policy | no | no |
| DDPG | continuous | off-policy | yes | optional prefill |
| DDPG+BC | continuous | off-policy | yes | required |
| DDPGfD | continuous | off-policy | prioritized | required |
| TD3 | continuous | off-policy | yes | optional prefill |
| TD3+BC | continuous | off-policy | yes | required |
| SAC | continuous | off-policy | yes | optional prefill |

Async and multi-agent support are backend responsibilities. They cannot be inferred
only from a model accepting batched tensors.

## SAC gradient clipping

SAC applies a global norm cap independently to critic 1, critic 2, and actor
gradients. The command-line default is:

```text
--grad-clip-norm 10.0
```

Set it to `0` to turn clipping off. `--grad-clip-adaptive` maintains an exponential
running norm for each of the three networks and clips at `--grad-clip-k` times that
network's estimate. When a positive hard norm is also configured, the adaptive value
can only make the threshold tighter; it cannot exceed the hard cap. A short warmup
uses the hard cap while the estimates settle.

The adaptive EMA is local to the learner process and is not restored from a
checkpoint. The entropy-temperature optimizer is not clipped. DQN, PPO, and the
DDPG/TD3 family do not currently expose these clipping flags.

## Episode metrics and dashboard sinks

Backends pass structured sections to `print_episode_metrics()`. The function preserves
the pretty or compact terminal output and also flattens the fields for registered
metric sinks. Sink failures are ignored so an optional monitor cannot terminate a
training run.

The bundled dashboard recognizes the shared keys `reward`, `progress`,
`critic_loss`, `actor_loss`, `env_steps_s`, `updates_s`, `alpha`, `finish`,
`collision`, and `stall`. Algorithms remain free to report additional fields; they
still appear in `/api/metrics` even when the current page has no chart for them.

See [Monitoring training](../guides/monitoring-training.md) for the launch command,
retention behavior, and dashboard limitations.

## Implementation conventions

- Shared arguments keep the same name across backends.
- Zero `max_steps` means no external step limit where the scene supports it.
- Every training checkpoint refreshes the portable Keras policy bundle.
- `run.py` can load a checkpoint without loading replay.
- A full off-policy resume restores replay when a matching snapshot is available.
- Multi-agent metrics count agents and transitions, not only environment steps.
- Episode output should go through `print_episode_metrics()` so optional metrics sinks
  receive the same values as the terminal log.
- Best-checkpoint evaluation uses one CPU thread by default to avoid starving the
  learner and collectors. Configure it with `--best-evaluation-cpu-threads`, or turn
  it off with `--no-best-checkpoint`.
- Public commands remain `train.py`, `run.py`, `recorder.py`, and `export.py`. Internal
  modules are not additional user-facing entry points.
