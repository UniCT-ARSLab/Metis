# Python reference

The Python runtime discovers the contract declared by a Godot scene and connects it
to the selected RL backend. Scenario-specific trainers are deliberately avoided.

## Public commands

### `python/train.py`

Parses shared arguments, probes the action space when necessary, and lazily loads a
learner. `--backend metis` is the default and dispatches to `python/algorithms`.
`--backend sb3` dispatches to the optional Stable-Baselines3 adapter in
`python/backends`. Metis remains the production/default backend. SB3 compatibility is
provided primarily for repeatable comparisons against its PyTorch implementations,
not as a claim that both backends expose every Metis feature.

`--algorithm auto` selects DQN for discrete actions, DDPG for continuous actions, and
PPO for hybrid actions. The selected backend then validates that contract.

`--total-timesteps N` adds a transition budget to the native trainers as well as the
SB3 adapter. With several agents, every agent transition counts. A collector may
finish its current synchronized batch after crossing the exact number.

`--policy-path` warm-starts only the policy from a Metis bundle, `.keras` model, full
`.h5` model, or `.weights.h5` file. It does not restore optimizer, critic, replay, or
episode state.

`--dashboard` starts the optional local metrics server on port `8770`. Change the port
with `--dashboard-port`. The Flask dependencies are deliberately kept out of the base
runtime and can be installed from `requirements-dashboard.txt`.

`--metrics-jsonl PATH` records the same flattened episode rows without starting the
dashboard. The backend benchmark uses this persistent stream to compare learning
curves on a transition axis.

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

### `python/backends/`

`sb3.py` adapts the Metis scene contract to Stable-Baselines3:

- `sb3_actions.py` maps one agent action to SB3's supported spaces;
- `sb3_vec_env.py` handles independent single-agent Godot processes;
- `sb3_multi_vec_env.py` groups shared-policy agents from the same Godot world;
- `sb3_state.py` handles checkpoint discovery and companion state.

The adapter supports DQN on discrete spaces; PPO on discrete or continuous spaces;
and DDPG, TD3, or SAC on continuous spaces. PPO also accepts a hybrid Metis contract
through a latent `Box` encoding. Every continuous component keeps its declared values;
each discrete component contributes one bounded latent logit per choice and is decoded
with `argmax`. Consequently, SB3 optimizes a Gaussian latent policy rather than Metis
PPO's exact mixed categorical/Gaussian distribution.

Multi-agent mode uses parameter sharing: each compatible agent is an SB3 vector lane,
while all actions belonging to one Godot process are grouped into one bridge step.
Godot worlds cannot reset one vector lane independently. The default behavior
therefore raises if only some agents terminate. With
`--sb3-multi-agent-partial-done reset-all`, still-active agents are marked truncated
and the complete world resets.

Collection is always synchronized from the learner's perspective, while socket waits
for separate Godot processes may overlap in worker threads. `async` is rejected. That
restriction is intentional: moving collection outside SB3's `learn()` loop would be a
new trainer rather than a comparison against SB3's normal implementation.

Install this optional backend from `requirements-sb3.txt`. It stores SB3/PyTorch
models as `.zip` files and off-policy replay as `.pkl`; those files are independent
of Metis Keras policy bundles.

### `python/core/`

- `models.py`: Keras model factories and compiled inference functions;
- `policy_artifact.py`: atomic `policy.keras` and manifest output;
- `replay_buffer.py`: uniform/prioritized replay, protected demonstrations, snapshots;
- `opponent_pool.py`: historical policy snapshots and opponent sampling;
- `training.py`: async collection, parallel stepping, TensorFlow setup, best-policy
  evaluation, health/recovery wiring, transition budgets, JSONL metrics, and shared
  utilities;
- `training_health.py`: health-state transitions, persistent alerts, collapse
  confirmation, and TensorFlow checkpoint recovery;
- `evaluation.py`: TensorFlow-free episode and evaluation summaries shared by native
  inference and the SB3 adapter.

Core modules must not know about concrete scenes such as Cars, Tanks, or Breakout.

### `python/envs/`

- `scenario.py`: Gymnasium wrapper around the TCP protocol;
- `process_manager.py`: Godot startup, readiness checks, logs, and shutdown.

This is the I/O boundary. Algorithm modules do not open sockets or construct Godot
commands directly.

### `python/tools/`

Contains random rollout and bridge diagnostics plus
`benchmark_backends.py`, which runs Metis and SB3 sequentially with common seeds,
transition budgets, and deterministic evaluation.

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
CLI -> algorithms (Metis) -> core
 |          |
 |          +---------------> envs
 |
 +--> backends/sb3 ----------> envs
```

`core` does not import concrete algorithms. `envs` imports neither TensorFlow nor
PyTorch. Godot does not depend on a Python learner class. Keeping this direction
prevents circular imports and allows collector workers to run without owning the
learner.

## Algorithm matrix

| Algorithm | Actions | Metis | SB3 | Demonstrations in Metis |
|---|---|---|---|---|
| DQN | discrete | yes | yes | optional prefill |
| PPO | discrete/continuous/hybrid | yes | discrete/continuous/hybrid encoding | no |
| DDPG | continuous | yes | yes | optional prefill |
| DDPG+BC | continuous | yes | no | required |
| DDPGfD | continuous | yes | no | required |
| TD3 | continuous | yes | yes | optional prefill |
| TD3+BC | continuous | yes | no | required |
| SAC | continuous | yes | yes | optional prefill |

Async and multi-agent support are backend responsibilities. They cannot be inferred
only from a model accepting batched tensors.

## SB3 compatibility matrix

| Capability | Metis backend | SB3 adapter |
|---|---|---|
| Synchronous single-agent | yes | yes |
| Asynchronous collectors | yes | no |
| Hybrid PPO | native mixed heads | latent Box adapter |
| Multi-agent parameter sharing | general shared contract | coordinated group reset |
| Simultaneous self-play | yes | shared current policy only |
| Historical opponent sampling | yes | no |
| Demonstration prefill and BC variants | yes | no |
| Keras policy/export pipeline | yes | no; SB3 `.zip` |

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

`TrainingHealthMonitor` adds `health_state`, `training_phase`, lifetime recovery
count, recovery cycle, and cycle attempt to metric rows. It emits a separate event
stream at `/api/health`; current state and event history are persisted beside the
training checkpoints. Native trainers connect the same monitor to
`BestCheckpointTracker`, so health decisions use isolated frozen evaluations rather
than raw episode noise.

The recovery contract is shared, while stabilization remains algorithm-aware:

- DQN restores the full checkpoint, synchronizes the target Q network, rejects stale
  async transitions, and waits for verification before learning again.
- PPO rejects rollout generations from the previous policy version and freezes
  optimization during verification. It has no replay-clearing phase.
- SAC and the DDPG/TD3 family can escalate from replay-preserving soft recovery to a
  hard recovery that removes online replay, retains protected DDPGfD demonstrations,
  synchronizes targets, and performs critic-only warmup.

Recovery limits belong to the current collapse cycle. Validated healthy evaluations
close the cycle without deleting lifetime recovery telemetry.

See [Monitoring training](../guides/monitoring-training.md) for the launch command,
retention behavior, and dashboard limitations.

## Implementation conventions

- Shared arguments keep the same name across backends.
- `--backend metis` remains the default; optional adapters must fail clearly on
  unsupported scene contracts.
- Zero `max_steps` means no external step limit where the scene supports it.
- Every native training checkpoint refreshes the portable Keras policy bundle.
- `run.py` can load a checkpoint without loading replay.
- A full off-policy resume restores replay when a matching snapshot is available.
- Multi-agent metrics count agents and transitions, not only environment steps.
- Episode output should go through `print_episode_metrics()` so optional metrics sinks
  receive the same values as the terminal log.
- Best-checkpoint evaluation uses one CPU thread by default to avoid starving the
  learner and collectors. Configure it with `--best-evaluation-cpu-threads`, or turn
  it off with `--no-best-checkpoint`.
- Automatic recovery requires best-checkpoint evaluation. It restores complete
  validated TensorFlow state, scales optimizer roles separately, republishes async
  policy snapshots, verifies the restore immediately, and escalates off-policy replay
  handling only when a soft attempt fails.
- Public commands remain `train.py`, `run.py`, `recorder.py`, and `export.py`. Internal
  modules are not additional user-facing entry points.
