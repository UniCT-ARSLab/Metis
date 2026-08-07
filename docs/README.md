# Metis documentation

You do not need to learn the whole framework before building a first agent. Pick the
guide closest to your task, then return to the reference when you need to understand a
node or runtime contract in more detail.

## Start here

1. Read the [architecture overview](reference/architecture.md) to understand the
   boundary between Godot, the bridge, and Python.
2. Read [Distribution and installation](reference/distribution.md) if you are using
   the Asset Library package or preparing a release.
3. Follow [Build a new agent and scenario](tutorials/new-agent-and-scenario.md).
4. Run `python/tools/random_rollout.py` before starting a long training job.
5. Use [Godot reference](reference/godot.md) and
   [Python reference](reference/python.md) while debugging.

## Scenario tutorials

| Goal | Tutorial |
|---|---|
| Learn the basic discrete workflow | [Breakout from scratch](tutorials/breakout-from-scratch.md) |
| Train two identical competitors | [Pong multi-agent](tutorials/pong-multi-agent.md) |
| Combine continuous motion and a discrete command | [Tanks 2v2](tutorials/tanks-hybrid-multi-agent.md) |
| Compare path-aware and sensor-only driving | [Autonomous driving](tutorials/autonomous-driving-path-vs-sensors.md) |
| Build a continuous 3D team task | [3D soccer](tutorials/soccer-continuous-multi-agent.md) |
| Train a URDF robot arm around obstacles | [Robot-arm reaching and grasping](tutorials/robot-arm-reaching-sim-to-real.md) |

Each tutorial includes the Godot scene contract, observations, actions, rewards,
terminal conditions, validation steps, training commands, and an inference command.
Training commands use the native Metis backend unless a section is explicitly labelled
as an optional SB3 comparison. Those SB3 examples always use synchronous collection
and save native `.zip` models; the tutorial's `python/run.py` command applies to the
Metis/Keras artifact.

## Practical guides

- [Manual demonstrations](guides/manual-demonstrations.md) covers recording, appending,
  replay prefill, and behavior cloning.
- [Exporting policies](guides/exporting-policies.md) covers Keras bundles, legacy H5
  files, TensorFlow Lite, ONNX, and deployment metadata.
- [Monitoring training](guides/monitoring-training.md) covers the optional live
  dashboard, metric fields, and SAC gradient clipping.
- [Benchmarking training backends](guides/benchmarking-backends.md) compares native
  Metis learners with the optional Stable-Baselines3 backend.
- [Adding an RL algorithm](guides/adding-an-rl-algorithm.md) explains backend
  registration, collectors, checkpoints, and required tests.
- [Extending the Godot side](guides/extending-godot.md) shows how to add observation,
  reward, event, progress, and action components.

## Algorithms

- [Algorithm guide](algorithms/README.md) compares every native learner and gives a
  practical selection order.
- [Shared training options](algorithms/common-options.md) documents collection,
  checkpoints, demonstrations, curriculum, recovery, rendering, and runtime flags.
- Individual guides: [DQN](algorithms/dqn.md), [PPO](algorithms/ppo.md),
  [DDPG](algorithms/ddpg.md), [DDPG+BC](algorithms/ddpg-bc.md),
  [DDPGfD](algorithms/ddpgfd.md), [TD3](algorithms/td3.md),
  [TD3+BC](algorithms/td3-bc.md), and [SAC](algorithms/sac.md).

## Reference

- [Architecture](reference/architecture.md): ownership, protocol, transitions,
  multi-agent semantics, and async collection.
- [Distribution and installation](reference/distribution.md): add-on layout, Python
  runtime setup, CLI, and release builds.
- [Godot](reference/godot.md): scene tree, component contracts, and physical reset.
- [Python](reference/python.md): public commands, internal packages, and algorithm
  support.

## Supported workflows

Metis currently supports:

- discrete, continuous, multi-discrete, and hybrid action spaces;
- native DQN, PPO, DDPG, DDPG+BC, DDPGfD, TD3, TD3+BC, and SAC;
- limited Stable-Baselines3 compatibility for controlled backend comparisons,
  including hybrid PPO encoding and coordinated multi-agent parameter sharing;
- one or more environments with synchronous or asynchronous collection;
- single-agent tasks and multi-agent parameter sharing;
- simultaneous self-play and historical opponent pools;
- Inspector-configured observations and rewards;
- scenario events, progress providers, curriculum, and seeded resets;
- manual demonstrations, full checkpoints, replay snapshots, and best-policy tracking;
- persistent training-health alerts and guarded opt-in recovery for native trainers;
- an optional local dashboard for metrics, health state, and recovery events;
- real-time or lockstep inference;
- Keras, TensorFlow Lite, and optional ONNX policy artifacts;
- Linux CPU/CUDA and Apple Silicon with TensorFlow Metal.

Visible training environments can use `project`, `light-gpu`, `cpu`, or `gpu` render
modes. `--render-env-count 1` is useful when several environments are collecting data
but only one preview is needed. See [Collection and rendering](../README.md#collection-and-rendering).

## Documentation conventions

An installed Metis runtime exposes one CLI:

```text
metis train
metis run
metis record
metis export
metis doctor
```

Source checkouts keep the equivalent `python/train.py`, `python/run.py`,
`python/recorder.py`, and `python/export.py` entry points. Both forms dispatch to the
same implementation; tutorials use the source form because they are tested from this
repository.

Modules under `python/algorithms`, `python/core`, and `python/envs` are implementation
details or extension points. Tutorials configure existing Godot components first and
add task-specific code only where the scene genuinely owns the behavior.

When documentation and runtime behavior disagree, inspect the scene and its tests.
Observation order, action components, rewards, terminal rules, and physics settings
are one training contract; update the relevant guide whenever that contract changes.
