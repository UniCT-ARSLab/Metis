# Metis

<p align="center">
  <img src="docs/logo.svg" alt="Metis logo" width="640">
</p>

**Modular Environment for Training Intelligent Systems**

Metis connects Godot simulations to reinforcement-learning code written with
Gymnasium and TensorFlow/Keras. Agents, sensors, rewards, episode rules, and world
physics live in Godot. Python reads that contract and handles data collection,
optimization, checkpoints, evaluation, and inference.

The aim is straightforward: changing the task should usually mean building a new
Godot scene, not writing another Python training program.

Metis is under active development, but it already supports discrete, continuous, and
hybrid control; multi-agent environments; asynchronous collection; demonstrations;
self-play; and portable policy exports.

Start with the [documentation index](docs/README.md). If you are building a task for
the first time, read [Build a new agent and scenario](docs/tutorials/new-agent-and-scenario.md).

## What is included

| Area | Current support |
|---|---|
| Simulation | Godot 4, 2D and 3D scenes, physics, seeded resets |
| Environment API | Generic Gymnasium wrapper and newline-delimited JSON over TCP |
| Action spaces | Discrete, continuous, multi-discrete, and hybrid |
| Observations | Methods, properties, body state, ray casts, targets, teams, and paths |
| Rewards | Per-agent components and scenario-level components |
| Algorithms | DQN, PPO, DDPG, DDPG+BC, DDPGfD, TD3, TD3+BC, and SAC |
| Collection | One or more Godot processes, synchronous or asynchronous |
| Multi-agent | Separate transitions with parameter sharing |
| Competition | Simultaneous self-play and historical opponent pools |
| Demonstrations | Manual recording, replay prefill, and behavior cloning |
| Persistence | Full checkpoints, replay snapshots, best-policy tracking, Keras bundles |
| Inference | Reproducible lockstep or real-time execution |
| Export | Keras, TensorFlow Lite, and optional ONNX |
| Platforms | Linux CPU/CUDA and Apple Silicon with TensorFlow Metal |

Metis ships its own TensorFlow/Keras trainers. Gymnasium defines the environment
interface; Stable-Baselines3 is not a runtime dependency.

## How the pieces fit

Godot owns the simulation:

- bodies, collision shapes, physics, and game rules;
- sensors and observations;
- action-space declarations;
- rewards, progress, events, and terminal conditions;
- reset randomization and task curriculum.

Python owns learning and orchestration:

- starting and stopping Godot processes;
- exposing each process as a Gymnasium environment;
- collecting replay or rollout data;
- updating Keras models;
- saving, evaluating, and running policies.

`BridgeServer` sits between them. Godot acts as the server because it owns the
authoritative world state. Python connects as a client and decides when the world
should reset or advance.

```text
python/train.py
    |
    +-- GodotProcessManager -> one or more Godot processes
    +-- ScenarioGymEnv      -> Gymnasium interface
    +-- algorithms/*        -> Keras learner
                                |
Godot                           |
    BridgeServer <--------------+
        ScenarioController
            Agent[]
            ScenarioEventSystem
            ScenarioRewardSystem
            ProgressProvider
```

The protocol exposes four operations: `spec`, `configure`, `reset`, and `step`.
`TCP_NODELAY` is enabled to avoid adding latency to the many small messages exchanged
during training.

## Installation

The project is tested with Godot 4.6.2. Nearby Godot 4 releases may work, but they are
not part of the regular test setup.

Create the virtual environment from the repository root:

```bash
python3 -m venv python/.venv
python/.venv/bin/python -m pip install --upgrade pip
python/.venv/bin/python -m pip install -r python/requirements.txt
```

For NVIDIA CUDA on Linux:

```bash
python/.venv/bin/python -m pip install -r python/requirements-linux-cuda.txt
```

For Apple Silicon:

```bash
xcode-select --install
python/.venv/bin/python -m pip install -r python/requirements-macos-metal.txt
```

The macOS requirements intentionally pin `tensorflow==2.18.1` and
`tensorflow-metal==1.2.0`. That Metal plugin does not match the ABI of newer
TensorFlow releases.

Set `GODOT_BIN` if Godot is not on `PATH`:

```bash
export GODOT_BIN=/path/to/Godot
```

## A first run

Before training, use a random rollout to check the scene contract and reset behavior:

```bash
python/.venv/bin/python python/tools/random_rollout.py \
  --godot-project godot \
  --godot-scene res://scenarios/pong/pong_scenario.tscn \
  --multi-agent \
  --steps 300 \
  --print-reward-terms \
  --no-headless
```

This example trains Breakout with DQN:

```bash
python/.venv/bin/python python/train.py \
  --algorithm dqn \
  --godot-project godot \
  --godot-scene res://scenarios/breakout/breakout_scenario.tscn \
  --num-envs 4 \
  --num-episodes 2000 \
  --max-steps-per-episode 0 \
  --checkpoint-dir checkpoints/breakout_dqn_v1 \
  --headless
```

Checkpoints also contain a portable `policy.keras` and a `policy.json` manifest. Run
the resulting policy with:

```bash
python/.venv/bin/python python/run.py \
  --checkpoint-dir checkpoints/breakout_dqn_v1 \
  --godot-project godot \
  --godot-scene res://scenarios/breakout/breakout_scenario.tscn \
  --episodes 20 \
  --no-headless
```

`run.py` reads the algorithm and the observation/action contract from `policy.json`.

## Building an agent in Godot

A typical agent scene looks like this:

```text
AgentBody
`-- Agent
    |-- ActionSpace
    |-- ObservationSystem
    `-- RewardSystem
```

`AgentBody` remains an ordinary Godot body and contains the task-specific mechanics:
movement, collisions, animation, projectiles, and manual controls. The child `Agent`
describes the RL interface.

### Actions

`ActionSpace` accepts:

- `DiscreteActionSet`, containing one or more `DiscreteAction` nodes;
- `ContinuousAction`, with a name, size, bounds, and optional exploration bounds;
- both at once for a hybrid agent.

A tank can therefore expose continuous throttle and steering together with a discrete
fire command. Python discovers those components from the scene. It has no built-in
knowledge of names such as `accelerate`, `steer`, or `shoot`.

### Observations

`ObservationSystem` keeps observation order and size stable. The supplied sources can
read methods and properties, body kinematics, ray casts, visible targets, team state,
and path-relative data.

The optional **Metis Inspector** editor plugin adds pickers for compatible methods and
properties. The selected values are stored as regular Godot scene properties, so the
runtime does not depend on the editor plugin.

Changing an observation's order, size, normalization, or meaning changes the model
contract. Existing policies are normally incompatible after such a change.

### Rewards, events, and progress

Put body-local terms under `Agent/RewardSystem`: motion cost, control smoothness,
sensor clearance, or a small time penalty. Put task rules under
`ScenarioRewardSystem`: goals, score, shared-world collisions, wins, and progress.

Each agent receives its own reward. A multi-agent scene does not silently sum every
agent's score.

`ScenarioEventSystem` separates facts from values. The scene can emit `ball_hit`; a
reward component decides whether that event is worth `0.02`, `1.0`, or nothing.

`ProgressProvider` is simply an ordered task metric. It may represent distance along a
track, bricks destroyed, an object's lift height, remaining health, or a phase of a
larger task. It does not have to use a `Path3D`.

See the [Godot reference](docs/reference/godot.md) for the complete node contract.

## Algorithms

`python/train.py` is the public training entry point.

| Algorithm | Action space | Good starting point for |
|---|---|---|
| `dqn` | discrete | A small set of categorical actions |
| `ppo` | discrete, continuous, hybrid | On-policy rollouts and composed spaces |
| `ddpg` | continuous | A minimal deterministic actor-critic baseline |
| `sac` | continuous | Stochastic exploration and robust continuous control |
| `td3` | continuous | Deterministic control with twin critics |
| `ddpg_bc` | continuous + demos | DDPG regularized by behavior cloning |
| `ddpgfd` | continuous + demos | Protected demonstration replay and pretraining |
| `td3_bc` | continuous + demos | TD3 with an adaptive behavior-cloning loss |

With `--algorithm auto`, Metis selects DQN for discrete spaces, DDPG for continuous
spaces, and PPO for hybrid spaces. Select other algorithms explicitly; action shape
alone is not enough to choose the best learner for a task.

## Episodes and decision frequency

`--max-steps-per-episode` is sent to Godot as well as enforced by Python. Set it to
zero when the scenario has reliable terminal or stall conditions and should have no
external step limit:

```text
--max-steps-per-episode 0
```

`--physics-frames-per-step N` holds one action for `N` physics ticks. At 60 Hz,
`N=4` gives the policy a 15 Hz decision rate. Step penalties, timeouts, and stall
windows count decisions rather than raw physics frames, so changing `N` also changes
their real-world duration.

## Collection and rendering

`--num-envs` starts independent Godot processes. The default asynchronous collector
lets each process advance at its own pace while the learner consumes transitions from
a queue:

```text
--num-envs 4 --collector-mode async
```

DQN and the continuous off-policy algorithms use replay buffers and lightweight CPU
policy copies in collector workers. PPO collects asynchronously but only updates from
a rollout produced by one frozen policy generation.

There is one learner. On a single GPU, multiple simulators feeding one model are
usually more useful than several learners competing for the same device. Small neural
networks may still show low GPU utilization because physics, sockets, and batch
assembly dominate the wall-clock time.

Use synchronous collection when reproducibility matters more than throughput:

```text
--collector-mode sync --parallel-env-steps
```

Rendering is independent of collection mode:

```text
--headless
--num-envs 4 --no-headless
--num-envs 4 --no-headless --render-env-count 1
```

Visible instances support these render modes:

| Mode | Behavior |
|---|---|
| `light-gpu` | OpenGL Compatibility; training default, with a Linux PRIME fallback |
| `project` | Uses the renderer configured by the Godot project |
| `cpu` | Mesa software rendering, mainly useful on Linux |
| `gpu` | Vulkan Forward+, mostly for visual inspection |

For example:

```text
--num-envs 4 --collector-mode async --no-headless \
  --render-env-count 1 --render-mode light-gpu
```

`--render-mode` only affects visible processes. If OpenGL falls back to `llvmpipe` on
Linux, `light-gpu` tries the discrete GPU reported by `switcherooctl`, unless PRIME or
DRI variables were already set explicitly.

## Multi-agent and self-play

With `--multi-agent`, each active agent contributes its own transition. Compatible
agents share model parameters while retaining separate observations, rewards,
terminal states, and diagnostics. Adding agents increases experience collection; it
does not create one model per agent or select a winner among them.

Team scenarios can use a historical opponent pool:

```text
--multi-agent \
--opponent-pool \
--opponent-snapshot-every 100 \
--opponent-pool-size 10 \
--opponent-current-probability 0.2
```

One team uses the current policy and the other uses a frozen snapshot. Opponent
transitions are excluded from the learner batch. DQN supports the opponent pool with
asynchronous collection. PPO, SAC, and the DDPG/TD3 family currently require a
synchronous collector when the pool is active.

Parameter sharing is not general multi-policy training. Independent policies with
separate networks and optimizers still require an explicit trainer extension.

## Curriculum and demonstrations

Python sends `training_episode`, an environment seed, and per-agent seeds to Godot.
Scenes can use them to expand spawn ranges, increase speed, randomize layouts, or
adjust tolerances. Curriculum state therefore follows the checkpoint episode.

`ScenarioController` also supports agent replication, randomized transforms,
progress-based spawning, and disabling cameras or UI in headless runs. Replication is
disabled while recording so that one agent remains under manual control.

Record demonstrations with `python/recorder.py`:

```bash
python/.venv/bin/python python/recorder.py \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --output demos/cars_demo.npz \
  --agent-id Car \
  --episodes 10 \
  --max-steps 800 \
  --no-headless
```

The dataset contains observations, applied actions, rewards, next observations,
terminal flags, agent IDs, episodes, and steps. See
[Manual demonstrations](docs/guides/manual-demonstrations.md) for prefill,
pretraining, and behavior-cloning workflows.

## Checkpoints and policies

Metis writes two kinds of artifacts:

- `ckpt-*` and `replay-*.npz` resume training;
- `policy.keras` and `policy.json` run or export a policy.

`--resume` selects the latest checkpoint in a directory. `--resume-checkpoint` selects
an exact checkpoint. `--policy-path` only warm-starts the policy network; optimizers,
critics, replay data, and episode counters start fresh. It accepts Metis bundles,
`.keras`, full `.h5` models, and `.weights.h5` files.

Trainers can evaluate frozen checkpoints and keep the best candidate under
`CHECKPOINT_DIR/best/`. Automatic scoring prefers success rate and uses mean reward as
a tie-breaker or as the primary metric when a scene exposes no success signal.

On `Ctrl+C`, the trainer saves the latest consistent state and closes its Godot
processes and sockets. Wait for the confirmation message before closing the terminal.

Run a policy file directly with:

```bash
python/.venv/bin/python python/run.py \
  --policy-path exports/my_policy/policy.keras \
  --godot-project godot \
  --godot-scene res://scenarios/my_scenario.tscn \
  --infinite \
  --no-time-limit \
  --no-headless
```

Export TensorFlow Lite and ONNX models with:

```bash
python/.venv/bin/python python/export.py \
  --policy checkpoints/my_run \
  --format all
```

TFLite uses TensorFlow. ONNX requires `python/requirements-export.txt`. The manifest
records observation order, action components, and decoding metadata. A deployment
outside Metis must still reproduce the sensor values and preprocessing performed in
Godot.

## Tutorials

- [Breakout from scratch](docs/tutorials/breakout-from-scratch.md): a discrete
  single-agent task with sparse events and optional shaping.
- [Pong multi-agent](docs/tutorials/pong-multi-agent.md): two identical agents,
  parameter sharing, self-play, and an opponent pool.
- [Tanks 2v2](docs/tutorials/tanks-hybrid-multi-agent.md): local sensors, teams, and a
  hybrid action space.
- [Autonomous driving](docs/tutorials/autonomous-driving-path-vs-sensors.md): compare
  path-aware control with a sensor-only policy.
- [3D soccer](docs/tutorials/soccer-continuous-multi-agent.md): continuous arcade
  movement around a physical ball.
- [Robot-arm reaching and grasping](docs/tutorials/robot-arm-reaching-sim-to-real.md):
  URDF import, joint control, IK, collision checks, demonstrations, and sim-to-real.

## Current boundaries

- Agents sharing one policy must expose the same observation and action contract.
- General independent multi-policy training is not implemented yet.
- Asynchronous historical opponent sampling is currently limited to DQN.
- A Keras model stores the mapping from numeric observations to actions. It does not
  contain the Godot scene, sensors, or scene-side preprocessing.
- The trainers are Metis implementations built with TensorFlow/Keras. They are not
  wrappers around Stable-Baselines3, and SB3 PyTorch checkpoints cannot be loaded
  directly.

These are extension points rather than hidden assumptions. See
[Adding an RL algorithm](docs/guides/adding-an-rl-algorithm.md) and
[Extending the Godot side](docs/guides/extending-godot.md).

## Repository layout

```text
godot/
  agents/       reusable agent scenes
  scenarios/    environments and task rules
  scripts/      bridge and Metis components
  addons/       editor plugins and third-party Godot add-ons

python/
  train.py      training entry point
  run.py        inference and evaluation
  recorder.py   manual demonstrations
  export.py     TFLite and ONNX export
  algorithms/   RL backends
  core/         models, replay, checkpoints, async collection, opponent pool
  envs/         Gymnasium wrapper and Godot process manager
  tools/        diagnostics and benchmarks
  tests/        Python test suite

docs/
  tutorials/    complete task walkthroughs
  guides/       reusable procedures and extension points
  reference/    runtime contracts and architecture
```

Run the Python tests with:

```bash
cd python
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

The design rule behind the project is simple: Godot describes the problem, Python
learns to solve it, and the boundary should remain clean enough that a new task does
not require another training stack.
