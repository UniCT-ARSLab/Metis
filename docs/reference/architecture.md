# Metis architecture

Metis separates simulation from learning through a small, explicit contract between
Godot and Python.

- Godot owns the world, physics, agents, observations, rewards, and terminal events.
- Python owns the Gymnasium interface, data collection, replay or rollout storage,
  native TensorFlow/Keras learners, optional learner adapters, checkpoints, and
  evaluation.
- A TCP bridge exchanges newline-delimited JSON messages and controls when the
  simulation advances.

This boundary lets a scene change without creating a new trainer and lets an
algorithm change without embedding TensorFlow in a Godot project.

## Runtime flow

```text
python/train.py
    |
    +-- selects a backend and algorithm
    +-- GodotProcessManager starts N processes
    +-- ScenarioGymEnv connects to each process
            |
            +-- BridgeServer
                    |
                    +-- ScenarioController
                            +-- Agent[]
                            +-- ScenarioEventSystem
                            +-- ScenarioRewardSystem
                            +-- ProgressProvider
```

Python sends `reset` at the start of an episode. Godot resets and randomizes the
scene, then returns the first observation. On every `step`, Python sends one action per
active agent. Godot applies those actions, advances `physics_frames_per_step` physics
ticks, and returns:

- the next observation;
- total reward and named reward terms;
- `terminated` for a natural task ending;
- `truncated` for an external limit;
- per-agent progress, events, and terminal reasons.

Learners must stop bootstrapping on `terminated`. A truncation is not a natural
terminal state and still has a meaningful next-state value.

## Scene contract

`BridgeServer.controller_path` must point to a `ScenarioController`. The controller
implements the bridge-facing operations:

- `get_spec()` or equivalent agent metadata;
- `configure(config)`;
- `reset_episode_with_request(request)`;
- `step(actions)`.

The supplied controller already implements this protocol. A scene should customize
composition, reset hooks, progress, and events instead of duplicating socket handling.

Every controlled body is registered in `controlled_agents` and normally owns an
`Agent` child. The body implements concrete behavior:

- `apply_action(action)`;
- `reset_all(original_transform, reset_rewards)`;
- optionally `is_terminal()`, `set_training_active()`, and `get_team_id()`.

The imperative `add_action()` and `add_observation()` APIs remain available for older
scenes. New scenes should prefer components under the `Agent` node so the contract is
visible in the Inspector.

## Action spaces

`ActionSpace` inspects its children and classifies the result:

- one `DiscreteActionSet`: `discrete`;
- one or more `ContinuousAction` nodes: `continuous`;
- mixed discrete and continuous components, or several discrete components: `hybrid`.

The Godot declaration is authoritative. Python reads component names, dimensions,
bounds, and ordering from the scenario specification.

## Observations

Each `ObservationSource` registers one or more values with `ObservationSystem`.
Scalars, booleans, `Vector2`, `Vector3`, and arrays are flattened by
`Agent.get_observation_vector()` in registration order.

That order and size must remain stable for the lifetime of a policy. Changing a
source, normalization, ordering, or semantic meaning creates a new model contract.

## Rewards, events, and progress

Rewards have two scopes:

- `RewardSystem` belongs to one agent and evaluates body-local state such as motion,
  controls, sensors, and effort;
- `ScenarioRewardSystem` evaluates task state such as goals, score, shared-world
  collisions, and progress.

Rewards remain per agent. Parameter sharing means shared network weights, not shared
episode returns.

`ScenarioEventSystem` gives scene events stable names. Reward and terminal components
can depend on those names without knowing which collision callback or area generated
them.

`ProgressProvider` supplies an ordered scalar used for shaping, diagnostics, and
curriculum. It is intentionally broader than path completion.

## Single-agent and multi-agent data

In single-agent mode, `step` accepts one action and exposes the first result channel.
In multi-agent mode, it accepts an action mapping and returns one channel per agent.

Agents with the same observation and action contract may use one shared policy. Their
transitions remain independent records in replay or rollout storage. Independent
policies require an explicit `agent_id -> policy_id` assignment and separate model
state; that general workflow is not implemented yet.

## Synchronous and asynchronous collection

The synchronous collector waits for a coordinated set of environment steps. It is
easier to reproduce and required by some self-play configurations.

The asynchronous collector lets each environment run independently. Workers publish
transitions to a bounded queue and periodically receive a new policy snapshot. Each
transition retains its worker, episode, and policy-version metadata, so data from
different workers is not confused. The trade-off is policy lag: a worker may finish
an episode using a slightly older policy.

PPO collects complete rollout generations and only trains on data produced by the
same frozen policy version. Off-policy algorithms can mix older data through replay by
design.

The optional Stable-Baselines3 adapter uses a synchronized `VecEnv`: each Godot
process still owns an independent scene and its socket wait can run concurrently, but
the learner receives a vector step only after every active lane has replied. Terminal
observations are preserved before the completed lane is reset. The adapter currently
exists for backend comparisons and does not implement asynchronous collection.

For hybrid PPO, a policy lane exposes a latent continuous `Box`. Continuous action
components retain their bounds, while discrete components are represented by logits
and decoded with `argmax`. This differs from the native PPO distribution and must be
reported as part of a comparison.

For multi-agent parameter sharing, one lane represents one compatible agent. Lanes
belonging to the same Godot process are stepped and reset as a group. Coordinated
terminal events work directly. Partial termination is rejected by default because an
individual lane cannot reset its shared physical world; the optional `reset-all` mode
truncates the other lanes before resetting that world.

## Metrics and live monitoring

Algorithm backends build named metric sections once an episode completes.
`print_episode_metrics()` owns both terminal formatting and delivery to optional
metric sinks, which keeps monitoring independent of collection mode and algorithm
control flow.

With `--dashboard`, a sink stores recent rows in memory and publishes them through a
local HTTP API and WebSocket. Browser batching changes only refresh frequency. It does
not change update scheduling or synchronize collectors. Metric recording is kept
small, WebSocket sends happen outside the shared history lock, sink exceptions are
isolated from training, and the dashboard thread exits with the Python process.

The dashboard is an observer, not part of the checkpoint contract. Its history is not
restored on resume and should not be used as the only experiment record.

Training health is a separate shared service. It consumes flattened numerical
telemetry and the results of isolated best-checkpoint evaluations, persists an atomic
state snapshot plus an event log, and publishes transitions to the dashboard when it
is present. Alerting does not mutate the learner.

With `--auto-recovery`, a confirmed collapse invokes the trainer's registered
checkpoint recovery handler on the learner thread. Full model, target-network, and
optimizer state comes from the validated best checkpoint. Optimizer roles receive
separate conservative learning-rate factors, async policy snapshots are refreshed,
queued experience carries a policy version, and a frozen verification is scheduled
immediately.

The first off-policy recovery keeps replay. A later hard attempt clears online replay,
preserves protected DDPGfD demonstrations, synchronizes target networks, removes
queued transitions from the previous policy era, and stages critic-only warmup before
actor updates resume. DQN uses the shared checkpoint and queue safeguards but has no
actor warmup. PPO marks old rollout generations stale and freezes policy optimization
during verification because replay reuse would violate its on-policy contract.

Attempt limits are scoped to one collapse cycle. A new best result or consecutive
healthy frozen evaluations closes that cycle and restores a fresh budget while the
lifetime recovery count remains available to monitoring.

## Checkpoints, replay, and policy bundles

Native TensorFlow checkpoints store the model variables, target networks, optimizers, and
counters registered by an algorithm. Off-policy replay is stored separately as
`replay-<episode>.npz`. A full resume restores both.

`policy.keras` and `policy.json` are deployment artifacts. They are sufficient for
`run.py` and export, but do not contain optimizer or replay state.

Best-checkpoint evaluation runs from a frozen checkpoint in a separate process. The
`best/` directory therefore contains evaluated candidates rather than the latest
chronological state.

Stable-Baselines3 uses its own PyTorch `.zip` model format and `.pkl` replay
snapshots. It cannot be resumed or exported as if it were a native Keras bundle.

## Runtime files

Godot process logs are written under `.runtime/godot_logs/`. This directory is ignored
by Git and can be removed when no Metis process is running. Checkpoints, exported
policies, and demonstration datasets are persistent user artifacts and should live in
their configured output directories.
