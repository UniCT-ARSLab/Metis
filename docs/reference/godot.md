# Godot reference

The Godot side of Metis declares agents and tasks with composable nodes configured in
the Inspector. Godot remains authoritative for physics, actions, observations,
rewards, and episode state.

## Recommended scene tree

```text
Scenario
|-- ScenarioController
|   |-- ScenarioRewardSystem
|   |-- ProgressProvider
|   `-- ScenarioEventSystem
|-- Agents
|   `-- AgentBody
|       `-- Agent
|           |-- ActionSpace
|           |-- ObservationSystem
|           `-- RewardSystem
`-- BridgeServer
```

Node names may differ. Exported `NodePath` properties must still point to the correct
objects.

## `Agent`

`Agent` is the RL-facing container attached to a physical body. It preserves
observation order, delegates action decoding to `ActionSpace`, and asks
`RewardSystem` for local reward terms.

Its parent owns concrete behavior: movement, animation, collision callbacks,
projectiles, and manual controls. Older imperative APIs such as `add_action()` and
`add_observation()` still work, but child components are easier to inspect and reuse.

## `ActionSpace`

Available components:

- `ContinuousAction`: component name, size, policy bounds, and optional exploration
  bounds;
- `DiscreteActionSet`: one categorical component containing `DiscreteAction` nodes;
- `DiscreteAction`: display name, target node, method, and method arguments.

Discrete actions can invoke their target methods automatically. Continuous values are
decoded by the body in `apply_action()` and assigned to its controls. A scene may
combine both forms to create a hybrid action space.

## `ObservationSystem`

An `ObservationSource` registers one or more numeric values. Supplied sources cover:

- body transform, velocity, and kinematics;
- normalized ray-cast distances and front clearance;
- target visibility and team-relative state;
- path-relative navigation;
- method calls through `MethodObservationSource`;
- direct property reads through `PropertyObservationSource`.

Method and property sources accept an optional `source_path` relative to the agent
body. An empty path reads the body itself.

The **Metis Inspector** plugin adds `Select...` menus for compatible methods and
properties. Selections are stored as ordinary `StringName` and `NodePath` values; the
runtime has no dependency on the editor plugin.

`PropertyObservationSource` also accepts sub-properties such as `velocity:x` and can
map numeric values from a source range to a normalized output range.

Observations should be finite, stable in size, and normalized where practical.
Resetting or teleporting a body must also reset and refresh its sources so a new
episode does not start with cached data from the previous physics frame.

## Local and scenario rewards

`RewardSystem` evaluates children derived from `RewardComponent`. Its context normally
contains:

- `agent`: the `Agent` node;
- `body`: the controlled body;
- `observations`: the current observation dictionary;
- `events`: accumulated local events;
- controller data such as step, progress, and terminal state.

Each component returns its weighted value and may implement
`reset_reward(context)`. Named values are included in diagnostics.

`ScenarioRewardSystem` evaluates `ScenarioRewardComponent` once per agent. Its context
adds `agent_id`, progress, scenario events as direct keys, and terminal/truncation
state. Optional `is_agent_stalled()` and `get_terminal_reason()` methods may finish one
agent without ending every other channel.

Use local rewards for behavior intrinsic to a body. Use scenario rewards for task
rules involving the shared world.

## Events

`ScenarioEventSystem` aggregates `ScenarioEventSource` children. Event sources keep
state per agent ID and return named values. Built-in sources cover:

- entering an `Area`;
- crossing an observation threshold;
- events triggered directly by scenario code.

An event records what happened. A reward component decides what that event is worth,
and an event source may independently declare a terminal reason.

## Progress

`ProgressProvider` exposes an ordered task metric. Included implementations are:

- `Path3DProgressProvider`;
- `MethodProgressProvider`, which calls a scene method.

Progress may mean track completion, bricks removed, target approach, object lift, or
another task phase. Avoid adding it to policy observations when the policy is intended
to remain independent of a particular map.

## `ScenarioController`

The controller handles:

- agent registration and replication;
- deterministic reset with environment and per-agent seeds;
- action application and physics-frame advancement;
- response assembly;
- per-agent terminal state and completed-agent caching;
- curriculum spawning through a progress provider;
- camera and UI shutdown in headless runs;
- single-agent manual recording mode.

Prefer controller hooks and child components to replacing `step()` or changing the
wire response format.

`physics_frames_per_step` sets the action repeat. `max_steps=0` disables the controller
step limit, so the scene must then provide reliable terminal or stall behavior.

## `BridgeServer`

The bridge accepts one TCP client, enables `TCP_NODELAY`, reads one JSON object per
line, and forwards `spec`, `configure`, `reset`, and `step` requests to the controller.

In lockstep mode the world advances only while serving a step. In real-time mode it
keeps processing with the latest action. The bridge does not know concrete agent
classes, reward values, or action names.

## Physical reset checklist

After teleporting a body or dynamic object:

1. Clear linear and angular velocity.
2. Clear current and previous control inputs.
3. Restore transforms for every dynamic object owned by the episode.
4. Force ray casts and sensors to refresh.
5. Reset observation sources, rewards, events, and progress.
6. Wait for extra physics frames only when the scene genuinely needs them.

An incomplete reset can return stale observations and contaminate replay with a
transition between two different episodes.

## Robot-arm collision handling

`URDFRobotArmAgentBody` performs separate environment and self-collision queries.
Parent-child links are ignored automatically. Additional mechanical overlaps can be
declared through `self_collision_ignored_link_pairs` or
`self_collision_ignored_link_sets`.

Support surfaces can belong to both `robot_obstacle` and
`robot_support_surface`. Only links listed in `support_contact_link_names` may touch
them. This lets a fixed base rest on the floor while keeping floor contact terminal
for the arm and tool.
