# Godot reference

The Godot side of Metis declares agents and tasks with composable nodes configured in the Inspector. Godot remains authoritative for physics, actions, observations, rewards, and episode state.

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

Node names may differ. Exported `NodePath` properties must still point to the correct objects.

## `Agent`

`Agent` is the RL-facing container attached to a physical body. It preserves observation order, delegates action decoding to `ActionSpace`, and asks `RewardSystem` for local reward terms.

Its parent owns concrete behavior: movement, animation, collision callbacks, projectiles, and manual controls. Older imperative APIs such as `add_action()` and `add_observation()` still work, but child components are easier to inspect and reuse.

## `ActionSpace`

Available components:

- `ContinuousAction`: component name, size, policy bounds, and optional exploration bounds;
- `DiscreteActionSet`: one categorical component containing `DiscreteAction` nodes;
- `DiscreteAction`: display name, target node, method, and method arguments.

Discrete actions can invoke their target methods automatically. Continuous values are decoded by the body in `apply_action()` and assigned to its controls. A scene may combine both forms to create a hybrid action space.

## `ObservationSystem`

An `ObservationSource` registers one or more numeric values. Supplied sources cover:

- body transform, velocity, and kinematics;
- normalized ray-cast distances and front clearance;
- target visibility and team-relative state;
- path-relative navigation;
- method calls through `MethodObservationSource`;
- direct property reads through `PropertyObservationSource`.

Method and property sources accept an optional `source_path` relative to the agent body. An empty path reads the body itself.

The **Metis** add-on adds `Select...` menus for compatible methods and properties. Selections are stored as ordinary `StringName` and `NodePath` values; exported scenes do not depend on the Inspector code.

`PropertyObservationSource` also accepts sub-properties such as `velocity:x` and can map numeric values from a source range to a normalized output range.

Observations should be finite, stable in size, and normalized where practical. Resetting or teleporting a body must also reset and refresh its sources so a new episode does not start with cached data from the previous physics frame.

## Local and scenario rewards

`RewardSystem` evaluates children derived from `RewardComponent`. Its context normally contains:

- `agent`: the `Agent` node;
- `body`: the controlled body;
- `observations`: the current observation dictionary;
- `events`: accumulated local events;
- controller data such as step, progress, and terminal state.

Each component returns its weighted value and may implement `reset_reward(context)`. Named values are included in diagnostics.

`ScenarioRewardSystem` evaluates `ScenarioRewardComponent` once per agent. Its context adds `agent_id`, progress, scenario events as direct keys, and terminal/truncation state. Optional `is_agent_stalled()` and `get_terminal_reason()` methods may finish one agent without ending every other channel.

Runtime scenario configuration is opt-in for reward components. Add property names to `scenario_config_properties` in the Inspector, or override `apply_scenario_config(config)`, when Python should be allowed to change them. Other configuration keys are ignored, so a generic key cannot silently overwrite a reward coefficient with the same name.

Use local rewards for behavior intrinsic to a body. Use scenario rewards for task rules involving the shared world.

### Observation margin reward

Add `ObservationMarginReward` under `Agent/RewardSystem` when a scalar observation should stay near a target instead of receiving a flat reward anywhere inside a threshold. Configure it in the Inspector; no task-specific Python code is required.

| Property | Meaning |
|---|---|
| `observation_name` | Scalar key in the current observation dictionary. |
| `target_value` | Desired center, in the observation's units. |
| `half_width` | Positive reference distance from the target to either band edge. |
| `scale` and inherited `weight` | Cost magnitude; positive values penalize distance. |
| `shape` | Linear (`0`) or quadratic (`1`, default) centering cost. |
| `gate_observation` | Optional observation key that enables the cost only inside the gate range. |
| `gate_min_value`, `gate_max_value` | Inclusive gate bounds. |

For `e = abs(observation - target_value) / half_width`, the component returns `-scale * weight * e` in linear mode or `-scale * weight * e * e` in quadratic mode. Cost is zero at the center, `-scale * weight` at either reference edge and continues beyond the band; it is not clipped or flat inside. Missing/nonfinite observations, an inactive gate or invalid width return zero. The component changes reward shaping only, not success or terminal rules. See [Training stability controls](../training-stability.md#observation-margin-reward) for scope and tuning cautions.

## Training wizard

The Train dialog reads each native algorithm's Python argument schema and generates its **All options** controls from the same defaults as the CLI. Its prominent fields include SAC demonstration protection, PPO expert datasets/BC/imitation settings, solved threshold, solved patience, plateau LR reduction and actor EMA where supported. Certification episode count, confidence, seed and cumulative recovery remain available in **All options**.

These settings are opt-in and do not bypass runtime validation. Solved certification requires single-agent, single-policy training; EMA supports the single-policy SAC/DDPG/TD3 family, while PPO imitation requires standard single-policy mode. The complete behavior and resume contract are described in [Training stability controls](../training-stability.md) and [Manual demonstrations](../guides/manual-demonstrations.md).

## Events

`ScenarioEventSystem` aggregates `ScenarioEventSource` children. Event sources keep state per agent ID and return named values. Built-in sources cover:

- entering an `Area`;
- crossing an observation threshold;
- events triggered directly by scenario code.

An event records what happened. A reward component decides what that event is worth, and an event source may independently declare a terminal reason.

## Progress

`ProgressProvider` exposes an ordered task metric. Included implementations are:

- `Path3DProgressProvider`;
- `MethodProgressProvider`, which calls a scene method.

Progress may mean track completion, bricks removed, target approach, object lift, or another task phase. Avoid adding it to policy observations when the policy is intended to remain independent of a particular map.

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

Prefer controller hooks and child components to replacing `step()` or changing the wire response format.

`physics_frames_per_step` sets the action repeat. `max_steps=0` disables the controller step limit, so the scene must then provide reliable terminal or stall behavior.

## `BridgeServer`

The bridge accepts one TCP client, enables `TCP_NODELAY`, reads one JSON object per line, and forwards `spec`, `configure`, `reset`, and `step` requests to the controller.

In lockstep mode the world advances only while serving a step. In real-time mode it keeps processing with the latest action. The bridge does not know concrete agent classes, reward values, or action names.

## Physical reset checklist

After teleporting a body or dynamic object:

1. Clear linear and angular velocity.
2. Clear current and previous control inputs.
3. Restore transforms for every dynamic object owned by the episode.
4. Force ray casts and sensors to refresh.
5. Reset observation sources, rewards, events, and progress.
6. Wait for extra physics frames only when the scene genuinely needs them.

An incomplete reset can return stale observations and contaminate replay with a transition between two different episodes.

## Robot-arm collision handling

The bundled URDF integration composes combined roll/pitch/yaw origins in URDF order after coordinate conversion to Godot axes. Reimport robot resources after an importer change before comparing visual, collision and joint transforms; a stale imported resource can retain the earlier transform. Single-axis origins alone do not exercise rotation-order differences.

`URDFRobotArmAgentBody` performs separate environment and self-collision queries. Environment collision checks query every imported robot collision shape and only report colliders in `obstacle_group`. Configure the query with `auto_detect_environment_collisions`, `environment_collision_mask`, and `max_collision_results_per_shape`.

Self-body checks are deliberately narrower. Enable `self_body_collision_enabled`, put the central body links in `self_body_link_names`, and list only distal links that can swing into that body in `self_check_link_names`. Adjacent shoulder links that overlap mechanically should stay out of the checker list. `collision_debug` and `self_collision_debug` print the first detected contact of an episode while a new URDF is being calibrated.
