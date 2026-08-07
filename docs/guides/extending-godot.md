# Extending Metis in Godot

Reusable Godot components live under `godot/addons/metis/runtime/agent/`. Mechanics
that belong to one task stay in the agent body or under `godot/scenarios`. In a
standalone project, keep custom components outside `addons/metis` so upgrading the
add-on cannot overwrite them.

## Choose the right extension point

| Need | Extension point |
|---|---|
| Read local state or a sensor | `ObservationSource` |
| Score local behavior | `RewardComponent` |
| Score task or shared-world state | `ScenarioRewardComponent` |
| Publish a collision or fact | `ScenarioEventSource` |
| Measure ordered task progress | `ProgressProvider` |
| Declare another action component | Child of `ActionSpace` |

Detect a fact once. For example, publish one collision event and let reward and
terminal components consume it. Repeating the collision query in several reward nodes
makes behavior harder to reason about.

## Adding an observation source

First check whether a generic adapter is enough:

```text
MethodObservationSource:
  source_path = ""               # Empty means the agent body
  method_name = Select...

PropertyObservationSource:
  observation_name = "health"
  source_path = ""
  property_path = Select...       # For example health or velocity:x
  normalize_numeric = true
  input_min = 0.0
  input_max = 100.0
  output_min = 0.0
  output_max = 1.0
```

The picker field remains editable, so sub-properties not shown by the direct-property
menu can still be entered manually. Write a new source only when an observation needs
its own state, physics query, or calculation.

This source returns normalized distance to a target:

```gdscript
extends "res://addons/metis/runtime/agent/observations/ObservationSource.gd"
class_name TargetDistanceObservationSource

@export var observation_name := "target_distance"
@export var target: Node3D
@export var max_distance := 20.0

var _body: Node3D


func register_observations(agent:Agent, body:Node) -> void:
    _body = body as Node3D
    agent.add_observation(observation_name, Callable(self, "_read_distance"))


func _read_distance() -> float:
    if _body == null or target == null:
        return 1.0
    var distance := _body.global_position.distance_to(target.global_position)
    return clampf(distance / maxf(max_distance, 0.000001), 0.0, 1.0)
```

In the Metis repository, save task-specific code beside its agent or scenario. In an
installed project, a path such as `res://rl_components/observations/` keeps it separate
from the add-on. Add the node below `ObservationSystem` and configure it in the
Inspector.

Observation-source rules:

- Register the same number of values every time.
- Return finite values.
- Clear episode-local state in `reset_source()`.
- Refresh ray casts or cached physics data in `refresh_source()`.
- Avoid global positions when the policy should generalize across maps.

## Adding a local reward

This example rewards forward speed only while a clearance signal is active:

```gdscript
extends "res://addons/metis/runtime/agent/reward_components/RewardComponent.gd"
class_name GatedSpeedReward

@export var gate_observation := "path_clear"
@export var speed_observation := "forward_speed"
@export var gate_threshold := 0.5
@export var reward_scale := 0.02


func compute_reward(context:Dictionary) -> float:
    var observations: Dictionary = context.get("observations", {})
    if float(observations.get(gate_observation, 0.0)) < gate_threshold:
        return 0.0
    var speed := maxf(float(observations.get(speed_observation, 0.0)), 0.0)
    return speed * reward_scale * weight
```

Below `RewardSystem`, the component receives `agent`, `body`, `observations`, local
`events`, and controller context. Give every component a clear `term_name` so logs are
useful.

If the component keeps memory between steps, clear it explicitly:

```gdscript
func reset_reward(_context:Dictionary = {}) -> void:
    # Clear counters and values from the previous episode.
    pass
```

`compute_reward()` returns the final value, including `weight`.

## Adding a scenario reward

Use a scenario component when the value depends on progress, teams, goals, or objects
outside one body:

```gdscript
extends "res://addons/metis/runtime/agent/scenario_reward_components/ScenarioRewardComponent.gd"
class_name PossessionScenarioReward

@export var event_name := "has_possession"
@export var reward_per_step := 0.005


func compute_reward(_agent:Node, context:Dictionary = {}) -> float:
    if not bool(context.get(event_name, false)):
        return 0.0
    return reward_per_step * weight
```

Scenario event names are merged directly into the context. Store per-agent state in a
dictionary keyed by `agent_id`, and clear it in both `reset_rewards()` and
`reset_agent(agent_id, context)`.

Optional methods include:

- `get_terminal_reason(agent_id)`;
- `is_agent_stalled(agent_id)`;
- `get_agent_terms(agent_id)`;
- `is_episode_end_only()` for a value computed only at episode end.

Python configuration does not write arbitrary reward properties. To expose a value as
a runtime setting, add its name to `scenario_config_properties` in the Inspector:

```gdscript
@export var possession_scale := 0.005


func _ready() -> void:
    scenario_config_properties = PackedStringArray(["possession_scale"])
```

For validation or derived values, override `apply_scenario_config(config)` instead.
Keep the whitelist small: changing reward semantics during a run also changes the
meaning of replay already collected.

## Adding an event source

This source emits an event when health reaches zero:

```gdscript
extends "res://addons/metis/runtime/agent/events/ScenarioEventSource.gd"
class_name HealthDepletedEventSource

@export var health_property := "health"

var _values := {}


func reset_events() -> void:
    _values.clear()


func reset_agent(agent:Node, context:Dictionary = {}) -> void:
    var agent_id := str(context.get("agent_id", agent.name))
    _values[agent_id] = false


func update_agent(agent:Node, context:Dictionary = {}) -> void:
    var agent_id := str(context.get("agent_id", agent.name))
    _values[agent_id] = float(agent.get(health_property)) <= 0.0


func get_agent_events(agent_id:String) -> Dictionary:
    return {event_name: bool(_values.get(agent_id, false))}
```

Set `event_name` and an optional `terminal_reason` in the Inspector. Respect the
controller-provided agent ID instead of assuming node names are globally unique.

## Adding a progress provider

Progress can be any mostly ordered metric:

```gdscript
extends "res://addons/metis/runtime/agent/progress/ProgressProvider.gd"
class_name BrickProgressProvider

@export var bricks_container: Node

var _initial_count := 0


func reset_provider() -> void:
    _initial_count = (
        bricks_container.get_child_count()
        if bricks_container != null
        else 0)


func measure_progress(_agent:Node, _context:Dictionary = {}) -> float:
    if bricks_container == null or _initial_count <= 0:
        return 0.0
    var remaining := bricks_container.get_child_count()
    return 1.0 - float(remaining) / float(_initial_count)
```

Assign the provider through `ScenarioController.progress_provider_path`. Progress
rewards and curriculum can then use it without knowing its concrete implementation.
Do not automatically add progress to policy observations.

## Adding an action component

Most tasks only need `ContinuousAction` and `DiscreteActionSet`. A custom child must at
least expose a specification:

```gdscript
func get_action_spec() -> Dictionary:
    return {
        "name": "ability",
        "size": 2,
        "action_type": "discrete",
        "names": ["none", "activate"],
    }
```

Implement `execute_action(action, default_target)` for automatic discrete execution.
Introducing an action type other than `discrete` or `continuous` also requires changes
to the Gymnasium parser, algorithms, and runner.

## Inspector workflow

1. Create the script in the directory for its component family.
2. Add a `class_name` so it appears in Godot's Add Node dialog.
3. Add the node under the correct aggregate system.
4. Configure names, paths, bounds, and `term_name`.
5. Inspect the resulting specification with a random rollout.

```bash
python/.venv/bin/python python/tools/random_rollout.py \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/my_scenario.tscn \
  --steps 200
```

## Review checklist

- Multiple agents do not accidentally share episode-local state.
- Reset clears every counter and cache.
- Seeded randomization is deterministic.
- Headless mode skips cameras, UI, and debug work.
- Observation and action order remain stable.
- Reward terms have unique names and sensible scale.
- One-shot events do not fire every step.
- Completed agents stop modifying the world.
- Reusable components contain no references to a specific example scene.

Keep one-off mechanics in their scene. Turning a single special case into a framework
node usually makes the Inspector harder to use without creating real reuse.
