# Build a new agent and scenario

This tutorial follows the normal Metis workflow without introducing a custom bridge
or Python trainer. The example is a small 3D body that learns to reach a target. It is
deliberately plain, but it exercises the same pieces used by larger tasks:

- an action space declared in Godot;
- component-based observations;
- local and scenario rewards;
- progress and a terminal event;
- seeded reset;
- validation, training, checkpoints, and inference.

## 1. Define the task on paper

Write down four things before opening the editor.

**Goal:** enter the target area using as few decisions as practical.

**Actions:**

```text
0 idle
1 forward
2 forward_left
3 forward_right
```

**Observations:**

```text
target_local_x       [-1, 1]
target_local_forward [-1, 1]
target_distance      [ 0, 1]
speed                [ 0, 1]
```

**Outcome:** a small step cost, distance-progress shaping, a goal bonus, natural
termination at the goal, and truncation at the configured step limit.

The target coordinates are relative to the body. Absolute world coordinates would
make it easier to memorize one layout and harder to reuse the policy elsewhere.

## 2. Create the files

```text
godot/agents/TargetSeeker/
    target_seeker.gd
    target_seeker.tscn

godot/scenarios/target_seeker/
    target_scenario.gd
    target_scenario.tscn
```

Build the agent scene:

```text
TargetSeeker                 CharacterBody3D
|-- MeshInstance3D
|-- CollisionShape3D
`-- Agent                    Agent.gd
    |-- ActionSpace          ActionSpace.gd
    |-- ObservationSystem    ObservationSystem.gd
    `-- RewardSystem         RewardSystem.gd
```

The conventional child names save configuration because `Agent` can discover those
systems automatically.

## 3. Implement the body

Attach `target_seeker.gd` to the root body:

```gdscript
extends CharacterBody3D
class_name TargetSeeker

@export var move_speed := 5.0
@export var turn_speed := 2.5
@export var target: Node3D
@export var target_distance_scale := 20.0

@onready var agent: Agent = $Agent

var _drive_input := 0.0
var _turn_input := 0.0
var _training_active := true


func _physics_process(delta:float) -> void:
    if not _training_active:
        return

    rotate_y(_turn_input * turn_speed * delta)
    var forward := -global_transform.basis.z
    var vertical_speed := velocity.y
    velocity = forward * (_drive_input * move_speed)
    velocity.y = vertical_speed
    if not is_on_floor():
        velocity += get_gravity() * delta
    move_and_slide()


func apply_action(action:Variant) -> Variant:
    clear_control()
    var action_id := int(action)
    if agent.act(action_id) != OK:
        return 0
    return action_id


func set_control(drive:float, turn:float) -> void:
    _drive_input = clampf(drive, -1.0, 1.0)
    _turn_input = clampf(turn, -1.0, 1.0)


func clear_control() -> void:
    _drive_input = 0.0
    _turn_input = 0.0


func get_target_local_observation() -> Vector2:
    if target == null:
        return Vector2.ZERO
    var offset_world := target.global_position - global_position
    var offset_local := global_transform.basis.inverse() * offset_world
    var scale := maxf(target_distance_scale, 0.001)
    return Vector2(
        clampf(offset_local.x / scale, -1.0, 1.0),
        clampf(-offset_local.z / scale, -1.0, 1.0))


func get_target_distance_observation() -> float:
    if target == null:
        return 1.0
    var distance := global_position.distance_to(target.global_position)
    return clampf(distance / maxf(target_distance_scale, 0.001), 0.0, 1.0)


func get_speed_observation() -> float:
    var horizontal_velocity := Vector2(velocity.x, velocity.z)
    return clampf(
        horizontal_velocity.length() / maxf(move_speed, 0.001),
        0.0,
        1.0)


func reset_all(original_transform:Variant, reset_rewards := true) -> void:
    set_training_active(true)
    clear_control()
    velocity = Vector3.ZERO
    if typeof(original_transform) == TYPE_TRANSFORM3D:
        transform = original_transform
    reset_physics_interpolation()
    agent.reset_observation_sources()
    if reset_rewards:
        agent.refresh_observation_sources()
        agent.reset_reward({"body": self})


func is_terminal() -> bool:
    return false


func set_training_active(enabled:bool) -> void:
    _training_active = enabled
    set_physics_process(enabled)
    if not enabled:
        clear_control()
        velocity = Vector3.ZERO
```

The body owns movement and state. `Agent` owns the RL declaration. There is no
`get_action_space()` or hand-built observation vector in this script.

## 4. Declare discrete actions

Under `Agent/ActionSpace`, add:

```text
Movement                    DiscreteActionSet
|-- Idle                    DiscreteAction
|-- Forward                 DiscreteAction
|-- ForwardLeft             DiscreteAction
`-- ForwardRight            DiscreteAction
```

Set `Movement.action_name = "movement"`. Leave each `target_path` empty so the agent
body is used as the default target.

| Node | `action_name` | `method_name` | `arguments` |
|---|---|---|---|
| Idle | `idle` | `set_control` | `[0.0, 0.0]` |
| Forward | `forward` | `set_control` | `[1.0, 0.0]` |
| ForwardLeft | `forward_left` | `set_control` | `[1.0, 1.0]` |
| ForwardRight | `forward_right` | `set_control` | `[1.0, -1.0]` |

Leave the legacy `DiscreteActionSet.names` array empty when child `DiscreteAction`
nodes are present. Do not register the same actions again with `agent.add_action()`.

## 5. Add observations

Add three `MethodObservationSource` nodes under `Agent/ObservationSystem`:

| Node | `observation_name` | `method_name` | Size |
|---|---|---|---:|
| TargetLocal | `target_local` | `get_target_local_observation` | 2 |
| TargetDistance | `target_distance` | `get_target_distance_observation` | 1 |
| Speed | `speed` | `get_speed_observation` | 1 |

Leave `source_path` empty. `Vector2` is flattened automatically, so the resulting
specification has `obs_dim=4`.

If a new method does not appear in the Inspector picker, save the script and scene,
fix any parser errors, and select the observation node again. The field remains
editable by hand.

Child order is observation order. Reordering sources changes the policy contract even
when `obs_dim` stays the same.

## 6. Add a local reward

Add `StepPenaltyReward` under `Agent/RewardSystem`:

```text
term_name = time
penalty = -0.001
weight = 1.0
```

This cost makes shorter solutions preferable. There is no generic movement bonus:
driving in circles is movement but not useful task progress.

## 7. Build the scenario

```text
TargetScenario                     Node3D, target_scenario.gd
|-- World
|   |-- Floor                      StaticBody3D
|   `-- Target                     Area3D
|       |-- MeshInstance3D
|       `-- CollisionShape3D
|-- Agents
|   `-- TargetSeeker               target_seeker.tscn
|-- ScenarioController             ScenarioController.gd
|   |-- ScenarioRewardSystem       ScenarioRewardSystem.gd
|   |   |-- Progress               ProgressDeltaScenarioReward.gd
|   |   `-- Goal                   EventScenarioReward.gd
|   |-- ProgressProvider           MethodProgressProvider.gd
|   `-- ScenarioEventSystem        ScenarioEventSystem.gd
|       `-- GoalReached            AreaReachedEventSource.gd
`-- BridgeServer                   bridge_server.gd
```

Configure these paths:

- `TargetSeeker.target -> World/Target`;
- `ScenarioController.controlled_agents -> Agents/TargetSeeker`;
- `BridgeServer.controller_path -> ../ScenarioController`;
- `GoalReached.area -> World/Target`;
- `ProgressProvider.source_path -> TargetScenario`.

The target area's collision mask must include the body layer.

## 8. Define progress and the goal

Attach this script to `TargetScenario`:

```gdscript
extends Node3D

@export var target: Node3D
@export var max_target_distance := 20.0


func get_progress(agent:Node) -> float:
    if target == null or not agent is Node3D:
        return 0.0
    var distance := (agent as Node3D).global_position.distance_to(
        target.global_position)
    return 1.0 - clampf(
        distance / maxf(max_target_distance, 0.001),
        0.0,
        1.0)
```

Configure `MethodProgressProvider`:

```text
method_name = get_progress
pass_agent_to_source = true
```

Configure `ProgressDeltaScenarioReward`:

```text
term_name = progress
progress_reward_scale = 2.0
backward_penalty_scale = 2.0
```

Configure the event and goal reward:

```text
GoalReached.event_name = goal_reached
GoalReached.terminal_reason = goal_reached
GoalReached.only_once = true

Goal.term_name = goal
Goal.event_name = goal_reached
Goal.reward = 5.0
Goal.only_once = true
```

The event states what happened. Its reward and terminal reason are configured
separately. Python receives local and scenario reward terms independently in `info`.

## 9. Reset and episode duration

Start with:

```text
ScenarioController.max_steps = 500
ScenarioController.physics_frames_per_step = 1
ScenarioController.randomize_reset = false
ScenarioController.deactivate_done_agents = true
```

`max_steps` produces a truncation. Set it to zero only when a task has another reliable
way to end or detect a stall.

Randomize the target through the controller's reset signal:

```gdscript
@export var target_spawn_half_extent := Vector2(7.0, 7.0)
@onready var scenario_controller: ScenarioController = $ScenarioController


func _ready() -> void:
    scenario_controller.episode_reset_started.connect(_on_episode_reset_started)


func _on_episode_reset_started(episode_seed:int) -> void:
    var rng := RandomNumberGenerator.new()
    rng.seed = episode_seed
    target.position.x = rng.randf_range(
        -target_spawn_half_extent.x,
        target_spawn_half_extent.x)
    target.position.z = rng.randf_range(
        -target_spawn_half_extent.y,
        target_spawn_half_extent.y)
```

Use the supplied seed so a reset can be reproduced. Validate a fixed target before
turning randomization on.

## 10. Validate the contract

```bash
python/.venv/bin/python python/tools/random_rollout.py \
  --godot-project godot \
  --godot-scene res://scenarios/target_seeker/target_scenario.tscn \
  --steps 500 \
  --print-reward-terms \
  --no-headless
```

Check:

- one agent, `obs_dim=4`, discrete actions, and four action names;
- every action moves in the intended direction;
- observations remain finite and change with the body;
- progress rises toward the target and falls away from it;
- the goal bonus fires once;
- entering the area returns `terminal_reason=goal_reached`;
- reset clears velocity, controls, source caches, and rewards.

If Godot disconnects, inspect `.runtime/godot_logs/godot_<port>.log` for the original
GDScript error.

## 11. Train and run

```bash
python/.venv/bin/python python/train.py \
  --algorithm auto \
  --godot-project godot \
  --godot-scene res://scenarios/target_seeker/target_scenario.tscn \
  --num-envs 4 \
  --num-episodes 1500 \
  --max-steps-per-episode 500 \
  --checkpoint-dir checkpoints/target_seeker_dqn_v1 \
  --headless
```

`auto` chooses DQN for this discrete space. Show one collector while the others remain
headless with:

```text
--no-headless --render-env-count 1 --render-mode light-gpu
```

Run the saved policy:

```bash
python/.venv/bin/python python/run.py \
  --checkpoint-dir checkpoints/target_seeker_dqn_v1 \
  --godot-project godot \
  --godot-scene res://scenarios/target_seeker/target_scenario.tscn \
  --episodes 20 \
  --epsilon 0.0 \
  --no-headless
```

Resume training with the original checkpoint directory and `--resume`. Loading
`policy.keras` with `--policy-path` is a warm start and does not restore replay,
optimizers, target networks, or counters.

## 12. Multi-agent, continuous, and hybrid variants

For parameter sharing, add compatible instances to `controlled_agents` and pass
`--multi-agent`. Each body contributes separate transitions to one shared model.

For continuous motion, replace the discrete set with:

```text
ActionSpace
|-- Drive       ContinuousAction, size=1, low=-1, high=1
`-- Steering    ContinuousAction, size=1, low=-1, high=1
```

Decode the vector in the body:

```gdscript
var values := agent.decode_continuous_action(action)
_drive_input = float(values[0])
_turn_input = float(values[1])
return values
```

A hybrid agent keeps continuous components and adds a `DiscreteActionSet`, for example
continuous locomotion plus discrete fire/reload actions. PPO handles the resulting
component dictionary.

## 13. Curriculum and demonstrations

A curriculum should change task difficulty, not reveal hidden state to the policy.
Use `training_episode` to expand spawn ranges, add obstacles, or tighten tolerances.
Keep each stage stable long enough to evaluate it.

Record a manual policy with:

```bash
python/.venv/bin/python python/recorder.py \
  --godot-project godot \
  --godot-scene res://scenarios/target_seeker/target_scenario.tscn \
  --output demonstrations/target_seeker_demo.npz \
  --episodes 20 \
  --no-headless
```

The body must accept the special action `"manual"` and return the action that was
actually applied. See [Manual demonstrations](../guides/manual-demonstrations.md).

## 14. Compatibility checklist

Start a fresh run when observation order or size, action structure, or physics meaning
changes. A policy may still load after reward-only changes, but old replay contains
returns from the previous task and should usually be discarded.

Before calling a new scene ready, verify:

- action names, order, and bounds;
- observation names, order, size, range, and reset behavior;
- local versus scenario reward ownership;
- one-shot events and terminal reasons;
- seeded reset reproducibility;
- clean headless execution;
- random rollout behavior;
- checkpoint, resume, and `run.py` loading.
