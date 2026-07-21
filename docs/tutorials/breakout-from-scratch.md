# Breakout from scratch

This tutorial builds a discrete single-agent task and explains why each part belongs
either to the paddle, the scenario, or the physics engine. The finished example is in:

```text
godot/agents/BreakoutPaddle/
godot/scenarios/breakout/
```

Use those files as the reference implementation while following the steps.

## 1. Define the learning problem

The paddle has three actions:

```text
0 idle
1 move_left
2 move_right
```

It observes five normalized values:

```text
paddle_x                    1
ball_relative_position     2
ball_velocity              2
```

The task ends when the ball enters the loss zone or the final brick is destroyed.

The reward used by the example is:

```text
time                -0.0001 per decision
ball_hit            +0.02
brick_destroyed     +1.0
level_cleared       +20.0
life_lost           -1.0
no_brick_progress   -5.0 and terminal after a long stall
```

The paddle does not observe a selected brick or receive a reward for aiming at one.
That shaping made the task easier to exploit and tied the policy to a particular
layout. It now learns the game from ball motion, paddle position, and outcomes.

## 2. Why the ball is a `RigidBody2D`

The ball should collide with walls, bricks, and the paddle through Godot physics. A
`RigidBody2D` gives us collision response, continuous collision detection, contact
signals, and a physical velocity that can be observed.

Configure it with:

```text
gravity_scale = 0
can_sleep = false
continuous_cd = CAST_SHAPE
contact_monitor = true
max_contacts_reported >= 8
```

Use a physics material with high bounce and no friction. The scenario still adjusts
the post-collision direction to keep the game arcade-like and avoid nearly horizontal
trajectories.

Do not assign a rigid body's transform every process frame. Reset it through
`_integrate_forces()` so transform and velocity change in the same physics update:

```gdscript
extends RigidBody2D
class_name BreakoutBall

var _reset_pending := false
var _pending_transform := Transform2D.IDENTITY
var _pending_velocity := Vector2.ZERO


func reset_ball(reset_transform:Transform2D, launch_velocity:Vector2) -> void:
    _pending_transform = reset_transform
    _pending_velocity = launch_velocity
    _reset_pending = true
    freeze = false
    sleeping = false


func stop_ball() -> void:
    _reset_pending = false
    set_deferred("freeze", true)
    call_deferred("reset_physics_interpolation")


func _integrate_forces(state:PhysicsDirectBodyState2D) -> void:
    if not _reset_pending:
        return
    state.transform = _pending_transform
    state.linear_velocity = _pending_velocity
    state.angular_velocity = 0.0
    _reset_pending = false
    call_deferred("reset_physics_interpolation")
```

This avoids the one-frame reset followed by a jump back to the old physics state.

## 3. Build the paddle scene

Create:

```text
BreakoutPaddle              CharacterBody2D
|-- Sprite2D
|-- CollisionShape2D
`-- Agent
    |-- ActionSpace
    |   `-- Actions         DiscreteActionSet
    |       |-- Idle        DiscreteAction
    |       |-- MoveLeft    DiscreteAction
    |       `-- MoveRight   DiscreteAction
    |-- RewardSystem
    |   `-- TimePenalty     StepPenaltyReward
    `-- ObservationSystem
        |-- PaddlePosition  MethodObservationSource
        |-- BallPosition    MethodObservationSource
        `-- BallVelocity    MethodObservationSource
```

Configure the action nodes:

| Node | `action_name` | `method_name` |
|---|---|---|
| Idle | `idle` | `stop` |
| MoveLeft | `move_left` | `move_left` |
| MoveRight | `move_right` | `move_right` |

Leave `target_path` empty. The default target is the paddle body. Do not fill the
legacy names array or call `agent.add_action()` for the same actions.

## 4. Implement paddle movement and observations

The full script is
[`breakout_paddle.gd`](../../godot/agents/BreakoutPaddle/breakout_paddle.gd). Its
important behavior is:

```gdscript
func _physics_process(_delta:float) -> void:
    if not _training_active:
        return
    if manual_control:
        _move_input = Input.get_axis("move_left", "move_right")
    velocity = Vector2(_move_input * speed, 0.0)
    move_and_slide()
    if lock_vertical_position:
        global_position.y = _locked_y
        velocity.y = 0.0
    global_position.x = clampf(
        global_position.x,
        horizontal_center - horizontal_limit,
        horizontal_center + horizontal_limit)


func apply_action(action:Variant) -> Variant:
    if str(action) == "manual":
        return apply_manual_action()
    manual_control = false
    stop()
    var action_id := int(action)
    if agent.act(action_id) != OK:
        return 0
    return action_id
```

Locking `y` after `move_and_slide()` prevents the paddle from being pushed vertically
when the ball hits it.

Observation methods return agent-relative and normalized values:

```gdscript
func get_paddle_x_observation() -> float:
    return clampf(
        (global_position.x - horizontal_center) / maxf(horizontal_limit, 0.001),
        -1.0,
        1.0)


func get_ball_relative_position_observation() -> Vector2:
    if ball == null:
        return Vector2.ZERO
    var relative := ball.global_position - global_position
    return Vector2(
        clampf(relative.x / observation_position_scale.x, -1.0, 1.0),
        clampf(relative.y / observation_position_scale.y, -1.0, 1.0))


func get_ball_velocity_observation() -> Vector2:
    if ball == null:
        return Vector2.ZERO
    return (ball.linear_velocity / observation_ball_speed_scale).clamp(
        Vector2(-1.0, -1.0),
        Vector2(1.0, 1.0))
```

Point the three `MethodObservationSource` nodes at these methods. Their combined size
is five.

`reset_all()` must clear movement, velocity, terminal state, physics interpolation,
observation-source caches, and local reward state. The repository script contains the
complete reset implementation.

## 5. Build the scenario

```text
BreakoutScenario                    Node2D, breakout_scenario.gd
|-- BridgeServer
|-- ScenarioController
|   |-- ProgressProvider            MethodProgressProvider
|   |-- ScenarioEventSystem
|   |   |-- BrickDestroyed          ManualScenarioEventSource
|   |   |-- LifeLost                ManualScenarioEventSource
|   |   |-- LevelCleared            ManualScenarioEventSource
|   |   `-- BallHitEvent            ManualScenarioEventSource
|   `-- ScenarioRewardSystem
|       |-- BrickReward             EventScenarioReward
|       |-- LifeLostPenalty         EventScenarioReward
|       |-- LevelReward             EventScenarioReward
|       |-- HitBall                 EventScenarioReward
|       `-- NoBrickProgress         ProgressStallScenarioReward
|-- BreakoutPaddle                  BreakoutPaddle.tscn
|-- Ball                            BreakoutBall
|-- Bricks                          Node2D
|   `-- Brick01...Brick27           StaticBody2D, group brick
|-- Walls                           Node2D
`-- LossZone                        Area2D
```

Set:

```text
BridgeServer.controller_path = ../ScenarioController
ScenarioController.controlled_agents = [../BreakoutPaddle]
ScenarioController.max_steps = 0
ProgressProvider.source_path = ../..
ProgressProvider.method_name = get_brick_progress
ProgressProvider.pass_agent_to_source = false
```

The scene has a reliable terminal condition and stall detector, so an unlimited step
count is acceptable.

## 6. Configure events and rewards

```text
BrickDestroyed.event_name = brick_destroyed

LifeLost.event_name = life_lost
LifeLost.terminal_reason = life_lost

LevelCleared.event_name = level_cleared
LevelCleared.terminal_reason = level_cleared

BallHitEvent.event_name = ball_hit
```

```text
BrickReward.event_name = brick_destroyed
BrickReward.reward = 1.0
BrickReward.only_once = false

LifeLostPenalty.event_name = life_lost
LifeLostPenalty.reward = -1.0

LevelReward.event_name = level_cleared
LevelReward.reward = 20.0

HitBall.event_name = ball_hit
HitBall.reward = 0.02

NoBrickProgress.terminate_on_stalled_progress = true
NoBrickProgress.stalled_progress_window_steps = 2500
NoBrickProgress.stalled_progress_penalty = -5.0
```

The ball-hit reward is intentionally small. It helps the early policy discover paddle
contact without making endless rallies more valuable than destroying bricks.

## 7. Scenario reset and arcade bounce

The complete implementation is
[`breakout_scenario.gd`](../../godot/scenarios/breakout/breakout_scenario.gd). It:

- collects the brick nodes at startup;
- resets every brick and the paddle terminal flag;
- randomizes launch position, direction, and speed from the episode seed;
- grows launch-position variation with `training_episode`;
- disables a brick exactly once when hit;
- emits the four scenario events;
- freezes the ball as soon as an episode ends.

The paddle controls rebound angle. A center hit sends the ball mostly upward; an edge
hit produces a diagonal trajectory:

```gdscript
func _apply_paddle_bounce(horizontal_offset:float) -> void:
    _set_ball_direction(Vector2(horizontal_offset, -1.0))
    ball_hit_event.trigger(str(paddle.name))


func _set_ball_direction(direction:Vector2) -> void:
    if direction.is_zero_approx():
        direction = Vector2(0.0, -1.0)
    var vertical_sign := signf(direction.y)
    if is_zero_approx(vertical_sign):
        vertical_sign = -1.0
    if absf(direction.y) < min_ball_vertical_ratio:
        direction.y = vertical_sign * min_ball_vertical_ratio
    ball.linear_velocity = direction.normalized() * _episode_ball_speed
    ball.angular_velocity = 0.0
```

`min_ball_vertical_ratio` prevents long horizontal loops between side walls.

For editor-only manual testing, set `manual_control=true` and
`auto_start_manual=true`. Save `manual_control=false` before training; the first Python
action also disables it defensively.

## 8. Collision setup

One simple layout is:

| Object | Layer | Mask |
|---|---:|---:|
| Ball | 1 | 1 |
| Paddle | 1 | 1 |
| Walls | 1 | 1 |
| Bricks | 1 | 1 |
| LossZone | 2 | 1 |

Every brick belongs to the `brick` group.

## 9. Validate before training

```bash
python/.venv/bin/python python/tools/random_rollout.py \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/breakout/breakout_scenario.tscn \
  --steps 200 \
  --print-reward-terms \
  --no-headless
```

Expected contract:

```text
action_type=discrete
obs_shape=(5,)
num_actions=3
```

Check that:

1. Identical seeds reproduce ball and paddle reset.
2. Different seeds move the launch position.
3. One brick hit emits one `brick_destroyed` event.
4. The loss zone returns `terminal_reason=life_lost`.
5. The last brick returns `terminal_reason=level_cleared`.
6. Ball speed stays close to the configured episode speed.
7. Different paddle contact points change rebound direction.
8. The ball freezes while Python is not advancing the environment.
9. No observation is `NaN` or outside its intended scale.

## 10. Train with DQN

```bash
python/.venv/bin/python python/train.py \
  --algorithm dqn \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/breakout/breakout_scenario.tscn \
  --num-envs 4 \
  --num-episodes 2000 \
  --max-steps-per-episode 0 \
  --batch-size 128 \
  --replay-warmup 5000 \
  --epsilon-start 1.0 \
  --epsilon-min 0.05 \
  --epsilon-decay 0.997 \
  --collector-mode async \
  --async-update-every 4 \
  --checkpoint-dir checkpoints/breakout_dqn_v1 \
  --headless
```

PPO also supports this discrete action space, but its on-policy settings are different.
Do not carry DQN replay, epsilon, or target-network options into a PPO command.

Use one visible preview without rendering every environment:

```text
--no-headless --render-env-count 1 --render-mode light-gpu
```

The log should eventually show more bricks per episode, longer useful rallies, and a
rising `level_cleared` rate. Evaluate without epsilon before deciding that the policy
has improved.

Resume the same run with the same checkpoint directory and `--resume`. Increase
`--num-episodes` to the new final episode number.

## 11. Run and evaluate the policy

```bash
python/.venv/bin/python python/run.py \
  --checkpoint-dir checkpoints/breakout_dqn_v1 \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/breakout/breakout_scenario.tscn \
  --episodes 20 \
  --epsilon 0.0 \
  --no-headless
```

Use `checkpoints/breakout_dqn_v1/best` to inspect the best evaluated policy. Resume
from the main chronological directory because it owns the replay snapshots.

For continuous viewing:

```text
--infinite --no-time-limit --epsilon 0.0 --no-headless
```

Record at least level-clear rate, mean bricks destroyed, episode length, and loss rate
over fixed seeds. Training reward alone includes exploratory actions and is not a fair
evaluation score.

## 12. Curriculum and randomization

The example gradually expands horizontal launch jitter:

```text
ball_position_jitter_x = 120
curriculum_final_ball_jitter_x = 420
curriculum_ramp_episodes = 800
reset_curriculum_enabled = true
```

`training_episode` comes from the checkpoint, so the schedule continues after resume.
Once the base task works, modestly randomize launch angle, ball speed, paddle width,
and brick layout. Use the reset seed for every random choice.

Change one source of difficulty at a time. A curriculum is useful when it presents a
learnable sequence of tasks; changing physics, observations, reward, and layout
together only makes regressions hard to diagnose.
