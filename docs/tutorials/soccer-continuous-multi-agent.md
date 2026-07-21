# Build a continuous-control soccer scenario

This tutorial designs a small 3D arcade soccer task. Two `CharacterBody3D` players,
one on each team, move around a field and kick a `RigidBody3D` ball into the opposing
goal. The first version is 1v1, but its contract can be reused for 2v2 or 3v3 with one
shared policy.

Soccer is not currently a ready-made scene in the repository. The names below are a
recommended layout for a new example.

## 1. Choose the action space

Use three continuous values:

```text
movement:      Continuous(2)  [throttle, steering]
kick_strength: Continuous(1)  [0, 1]
```

Continuous kick strength allows small touches, passes, and stronger shots. A kick is
applied only when all of these conditions hold:

- the input is above a small intention threshold;
- the ball is inside a short area in front of the player;
- the cooldown has expired.

This remains a continuous action space, so SAC is a good first trainer. If you later add
tackle, jump, or player switching, those commands are natural discrete components and
the action space can become hybrid, trained with PPO.

Do not change the action contract in the middle of a run. A model trained with three
continuous outputs cannot load into a different set of heads.

## 2. Use parameter sharing across teams

Create two instances of the same player scene:

```text
RedPlayer   team_id = 0
BluePlayer  team_id = 1
```

Both use one policy. Each instance receives references to its own goal and opposing
goal, and observations are expressed in the player's local frame. The network therefore
does not need a color-specific coordinate system.

With four Godot environments in 1v1, eight agents collect experience for one actor and
two SAC critics. Adding players increases data collection; it does not create a model
per player.

## 3. Define observations in the player's frame

A useful first contract is:

| Group | Size | Values |
|---|---:|---|
| body motion | 2 | signed forward speed, absolute speed |
| previous input | 3 | throttle, steering, kick strength |
| ball | 4 | local X/Z position and local X/Z velocity |
| opponent goal | 3 | local direction X/Z and normalized distance |
| own goal | 3 | local direction X/Z and normalized distance |
| kick state | 2 | ball in range, cooldown ready |
| players | 7 | enemy visible plus three enemy and three ally rays |

Total: 24 scalar values.

Avoid global positions, player names, or team colors. The same physical situation should
produce the same vector after it is transformed into either player's frame.

Direct ball position is acceptable for a first simulation. Replace it with raycasts or
vision later if that better matches deployment, but treat that as a new observation
contract and a new training run.

## 4. Create `soccer_player.tscn`

Start with:

```text
SoccerPlayer (CharacterBody3D) [soccer_player.gd]
├── MeshInstance3D
├── CollisionShape3D
├── KickArea (Area3D)
│   └── CollisionShape3D
├── VisionSensors (Node3D)
│   ├── VisionLeft (RayCast3D)
│   ├── VisionCenter (RayCast3D)
│   └── VisionRight (RayCast3D)
└── Agent
    ├── ActionSpace
    │   ├── Movement (ContinuousAction)
    │   └── KickStrength (ContinuousAction)
    ├── ObservationSystem
    │   ├── BodySpeed (BodySpeedObservationSource)
    │   ├── MoveInput (MethodObservationSource)
    │   ├── RotationInput (MethodObservationSource)
    │   ├── KickInput (MethodObservationSource)
    │   ├── BallPosition (MethodObservationSource)
    │   ├── BallVelocity (MethodObservationSource)
    │   ├── OpponentGoalDirection (MethodObservationSource)
    │   ├── OpponentGoalDistance (MethodObservationSource)
    │   ├── OwnGoalDirection (MethodObservationSource)
    │   ├── OwnGoalDistance (MethodObservationSource)
    │   ├── BallInRange (MethodObservationSource)
    │   ├── KickReady (MethodObservationSource)
    │   └── TeamVision (TeamRaycastObservationSource)
    └── RewardSystem
        ├── TimePenalty (StepPenaltyReward)
        └── WastedKickPenalty (EventReward)
```

Place `KickArea` immediately in front of the player. A short box is easier to reason
about than a sphere around the whole body: the player has to face the ball before a kick
is valid.

Configure the action nodes:

```text
Movement:
  action_name = "movement"
  size = 2
  low = -1
  high = 1

KickStrength:
  action_name = "kick_strength"
  size = 1
  low = 0
  high = 1
```

## 5. Implement the player body

This is a compact starting point. It leaves observations as small methods that can be
wired through `MethodObservationSource` nodes.

```gdscript
extends CharacterBody3D
class_name SoccerPlayer

signal valid_kick(player: SoccerPlayer, strength: float)

@export var team_id := 0
@export var move_speed := 8.0
@export var acceleration := 20.0
@export var turn_speed_degrees := 160.0
@export var kick_impulse := 10.0
@export var kick_cooldown := 0.35
@export var min_kick_input := 0.15
@export var distance_scale := 30.0
@export var ball: RigidBody3D
@export var own_goal: Node3D
@export var opponent_goal: Node3D

@onready var agent: Agent = $Agent
@onready var kick_area: Area3D = $KickArea

var _move_input := 0.0
var _rotation_input := 0.0
var _kick_input := 0.0
var _cooldown_left := 0.0
var _terminal := false


func _physics_process(delta: float) -> void:
	_cooldown_left = maxf(_cooldown_left - delta, 0.0)
	rotate_y(-_rotation_input * deg_to_rad(turn_speed_degrees) * delta)

	var desired := global_transform.basis.z * _move_input * move_speed
	var flat := Vector3(velocity.x, 0.0, velocity.z)
	flat = flat.move_toward(desired, acceleration * delta)
	velocity.x = flat.x
	velocity.z = flat.z
	if not is_on_floor():
		velocity += get_gravity() * delta
	move_and_slide()

	_try_kick()


func apply_action(action: Variant) -> Variant:
	var values := agent.decode_continuous_action(action)
	_move_input = clampf(float(values[0]), -1.0, 1.0)
	_rotation_input = clampf(float(values[1]), -1.0, 1.0)
	_kick_input = clampf(float(values[2]), 0.0, 1.0)
	return [_move_input, _rotation_input, _kick_input]


func _try_kick() -> void:
	if _kick_input < min_kick_input or _cooldown_left > 0.0:
		return
	if ball == null or not kick_area.overlaps_body(ball):
		agent.add_reward_event("wasted_kick", 1.0)
		_cooldown_left = kick_cooldown
		return

	var direction := ball.global_position - global_position
	direction.y = 0.15
	direction = direction.normalized()
	ball.apply_central_impulse(direction * kick_impulse * _kick_input)
	_cooldown_left = kick_cooldown
	valid_kick.emit(self, _kick_input)
```

Do not smooth `kick_strength` in the body. It represents an impulse request, not a
persistent motor command. If action smoothing is used by the trainer, set it to zero for
this baseline or move the kick to a discrete action.

## 6. Add observation methods

Convert world vectors through the player's inverse basis:

```gdscript
func get_ball_relative_position_observation() -> Vector2:
	if ball == null:
		return Vector2.ZERO
	var local := global_transform.basis.inverse() * (ball.global_position - global_position)
	return Vector2(
		clampf(local.x / distance_scale, -1.0, 1.0),
		clampf(local.z / distance_scale, -1.0, 1.0)
	)


func get_ball_velocity_observation() -> Vector2:
	if ball == null:
		return Vector2.ZERO
	var local := global_transform.basis.inverse() * ball.linear_velocity
	return (Vector2(local.x, local.z) / 15.0).limit_length(1.0)


func get_goal_direction_observation(opponent: bool) -> Vector2:
	var goal := opponent_goal if opponent else own_goal
	if goal == null:
		return Vector2.ZERO
	var local := global_transform.basis.inverse() * (goal.global_position - global_position)
	var flat := Vector2(local.x, local.z)
	return flat.normalized() if flat.length() > 0.0001 else Vector2.ZERO


func get_goal_distance_observation(opponent: bool) -> float:
	var goal := opponent_goal if opponent else own_goal
	if goal == null:
		return 0.0
	return clampf(global_position.distance_to(goal.global_position) / distance_scale, 0.0, 1.0)


func get_ball_in_range_observation() -> float:
	return 1.0 if ball != null and kick_area.overlaps_body(ball) else 0.0


func get_kick_ready_observation() -> float:
	return 1.0 if _cooldown_left <= 0.0 else 0.0
```

Bind the Boolean argument on the goal observation nodes so the same methods can serve
both goals. Use `TeamRaycastObservationSource` for ally and enemy channels.

## 7. Configure the ball and field

The ball should be a `RigidBody3D` with:

- a sphere collision shape;
- low but nonzero linear damping;
- enough continuous collision quality for the maximum kick impulse;
- a physics material with moderate bounce and low friction;
- constrained motion if you want strictly planar arcade play.

Create floor, side walls, two physical goal frames, and two `Area3D` goal volumes behind
the goal line. Goal areas should detect the ball but not push it.

Use collision layers deliberately. One workable layout is:

| Layer | Contents |
|---:|---|
| 1 | players |
| 2 | arena and goal frames |
| 3 | ball |
| 4 | goal and kick areas |

The exact numbers do not matter; consistent masks do.

## 8. Build `soccer_1v1.tscn`

```text
SoccerScenario (Node3D) [soccer_scenario.gd]
├── BridgeServer
├── ScenarioController
│   ├── ScenarioEventSystem
│   │   ├── RedScored
│   │   ├── BlueScored
│   │   ├── ValidKick
│   │   └── BallProgress
│   └── ScenarioRewardSystem
├── Field
│   ├── RedGoalArea
│   └── BlueGoalArea
├── Ball
├── RedPlayer
└── BluePlayer
```

Add both players to `controlled_agents`. Give each player the correct `own_goal` and
`opponent_goal` references. The scenario script owns team outcomes and reset logic.

Recommended sparse rewards:

| Event | Recipient | Reward |
|---|---|---:|
| goal scored | scoring team | `+5.0` |
| goal conceded | defending team | `-5.0` |
| valid kick | kicker | `+0.02` |
| wasted kick | local player | `-0.01` |

Add a small team-relative ball progress reward:

```text
delta(distance_to_own_goal - distance_to_opponent_goal)
```

For one team, increasing this value means moving the ball towards the opponent's goal;
for the other team the goal references are swapped. Normalize by field length and keep
the coefficient small enough that scoring remains the dominant objective.

Do not heavily reward touches. Otherwise two agents can learn to exchange harmless
contacts without ever attempting a goal.

## 9. Reset the match

At episode reset:

1. restore both player transforms;
2. zero player velocity, controls, cooldown, and terminal state;
3. place the ball near the center with small seeded position variation;
4. zero ball linear and angular velocity;
5. reset physics interpolation;
6. reset event and reward state.

When a goal is detected, freeze the ball, emit opposite terminal events for both teams,
and let `ScenarioController` perform the next reset. Guard the handler so overlapping
area signals cannot score twice.

## 10. Validate the environment

```bash
python/.venv/bin/python python/tools/random_rollout.py \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/soccer/soccer_1v1.tscn \
  --multi-agent \
  --steps 500 \
  --print-reward-terms \
  --no-headless
```

Expect `action_type=continuous`, `action_size=3`, two agents, and equal observation
dimensions. Then verify:

1. Goal directions are mirrored correctly.
2. Zero kick input never applies an impulse.
3. Stronger valid input produces a stronger impulse.
4. The cooldown prevents repeated kicks every physics frame.
5. A goal gives opposite rewards and terminates both players.
6. Reset clears every velocity and cooldown.

## 11. Train 1v1 with SAC

Begin with current-vs-current parameter sharing and the async collector:

```bash
python/.venv/bin/python python/train.py \
  --algorithm sac \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/soccer/soccer_1v1.tscn \
  --num-envs 4 \
  --num-episodes 4000 \
  --max-steps-per-episode 700 \
  --batch-size 128 \
  --replay-warmup 12000 \
  --random-exploration-episodes 50 \
  --action-smoothing 0.0 \
  --checkpoint-dir checkpoints/soccer_1v1_sac_v1 \
  --multi-agent \
  --collector-mode async \
  --headless
```

Keep `--physics-frames-per-step 1` initially. Contact, cooldown, and kick timing all
depend on the decision rate.

Once the baseline scores reliably, a historical opponent pool can reduce strategic
forgetting. SAC currently requires synchronous collection for that mode:

```bash
python/.venv/bin/python python/train.py \
  --algorithm sac \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/soccer/soccer_1v1.tscn \
  --num-envs 4 \
  --num-episodes 6000 \
  --policy-path checkpoints/soccer_1v1_sac_v1 \
  --checkpoint-dir checkpoints/soccer_1v1_sac_pool_v1 \
  --multi-agent \
  --collector-mode sync \
  --opponent-pool \
  --opponent-snapshot-every 100 \
  --opponent-pool-size 12 \
  --opponent-current-probability 0.2 \
  --headless
```

This creates a new run initialized from the baseline policy. It is not a resume of the
old optimizer and replay state.

## 12. Use a curriculum

Introduce difficulty in stages without changing action or observation dimensions.

**Ball control**

- shorter field and wider goals;
- ball spawned near one player;
- reduced maximum speed and kick impulse;
- no wasted-kick penalty.

**Full 1v1**

- center kickoff;
- normal field and goals;
- small spawn and yaw randomization;
- final outcome rewards.

**Generalization**

- varied ball spawn;
- moderate mass and damping variation;
- slightly varied goal width;
- held-out seeds for evaluation.

Use `scenario_configured` and `training_episode` to choose the stage. Keep evaluation
fixed while training conditions change.

## 13. Expand to 2v2

Create a second scene with four explicit instances:

```text
RedPlayer1, RedPlayer2
BluePlayer1, BluePlayer2
```

Add all four to `controlled_agents` and to the scenario's player list. Explicit scene
instances are clearer than automatic replication here because each one has a team,
spawn, own goal, and opponent goal.

The existing ally rays become useful without changing `obs_dim`. Team goal rewards go
to both teammates, while local kick and movement terms remain per agent.

Warm-starting from the 1v1 policy is reasonable if the contract is identical, but begin
a new replay buffer. Four-player trajectories have a different state distribution even
when the network shape is unchanged.

A shared policy can still develop roles from observations: the closest player attacks,
the deeper player defends, and a well-positioned teammate can receive a pass. Add an
explicit role observation only when the assignment will also exist in production.

## 14. Evaluate more than win rate

Two copies of the same weak policy tend towards a 50% win rate. Record:

- goals scored and conceded per side;
- valid and wasted kicks;
- average kick strength;
- approximate possession;
- ball progress towards each goal;
- timeout rate and mean time to score;
- results against scripted bots and frozen snapshots.

Use fixed seeds and both starting sides. A robust policy should improve against stable
references, not merely track the latest version of itself.

## 15. Run the policy

```bash
python/.venv/bin/python python/run.py \
  --policy-path checkpoints/soccer_1v1_sac_v1 \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/soccer/soccer_1v1.tscn \
  --multi-agent \
  --episodes 20 \
  --execution-mode realtime \
  --no-headless
```

After the 1v1 and 2v2 baselines are stable, useful extensions include a directional
kick, stamina, a discrete tackle action, explicit goalkeeper roles, and a rated
opponent league. Add them one at a time so changes in learning behavior remain
traceable.
