# Pong with shared-policy self-play

Pong is a useful first multi-agent project because the rules are small enough to
inspect, while the training setup contains most of the ideas used by larger competitive
scenarios: mirrored observations, parameter sharing, simultaneous self-play, and an
optional pool of historical opponents.

The repository includes a working version:

- [`pong_paddle.tscn`](../../godot/agents/PongPaddle/pong_paddle.tscn)
- [`pong_paddle.gd`](../../godot/agents/PongPaddle/pong_paddle.gd)
- [`pong_scenario.tscn`](../../godot/scenarios/pong/pong_scenario.tscn)
- [`pong_scenario.gd`](../../godot/scenarios/pong/pong_scenario.gd)

This guide explains how that example is assembled and why each part exists.

## 1. Decide what is being trained

There are two paddles, but only one trainable policy. Both agents expose the same
observation and action spaces, so their transitions can train the same network. This is
called **parameter sharing**.

It is distinct from the number of environments:

- two paddles in one scene means two agents contribute experience;
- four Godot instances means four independent matches are collected at once;
- one shared policy still produces every trainable action.

During ordinary simultaneous self-play, both teams use the current policy. When the
historical opponent pool is enabled, one side may instead use a frozen earlier snapshot.
Only transitions from the current learner enter training in those matches.

## 2. Design an agent-centric contract

Each paddle has three discrete actions:

| ID | Name | Effect |
|---:|---|---|
| `0` | `idle` | stop |
| `1` | `move_up` | move towards the top wall |
| `2` | `move_down` | move towards the bottom wall |

The observation vector has six scalar values:

| Observation | Size | Meaning |
|---|---:|---|
| `paddle_y` | 1 | normalized vertical position |
| `ball_relative_position` | 2 | ball position relative to this paddle |
| `ball_velocity` | 2 | normalized ball velocity |
| `opponent_delta_y` | 1 | opponent height relative to this paddle |

The right paddle sees a mirrored horizontal frame. Its own side is still "near" and
the opponent's side is still "far" from the policy's point of view. Without this mirror,
one network would have to learn two coordinate conventions for the same task.

In [`pong_paddle.gd`](../../godot/agents/PongPaddle/pong_paddle.gd), the mirror is the
`view_direction` export:

```gdscript
func get_ball_relative_position_observation() -> Vector2:
	var relative := ball.global_position - global_position
	relative.x *= view_direction
	return Vector2(
		clampf(relative.x / 640.0, -1.0, 1.0),
		clampf(relative.y / maxf(vertical_limit, 0.001), -1.0, 1.0)
	)
```

Set `view_direction = 1` on the left paddle and `-1` on the right paddle. The team ID
does not need to be part of the observation because the spatial frame is already
canonical.

## 3. Build `PongPaddle.tscn`

Use a `CharacterBody2D` root and add the usual visual and collision nodes. The Metis
branch should look like this:

```text
PongPaddle (CharacterBody2D)
└── Agent
    ├── ActionSpace
    │   └── Actions (DiscreteActionSet)
    │       ├── Idle (DiscreteAction)
    │       ├── MoveUp (DiscreteAction)
    │       └── MoveDown (DiscreteAction)
    ├── ObservationSystem
    │   ├── PaddleY (MethodObservationSource)
    │   ├── BallPosition (MethodObservationSource)
    │   ├── BallVelocity (MethodObservationSource)
    │   └── OpponentY (MethodObservationSource)
    └── RewardSystem
        ├── TimePenalty (StepPenaltyReward)
        └── AlignmentReward (FunctionRewardComponent)
```

Configure each `DiscreteAction` with a method on the paddle body:

| Node | `action_name` | `method_name` |
|---|---|---|
| `Idle` | `idle` | `stop` |
| `MoveUp` | `move_up` | `move_up` |
| `MoveDown` | `move_down` | `move_down` |

The movement methods only set the requested direction. `_physics_process()` applies
velocity and clamps the body to the arena:

```gdscript
func stop() -> void:
	_move_input = 0.0

func move_up() -> void:
	_move_input = -1.0

func move_down() -> void:
	_move_input = 1.0
```

`apply_action()` delegates the ID to `Agent.act()`. Keeping dispatch in the action
nodes makes the scene visible and editable from the Inspector.

## 4. Add local shaping carefully

The example uses two local terms:

- `TimePenalty`: `-0.00005` per decision;
- `AlignmentReward`: `0.05 * delta_alignment` while the ball approaches.

The alignment reward measures whether the absolute vertical gap to the ball shrank
since the previous observation. It is disabled while the ball moves away. This rewards
useful tracking without paying the paddle forever for merely standing near the ball.

Outcome rewards remain much larger:

- successful return: `+0.02`;
- point won: `+1.0`;
- point lost: `-1.0`.

That scale keeps winning the point as the real objective. If alignment dominates,
paddles can learn attractive movement that does not improve the score.

## 5. Build the match scene

Create this structure:

```text
PongScenario (Node2D)
├── BridgeServer
├── ScenarioController
│   ├── ScenarioEventSystem
│   │   ├── RallyHit
│   │   ├── PointWon
│   │   └── PointLost
│   └── ScenarioRewardSystem
│       ├── RallyReward
│       ├── WinReward
│       └── LossPenalty
├── LeftPaddle
├── RightPaddle
├── Ball
├── TopWall
├── BottomWall
├── LeftGoal
└── RightGoal
```

In `ScenarioController.controlled_agents`, add both paddles. Assign `team_id = 0` to
the left paddle and `team_id = 1` to the right paddle. Connect each paddle's `ball` and
`opponent` exports to the corresponding scene nodes.

The event and reward mapping is:

| Event | Terminal reason | Reward |
|---|---|---:|
| `rally_hit` | none | `+0.02` to the hitter |
| `point_won` | `point_won` | `+1.0` to the winner |
| `point_lost` | `point_lost` | `-1.0` to the loser |

Both paddles become terminal on a point. A shared episode boundary matters here: if
only the loser terminated, the two agents would disagree about which match a transition
belongs to.

## 6. Reset and serve variation

At reset, restore the ball transform, randomize its vertical position, choose a serve
side, and vary the vertical component. The included scenario seeds its own
`RandomNumberGenerator`, so a fixed episode seed remains reproducible.

The ball also uses an arcade bounce. The outgoing vertical direction depends on where
it touched the paddle, with a small minimum angle to prevent endless horizontal or
wall-to-wall loops.

The built-in curriculum changes ball speed and serve angle:

| Training episode | Ball speed | Serve angle range |
|---:|---:|---|
| `0-299` | `300` | narrow |
| `300-799` | `380` | medium |
| `800+` | `460` | full |

Curriculum state changes the task distribution, not the policy input. The action and
observation dimensions remain fixed.

## 7. Validate before training

Run a short random rollout:

```bash
python/.venv/bin/python python/tools/random_rollout.py \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/pong/pong_scenario.tscn \
  --multi-agent \
  --steps 300 \
  --print-reward-terms \
  --no-headless
```

Check the contract rather than judging the random policy:

1. Both paddle IDs appear in the scenario spec.
2. Both use the same observation and action dimensions.
3. `team_id` values are `[0, 1]` with no unassigned agent.
4. Mirrored observations have the same meaning on each side.
5. A goal emits one win, one loss, and terminal state for both paddles.
6. Reset returns paddles and ball to valid positions.

## 8. Train with simultaneous self-play

Start with ordinary self-play. It is easier to debug and gives the policy a useful
baseline before historical opponents are introduced.

```bash
python/.venv/bin/python python/train.py \
  --algorithm dqn \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/pong/pong_scenario.tscn \
  --num-envs 4 \
  --num-episodes 2500 \
  --max-steps-per-episode 0 \
  --batch-size 128 \
  --replay-warmup 8000 \
  --epsilon-start 1.0 \
  --epsilon-min 0.05 \
  --epsilon-decay 0.997 \
  --target-update-every 20 \
  --checkpoint-dir checkpoints/pong_shared_dqn_v1 \
  --weights-path pong_shared_dqn_v1.weights.h5 \
  --multi-agent \
  --collector-mode async \
  --no-opponent-pool \
  --headless
```

`--max-steps-per-episode 0` relies on the scene's terminal events and its own
`ScenarioController.max_steps`. Keep a reliable timeout somewhere; a physics loop must
not hold a collector forever.

For Pong, keep `--physics-frames-per-step 1` unless you have explicitly tested another
decision rate. Frame skipping changes how often the paddle can react.

### Optional SB3 comparison

Pong is also suitable for the limited SB3 adapter because a point terminates both
paddles together:

```bash
python/.venv-sb3/bin/python python/train.py \
  --backend sb3 \
  --algorithm dqn \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/pong/pong_scenario.tscn \
  --num-envs 4 \
  --total-timesteps 250000 \
  --max-steps-per-episode 1500 \
  --batch-size 128 \
  --multi-agent \
  --collector-mode sync \
  --evaluation-episodes 50 \
  --checkpoint-dir checkpoints/pong_sb3_dqn_v1 \
  --headless
```

Each paddle is one SB3 vector lane and both lanes use the same DQN. This is
current-policy simultaneous self-play, not historical opponent sampling. SB3
collection remains synchronous even with several Godot processes; `async` and
`--opponent-pool` are native Metis features.

## 9. Add historical opponents

Current-vs-current self-play can cycle: a policy learns to exploit its latest opponent
and forgets how to handle older strategies. The opponent pool keeps frozen snapshots
and samples one for the opposing team.

Resume the baseline run with:

```bash
python/.venv/bin/python python/train.py \
  --algorithm dqn \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/pong/pong_scenario.tscn \
  --num-envs 4 \
  --num-episodes 4000 \
  --max-steps-per-episode 0 \
  --batch-size 128 \
  --replay-warmup 8000 \
  --epsilon-min 0.05 \
  --epsilon-decay 0.997 \
  --target-update-every 20 \
  --checkpoint-dir checkpoints/pong_shared_dqn_v1 \
  --weights-path pong_shared_dqn_v1.weights.h5 \
  --multi-agent \
  --collector-mode async \
  --opponent-pool \
  --opponent-snapshot-every 100 \
  --opponent-pool-size 10 \
  --opponent-current-probability 0.2 \
  --opponent-sampling uniform \
  --resume \
  --headless
```

The important options are:

| Option | Purpose |
|---|---|
| `--opponent-snapshot-every 100` | freeze the current policy every 100 completed episodes |
| `--opponent-pool-size 10` | retain at most ten historical policies |
| `--opponent-current-probability 0.2` | keep 20% current-vs-current matches |
| `--opponent-sampling uniform` | sample uniformly among eligible snapshots |

By default, Metis alternates which team is the learner. `--learner-team 0` is useful for
diagnosis, but fixing one side for a full run can introduce a side bias.

DQN supports the opponent pool with the asynchronous collector. Other trainers reject
that combination when their async path cannot preserve the learner/opponent assignment;
use `--collector-mode sync` when the CLI reports that restriction.

The pool lives under the checkpoint directory:

```text
checkpoints/pong_shared_dqn_v1/
├── checkpoint
├── ckpt-*.index
├── replay-*.npz
└── opponents/
    ├── manifest.json
    └── policy-*.weights.h5
```

Changing the observation or action contract invalidates old snapshots, just as it
invalidates the main model.

## 10. Read the logs

Useful signals are:

- point duration and reward distribution;
- point wins per team, not only aggregate reward;
- replay size and update rate;
- action balance between idle, up, and down;
- opponent mode such as `current` or `snapshot:500`.

A longer rally is not automatically better. A policy can prolong play without learning
to win. Evaluate against fixed snapshots and a simple scripted paddle so that both
offense and defense remain visible.

## 11. Run the trained policy

Metis saves a portable policy artifact alongside checkpoints. Run it with:

```bash
python/.venv/bin/python python/run.py \
  --policy-path checkpoints/pong_shared_dqn_v1 \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/pong/pong_scenario.tscn \
  --multi-agent \
  --episodes 20 \
  --epsilon 0.0 \
  --execution-mode realtime \
  --no-headless
```

`run.py` reads the algorithm from the policy manifest when `--algorithm auto` is left at
its default. Realtime mode is intended for watching; lockstep mode is better for
repeatable evaluation.

## 12. Common failure modes

**Both paddles move the same way on screen.** This can be correct under parameter
sharing, but verify the horizontal mirror. Equal actions should have equal meanings in
each paddle's local frame.

**Every point ends immediately.** Check goal collision masks, spawn positions, and ball
launch direction. Use the random rollout before changing the learner.

**The paddle follows the ball while it moves away.** Gate the alignment reward by the
incoming direction, as the included example does.

**Training beats only the latest policy.** Add the opponent pool and evaluate against
several snapshots. A pool reduces forgetting; it does not replace a proper evaluation
suite.
