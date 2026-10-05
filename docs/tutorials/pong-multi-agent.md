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
- [`pong_ball.gd`](../../godot/scenarios/pong/pong_ball.gd)

This guide explains how that example is assembled and why each part exists. There are
no demonstrations and no curriculum: both paddles learn from scratch against each other.

### If you trained an earlier version

Self-play checkpoints swing in quality. Every configuration measured for this page
reached near-perfect defense and then lost part of it a few hundred points later
(greedy self-play, 40 points per row, a return being any ball sent back):

| Run | Checkpoint | Balls returned | Returns per point (median) |
|---|---:|---:|---:|
| earlier scene, earlier command | 1300 | 79% | 2 |
| earlier scene, earlier command | 1900 | 99.9% | 33 |
| current scene and command | 950 | 97% | 22 |
| current scene and command | 2100 | 99% | 36 |
| current scene and command | 3225 | 89% | 5 |

The earlier-scene policies were measured with corrected physics at their final ball
speed of 460 pixels per second, and checkpoint 1900 scored the same on the earlier scene
itself. The others were measured in the game as it was then, with the ball accelerating
to 700; section 6 explains why the cap is now 1200.

The earlier instructions turned that swing into paddles that play badly in two ways.
`run.py` loaded the *latest* checkpoint, wherever the swing happened to be. And `best/`,
selected with the default `--best-metric auto`, stayed on checkpoint 1300 for the rest
of the run. Later checkpoints defended well enough for their evaluation points to reach
the step limit, and a point that reaches the limit counts as a failure for both paddles.
With `--best-metric reward_mean`, as in section 8, `best/` kept checkpoint 2100 while the
latest one fell to 89%.

The earlier scene also had two physics defects, described in section 5: a ball trapped
against a paddle was paid on every contact, and paddles could be pushed sideways. They
did not show up in the measurements above. They did once the paddles could travel all
the way to the walls: a policy exploited the first one within 300 points, and that
checkpoint became `best/`.

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

Metis can also train two genuinely independent policies, one per side. That is useful
for experiments with asymmetric roles, but it is not the recommended first Pong run:
the shared policy reuses twice as much relevant experience and avoids one side becoming
temporarily much stronger than the other.

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
| `paddle_y` | 1 | vertical position, -1 at the top of the paddle's travel and +1 at the bottom |
| `ball_relative_position` | 2 | ball position relative to this paddle, divided by the field size |
| `ball_velocity` | 2 | ball velocity divided by the maximum ball speed |
| `opponent_delta_y` | 1 | opponent height relative to this paddle |

The right paddle sees a mirrored horizontal frame. Its own side is still "near" and
the opponent's side is still "far" from the policy's point of view. Without this mirror,
one network would have to learn two coordinate conventions for the same task.

In [`pong_paddle.gd`](../../godot/agents/PongPaddle/pong_paddle.gd), the mirror is the
`view_direction` export:

```gdscript
func get_ball_relative_position_observation() -> Vector2:
	if ball == null:
		return Vector2.ZERO
	var relative := ball.global_position - global_position
	relative.x *= view_direction
	return Vector2(
		clampf(relative.x / maxf(observation_scale.x, 0.001), -1.0, 1.0),
		clampf(relative.y / maxf(observation_scale.y, 0.001), -1.0, 1.0)
	)
```

Set `view_direction = 1` on the left paddle and `-1` on the right paddle. The team ID
does not need to be part of the observation because the spatial frame is already
canonical.

### Scale observations by the field, not by a guess

The clamp is a safety net. It must almost never be reached, because a clamped value
throws information away: every ball position beyond the limit looks the same.

An earlier version of this example divided the horizontal distance by 640 pixels and
the vertical distance by the paddle's travel. With 1194 pixels between the paddles, the
ball read `1.0` for the whole opponent half of the field, and a paddle near one wall saw
the entire other half of the field as `1.0` as well. Measured over random and scripted
rallies, **50-59% of all decisions had at least one saturated ball value**. The paddles
still learned to defend with those scales, so this was not what broke training, but
every clamped value is information the policy never gets.

The scenario now measures the play field from the walls and goals at startup and passes
the result to both paddles:

```gdscript
paddle.vertical_center = _field.get_center().y
paddle.vertical_limit = maxf(1.0, _field.size.y * 0.5 - half - 1.0)
paddle.observation_scale = _field.size
paddle.ball_speed_scale = max_ball_speed
```

With these scales no ball value saturates during play. If you resize the arena, nothing
needs to be retuned by hand. `auto_calibrate_from_geometry` on the scenario turns this
off, which is only useful to reproduce the old behaviour.

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

A paddle is driven either by the policy or by the keyboard. `manual_control` reads the
`move_up` and `move_down` input actions; the paddle registers them (W/S and the arrow
keys) at startup when the project's input map does not define them.

## 4. Add local shaping carefully

The only local term is `AlignmentReward`: `0.05 * delta_alignment` while the ball
approaches.

The paddle predicts where the ball will cross its contact line, folding the path on the
top and bottom walls, and measures how far it is from that point. The reward is the
reduction of that gap since the previous decision. It is disabled while the ball moves
away, so the paddle is not paid forever for standing near the ball.

Two details matter:

- **The target is the intercept, not the ball's current height.** Chasing the current
  height sends the paddle the wrong way whenever a steep shot is about to bounce off a
  wall.
- **Half a paddle counts as aligned.** The bounce angle depends on where the ball
  touches the paddle, and edge hits produce the steep, hard-to-return shots. Paying for
  a perfectly centred paddle would push every return towards the flattest angle.
  `alignment_deadzone_ratio` sets that tolerance.

Outcome rewards remain much larger:

- successful return: `+0.02`;
- point won: `+1.0`;
- point lost: `-1.0`.

That scale keeps winning the point as the real objective. If alignment dominates,
paddles can learn attractive movement that does not improve the score.

There is no time penalty. In a symmetric zero-sum game it costs both players the same
and carries no information, and since a point ends when someone loses, it would in
principle reward conceding sooner.

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

### End the point in the same tick

An `Area2D` reports `body_entered` at the start of the following physics tick. Used as
the goal detector, that ends every point one decision late, with the ball already inside
the goal in the last observation. Worse, if an episode is truncated in the same step in
which the ball enters a goal, the queued signal arrives after the reset and ends the new
point at its first step.

The ball therefore checks the goal lines itself and emits `goal_line_crossed` in the tick
in which it reaches one. The goal areas remain as a fallback and ignore a signal when
the ball is no longer at the goal line.

### Pay for returns, not contacts

The rounded end of a paddle and a wall can squeeze the ball between them. Squeezed there,
the ball touches the paddle on every physics tick. `move_and_collide()` also discards the
motion left after a contact, so a ball that collides at the start of every tick barely
moves and stays trapped.

An earlier version paid `+0.02` for every contact. Once the paddles could travel all the
way to the walls, a shared policy found this within 300 points. In one evaluation point a
paddle collected 2585 contacts. The point ran into the time limit instead of being lost,
and that checkpoint was selected as the best one.

The ball now spends the rest of its motion on the new heading after each contact. It
emits `paddle_hit` only when a paddle actually returns it, meaning the ball was moving
towards that paddle before the bounce. Only returns pay the rally reward and speed the
ball up.

### Keep the paddle on its line

`move_and_slide()` with a purely vertical velocity still slides along the round ball, and
that pushes the paddle sideways. In one evaluation a right paddle ended a point at
`x = 127` instead of `x = 1249`, and the next serve started next to the left goal. The
paddle pins its `x` to its spawn position after every move, and the serve position is
computed from the spawn positions rather than from wherever the paddles are.

## 6. Reset, serve, and rally speed

At reset, the ball is placed at the midpoint between the paddles with a random vertical
offset, a random serve side, and a random vertical component. The included scenario
seeds its own `RandomNumberGenerator`, so a fixed episode seed remains reproducible.

The ball also uses an arcade bounce. The outgoing vertical direction depends on where
it touched the paddle, with a small minimum angle to prevent endless horizontal or
wall-to-wall loops.

Every point starts at the same speed and gets faster with each paddle hit, up to a cap:

| Scenario export | Value | Meaning |
|---|---:|---|
| `ball_start_speed` | `300` | serve speed, in pixels per second |
| `hit_speed_gain` | `1.07` | multiplier at every paddle hit, about 21 hits to reach the cap |
| `max_ball_speed` | `1200` | cap |

This is a rule of the game, not a curriculum: it does not depend on the training
episode, so it behaves the same in training, evaluation and `run.py`, in sync and async
collection. Early in a rally the ball is slow and easy; a long rally demands anticipation.

### Choosing the cap

The cap decides what a good paddle has to learn. Two scripted players, deciding every 3
physics frames, show it: one moves to the predicted intercept and goes back to the middle
while the ball is away, the other parks at the top of its travel. Both faced a scripted
opponent that aims each return as far as possible from where they stand:

| Constant ball speed | Points lost by the parked paddle | Points lost by the centred paddle |
|---:|---:|---:|
| 700 | 0 / 20 | 0 / 20 |
| 1000 | 7 / 20 | 0 / 20 |
| 1200 | 19 / 20 | 0 / 20 |
| 1400 | 17 / 20 | 0 / 20 |
| 1600 | 17 / 20 | 7 / 20 |

With a 700 cap, parking at one end costs nothing, so a policy has no reason to come back to
the middle, and it doesn't: a policy trained with that cap kept its paddle within the last
10% of its travel in 77% of the decisions taken while the ball was moving away. At
1200 parking loses almost every point while a paddle that waits in the middle loses none.
In the real game, starting at 300 and accelerating by 1.07 per return, the same pair gave
19 lost points in 20 for the parked paddle, and 20 rallies out of 20 reaching the 90-second
limit for the centred one. The game stays winnable for a good player, so a strong policy can
rally forever, which is what you want to see when you watch it.

Retraining with the 1200 cap and the command of section 8 reduced parking without removing
it. Parking is measured as decisions taken with the ball moving away and the paddle in the
last 10% of its travel:

| Cap | Checkpoint | Balls returned | Returns per point (median) | Parked |
|---:|---|---:|---:|---:|
| 700 | 2100, `best/` | 99% | 36 | 77% |
| 1200 | 1600, `best/` | 96% | 24.5 | 36% |
| 1200 | 4000, latest | 92% | 13 | 41% |

The two caps are different games, so the return rates are not directly comparable: with
the 700 cap 27 of 40 rallies reached the 90-second limit, with 1200 none did. Parking costs
a point only once the ball is fast and the opponent aims away from the paddle, so the
pressure to wait in the middle grows as the policy learns to aim.

`ball_speed_scale` on the paddles follows `max_ball_speed`, so the observed velocity stays
inside `[-1, 1]` at any speed. Change the cap and the observation follows.

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
7. No ball value in the observation sits at `±1` during a rally.
8. Neither paddle leaves its horizontal line, and every serve starts from the middle.
9. No single point pays a paddle much more than `+1`. A larger total means contacts are
   being paid, not returns.

Two things about the tool's output are easy to misread. In multi-agent mode the printed
`reward` is the mean over the agents, so a goal prints about zero: `+1` and `-1` cancel.
Read the per-agent `scenario_rewards` instead. The rollout also stops at the first
point, and it does not send the trainer's configuration, so it runs at one physics frame
per decision with the scene's own `max_steps`.

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
  --num-episodes 4000 \
  --max-steps-per-episode 1800 \
  --physics-frames-per-step 3 \
  --batch-size 128 \
  --replay-warmup 8000 \
  --epsilon-start 1.0 \
  --epsilon-min 0.05 \
  --epsilon-decay-transitions 200000 \
  --target-update-steps 8000 \
  --critic-loss huber \
  --grad-clip-norm 10 \
  --best-metric reward_mean \
  --checkpoint-dir checkpoints/pong_shared_dqn_v1 \
  --weights-path checkpoints/pong_shared_dqn_v1/final.weights.h5 \
  --multi-agent \
  --collector-mode async \
  --dashboard \
  --headless
```

Several of those settings differ from the defaults, and each one matters.

**`--weights-path`** keeps the final weights inside the run's directory. Without it they
are written to `generic_dqn_weights.weights.h5` in the directory you launched from, where
`run.py --load-from auto` can later pick them up instead of the checkpoint you meant.

**`--physics-frames-per-step 3`: 20 decisions per second.** At one frame per decision
the paddle moves 7 pixels per action, and a ball crossing lasts 150-300 decisions,
beyond the roughly 100-step horizon of `gamma = 0.99`. Three frames per decision move
the paddle 21 pixels per action, bring a crossing within that horizon, and collect a
third of the transitions for the same amount of play. It is the same value the Breakout
example uses. One frame per decision also learns once the physics is right; three is
the cheaper and better-conditioned choice, not a requirement.

**`--max-steps-per-episode 1800` is a truncation, not a rule of the game.** Python sends
this value to the scene before every episode, overriding `ScenarioController.max_steps`;
`0` disables the limit completely. Without a limit, nothing bounds a point between two
good paddles except exploration noise: a greedy pair rallies for ever, and everything
counted in episodes slows down while it does. When the limit is hit, the episode is *truncated*, and DQN bootstraps
from the next state instead of treating it as the end of the game. The policy therefore
learns the value of a game that goes on for ever; the limit only gives the collector a
chance to reset. At 3 frames per decision, 1800 decisions are 90 seconds of play.

**`--epsilon-decay-transitions` and `--target-update-steps` count transitions.**
`--epsilon-decay` and `--target-update-every` count episodes, and a Pong episode is one
point. Points grow from a few seconds to minutes as the paddles improve, so an
episode-based schedule slows down exactly when learning speeds up. Counting transitions
pins exploration and target refreshes to the data actually collected. Note that each
decision of each paddle is one transition, so an environment step adds two.

**`--critic-loss huber` and `--grad-clip-norm 10`** keep one large temporal-difference
error from dominating an update. Rewards here are within `±1` per step, which is the
setting Huber loss was designed for.

**`--best-metric reward_mean`** picks the checkpoint to keep. The default, `auto`, ranks
by success rate first. In self-play one of the two paddles wins every decided point, so
the success rate is 50% by construction, and a point that runs into the evaluation's step
limit counts as two failures. With `auto`, a policy that defends well enough to reach
that limit ranks *below* a weaker one. `reward_mean` averages both paddles: `+1` and `-1`
cancel, and what remains grows with the number of returns per point.

**`--collector-mode async`** is also the mode whose episode count means what you expect.
In sync mode one "episode" is one round of all environments, so with `--num-envs 4`
every episode-based count, including `--num-episodes`, is four times larger in points.

### Optional independent policies

To give each side its own learner, either set a distinct `policy_id` on each `Agent`
node or assign policies from the existing teams. The latter needs no scene change:

```bash
python/.venv/bin/python python/train.py \
  --algorithm dqn \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/pong/pong_scenario.tscn \
  --num-envs 4 \
  --num-episodes 4000 \
  --max-steps-per-episode 1800 \
  --physics-frames-per-step 3 \
  --batch-size 128 \
  --replay-warmup 8000 \
  --critic-loss huber \
  --grad-clip-norm 10 \
  --checkpoint-dir checkpoints/pong_independent_dqn_v1 \
  --multi-agent \
  --multi-policy \
  --policy-assignment team \
  --collector-mode async \
  --headless
```

`--target-update-steps` is refused with `--multi-policy`, where the learners still sync
their target networks per episode, so this command keeps the episode-based default.

The resolved policy IDs are `team_0` and `team_1`. Each gets its own Q-network,
target network, optimizer, epsilon, and replay buffer. They are saved together, not
selected as competing checkpoint candidates. Async workers receive both networks in
one versioned snapshot, and each transition is routed back to the matching replay
buffer. Run the pair with:

```bash
python/.venv/bin/python python/run.py \
  --checkpoint-dir checkpoints/pong_independent_dqn_v1 \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/pong/pong_scenario.tscn \
  --multi-agent \
  --multi-policy \
  --physics-frames-per-step 3 \
  --max-steps 0 --no-time-limit \
  --episodes 20 \
  --execution-mode realtime \
  --no-headless
```

Use this as a comparison against the shared-policy baseline. A fair comparison keeps
the environment seeds and transition budget fixed; counting only episodes gives the
two setups different amounts of learner data.

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
  --max-steps-per-episode 1800 \
  --physics-frames-per-step 3 \
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

The SB3 run evaluates its `.zip` model through `--evaluation-episodes`. The current
`python/run.py` loads Metis/Keras artifacts only.

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
  --num-episodes 6000 \
  --max-steps-per-episode 1800 \
  --physics-frames-per-step 3 \
  --batch-size 128 \
  --replay-warmup 8000 \
  --epsilon-min 0.05 \
  --epsilon-decay-transitions 200000 \
  --target-update-steps 8000 \
  --critic-loss huber \
  --grad-clip-norm 10 \
  --best-metric reward_mean \
  --checkpoint-dir checkpoints/pong_shared_dqn_v1 \
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

Keep every task setting identical to the first run, `--physics-frames-per-step` above
all: a policy resumed at a different decision rate is solving a different problem.

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
invalidates the main model. So does changing the observation scales or the maximum ball
speed, because they change what each observed number means.

## 10. Read the numbers correctly

Self-play hides progress from the usual metrics, because both sides improve together.

- **Reward is about zero by construction.** Every decided point pays `+1` to one paddle
  and `-1` to the other. What is left is the shaping and the `+0.02` per return.
- **Success rate is 50% by construction**, since one of the two paddles wins every
  decided point. The training log even shows it as 0%, because the DQN trainer's own list
  of success events does not include `point_won`; the frozen evaluation does.
- **Returns per point is the signal.** Two random paddles return fewer than one ball
  per point. As the shared policy learns to defend, rallies get longer, the ball gets faster,
  and more points reach the time limit.

Useful signals are:

- returns per point, and the fraction of points that end in a truncation;
- how often a paddle waits at one end of its travel while the ball moves away (section 6);
- point duration;
- replay size and update rate;
- action balance between idle, up, and down;
- opponent mode such as `current` or `snapshot:500`.

A longer rally is not automatically better. A policy can prolong play without learning
to win. To see both offense and defense, evaluate against fixed snapshots, or play
against the policy yourself.

## 11. Run the trained policy

Run the best checkpoint with:

```bash
python/.venv/bin/python python/run.py \
  --algorithm dqn \
  --checkpoint-dir checkpoints/pong_shared_dqn_v1/best \
  --load-from checkpoint \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/pong/pong_scenario.tscn \
  --multi-agent \
  --physics-frames-per-step 3 \
  --max-steps 0 --no-time-limit \
  --infinite \
  --epsilon 0.0 \
  --execution-mode realtime \
  --no-headless
```

Two of those arguments are easy to forget, and both make a good policy look broken.

**`--physics-frames-per-step 3` must match training.** `run.py` defaults to 1. Without
it the policy acts three times as often as it learned to, and every action moves the
paddle a third of the distance it expects.

**`--max-steps 0 --no-time-limit` lets the game go on.** `run.py` otherwise stops every
episode after 1000 decisions. Training truncated long points only so that its collectors
could reset; in `run.py` a strong policy is expected to keep the rally going.

`best/` holds the checkpoint chosen by `--best-metric`. The checkpoint directory itself
also holds a portable `policy.keras`, but of the *latest* checkpoint, which self-play can
leave well below the best one. Realtime mode is intended for watching; lockstep mode is
better for repeatable evaluation.

### Export the policy

`best/` holds TensorFlow checkpoints. Turn the best one into a portable Keras bundle, then
into TensorFlow Lite:

```bash
python/.venv/bin/python python/run.py \
  --algorithm dqn \
  --checkpoint-dir checkpoints/pong_shared_dqn_v1/best \
  --load-from checkpoint \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/pong/pong_scenario.tscn \
  --multi-agent \
  --headless \
  --export-policy-dir checkpoints/pong_shared_dqn_v1/export \
  --export-policy-only

python/.venv/bin/python python/export.py \
  --policy checkpoints/pong_shared_dqn_v1/export \
  --format tflite \
  --tflite-quantization none
```

The first command starts Godot only to read the scenario's contract; it runs no episode.
The export directory then holds `policy.keras`, `policy.tflite` and the `policy.json`
manifest. The TFLite model takes one observation of shape `[1, 6]` in `float32` and returns
the three Q-values for `idle`, `move_up` and `move_down`; the action is the largest one.
Unquantized, it matched the Keras model to within `1e-5` and chose the same action on
10,000 random observations.

`run.py` loads Keras policies only. Watch the exported one with `--policy-path`, which
reads the algorithm from the manifest:

```bash
python/.venv/bin/python python/run.py \
  --policy-path checkpoints/pong_shared_dqn_v1/export \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/pong/pong_scenario.tscn \
  --multi-agent \
  --physics-frames-per-step 3 \
  --max-steps 0 --no-time-limit \
  --infinite \
  --epsilon 0.0 \
  --execution-mode realtime \
  --no-headless
```

The TFLite file is for running the policy outside Metis, for example on an embedded
device. Whatever runs it has to rebuild the same six observations, with the same field
scales and the same maximum ball speed, and decide every 3 physics frames.

## 12. Common failure modes

**Both paddles move the same way on screen.** This can be correct under parameter
sharing, but verify the horizontal mirror. Equal actions should have equal meanings in
each paddle's local frame.

**Every point ends immediately.** Check goal collision masks, spawn positions, and ball
launch direction. Use the random rollout before changing the learner.

**The paddle follows the ball while it moves away.** Gate the alignment reward by the
incoming direction, as the included example does.

**A policy that trained well plays badly in `run.py`.** Check that
`--physics-frames-per-step` is the same in training and `run.py`.

**Paddles miss easy balls after a long run.** Check which checkpoint you are running.
Self-play quality swings by 10-20 points of return rate within a few hundred points of
training, so run `best/`, and select it with a metric that can see defense.

**Episode counts barely move once the paddles defend.** Check `--max-steps-per-episode`.
With `0`, only exploration noise ends a rally between two good paddles.

**One paddle earns far more than `+1` in a point.** The ball is trapped against the paddle
and every contact is being paid. Pay only for returns, and keep the motion left after a
contact.

**Training beats only the latest policy.** Add the opponent pool and evaluate against
several snapshots. A pool reduces forgetting; it does not replace a proper evaluation
suite.
