# Tanks: hybrid actions and multi-agent self-play

This scenario puts two red tanks against two blue tanks in an arena they do not know in
advance. A tank must move through local geometry, identify enemies in its field of view,
and decide when to fire. It never receives the map or an opponent's world position.

The example combines continuous movement with a discrete weapon command, making it a
compact introduction to Metis hybrid action spaces.

Working files:

- [`tank.tscn`](../../godot/agents/Tank/tank.tscn)
- [`tank.gd`](../../godot/agents/Tank/tank.gd)
- [`tank_missile.tscn`](../../godot/scenarios/tanks/tank_missile.tscn)
- [`tanks_scenario.tscn`](../../godot/scenarios/tanks/tanks_scenario.tscn)
- [`tanks_scenario.gd`](../../godot/scenarios/tanks/tanks_scenario.gd)

## 1. Define the task

Each tank should learn to:

- drive without colliding with walls;
- search using only local observations;
- distinguish allies from enemies;
- fire when a useful shot is available;
- preserve its own health;
- cooperate through a shared team outcome.

The policy should not receive the active map index, absolute coordinates, or direct
enemy transforms. Those shortcuts make training easier but produce a map-specific
controller.

The baseline gives all four tanks one shared policy. Team ID is used by the scenario
and opponent pool, while mirrored or body-local observations give the network a common
spatial convention. This parameter-sharing setup is the best first validation run.
Metis can also assign one independent policy per team later in the tutorial.

## 2. Declare the hybrid action space

The action contains two components:

```text
movement: Continuous(2)
weapon:   Discrete(2)
```

`movement[0]` is throttle in `[-1, 1]`, and `movement[1]` is steering in `[-1, 1]`.
The weapon IDs are:

| ID | Name | Method |
|---:|---|---|
| `0` | `hold_fire` | `hold_fire()` |
| `1` | `fire` | `request_fire()` |

Build it in the Inspector:

```text
Agent
└── ActionSpace
    ├── Movement (ContinuousAction, size = 2)
    └── Weapon (DiscreteActionSet)
        ├── HoldFire (DiscreteAction)
        └── Fire (DiscreteAction)
```

The body decodes the continuous part and dispatches the discrete part through `Agent`:

```gdscript
func apply_action(action: Variant) -> Variant:
	clear_inputs()
	if not _alive or typeof(action) != TYPE_DICTIONARY:
		return {"movement": [0.0, 0.0], "weapon": 0}

	var movement := agent.decode_continuous_action(action)
	_throttle_input = clampf(float(movement[0]), -1.0, 1.0)
	_rotation_input = clampf(float(movement[1]), -1.0, 1.0)

	var weapon_action := int(action.get("weapon", 0))
	agent.act_discrete(weapon_action, "weapon")
	return {
		"movement": [_throttle_input, _rotation_input],
		"weapon": weapon_action,
	}
```

PPO is the built-in trainer for hybrid spaces. DQN handles a single discrete set, while
SAC, TD3, and DDPG variants handle continuous spaces.

## 3. Design local observations

The included tank combines body state with two sensor groups:

| Group | Values | Purpose |
|---|---|---|
| body speed | normalized velocity | understand current motion |
| previous controls | throttle and steering | reduce control ambiguity |
| health | `[0, 1]` | trade aggression against survival |
| reload ready | `0/1` | avoid firing during cooldown |
| alive | `0/1` | stable state for shared batches |
| obstacle rays | normalized hit distances | navigate walls |
| team vision rays | enemy and ally signals | find targets without coordinates |

The tree is:

```text
Agent
└── ObservationSystem
    ├── BodySpeed (BodySpeedObservationSource)
    ├── Throttle (MethodObservationSource)
    ├── Steering (MethodObservationSource)
    ├── Health (MethodObservationSource)
    ├── ReloadReady (MethodObservationSource)
    ├── Alive (MethodObservationSource)
    ├── Obstacles (RaycastObservationSource)
    └── TeamVision (TeamRaycastObservationSource)
```

`RaycastObservationSource` reports normalized distance to geometry. `TeamRaycastObservationSource`
uses the observed body's `team_id` to fill separate ally and enemy channels. In the
example, three forward vision rays produce `enemy_left`, `enemy_center`,
`enemy_right` and the corresponding ally channels.

Ray length limits perception. A target beyond a wall is invisible because the wall is
the first collider. Make sure obstacle layers and body layers are configured so the
first hit has the same meaning in every map.

## 4. Keep observations independent of the layout

Three arena scenes are supplied:

- [`easy_map.tscn`](../../godot/scenarios/tanks/easy_map.tscn)
- [`medium_map.tscn`](../../godot/scenarios/tanks/medium_map.tscn)
- [`hard_map.tscn`](../../godot/scenarios/tanks/hard_map.tscn)

The policy receives none of their names or indices. It sees only local rays and its own
state. This is what allows the same model to work on layouts that were not present in
the training set.

Instantiate exactly one layout under `Environment/LayoutContainer`. Hiding a scene does
not guarantee that its physics bodies stop colliding. Removing inactive layouts from
the scene tree prevents invisible walls from surviving a visual switch.

The scenario chooses layouts by seeded curriculum:

| Training episode | Available layouts |
|---:|---|
| `0-499` | easy |
| `500-1499` | easy and medium |
| `1500+` | easy, medium, and hard |

## 5. Configure local rewards

The tank's own `RewardSystem` currently contains:

| Term | Value | Reason |
|---|---:|---|
| time | `-0.0005` per step | discourage passive matches |
| invalid fire | `-0.01` | discourage cooldown spam |

`request_fire()` emits `invalid_fire` only when the shot cannot be created. The weapon
still needs a cooldown in game logic; a penalty alone should not enforce fire rate.

Keep the local shaping small. Combat outcomes are assigned by the scenario because they
involve more than one agent.

## 6. Model missiles and damage

A missile records the tank that fired it, moves independently, and reports damage to
the victim. At minimum it needs:

- an owner reference;
- speed and lifetime;
- a collision layer that hits tanks and walls;
- protection from immediately striking its owner;
- one-shot damage handling before it is freed.

The tank emits:

```gdscript
signal damage_received(victim: BattleTank, attacker: BattleTank, amount: float)
signal destroyed(victim: BattleTank, killer: BattleTank)
```

These signals keep combat mechanics in the tank and credit assignment in the scenario.

## 7. Build the scenario event graph

Use this Metis structure:

```text
TanksScenario
├── BridgeServer
├── ScenarioController
│   ├── ScenarioEventSystem
│   │   ├── EnemyDamage
│   │   ├── EnemyKill
│   │   ├── TeamKillAssist
│   │   ├── DamageTaken
│   │   ├── Destroyed
│   │   ├── TeamWon
│   │   ├── TeamLost
│   │   └── MatchDraw
│   └── ScenarioRewardSystem
├── Environment
│   └── LayoutContainer
└── Agents
    ├── RedTank1
    ├── RedTank2
    ├── BlueTank1
    └── BlueTank2
```

Add all four tank paths to `ScenarioController.controlled_agents`. Set red `team_id` to
`0` and blue `team_id` to `1`.

The included rewards are:

| Event | Recipient | Reward |
|---|---|---:|
| enemy damage | attacker | `+0.25 * normalized_damage` |
| enemy kill | killer | `+1.0` |
| team kill assist | killer's team | `+0.2` |
| damage taken | victim | `-0.15 * normalized_damage` |
| destroyed | victim | `-1.0` |
| team won | winning team | `+3.0` |
| team lost | losing team | `-3.0` |
| draw | everyone | `-0.25` |

Damage is normalized by maximum health before it becomes an event amount. This keeps
the scale stable if health or missile damage changes.

The match ends when one team has no living tanks. Every agent receives a terminal
reason in that same scenario step. A destroyed tank can remain in the controlled list
with collisions, visibility, and movement disabled until the shared reset.

## 8. Reset the full match atomically

On reset:

1. clear every live projectile;
2. replace the current layout;
3. restore tank transforms, health, cooldown, visibility, and collision masks;
4. zero movement inputs and velocity;
5. reset observation sources and reward state.

The included `ScenarioController` also applies a small position and yaw jitter. That
variation is useful as long as it cannot spawn a tank inside a wall or with an immediate
line of fire.

## 9. Validate with random actions

Before spending time on PPO, inspect a rollout:

```bash
python/.venv/bin/python python/tools/random_rollout.py \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/tanks/tanks_scenario.tscn \
  --multi-agent \
  --steps 500 \
  --print-reward-terms \
  --no-headless
```

Verify all of the following:

1. The action type is `hybrid`, with `movement: 2` and `weapon: 2`.
2. Four agents and two teams appear in the spec.
3. Walls shorten obstacle rays and block team vision.
4. Allies fill ally channels and enemies fill enemy channels.
5. Fire cooldown and invalid-fire penalties agree.
6. Damage and kill credit go to the intended agents.
7. All agents terminate together when a team is eliminated.
8. A timeout is truncated, not counted as a win.

## 10. Train the shared policy

Start with simultaneous self-play:

```bash
python/.venv/bin/python python/train.py \
  --algorithm ppo \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/tanks/tanks_scenario.tscn \
  --num-envs 4 \
  --num-episodes 5000 \
  --max-steps-per-episode 1000 \
  --physics-frames-per-step 1 \
  --batch-size 256 \
  --ppo-epochs 4 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-ratio 0.2 \
  --learning-rate 3e-4 \
  --entropy-coef 0.01 \
  --checkpoint-dir checkpoints/tank_battle_ppo_v1 \
  --multi-agent \
  --collector-mode async \
  --headless
```

PPO is on-policy and has no replay buffer. Do not change the action contract or agent
population halfway through a run and call it an equivalent resume; both alter the
trajectory distribution.

Keep one physics frame per action until projectile speed, cooldown, and sensor timing
are validated. A larger frame skip can let a fast missile cross a collider or leave too
few steering decisions near walls.

### Optional SB3 comparison

Metis is the default backend for this task. The limited SB3 adapter can run a
comparison because Tanks terminates every combatant when the match ends:

```bash
python/.venv-sb3/bin/python python/train.py \
  --backend sb3 \
  --algorithm ppo \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/tanks/tanks_scenario.tscn \
  --num-envs 4 \
  --total-timesteps 500000 \
  --max-steps-per-episode 1000 \
  --batch-size 256 \
  --ppo-epochs 4 \
  --device cpu \
  --multi-agent \
  --collector-mode sync \
  --evaluation-episodes 40 \
  --checkpoint-dir checkpoints/tank_battle_sb3_ppo_v1 \
  --headless
```

This is not the same policy distribution as native Metis PPO. SB3 receives a latent
continuous `Box`: the two movement values remain continuous, while the two weapon
choices are represented by two logits and decoded with `argmax`. Startup prints
`sb3_action=hybrid_box(...)` to make that distinction visible.

The example keeps PPO on CPU because this small MLP usually gains little from GPU
execution and SB3 itself warns about the extra overhead. Benchmark both devices before
changing it for a larger policy.

SB3 does not support Metis's asynchronous collector or historical opponent pool. It
does provide current-policy simultaneous self-play here because all four lanes share
the same policy. Keep the default `--sb3-multi-agent-partial-done error`; a partial
death should be handled by the match rules rather than silently resetting the world.

The SB3 run evaluates its `.zip` model through `--evaluation-episodes`. The current
`python/run.py` loads native Metis/Keras artifacts only.

## 11. Add an opponent pool

Once the current policy can navigate, hit targets, and finish matches against itself,
train against historical snapshots:

```bash
python/.venv/bin/python python/train.py \
  --algorithm ppo \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/tanks/tanks_scenario.tscn \
  --num-envs 4 \
  --num-episodes 8000 \
  --max-steps-per-episode 1000 \
  --policy-path checkpoints/tank_battle_ppo_v1 \
  --checkpoint-dir checkpoints/tank_battle_ppo_pool_v1 \
  --multi-agent \
  --collector-mode sync \
  --opponent-pool \
  --opponent-snapshot-every 100 \
  --opponent-pool-size 12 \
  --opponent-current-probability 0.2 \
  --headless
```

PPO currently requires the synchronous collector when historical opponent sampling is
enabled. `--policy-path` warm-starts a new run from the baseline policy. It does not
restore the previous optimizer or create a continuation of the old checkpoint series.

An opponent pool reduces strategic forgetting, but it is not independent multi-policy
training. There is still one learner and one shared model artifact.

### Optional independent team policies

To train red and blue as two live, independent learners, replace the opponent-pool
flags with:

```text
--multi-agent
--multi-policy
--policy-assignment team
--collector-mode async
```

Metis creates `team_0` and `team_1`, each with its own PPO model and optimizer, and
saves both in one atomic checkpoint. Async rollout generations keep the complete
policy group frozen until every environment has finished, then update each model only
from its assigned agents. This mode cannot currently be combined with the historical
opponent pool. It is useful when teams have genuinely different roles;
for a symmetric battle, compare it against parameter sharing rather than assuming
that two networks will learn faster.

## 12. Evaluate and run

Run the native Metis policy in realtime:

```bash
python/.venv/bin/python python/run.py \
  --policy-path checkpoints/tank_battle_ppo_v1 \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/tanks/tanks_scenario.tscn \
  --multi-agent \
  --episodes 20 \
  --execution-mode realtime \
  --no-headless
```

For repeatable evaluation, switch to lockstep and run several fixed seeds. Track at
least:

- win rate by starting side and map;
- damage dealt and received;
- hit rate and invalid-fire rate;
- survival time;
- collisions and distance travelled;
- performance against several historical snapshots.

A rising training reward is not enough. A tank may exploit one opponent behavior or
one layout while becoming less robust elsewhere.

## 13. Sensible next steps

After the baseline works, add complexity one change at a time:

- unseen evaluation layouts;
- randomized spawn formations;
- friendly-fire mechanics and penalties;
- limited ammunition;
- separate turret rotation as another continuous action;
- role observations for larger teams;
- recurrent policies when partial observability becomes the main bottleneck.

Keep the local observation contract stable across maps. The point of this scenario is
to learn behavior from perception, not to memorize a level identifier.
