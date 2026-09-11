# DQN

Deep Q-Network learns a value for each discrete action and chooses with an
epsilon-greedy policy. It is an off-policy algorithm: collected transitions go into a
replay buffer and may be reused many times.

## When to use it

Use DQN when the action is one choice from a small finite set: move left/stay/right,
select a weapon, or choose one tactical command. Breakout and Pong paddles are natural
examples. It is easy to inspect and usually a better fit than encoding three discrete
actions as a continuous scalar.

Do not use it for continuous motor commands or a large Cartesian product of discrete
choices. Use PPO for native hybrid/multi-discrete policies, or SAC/TD3 for continuous
control.

## DQN flags

The complete CLI is this table plus [Shared training options](common-options.md).

| Flag | Meaning |
|---|---|
| `--learning-rate LR` | Q-network optimizer rate. Default: `1e-3`. |
| `--target-update-every N` | Completed episodes between target-network synchronisations. Default: `20`. |
| `--target-update-steps N` | Use accepted transitions instead of episodes for target-network synchronisation. `0` keeps the episode schedule. |
| `--replay-warmup N` | Transitions required before updates begin. Default: `500`. |
| `--replay-capacity N` | Replay capacity. Default: `100000`. |
| `--epsilon-start X` | Initial random-action probability. Default: `1.0`. |
| `--epsilon-min X` | Exploration floor. Default: `0.05`. |
| `--epsilon-decay X` | Explicit per-episode multiplier. If omitted, Metis derives it from the remaining episode budget. |
| `--epsilon-decay-horizon-fraction X` | Fraction of remaining episodes used by the derived schedule. Default: `0.5`. |
| `--epsilon-decay-transitions N` | Linearly anneal epsilon from its start to its floor over exactly `N` accepted transitions. The alias is `--epsilon-decay-steps`; `0` keeps episode mode. |
| `--log-action-every N` | Episode interval for action-distribution diagnostics. Default: `1`; `0` disables them. |
| `--weights-path PATH` | Legacy/final Q-network weights output. Default: `generic_dqn_weights.weights.h5`. |

DQN also accepts the replay and demonstration flags in the shared page. Demonstration
BC trains the discrete policy from recorded actions before Q-learning; replay prefill
provides complete transitions.

The episode and transition epsilon schedules are mutually exclusive. Transition mode
is preferable when episode lengths change during learning or when two implementations
must receive the same exploration budget. Its transition counter and schedule
parameters are stored in the TensorFlow checkpoint; resume rejects a missing or
different schedule instead of restarting the anneal.

## Example

```bash
python/.venv/bin/python python/train.py \
  --algorithm dqn \
  --godot-project godot \
  --godot-scene res://scenarios/breakout/breakout_scenario.tscn \
  --num-envs 8 \
  --num-episodes 2500 \
  --max-steps-per-episode 0 \
  --replay-warmup 10000 \
  --replay-capacity 200000 \
  --epsilon-start 1.0 --epsilon-min 0.05 \
  --collector-mode async \
  --checkpoint-dir checkpoints/breakout_dqn \
  --headless --dashboard
```

Watch the frozen success rate, epsilon, replay occupancy, and action histogram. A
reward increase with one action taking almost every decision can indicate reward
exploitation rather than useful control.
