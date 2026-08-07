# DDPG

Deep Deterministic Policy Gradient is an off-policy continuous-control actor-critic.
The actor is deterministic; exploration is added externally to its action.

## When to use it

Use DDPG as a compact baseline, for comparisons with older continuous-control work,
or when deterministic actions and implementation simplicity are central. It can be
sample-efficient on smooth, well-shaped tasks.

DDPG is sensitive to critic overestimation and hyperparameters. For a new
deterministic-control task, TD3 is usually the safer first experiment. SAC is often
better when broad stochastic exploration is needed.

## Flags

DDPG adds no variant-only flags. Its complete CLI consists of:

- all [Shared training options](common-options.md);
- every option under **Deterministic continuous family** on that page;
- the common replay and demonstration options on that page.

Important first choices are `--exploration-noise`, `--exploration-noise-kind`,
`--replay-warmup`, actor/critic learning rates, and `--action-smoothing`. Disable
`--actor-drive-regularization` for arbitrary robot-joint actions.

## Example

```bash
python/.venv/bin/python python/train.py \
  --algorithm ddpg \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --num-envs 8 --num-episodes 5000 \
  --replay-warmup 10000 --replay-capacity 200000 \
  --exploration-noise 0.20 --exploration-noise-min 0.02 \
  --collector-mode async \
  --checkpoint-dir checkpoints/cars_ddpg \
  --headless --dashboard
```

Track actor output saturation, critic scale, collisions, replay age, and deterministic
frozen evaluation. Rising training reward under noisy actions does not guarantee that
the deterministic actor is improving.

