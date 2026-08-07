# DDPG+BC

DDPG+BC combines deterministic Q-guided learning with an actor behavior-cloning loss.
Unlike a one-time BC pretrain, the imitation term can remain active and decay while
online reinforcement learning proceeds.

## When to use it

Use it when demonstrations are trustworthy but imperfect and the policy should refine
them online. It is useful for continuous robot or vehicle control where unguided DDPG
rarely reaches useful states.

Do not use it merely because a recorder exists. Inconsistent actions for similar
observations can pull the actor in incompatible directions. Validate a BC-only policy
with `--stop-after-bc` before spending time on RL.

## Flags

The complete CLI is [Shared training options](common-options.md), the **Deterministic
continuous family** section there, plus:

| Flag | Meaning |
|---|---|
| `--demo-bc-weight-start X` | BC loss weight at the first policy update. Default: `1.0`. |
| `--demo-bc-weight-end X` | BC weight after decay. Default: `0.05`. |
| `--demo-bc-decay-updates N` | Policy updates over which BC weight decays. Default: `100000`. |
| `--demo-q-weight-start X` | Initial Q-objective weight. Default: `0`, allowing BC-only actor updates first. |
| `--demo-q-weight-end X` | Final Q-objective weight. Default: `1`. |
| `--demo-q-weight-ramp-updates N` | Policy updates used to ramp the Q objective. Default: `50000`. |
| `--demo-q-filter` | Apply BC only where the critic rates the expert action above the current policy. Off by default. |

The Q filter controls the BC term; it does not make a poorly calibrated Q gradient
safe. Use critic warmup/audit and gradient telemetry before trusting Q-guided actor
updates on a narrow offline dataset.

## Example

```bash
python/.venv/bin/python python/train.py \
  --algorithm ddpg_bc \
  --godot-project godot --godot-scene res://my_task.tscn \
  --demo-path python/demos/my_task/train.npz \
  --demo-validation-path python/demos/my_task/val.npz \
  --demo-bc-epochs 30 --stop-after-bc \
  --bc-actor-weights-path checkpoints/my_ddpg_bc/actor_bc.weights.h5 \
  --checkpoint-dir checkpoints/my_ddpg_bc --headless
```

Remove `--stop-after-bc` only after the deterministic frozen evaluation is credible.

