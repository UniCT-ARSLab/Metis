# DDPGfD

DDPG from Demonstrations extends DDPG with demonstration pretraining and prioritized
replay that keeps expert transitions influential after online data begins to dominate
the buffer.

## When to use it

Use DDPGfD for continuous tasks with sparse success and a small set of high-quality,
complete demonstrations. It is appropriate when losing those transitions through
uniform replay dilution would be costly.

It is less attractive when demonstrations are noisy, come from a different physics or
reward contract, or lack valid `next_obs` and termination data. Use imitation-only BC
labels instead of pretending such labels are replay transitions.

## Flags

The complete CLI is [Shared training options](common-options.md), the **Deterministic
continuous family** section, the DDPG+BC schedule below, and the DDPGfD flags.

| Flag | Meaning |
|---|---|
| `--demo-bc-weight-start X` / `--demo-bc-weight-end X` | Initial and final online imitation weights. Defaults: `1.0` / `0.05`. |
| `--demo-bc-decay-updates N` | BC decay duration. Default: `100000` policy updates. |
| `--demo-q-weight-start X` / `--demo-q-weight-end X` | Initial and final Q-objective weights. Defaults: `0` / `1`. |
| `--demo-q-weight-ramp-updates N` | Q-objective ramp. Default: `50000`. |
| `--demo-q-filter` | Clone only expert actions preferred by the current critic. Off by default. |
| `--ddpgfd-pretrain-updates N` | Learner updates performed from demonstrations before online collection dominates. Default: `1000`. |
| `--ddpgfd-priority-alpha X` | Strength of priority-based sampling. Default: `0.3`. |
| `--ddpgfd-priority-beta X` | Importance-sampling correction exponent. Default: `1.0`. |
| `--ddpgfd-demo-priority-bonus X` | Extra priority retained by demonstration transitions. Default: `1.0`. |
| `--ddpgfd-actor-priority-weight X` | Actor contribution to priority. Default: `1e-3`. |

## Example

```bash
python/.venv/bin/python python/train.py \
  --algorithm ddpgfd \
  --godot-project godot --godot-scene res://my_robot_task.tscn \
  --demo-path python/demos/robot/train.npz \
  --demo-validation-path python/demos/robot/val.npz \
  --demo-bc-epochs 30 --ddpgfd-pretrain-updates 3000 \
  --replay-capacity 200000 --require-replay-buffer \
  --collector-mode async \
  --checkpoint-dir checkpoints/robot_ddpgfd \
  --headless --dashboard
```

Inspect sampling composition as well as reward. A very large demonstration bonus can
prevent adaptation; a very small one silently turns the run back into ordinary DDPG.

