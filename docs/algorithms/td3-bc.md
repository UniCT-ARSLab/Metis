# TD3+BC

TD3+BC combines TD3's twin-critic deterministic control with behavior cloning. Metis
can schedule BC and Q weights separately and can filter the BC term using critic
preference.

## When to use it

Use TD3+BC when a deterministic actor already has useful demonstrations and should
improve through online interaction. It is often a better demonstration-guided choice
than DDPG+BC when Q overestimation is a concern.

It is not automatically safe for narrow offline datasets. A critic can rank obvious
bad actions correctly while still providing a harmful local action gradient. Use
`--stop-after-bc`, critic-only warmup/audit, and a short protected canary before a long
actor run.

## Flags

The complete CLI is [Shared training options](common-options.md), the **Deterministic
continuous family** section, all TD3 flags, and:

| Flag | Meaning |
|---|---|
| `--demo-bc-weight-start X` / `--demo-bc-weight-end X` | Initial/final online BC weights. Defaults: `1.0` / `0.05`. |
| `--demo-bc-decay-updates N` | BC decay duration. Default: `100000`. |
| `--demo-q-weight-start X` / `--demo-q-weight-end X` | Initial/final Q-objective weights. Defaults: `0` / `1`. |
| `--demo-q-weight-ramp-updates N` | Q-objective ramp duration. Default: `50000`. |
| `--demo-q-filter` | Apply imitation only when expert Q exceeds current-policy Q. Off by default. |
| `--td3-bc-alpha X` | TD3+BC Q-loss scaling coefficient. Default: `2.5`. |
| `--critic2-weights-path PATH` | Second critic output. |
| `--td3-policy-delay N` | Actor update delay. Default: `2`. |
| `--td3-target-policy-noise X` / `--td3-target-noise-clip X` | Target smoothing scale/clip. Defaults: `0.2` / `0.5`. |

`--demo-q-filter-start-policy-updates` delays filtering until the critic has had time
to adapt. `--gradient-telemetry-every` reports Q versus BC gradient conflict and actor
deviation; both are documented in the shared demonstration table.

## Staged example

First validate the clone:

```bash
python/.venv/bin/python python/train.py \
  --algorithm td3_bc \
  --godot-project godot --godot-scene res://my_task.tscn \
  --demo-path python/demos/my_task/train.npz \
  --demo-validation-path python/demos/my_task/val.npz \
  --demo-bc-epochs 30 --stop-after-bc \
  --checkpoint-dir checkpoints/my_td3bc_bc --headless
```

For online refinement, remove `--stop-after-bc`, warm-start from the saved BC actor,
keep complete demonstrations in replay, and use a slow Q ramp. Never mix
imitation-only corrective labels into replay.

