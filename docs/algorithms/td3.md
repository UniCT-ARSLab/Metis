# TD3

Twin Delayed DDPG reduces DDPG's overestimation with two critics, delayed actor
updates, and smoothed target actions. The deployed actor remains deterministic.

## When to use it

Use TD3 when continuous actions should be deterministic and DDPG critics are unstable
or overoptimistic. It is a sensible baseline for vehicles, manipulators, and other
smooth controls where external action noise is acceptable during training.

Use SAC instead when stochastic exploration is central. Use PPO for hybrid actions or
when an on-policy experiment is required.

## Flags

The complete CLI is [Shared training options](common-options.md), the **Deterministic
continuous family** section there, plus:

| Flag | Meaning |
|---|---|
| `--critic2-weights-path PATH` | Second critic weights output. Default: `generic_td3_critic2.weights.h5`. |
| `--td3-policy-delay N` | Critic updates per actor update. Default: `2`. |
| `--td3-target-policy-noise X` | Gaussian noise applied only to target actions. Default: `0.2`. |
| `--td3-target-noise-clip X` | Absolute clip for target action noise. Default: `0.5`. |

Target-policy noise is not environment exploration. The former smooths Bellman
targets; `--exploration-noise` changes actions actually sent to Godot.

## Example

```bash
python/.venv/bin/python python/train.py \
  --algorithm td3 \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --num-envs 8 --num-episodes 6000 \
  --replay-warmup 10000 --replay-capacity 200000 \
  --td3-policy-delay 2 \
  --td3-target-policy-noise 0.2 --td3-target-noise-clip 0.5 \
  --collector-mode async \
  --checkpoint-dir checkpoints/cars_td3 \
  --headless --dashboard
```

Compare both critic losses and the minimum Q used by the target. A low loss alone does
not prove useful action gradients; deterministic frozen evaluation remains the policy
test.

