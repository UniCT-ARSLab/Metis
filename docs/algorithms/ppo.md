# PPO

Proximal Policy Optimization is an on-policy actor-critic. Metis PPO has native
categorical and Gaussian heads, so it supports discrete, continuous, multi-discrete,
and hybrid action contracts without flattening them into an artificial scalar.

## When to use it

Use PPO for hybrid controls, stable on-policy experiments, or tasks where replay from
old behavior is undesirable. It is a strong baseline for Tanks-style movement plus a
discrete command. It can also train a small bounded residual around a frozen base
policy with `--policy-mode residual`.

PPO is generally less sample-efficient than off-policy SAC/TD3 because rollout data is
used for a limited number of epochs. Avoid very short, highly correlated batches.

## PPO flags

The complete CLI is this table plus [Shared training options](common-options.md).

| Flag | Meaning |
|---|---|
| `--ppo-epochs N` | Optimization passes over each rollout. Default: `4`. |
| `--ppo-rollout-steps N` | Steps per worker and generation. `0` collects one full episode per worker; `N>0` uses fixed GAE rollouts across resets. Fixed rollouts are currently single-agent only. |
| `--gae-lambda X` | GAE bias/variance coefficient. Default: `0.95`. |
| `--clip-ratio X` | PPO probability-ratio clip. Default: `0.2`. |
| `--learning-rate LR` | Shared/actor optimizer rate. Default: `3e-4`. |
| `--value-learning-rate LR` | Residual mode: separate value-tower optimizer rate. `0` retains the standard single optimizer. |
| `--value-loss-coef X` | Value loss weight. Default: `0.5`. |
| `--entropy-coef X` | Entropy bonus weight. Default: `0.01`. |
| `--ppo-grad-clip X` | Global gradient norm cap; `0` disables it. |
| `--ppo-target-kl X` | Stop remaining PPO epochs when non-negative approximate KL exceeds the safety threshold; `0` disables it. |
| `--initial-log-std X` | Initial continuous-policy log standard deviation. Default: `-0.5`. |
| `--ppo-log-std-min X` / `--ppo-log-std-max X` | Bounds preventing premature deterministic collapse or runaway exploration. Defaults: `-20` / `2`. |
| `--weights-path PATH` | Legacy/final PPO model weights output. Default: `generic_ppo_hybrid.weights.h5`. |

## Protected residual mode

| Flag | Meaning |
|---|---|
| `--policy-mode {standard,residual}` | Select ordinary PPO or a bounded correction around a base policy. Default: `standard`. |
| `--base-policy PATH` | Frozen `.keras`, `.h5`, or weights artifact used by residual mode. |
| `--residual-delta-max X` | Maximum absolute correction per continuous action channel. Default: `0.005`. |
| `--residual-gate-config SPEC` | JSON path or `outer,inner,err_start[,err_size]` observation-space gate. |
| `--residual-update-mask {gate,all}` | Train only where the gate is active or on every state. Default: `gate`. |
| `--freeze-base-policy / --no-freeze-base-policy` | Exclude the base from trainable variables. Keep enabled for protected residual learning. |

Residual mode uses a linear policy mean followed by one adapter `tanh`, a separate
value tower when requested, and checkpoint manifests that validate base-policy hash,
dimensions, gate, optimizer, log-std, and RNG state on resume. It currently rejects
async collection; use `--collector-mode sync` until versioned residual snapshots have
their own validation coverage.

## Hybrid example

```bash
python/.venv/bin/python python/train.py \
  --algorithm ppo \
  --godot-project godot \
  --godot-scene res://scenarios/tanks/tanks_scenario.tscn \
  --num-envs 8 --multi-agent \
  --ppo-rollout-steps 256 \
  --batch-size 256 --ppo-epochs 4 \
  --collector-mode sync \
  --checkpoint-dir checkpoints/tanks_ppo \
  --headless --dashboard
```

For residual PPO, add `--policy-mode residual`, `--base-policy`, the gate and bound,
`--freeze-base-policy`, and `--collector-mode sync`. Start from a zero residual and
validate the unchanged base behavior before the first update.

