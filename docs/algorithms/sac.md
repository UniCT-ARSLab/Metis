# SAC

Soft Actor-Critic is an off-policy stochastic actor-critic for continuous actions. It
uses twin critics and an entropy temperature, so exploration is learned as part of the
policy rather than added as external OU/Gaussian noise.

## When to use it

SAC is usually the first Metis choice for difficult continuous control: robot arms,
driving, and tasks where the policy must explore several plausible motions. Replay
makes it more sample-efficient than PPO on many environments.

Do not use SAC for discrete or hybrid action contracts. It can also be unnecessarily
noisy for a task already solved by a precise deterministic controller; TD3 or protected
residual PPO may then be easier to validate.

## SAC flags

The complete CLI is this table plus [Shared training options](common-options.md).

| Flag | Meaning |
|---|---|
| `--tau X` | Target critic Polyak coefficient. Default: `0.005`. |
| `--actor-learning-rate LR` / `--critic-learning-rate LR` | Actor and twin-critic rates. Defaults: `3e-4` / `3e-4`. |
| `--alpha-learning-rate LR` | Entropy-temperature optimizer rate. Default: `3e-4`. |
| `--resume-actor-learning-rate LR` / `--resume-alpha-learning-rate LR` | Conservative rates applied after checkpoint restore. Defaults: `1e-5` / `1e-5`. |
| `--initial-alpha X` | Initial/fixed entropy coefficient. Default: `0.2`. |
| `--tune-alpha / --no-tune-alpha` | Automatically fit alpha to target entropy. Enabled by default. |
| `--target-entropy X` | Explicit entropy target; omitted derives it from the action size. |
| `--min-alpha X` | Floor for tuned alpha. Default: `0` (no floor). |
| `--log-std-min X` / `--log-std-max X` | Actor distribution bounds. Defaults: `-20` / `2`. |
| `--action-smoothing X` | Environment-action smoothing; `0` disables it. |
| `--random-exploration-episodes N` | Fully random collection before policy actions. Default: `15`. |
| `--replay-warmup N` | Replay transitions required before ordinary updates. Default: `10000`. |
| `--replay-capacity N` | Replay capacity. Default: `200000`. |
| `--critic-warmup-updates N` | Critic-only updates after actor warm-start/resume/BC. Default: `2000`. |
| `--target-update-every N` | Critic updates between target updates. Default: `1`. |
| `--policy-update-every N` | Critic updates between actor/alpha updates. Default: `2`. |
| `--grad-clip-norm X` | Hard global norm cap for actor and critics; `0` disables it. Default: `10`. |
| `--grad-clip-adaptive` | Use `k * EMA(gradient norm)`, still bounded by the hard cap. Off by default. |
| `--grad-clip-k X` | Adaptive clip multiplier. Default: `3`. |
| `--actor-anchor-coef X` | Keep a resumed/warm-started actor near its startup copy. Default: `0`. |
| `--actor-anchor-log-std-coef X` | Relative log-std weight in the anchor. Default: `0.1`. |
| `--actor-weights-path PATH` | Actor output. Default: `generic_sac_actor.weights.h5`. |
| `--critic-weights-path PATH` | Legacy critic path alias. |
| `--critic1-weights-path PATH` / `--critic2-weights-path PATH` | Twin critic outputs. |

## CAPS and reset curriculum

| Flag | Meaning |
|---|---|
| `--caps` | Enable temporal and spatial action-consistency regularization. Off by default. |
| `--caps-lambda-temporal X` | Temporal smoothness weight. Default: `1`. |
| `--caps-lambda-spatial X` | Nearby-state smoothness weight. Default: `1`. |
| `--caps-sigma X` | Observation perturbation standard deviation. Default: `0.05`. |
| `--reset-progress-curriculum` | Gradually widen reset progress. Off by default. |
| `--reset-progress-start-max X` / `--reset-progress-end-max X` | Initial/final reset ranges. Defaults: `0.025` / `0.35`. |
| `--reset-progress-ramp-episodes N` | Range ramp duration. Default: `400`. |

CAPS is not a universal improvement. Strong smoothness from episode zero can teach a
robot to remain still before it learns to reach. Introduce it after movement is
reliable, lower the lambdas, or keep it disabled for early curriculum stages.

## Example

```bash
python/.venv/bin/python python/train.py \
  --algorithm sac \
  --godot-project godot \
  --godot-scene res://scenarios/robotarms/openarm_scenario.tscn \
  --num-envs 8 --num-episodes 8000 \
  --max-steps-per-episode 300 --physics-frames-per-step 3 \
  --batch-size 256 --replay-warmup 20000 \
  --random-exploration-episodes 40 \
  --grad-clip-adaptive --grad-clip-norm 10 \
  --min-alpha 0.02 --no-caps \
  --collector-mode async \
  --checkpoint-dir checkpoints/openarm_sac \
  --headless --dashboard
```

Watch deterministic frozen success, alpha, actor distribution bounds, both critics,
gradient norms, and replay composition. Low GPU utilization is normal for small MLPs
when Godot stepping and socket collection are the bottleneck.

