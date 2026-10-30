# PPO

Proximal Policy Optimization is an on-policy actor-critic. Metis PPO has native categorical and Gaussian heads, so it supports discrete, continuous, multi-discrete, and hybrid action contracts without flattening them into an artificial scalar.

## When to use it

Use PPO for hybrid controls, stable on-policy experiments, or tasks where replay from old behavior is undesirable. It is a strong baseline for Tanks-style movement plus a discrete command. It can also train a small bounded residual around a frozen base policy with `--policy-mode residual`.

PPO is generally less sample-efficient than off-policy SAC/TD3 because rollout data is used for a limited number of epochs. Avoid very short, highly correlated batches.

Frozen evaluation and `run.py` decode PPO's categorical heads as logits, including a scene with just one discrete component; they do not treat the actor-critic's logits/value outputs as DQN Q-values. This applies to shared-policy and independent-policy inference. Continuous action decoding retains its own mean/bounds contract.

Asynchronous collection uses an independent CPU random generator for each worker and policy, for both categorical and Gaussian draws. `--env-seed-base` seeds these streams; worker scheduling does not change another worker's stream. This isolates sampling randomness, not whole-training trajectories: rollout ordering, policy updates and environment execution can still vary. Synchronous sampling and the protected residual RNG path retain their existing behavior.

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

Demonstration imitation is available for standard single-policy PPO only. It is not combined with residual mode, whose labels would need to describe residual corrections rather than absolute expert actions.

| Flag | Meaning |
|---|---|
| `--policy-mode {standard,residual}` | Select ordinary PPO or a bounded correction around a base policy. Default: `standard`. |
| `--base-policy PATH` | Frozen `.keras`, `.h5`, or weights artifact used by residual mode. |
| `--residual-delta-max X` | Maximum absolute correction per continuous action channel. Default: `0.005`. |
| `--residual-gate-config SPEC` | JSON path or `outer,inner,err_start[,err_size]` observation-space gate. |
| `--residual-update-mask {gate,all}` | Train only where the gate is active or on every state. Default: `gate`. |
| `--freeze-base-policy / --no-freeze-base-policy` | Exclude the base from trainable variables. Keep enabled for protected residual learning. |

Residual mode uses a linear policy mean followed by one adapter `tanh`, a separate value tower when requested, and checkpoint manifests that validate base-policy hash, dimensions, gate, optimizer, log-std, and RNG state on resume. It currently rejects async collection; use `--collector-mode sync` until versioned residual snapshots have their own validation coverage.

## Demonstrations

PPO can pretrain its policy from expert data with behavior cloning and optionally keep an auxiliary imitation loss during on-policy training. Both are off by default. Expert samples stay in a separate, read-only dataset: they never enter PPO rollouts, old action probabilities, returns, GAE, or the accepted-transition budget. This is PPO with an optional supervised objective, not replay-based PPO or a guarantee against forgetting.

Use `--demo-path` for existing expert NPZ recordings or `--demo-bc-path` for imitation-only labels; both accept repeated paths and PPO reads only the observations and action labels. For continuous control, each file needs finite `obs` with shape `(N, obs_dim)` and `actions` with shape `(N, action_size)` in the scenario's action units and bounds. Discrete labels must be integers in range; one discrete head accepts `(N,)` or `(N, 1)`, and multiple discrete heads use `(N, number_of_heads)`. Hybrid files require explicit `discrete_actions` and `continuous_actions` arrays in action-contract head order. Object/pickled arrays, invalid dimensions, nonfinite values, out-of-range labels and empty datasets are rejected. Optional `action_type` metadata must match the scene. PPO does not normalize observations or convert a SAC actor/checkpoint into a PPO model.

In Godot, select PPO and set **Expert dataset (NPZ)**, **BC pretraining epochs**, and optionally **Expert imitation weight**. Additional demo options, including held-out validation and BC-only execution, are available in **All options**. A single dataset path can be entered in the prominent field; multiple sources can be passed as repeated CLI flags.

| Flag | Meaning |
|---|---|
| `--demo-path PATH` / `--demo-bc-path PATH` | Repeatable expert datasets, used only for supervised imitation. |
| `--demo-max-transitions N` | Cap expert training samples across all input sources; `0` loads all. |
| `--demo-bc-epochs N` | Policy pretraining epochs before PPO; default `0`. |
| `--demo-bc-batch-size N` | Expert batch size for BC and auxiliary imitation; default `128`. |
| `--demo-bc-learning-rate LR` | Separate BC optimizer rate; omitted uses `--learning-rate`. |
| `--demo-validation-path PATH` | Repeatable held-out datasets; restore the BC epoch with the lowest validation imitation loss. |
| `--demo-bc-on-resume` | Explicitly repeat BC after checkpoint restore; otherwise BC is skipped. |
| `--bc-actor-weights-path PATH` | Save the cloned PPO actor-critic model before RL; default is a `_bc.weights.h5` sibling of `--weights-path`. |
| `--stop-after-bc` | Save the cloned policy and checkpoint, then stop before collecting rollouts; does not run an automatic evaluation. |
| `--ppo-demo-coef X` | Nonnegative auxiliary imitation weight during PPO; default `0`, disabled. |
| `--ppo-demo-decay-updates N` | Linear decay over PPO optimizer minibatch updates; `0` keeps a constant weight. |
| `--ppo-demo-final-coef X` | Final weight after decay; default `0`, no greater than the initial weight. |

The supervised loss is the sum of one mean categorical cross-entropy per discrete head and a mean squared error between the continuous policy mean and expert action labels. It does not fit the value head or Gaussian log standard deviation directly, although a shared feature trunk can affect value predictions. PPO's entropy coefficient and initial log standard deviation still control stochastic exploration after BC; a good deterministic clone alone does not ensure successful stochastic rollouts. Expert batches are sampled uniformly with replacement from a private stream indexed by the PPO optimizer update counter and training seed. There is no trajectory balancing or task-specific grasping logic.

Both synchronous and asynchronous standard single-policy PPO use this objective. The effective loss is `PPO loss + imitation weight * expert loss`. The existing KL early stop checks the on-policy distribution after each epoch and can stop further epochs; it is not a hard bound on imitation-driven policy drift. Metrics include `demo_loss`, `demo_coefficient`, and `demo_samples` per PPO update call, excluding BC pretraining. `demo_coefficient` is the mean weight used across that call's minibatches. With decay enabled, zero weight is reached after the configured number of optimizer minibatch steps, not environment transitions or episodes. Independent multi-policy and residual demo training are explicitly rejected rather than silently ignored.

Provide the expert files again on resume; they are not copied into the checkpoint. Auxiliary-imitation checkpoints bind the loaded samples, action contract, weight schedule, expert batch size and seed inside the TensorFlow checkpoint, with a `.ppo_demos.json` sidecar for inspection. The embedded contract survives best-checkpoint staging, promotion and recovery copies even when the sidecar is absent. Resume rejects changes or removal of the saved auxiliary objective; use a new run with `--policy-path` for an intentional treatment change. Legacy PPO checkpoints without a saved demo contract can opt into imitation explicitly. The restored optimizer counter governs the decay and expert sampling after resume or recovery; these guarantees do not claim bitwise continuation of the whole Godot training trajectory.

For a first clone-only check on an existing continuous scene, append these options to the normal PPO scene command:

```bash
--demo-path /absolute/path/expert.npz --demo-bc-epochs 60 --stop-after-bc --bc-actor-weights-path /absolute/path/ppo_bc.weights.h5
```

Evaluate the saved clone before RL, then start a fresh checkpoint directory and append the following options for an imitation-guided PPO trial. The coefficient below is an initial trial setting, not a tuned recommendation for every task:

```bash
--policy-path /absolute/path/ppo_bc.weights.h5 --demo-path /absolute/path/expert.npz --ppo-demo-coef 0.1
```

## Hybrid example

```bash
python/.venv/bin/python python/train.py \
  --algorithm ppo \
  --godot-project godot \
  --godot-scene res://scenarios/tanks/tanks_scenario.tscn \
  --num-envs 8 --multi-agent \
  --ppo-rollout-steps 0 \
  --batch-size 256 --ppo-epochs 4 \
  --collector-mode sync \
  --checkpoint-dir checkpoints/tanks_ppo \
  --headless --dashboard
```

This multi-agent example uses episode rollouts; fixed-length `--ppo-rollout-steps N` rollouts are currently single-agent only. Residual PPO also requires a compatible single-agent continuous scene: use `--policy-mode residual`, `--base-policy`, the gate and bound, `--freeze-base-policy`, and `--collector-mode sync`. Start from a zero residual and validate the unchanged base behavior before the first update.

PPO supports the shared opt-in solved/LR controls but not off-policy actor EMA. See [Training stability controls](../training-stability.md) before combining frozen qualification, independent certification and auxiliary imitation.

