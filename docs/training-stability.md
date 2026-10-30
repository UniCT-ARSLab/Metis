# Training stability controls

Native Metis trainers offer optional evaluation-driven learning rates, solved-task stopping with independent simulator certification, cumulative recovery reductions, and actor averaging for continuous off-policy learners. All additions are disabled by default. They change the training or model-selection procedure when enabled and must be declared in experiment configurations. Implementing a control does not by itself establish improved task performance.

## Supported combinations

| Control | Native scope | Important boundary |
|---|---|---|
| Plateau/on-best LR reduction | All native algorithm families | Requires frozen best-checkpoint evaluation; choose one decay mode. |
| Cumulative recovery LR reduction | All native algorithm families | Requires best-checkpoint support; rollback still needs explicit automatic recovery. |
| Solved qualification and certification | All native algorithm families | Single-agent, single-policy only; a passing held-out test is mandatory before stopping. |
| Actor EMA | Single-policy SAC and DDPG/TD3 families, including BC/fD variants | Not PPO/DQN; collectors use raw weights and deployment uses the averaged artifact. |
| Observation margin reward | Any compatible Godot agent reward system | Scene-owned shaping; it neither detects success nor certifies a policy. |

These runtime controls are not features of the optional SB3 comparison adapter. SAC protected replay and PPO auxiliary imitation can be combined with supported stability controls, but they solve different problems: retaining expert transitions or adding a supervised loss does not certify the learned policy. See [Manual demonstrations](guides/manual-demonstrations.md).

## Solved qualification and certification

Add `--stop-when-solved 0.95 --solved-patience 3` to qualify a candidate after three consecutive frozen evaluations reach the requested selection success rate. A failed, stale or insufficient evaluation does not count. Qualification alone never stops training. Metis pins the exact candidate files, evaluates that candidate on a separate episode deck, and stops only if the certification checks pass. The learner does not update while this synchronous certification is running; async collectors may wait on their bounded queues. On failure, timeout or incomplete evaluation, training continues and another full qualification sequence is required.

When solved stopping or LR scheduling is enabled, frozen evaluation uses `--best-evaluation-training-episode` if provided, otherwise the fixed `--num-episodes` difficulty rather than the live episode number. Solved evaluations additionally disable reverse curriculum, so a policy is not certified solely from near-goal bootstrap resets. The task must support the declared evaluation contract; a reported mixture of seeded/non-regular resets fails certification. Adaptive curricula qualify for stopping only after an evaluation at level 1.0, not on the evaluation that promotes an earlier stage. This does not automatically verify every task-specific reset or curriculum implementation.

Certification uses `--solved-certification-episodes` fresh single-agent episodes, default `300`, beginning at `--solved-certification-seed`, default `1000000000`. Each attempt reserves a distinct consecutive seed block before launching the evaluator. Seeds must be disjoint from the selection deck and beyond the declared training reset-seed range. Later blocks skip the selection namespace if necessary; exhaustion of the uint32 seed space disables further certification attempts without stopping training or declaring success. Certification attempts and their spent seed blocks never rewind on resume or recovery. Do not independently reuse the reserved seeds for other training or tuning.

The pass criterion is stronger than observing 95% successes: the exact one-sided Clopper-Pearson lower bound must reach the configured threshold. With `--solved-certification-confidence 0.95`, the total error budget is 0.05 for this checkpoint-directory-scoped run. Attempt `k` receives `0.05 / (k * (k + 1))`, and that allowance is divided between the overall result and all observed target-region checks. Every required region from the qualifying evaluation must appear and pass. Repeated failed tests therefore do not each receive a fresh 5% error budget. Small or unevenly sampled regions may need a larger episode count to pass even with no observed failures.

These bounds assume independent Bernoulli episodes under the declared simulator/reset distribution. Correlated resets, an incorrect success signal, narrow target coverage, simulator mismatch or physical deployment can invalidate a broader interpretation. A 100% probability threshold cannot be certified with a finite sample and is rejected. Certification currently rejects multi-agent and independent multi-policy training rather than treating correlated agent outcomes as independent episodes. No robot-safety or universal-generalisation certification is claimed.

Each attempt retains its frozen policy and `report.json` under `CHECKPOINT_DIR/certification/attempt-K/`. The report records the policy fingerprint checked before and after evaluation, evaluation settings, results, bounds and pass/fail reason. A successful candidate is the deployable certified model; later live learner weights are not substituted for it. The final ordinary training checkpoint is still saved separately for continuation. Use the reported `.keras` policy directly, or the reported TensorFlow checkpoint prefix when artifact mode is not used. Failed candidates and reports remain available for audit and are never called solved.

## Learning-rate reduction

`--lr-decay-on-plateau 0.3 --lr-plateau-patience 6` multiplies registered optimizer learning rates after six frozen evaluations without a new best. The plateau counter resets after improvement, reduction or a curriculum-baseline change. `--lr-decay-on-best 0.3` is an alternative opt-in mode that reduces rates on every new best; it cannot be combined with plateau mode and is not a recommended universal default. `--lr-min-scale 0.01` bounds scheduler reductions relative to each optimizer's first controlled rate. Existing lower manually configured rates are not raised.

All registered optimizer roles are controlled, including SAC actor, critics and entropy temperature, and PPO's separate value optimizer when present. The scheduler does not assume that one factor is optimal for all tasks. Optimizer rates and monotone ceilings are saved in `training_control.json`; resume reapplies the ceilings so loading an older optimizer cannot undo prior reductions. Configuration changes that alter the saved control/evaluation contract require a new checkpoint directory. If extending a run, use an explicit fixed evaluation difficulty so a larger episode budget does not silently change the validation task. Qualification starts again after resume; certification attempts and LR ceilings are retained.

`--recovery-lr-cumulative` additionally compounds the existing role-specific recovery factors from the pre-restore learning rates, down to `--recovery-min-lr-scale`. Without this flag, the legacy absolute-from-initial recovery behavior remains. An active LR scheduler still prevents recovery from increasing rates above its saved ceilings. Recovery factors are intentionally conservative and may hit the minimum quickly when compounded; retune `--recovery-actor-lr-factor`, `--recovery-critic-lr-factor`, `--recovery-alpha-lr-factor` or `--recovery-policy-lr-factor` for the intended run. Controller state must accompany the checkpoint directory: copying only an old checkpoint into a new directory starts a different control/certification scope.

## Actor EMA

Single-policy SAC and the DDPG/TD3 family support `--actor-ema-decay 0.995`. A value of `0` disables averaging; enabled decay must be strictly between 0 and 1. The moving average is initialised after restore/warm-start/pretraining, updated only after actual actor optimizer steps, and preserved separately from raw actor and optimizer state. Critic-only warmup and skipped/delayed TD3 actor steps do not update it. Explicit BC on resume resets the average to the new clone. Changing an existing EMA decay requires a new warm-start run. Independent multi-policy EMA is explicitly unsupported.

Collectors and learner targets continue using their original policies. Frozen evaluation, saved `policy.keras` artifacts and certification use the EMA actor; manifests label `inference.policy_weights` as `ema`. A promoted best retains both the exact evaluated EMA artifact and a matching full raw-learner/EMA checkpoint for recovery. Raw and EMA best-comparison baselines are not mixed. Resuming an EMA checkpoint requires the matching EMA option rather than silently dropping its saved state; recovery from a legacy raw-only checkpoint reinitialises the average to that restored actor. Ordinary checkpoint loading remains the raw learning path; deploy the saved `.keras` artifact when the averaged policy is intended. No EMA-specific inference engine or `run.py` switch is needed. PPO and DQN do not expose this actor-EMA option.

EMA can reduce short-term variation but can also lag improvements or follow a sustained collapse. Weight averaging is not action averaging and does not guarantee a better policy. Compare raw and averaged configurations on independent policies before making quality claims or changing defaults.

## Observation margin reward

Godot's `ObservationMarginReward` is a generic `RewardComponent` configured with `observation_name`, `target_value`, `half_width`, `scale`, and linear or quadratic `shape`. It returns a continuous negative centering cost: zero at the desired value, `-scale * weight` at either reference band edge, and a larger penalty beyond the edge. The interior is not a flat success region. Values use the observation's units, which may already be normalised; no millimetre or OpenArm-specific assumption is built in.

An optional observation gate limits when the term is active. Missing/nonfinite values and invalid width return zero instead of contaminating reward with NaNs. This component does not change termination or success thresholds. Its cost changes the task objective; tune its scale against speed and other rewards and validate task outcomes. Add it through the Godot node inspector; the existing `ObservationThresholdReward` remains unchanged.

## Editor and first trial

The Train wizard exposes solved threshold, solved patience and plateau LR factor for native algorithms, plus actor EMA for SAC/DDPG/TD3. Certification size/confidence/seed, scheduler floors, alternative on-best decay and cumulative recovery are available in All options. All parser/schema and runtime controls share the same CLI defaults.

For an initial SAC trial, append the following to the normal scene command, using a new checkpoint directory. These are trial settings, not validated hyperparameters for a particular robot or task:

```bash
--stop-when-solved 0.95 --solved-patience 3 --solved-certification-episodes 1000 --lr-decay-on-plateau 0.3 --lr-plateau-patience 6 --recovery-lr-cumulative --actor-ema-decay 0.995
```

PPO can use the solved/LR controls together with [separate expert imitation](algorithms/ppo.md#demonstrations), but not the off-policy actor-EMA flag. Changing the reward component is a scene edit and is deliberately not performed automatically by the training controller.
