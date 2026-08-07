# Shared training options

The native trainers share environment processes, collectors, checkpointing,
evaluation, curriculum, health monitoring, and observability. Their complete command
line is therefore the union of this page and the algorithm-specific page.

Boolean flags use Python's paired form: `--headless` enables a setting and
`--no-headless` disables it. The table shows both only where the distinction helps.

## Entrypoint and task

| Flag | Meaning |
|---|---|
| `--algorithm NAME` | Native learner: `dqn`, `ppo`, `ddpg`, `ddpg_bc`, `ddpgfd`, `td3`, `td3_bc`, `sac`, or `auto`. |
| `--backend {metis,sb3}` | Select the native Metis backend or the limited SB3 comparison backend. Default: `metis`. |
| `--probe-port PORT` | Port used by `train.py` while probing the Godot action and observation specification. |
| `--godot-bin PATH` | Godot executable. May also come from `GODOT_BIN`. |
| `--godot-project PATH` | Godot project directory. |
| `--godot-scene RES_PATH` | Scenario scene, such as `res://scenarios/pong/pong_scenario.tscn`. |
| `--num-envs N` | Number of independent Godot instances. Default: `1`. |
| `--base-port PORT` | First TCP port; subsequent environments use consecutive ports. Default: `6200`. |
| `--num-episodes N` | Episode budget. Default: `500`. |
| `--total-timesteps N` | Optional transition budget; `0` disables it. A small in-flight batch may finish. |
| `--max-steps-per-episode N` | Python/Godot episode cap. `0` relies only on terminal scenario states. |
| `--batch-size N` | Learner minibatch size. Default: `128`. |
| `--gamma VALUE` | Reward discount. Default: `0.99`. |
| `--env-seed-base N` | Base seed assigned to environments. Default: `100`. |
| `--episode-seed-multiplier N` | Separates episode seed ranges. Default: `1000`. |
| `--env-timeout SECONDS` | TCP environment timeout. Default: `30`. |
| `--agent-id ID` | Train a named agent when the scenario exposes more than one candidate. |
| `--network-layers W [W ...]` | Optional hidden-layer widths for the actor/critic/Q networks. Omit it to use the algorithm's reference architecture (see below). |

### Network architecture

`--network-layers` is optional. Left out, each algorithm builds the architecture from its
reference paper:

| Algorithm | Default hidden layers | Source |
|---|---|---|
| SAC | `256 256` | Haarnoja et al. 2018 |
| TD3, TD3+BC | `400 300` | Fujimoto et al. 2018 |
| DDPG, DDPG+BC, DDPGfD | `400 300` | Lillicrap et al. 2015 |
| PPO | `64 64` | Schulman et al. 2017, MuJoCo MLP |
| DQN | `64 64` | Mnih et al. 2015 is convolutional and defines no MLP; this is the usual vector-observation choice and matches SB3 |

Pass a space-separated list of widths to override it:

```bash
python python/train.py --algorithm sac --network-layers 512 512 256 ...
```

The list applies to every network the algorithm builds — the actor, both critics, and the
DQN Q network. The SB3 backend ignores it and keeps its own `--network`.

**Checkpoints built before this change used `256 256 128` for every algorithm.** A TF
checkpoint only restores into the architecture it was saved with, so resuming, evaluating
or exporting one of those needs the old widths spelled out:

```bash
python python/run.py --load-from checkpoints/OLD_RUN --network-layers 256 256 128 ...
```

`policy.keras` bundles are unaffected: they carry their own architecture and load either
way. The same rule applies to any run of your own — record the widths you trained with, or
keep the default and it stays implicit.

Frozen best-policy evaluations inherit the flag automatically: the trainer forwards it to
the `run.py` subprocess it spawns, so evaluations rebuild the same architecture.

## Godot execution and rendering

| Flag | Meaning |
|---|---|
| `--headless / --no-headless` | Headless is the fast default. `--no-headless` opens training windows and runs physics at visible real-time pacing. |
| `--render-env-count N` | In visible training, render only `N` environments; when omitted all environments render. |
| `--render-mode {project,cpu,light-gpu,gpu}` | Keep project settings, use Mesa CPU rendering, compatibility/OpenGL rendering, or full Vulkan. Default: `light-gpu`. |
| `--godot-debug` | Preserve verbose Godot output useful for scene and bridge debugging. |
| `--physics-frames-per-step N` | Physics ticks advanced per Python action. This is frame skip/control-rate downsampling, not a reward multiplier. Default: `1`. |
| `--parallel-env-steps / --no-parallel-env-steps` | Dispatch synchronous `env.step()` calls concurrently. Enabled by default. |
| `--lockstep-idle-sleep-usec N` | Bridge idle backoff. Lower values reduce latency but consume more CPU. |
| `--lockstep-spin-polls N` | Empty TCP polls performed before sleeping; `0` disables spinning. |

## Collection and update scheduling

| Flag | Meaning |
|---|---|
| `--collector-mode {sync,async}` | Batched lockstep collection or independent collector workers. Native default: `async`. |
| `--async-queue-capacity N` | Maximum pending collector events before backpressure. Default: `256`. |
| `--async-policy-sync-steps N` | Collector steps between checks for new learner weights. Default: `100`. |
| `--async-policy-publish-updates N` | Learner updates between published policy snapshots. Default: `100`. |
| `--async-updates-per-step N` | Updates scheduled at each collection interval. Used by sync and async modes despite the historical name. Default: `1`. |
| `--async-update-basis {transitions,env_steps}` | Count individual agent transitions or environment events. Use `transitions` for multi-agent scaling. |
| `--async-update-every N` | Collected units per update interval. Default: `4`; `--async-update-every-steps` is a legacy alias. |
| `--async-max-updates-per-env-step N` | Safety cap on updates per Godot step; `0` disables the cap. Default: `1`. |
| `--async-drain-max-events N` | Events moved from the async queue before scheduled learning. Default: `64`. |
| `--async-replay-save / --no-async-replay-save` | Save replay snapshots in a background thread. Enabled by default. |

Async collectors deliberately act on slightly stale published policies. Transitions
retain their worker and policy context; they are not mixed between unrelated
independent policies. PPO uses versioned on-policy batches and has additional rollout
constraints described on its own page.

## Multi-agent, multi-policy, and opponents

| Flag | Meaning |
|---|---|
| `--multi-agent` | Treat all compatible agents returned by the scenario as participants. |
| `--multi-policy` | Maintain independent policies instead of parameter sharing. |
| `--policy-assignment {auto,policy_id,team,agent}` | Map agents to policies from scenario metadata, team, or agent identity. |
| `--train-policy ID` | Policy to optimize; repeat the flag to select several policies. |
| `--opponent-pool` | Enable historical opponent sampling. |
| `--opponent-pool-dir PATH` | Snapshot directory for opponent policies. |
| `--opponent-snapshot-every N` | Learner updates between opponent snapshots. Default: `100`. |
| `--opponent-pool-size N` | Maximum historical policies retained. Default: `10`. |
| `--opponent-current-probability P` | Probability of playing the current policy rather than history. Default: `0.2`. |
| `--opponent-sampling {uniform,latest}` | Sample all retained opponents uniformly or prefer the latest. |
| `--learner-team ID` | Team controlled by the learner when the other team is sampled from history. |

Independent native multi-policy training supports sync and async collection. Historical
opponent pools have narrower compatibility: DQN supports async sampling, while the
continuous and PPO opponent-pool paths require synchronous collection.

## Checkpoints, policies, and replay

| Flag | Meaning |
|---|---|
| `--checkpoint-dir PATH` | Run checkpoint directory. |
| `--checkpoint-every N` | Completed episodes between checkpoints. Default: `25`. |
| `--keep-checkpoints N` | Number of regular TensorFlow checkpoints retained. Default: `5`. |
| `--resume` | Restore the latest checkpoint in `--checkpoint-dir`. |
| `--resume-checkpoint PATH` | Restore a specific checkpoint rather than the latest one. |
| `--policy-path PATH` | Warm-start policy/actor only from `.keras`, full `.h5`, or `.weights.h5`; critics and optimizer state are fresh. |
| `--save-replay-buffer / --no-save-replay-buffer` | Persist off-policy replay alongside checkpoints. |
| `--require-replay-buffer` | Fail a resume when the matching replay snapshot is absent instead of starting with empty replay. |

`--resume` and `--policy-path` are not interchangeable. Resume is for continuing the
same experiment contract; policy warm-start is for transferring only the policy into
a fresh optimization state.

## Demonstrations

These flags appear on replay-based trainers. PPO currently learns from on-policy
rollouts rather than replay demonstrations.

| Flag | Meaning |
|---|---|
| `--demo-path PATH` | Add a complete transition dataset. Repeatable. It can feed BC and replay. |
| `--demo-prefill / --no-demo-prefill` | Insert complete demonstration transitions into replay. Enabled by default. |
| `--demo-max-transitions N` | Per-source transition cap; `0` loads all transitions. |
| `--demo-bc-epochs N` | Behavior-cloning epochs before RL; `0` disables BC. |
| `--demo-bc-batch-size N` | BC minibatch size. Default: `128`. |
| `--demo-bc-learning-rate LR` | Optional BC optimizer rate; omitted uses the trainer's policy/actor rate. |
| `--demo-bc-on-resume` | Repeat BC after restoring a checkpoint. Off by default. |
| `--demo-validation-path PATH` | Held-out demonstrations used to restore the BC epoch with the lowest action MSE. Repeatable where supported. |
| `--demo-bc-path PATH` | Imitation-only `(obs, action)` labels; never inserted into replay. |
| `--demo-replay-only-path PATH` | Complete transitions inserted only into replay and excluded from BC. Deterministic family only. |
| `--bc-actor-weights-path PATH` | Save the actor immediately after BC, before critic warmup or RL updates. |
| `--stop-after-bc` | Save and frozen-evaluate the cloned actor, then exit before RL. |
| `--stop-after-critic-warmup` | Deterministic family: audit the warmed critics and stop before actor updates. |
| `--gradient-telemetry-every N` | Deterministic family: periodically measure Q/BC gradient norms, cosine, and actor drift; `0` disables it. |
| `--critic-audit-win-rate P` | Minimum per-sample good-versus-bad Q ranking used by the warmup audit. Default: `0.9`. |
| `--critic-audit-cell-margin-tol X` | Allowed negative per-cell mean Q margin in the warmup audit. Default: `0`. |
| `--demo-q-filter-start-policy-updates N` | Delay Q-filter activation until this many post-warmup policy updates. |

Only use a transition in replay when its `next_obs`, reward, and termination were
produced by executing the recorded action. Corrective labels for learner-generated
next states belong in `--demo-bc-path`, not replay.

## Deterministic continuous family

DDPG, DDPG+BC, DDPGfD, TD3, and TD3+BC share these options. The variant pages list
the flags layered on top.

| Flag | Meaning |
|---|---|
| `--tau X` | Polyak coefficient for target networks. Default: `0.005`. |
| `--actor-learning-rate LR` | Deterministic actor optimizer rate. Default: `1e-4`. |
| `--critic-learning-rate LR` | Critic optimizer rate. Default: `1e-3`. |
| `--exploration-noise X` | Initial action-noise scale. Default: `0.2`. |
| `--exploration-noise-min X` | Lower exploration-noise bound. Default: `0.02`. |
| `--exploration-noise-decay X` | Per-episode multiplicative decay. Default: `0.995`. |
| `--exploration-noise-kind {ou,gaussian}` | Temporally correlated Ornstein-Uhlenbeck noise or independent Gaussian noise. Default: `ou`. |
| `--ou-theta X` | OU mean-reversion rate. Default: `0.15`. |
| `--action-smoothing X` | Blend consecutive actions; `0` disables smoothing. Default: `0.2`. |
| `--random-exploration-episodes N` | Fully random episodes before actor-driven collection. Default: `15`. |
| `--actor-drive-prior X` | Optional first-action drive prior used by driving-style tasks. Default: `0.75`. |
| `--actor-steering-prior X` | Optional second-action steering prior. Default: `0`. |
| `--actor-drive-regularization X` | Weight pulling the drive channel toward its target. Default: `0.05`; set `0` when those channel semantics do not apply. |
| `--actor-drive-target X` | Target used by drive regularization. Default: `0.65`. |
| `--reset-progress-curriculum` | Gradually widen scenario reset progress during training. Off by default. |
| `--reset-progress-start-max X` / `--reset-progress-end-max X` | Initial and final maximum reset progress. Defaults: `0.025` / `0.35`. |
| `--reset-progress-ramp-episodes N` | Episodes over which reset progress widens. Default: `400`. |
| `--replay-warmup N` | Replay transitions required before ordinary learning. Default: `500`. |
| `--replay-capacity N` | Replay capacity. Default: `100000`. |
| `--critic-warmup-updates N` | Critic-only updates after BC, resume, or actor warm-start. Default: `2000`. |
| `--target-update-every N` | Critic updates between target-network updates. Default: `1`. |
| `--actor-weights-path PATH` | Final/legacy actor weights output. |
| `--critic-weights-path PATH` | First critic weights output. |

The drive prior flags predate task-neutral action metadata. They are useful only when
the first channels really mean drive and steering; for arbitrary robot joints, disable
`--actor-drive-regularization` instead of imposing that semantic prior.

## Best-policy evaluation and curriculum

| Flag | Meaning |
|---|---|
| `--best-checkpoint / --no-best-checkpoint` | Enable isolated deterministic evaluation and best-policy retention. Enabled by default. |
| `--best-checkpoint-dir PATH` | Best-policy directory; default is `CHECKPOINT_DIR/best`. |
| `--keep-best-checkpoints N` | Best checkpoints retained. Default: `3`. |
| `--best-evaluation-every N` | Training episodes between frozen evaluations. Default: `100`. |
| `--best-evaluation-episodes N` | Episodes in each evaluation. Default: `20`. |
| `--best-evaluation-seed N` | First deterministic evaluation seed. Default: `10000`. |
| `--best-evaluation-training-episode N` | Fixed episode/curriculum value passed to Godot during every evaluation. |
| `--best-evaluation-follow-curriculum` | Evaluate at the live training difficulty. Enabled by default. |
| `--best-evaluation-max-steps N` | Evaluation time limit; `0` means unlimited. |
| `--best-evaluation-port PORT` | Port for the isolated evaluator. |
| `--best-evaluation-timeout SECONDS` | Wall-time timeout for one evaluation. Default: `1800`. |
| `--best-final-drain-timeout SECONDS` | Wait for an in-flight final evaluation on normal shutdown. Default: `120`. |
| `--best-evaluation-device {cpu,auto}` | Keep evaluation on CPU or allow automatic placement. Default: `cpu`. |
| `--best-evaluation-cpu-threads N` | Evaluator CPU thread count. Default: `1`. |
| `--best-metric {auto,success_rate,reward_mean,task_progress,lexicographic}` | Ranking rule. Lexicographic prioritizes success, then fewer collisions, then progress. |
| `--adaptive-curriculum` | Expose and persist a framework-managed `curriculum_level`. |
| `--curriculum-initial-level X` | Fresh-run level in `[0,1]`. Default: `0`. |
| `--curriculum-level-step X` | Promotion/demotion step. Default: `0.1`. |
| `--curriculum-promotion-metric {success_rate,task_progress,reward_mean}` | Frozen-evaluation mastery metric. |
| `--curriculum-promotion-threshold X` | Promotion threshold. Default: `0.70`. |
| `--curriculum-promotion-evaluations N` | Consecutive confirmations required. Default: `2`. |
| `--curriculum-min-policy-updates N` | New policy updates required between stage transitions. Default: `100`. |
| `--curriculum-demotion-threshold X` | Optional lower threshold for moving back one stage. Disabled when omitted. |
| `--curriculum-demotion-evaluations N` | Consecutive demotion confirmations. Default: `3`. |
| `--curriculum-stage-names CSV` | Optional task-owned display names, for example `A,B,C,D,E,F`. |

## Health and automatic recovery

| Flag | Meaning |
|---|---|
| `--health-monitor / --no-health-monitor` | Persist health state from frozen evaluations. Enabled by default. |
| `--auto-recovery` | Opt in to checkpoint rollback and learning-rate reduction after confirmed collapse. |
| `--health-warning-drop X` / `--health-critical-drop X` | Relative evaluation drops for warning and critical state. Defaults: `0.35` / `0.60`. |
| `--health-collapse-patience N` | Consecutive critical evaluations required. Default: `2`. |
| `--health-plateau-evaluations N` | Evaluations without a new best before plateau warning; `0` disables it. Default: `6`. |
| `--health-min-success-baseline X` | Minimum success reference needed for success-based collapse detection. Default: `0.05`. |
| `--health-reward-scale-floor X` | Stabilizes relative reward comparisons near zero. Default: `1`. |
| `--health-state-path PATH` | Health JSON path; defaults inside the checkpoint directory. |
| `--recovery-lr-factor X` | Legacy fallback LR multiplier for optimizer roles without a specific factor. |
| `--recovery-actor-lr-factor X` | Actor LR multiplier; effective default `1/30`. |
| `--recovery-critic-lr-factor X` | Critic LR multiplier; effective default `1/3`. |
| `--recovery-alpha-lr-factor X` | SAC temperature LR multiplier; effective default `1/30`. |
| `--recovery-policy-lr-factor X` | DQN/PPO LR multiplier; effective default `1/4`. |
| `--recovery-min-lr-scale X` | Lower bound on cumulative LR scaling. Default: `0.01`. |
| `--recovery-max-attempts N` | Attempts per recovery cycle. Default: `3`. |
| `--recovery-hard-after-attempt N` | Attempt that starts clearing stale online replay. Default: `2`. |
| `--recovery-critic-warmup-updates N` | Critic-only updates after continuous off-policy rollback. Default: `3000`. |
| `--recovery-policy-update-every N` | Minimum critic updates between actor updates during stabilization. Default: `4`. |
| `--recovery-cycle-reset-evaluations N` | Healthy evaluations needed to close a cycle. Default: `2`. |
| `--recovery-min-evaluations N` | Evaluations required before rollback may spend an attempt. Default: `5`. |
| `--recovery-require-success-baseline` | Require meaningful success before success-based rollback. Enabled by default. |
| `--recovery-cooldown-evaluations N` | Evaluations before another recovery. Default: `2`. |
| `--recovery-keep-diagnostics N` | Pre-recovery diagnostic checkpoints retained. Default: `3`. |

Recovery is a guardrail, not an optimizer. A wrong reward, unreachable target, invalid
collision geometry, or incompatible checkpoint should be fixed at the task contract.

## Logs, dashboard, and TensorFlow

| Flag | Meaning |
|---|---|
| `--log-format {pretty,compact}` | Readable episode blocks or one-line machine-friendly logs. |
| `--log-details` | Continuous off-policy trainers: include additional action/update diagnostics. |
| `--dashboard` | Start the local HTTP/WebSocket dashboard. Off by default. |
| `--dashboard-port PORT` | Dashboard port. Default: `8770`. |
| `--metrics-jsonl PATH` | Write the same episode metrics to JSONL. |
| `--gpu-memory-growth / --no-gpu-memory-growth` | Let TensorFlow grow CUDA memory on demand. Enabled by default on supported systems. |
| `--tf-compile-learner / --no-tf-compile-learner` | Compile supported learner steps into TensorFlow graphs. Enabled by default. |
| `--tf-xla / --no-tf-xla` | Enable XLA JIT. Off by default because small Metis networks are often slower with it. |

## Checking the installed version

Flags can evolve. Ask the exact checkout or installed package before launching a long
run:

```bash
PYTHONPATH=python python/.venv/bin/python -m algorithms.td3_bc --help
# The public entrypoint still runs the trainer:
python/.venv/bin/python python/train.py --algorithm td3_bc [OPTIONS]
```
