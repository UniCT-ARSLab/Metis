# Monitoring training

Metis can serve a small live dashboard from the training process. It is useful for
spotting a stalled collector, a saturated queue, a collapsing reward, or a loss that
starts to diverge without reading a long terminal log.

The dashboard is optional and has no effect on a normal training command. The shared
health monitor is separate: it runs even without the web page and is enabled by
default.

## Install the dashboard dependencies

From the repository root:

```bash
python/.venv/bin/python -m pip install -r python/requirements-dashboard.txt
```

This adds Flask and Flask-Sock. Chart.js and the browser styling code are shipped with
Metis, so opening the page does not require an internet connection.

## Start it with a trainer

Add `--dashboard` to any `python/train.py` command:

```bash
python/.venv/bin/python python/train.py \
  --algorithm sac \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --num-envs 4 \
  --checkpoint-dir checkpoints/cars_sac_v1 \
  --dashboard \
  --headless
```

The trainer prints the address after startup:

```text
Dashboard live: http://127.0.0.1:8770
```

Use `--dashboard-port PORT` when that port is already occupied. The server binds only
to the local machine and does not provide authentication or remote access.

## What the page shows

The cards and charts consume the same per-episode values printed by the trainer:

| Panel | Metric keys |
|---|---|
| Reward | `reward` |
| Progress | `progress` with `mean` and `max` |
| Losses | `critic_loss`, `actor_loss` |
| Throughput | `env_steps_s`, `updates_s` |
| Entropy temperature | `alpha` |
| Training rollout outcomes | `finish`, `collision`, `stall`; arm collision timing is `before/at/after` success |
| Frozen policy evaluation | deterministic regular and overall success from checkpoint evaluations |

Training rollout success and frozen-evaluation success are intentionally separate.
Rollouts used for learning may contain random actions, exploration noise, or
stochastic SAC actions. Their success line can therefore remain at zero while the
same checkpoint succeeds in deterministic evaluation. Use rollout outcomes to inspect
the experience entering the learner, and frozen evaluation to judge checkpoint
quality, curriculum promotion, and recovery decisions.

Above the charts, the health panel reports two related values:

- **Health state**: `warming_up`, `healthy`, `warning`, `critical`, or `disabled`.
- **Training phase**: `training`, `curriculum`, `recovering`, `verifying`,
  `stabilizing`, or `finished`.

The accompanying reason is the useful part. It names the frozen metric that dropped,
the number of confirmations still required, a plateau, a numerical failure, or the
checkpoint used for recovery. The recent event list makes transitions visible even
when episodes complete slowly. The right side separates lifetime recovery count from
the attempt count in the current recovery cycle. `Verification: pending` means policy
updates are temporarily held while an isolated evaluator checks the restored policy.

Off-policy logs separate `replay_warmup_left` from `critic_warmup_left`. The first
counts transitions still needed before learning starts and uses
`exploration=warmup_random`; the second belongs to recovery or resume and counts critic
updates while the actor is intentionally frozen. Treating both as one `warmup_left`
made healthy data collection look like a stalled recovery.

When adaptive curriculum is enabled, a `curriculum_promoted` health event marks the
level change. Best-checkpoint and health comparisons then start a new baseline because
reward, success gates, and target distribution are no longer directly comparable to
the previous stage. Evaluations made before
`--curriculum-min-policy-updates` are still useful diagnostics, but they cannot count
toward promotion; this prevents a task that begins near a solution from advancing a
policy whose actor has never been trained.

Demotion is optional. Set `--curriculum-demotion-threshold` below the promotion
threshold and use `--curriculum-demotion-evaluations` to require repeated failures
before moving back one level. The gap between thresholds provides hysteresis and
prevents one noisy evaluation from making the curriculum oscillate. A demotion also
starts a new frozen-evaluation baseline at the restored level.

Not every metric exists for every algorithm or scene. For example, DQN reports a
single `loss`, PPO reports policy and value losses, and only SAC has a learned
`alpha`. A panel with no line therefore means that the current backend did not emit
that field; it does not by itself indicate a training failure.

The **update every** selector controls how many completed episodes the server batches
before pushing a browser update. It reduces redraw frequency for fast experiments but
does not alter environment steps, learner updates, policy synchronization, or the
terminal log.

## History and lifetime

The server keeps the latest 20,000 metric rows in memory. The bundled page loads that
retained HTTP history, while a raw WebSocket connection receives a snapshot of the
latest 1,000 rows. Charts downsample long series for rendering. The HTTP API is also
available at:

```text
http://127.0.0.1:8770/api/metrics
http://127.0.0.1:8770/api/metrics?since=500
http://127.0.0.1:8770/api/health
```

Episode chart history is not written to disk unless `--metrics-jsonl` is used. Health
state is always persistent while monitoring is enabled:

```text
CHECKPOINT_DIR/training_health.json
CHECKPOINT_DIR/training_health_events.jsonl
```

The first file is an atomic current snapshot. The second is an append-only event
history. Both are restored when the trainer resumes from the same checkpoint run.

## How collapse detection works

Metis does not decide that training collapsed because one actor or critic loss moved
up. Loss scales differ by algorithm and reward scale, and a temporary spike can be
normal. The authoritative signal is the existing frozen-policy evaluation:

1. a checkpoint is evaluated deterministically in an isolated Godot process;
2. its success rate, task progress, pose diagnostics, and mean reward are compared with
   the validated best checkpoint when those fields are available;
3. a moderate relative drop produces `warning`;
4. a critical drop must repeat for `--health-collapse-patience` evaluations before
   the state becomes `critical`.

Automatic intervention also has a maturity gate. By default Metis waits for five
frozen evaluations. An explicit `--best-metric success_rate` also requires a
non-trivial success baseline. This prevents an early policy with zero successes from
repeatedly rolling back to another equally immature checkpoint.

`--best-metric task_progress` is intended for sparse tasks that expose
`track_progress` but have not produced a success yet. It ranks success first, then
average progress, position and orientation quality, achieved hold frames, and finally
reward. Once any policy succeeds, success rate still dominates. `auto` uses the same
task-aware ordering when progress diagnostics are present and falls back to success
plus reward otherwise. For tasks with neither progress nor a success signal, use
`--best-metric reward_mean`.

Useful controls are:

```text
--health-warning-drop 0.35
--health-critical-drop 0.60
--health-collapse-patience 2
--health-plateau-evaluations 6
```

A plateau raises an alert but does not cause rollback: returning to an older policy
does not solve a policy that simply stopped improving.

NaN or infinite telemetry is considered critical immediately. Recovery still needs a
previously validated best checkpoint.

## Automatic recovery

Alerts are enabled by default; intervention is not. Enable it explicitly:

```bash
python/.venv/bin/python python/train.py \
  ... \
  --best-checkpoint \
  --auto-recovery \
  --dashboard
```

On confirmed collapse, native Metis trainers archive the current TensorFlow state,
restore the full best checkpoint, reduce learning rates by optimizer role, and
publish the restored policy to async collectors. Experience tagged with an older
policy version is rejected. An isolated frozen evaluation is requested immediately;
policy updates remain paused until that result is available.

Recovery is staged:

1. A **soft** attempt keeps off-policy replay and restores the validated policy.
2. A failed verification escalates later attempts to **hard** recovery. Online replay
   is cleared, protected DDPGfD demonstrations are retained, target networks are
   synchronized, and queued async experience and update credit are discarded.
3. SAC, DDPG, DDPG+BC, DDPGfD, TD3, and TD3+BC then train critics alone before
   resuming less frequent actor updates.
4. DQN has no actor/critic split, so it synchronizes the target Q network and resumes
   after verification. PPO has no replay: it rejects stale rollout generations and
   skips policy optimization during verification.

The main safeguards are:

```text
--recovery-actor-lr-factor 0.033333
--recovery-critic-lr-factor 0.333333
--recovery-alpha-lr-factor 0.033333
--recovery-policy-lr-factor 0.25
--recovery-min-lr-scale 0.01
--recovery-max-attempts 3
--recovery-hard-after-attempt 2
--recovery-critic-warmup-updates 3000
--recovery-policy-update-every 4
--recovery-cycle-reset-evaluations 2
--recovery-min-evaluations 5
--recovery-cooldown-evaluations 2
--recovery-keep-diagnostics 3
```

Role-specific factors are applied relative to the run's configured learning rates,
without going below the minimum scale. The legacy `--recovery-lr-factor` remains a
common fallback when explicitly supplied.

`--recovery-max-attempts` applies to one collapse cycle. A new best checkpoint or the
configured number of healthy frozen evaluations closes that cycle, so a later,
independent collapse receives a fresh attempt budget. The lifetime counter remains
available for diagnosis. Reaching the limit means Metis will keep training and
alerting but will not loop indefinitely over the same failing checkpoint.

Recovery events are printed in the terminal, persisted to JSONL, and shown in the
dashboard. Use `--no-health-monitor` to disable the complete monitor or
`--no-auto-recovery` to retain alerts without intervention.

This mechanism covers DQN, PPO, SAC, DDPG, DDPG+BC, DDPGfD, TD3, and TD3+BC through
their shared Metis checkpoint contract. The SB3 comparison adapter reports finite
telemetry state but does not claim equivalent frozen-evaluation rollback.

## SAC gradient clipping

SAC applies fixed global-norm clipping to its two critics and actor:

```text
--grad-clip-norm 10.0
```

That default is a guard against an unusually large optimizer step. Set the value to
`0` to disable clipping. Adaptive mode keeps a separate exponential running gradient
norm for each network and uses `k` times that value, never exceeding the hard cap:

```text
--grad-clip-adaptive --grad-clip-k 3.0 --grad-clip-norm 10.0
```

The first few updates use the hard cap while the running estimates settle. Adaptive
statistics are runtime state rather than checkpoint state, so they are rebuilt after
a resume. The entropy-temperature gradient is not clipped. The trainer prints the
active mode when the SAC learner starts.

Gradient clipping limits the size of an update; it does not repair invalid rewards,
bad observations, unbounded targets, or an unsuitable learning rate. If losses become
non-finite, inspect the environment contract and replay data as well as the clip
settings.

## Adding metrics from a backend

Algorithm modules should format their episode output through
`core.training.print_episode_metrics()`. That function writes the terminal log and
forwards a flat metric row to registered sinks. Reuse the established keys above when
the meanings match so the dashboard can display a new backend without special-case
code.

Metric sinks run in the learner process and must return quickly. The bundled server
stores rows under a lock and performs WebSocket sends outside that lock; a dashboard
error is isolated from the training loop.
