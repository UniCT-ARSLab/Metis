# Monitoring training

Metis can serve a small live dashboard from the training process. It is useful for
spotting a stalled collector, a saturated queue, a collapsing reward, or a loss that
starts to diverge without reading a long terminal log.

The dashboard is optional and has no effect on a normal training command.

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
| Outcomes | `finish`, `collision`, `stall` |

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
```

History is not written to disk and disappears when training exits. Checkpoints,
replay snapshots, policies, and best-policy evaluation remain the persistent record
of a run.

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
