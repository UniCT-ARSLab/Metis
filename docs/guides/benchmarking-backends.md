# Benchmarking Metis and Stable-Baselines3

Metis includes a limited Stable-Baselines3 compatibility adapter so the same Godot
task can be trained by two independent learner implementations. Its purpose is
comparison. Metis remains the default backend, and compatibility does not mean that
every native feature has an SB3 equivalent.

## What a fair comparison fixes

Comparing "1000 episodes" is misleading when episode length differs. The benchmark
runner instead gives each backend the same number of collected transitions and fixes:

- the Godot scene and process count;
- environment and evaluation seeds;
- the maximum episode length;
- the algorithm family and shared hyperparameters;
- the deterministic evaluation episode count.

Training runs are sequential. This avoids two learners competing for the same GPU,
CPU cores, and Godot processes. Several seeds are required because one RL run is not a
reliable result.

Intermediate checkpoints and replay snapshots are disabled during benchmark runs.
Both can pause a learner for very different amounts of time and would contaminate the
wall-clock comparison. Each trainer still writes its final model for evaluation.

The implementations will still differ internally. Replay sampling, network
initialization, optimizer details, target updates, and exploration schedules are part
of what the benchmark is measuring.

## Install the two backends

The native backend uses TensorFlow/Keras:

```bash
python3 -m venv python/.venv
python/.venv/bin/python -m pip install -r python/requirements.txt
```

SB3 uses PyTorch. Keeping it in a separate environment avoids dependency conflicts:

```bash
python3 -m venv python/.venv-sb3
python/.venv-sb3/bin/python -m pip install -r python/requirements-sb3.txt
```

Both interpreters execute source files from the same repository, so no package
installation step is needed.

## Run a comparison

This example compares DQN on Breakout:

```bash
python/.venv/bin/python python/tools/benchmark_backends.py \
  --name breakout_dqn_250k \
  --algorithm dqn \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/breakout/breakout_scenario.tscn \
  --total-timesteps 250000 \
  --evaluation-episodes 30 \
  --evaluation-max-steps 1500 \
  --seeds 100 101 102 \
  --num-envs 4 \
  --metis-python python/.venv/bin/python \
  --sb3-python python/.venv-sb3/bin/python \
  -- \
  --batch-size 128 \
  --gamma 0.99
```

Arguments after `--` are forwarded to both trainers. Do not put backend-specific
options there. Run separate experiments when a parameter has different semantics in
the two implementations.

The runner uses synchronous vector collection for both backends. SB3's vector
environment advances the independent Godot processes concurrently, but the learner
receives one synchronized batch. This is closer to a controlled comparison than
comparing SB3 against Metis's asynchronous queue.

SB3 does not use Metis's asynchronous collector. `--backend sb3 --collector-mode
async` fails explicitly. Replacing SB3's collection loop with the Metis queue would no
longer be a clean comparison with the upstream algorithm.

## Output

The command creates `benchmarks/NAME/`:

```text
config.json
runs.jsonl
report.json
summary.csv
metis/seed-100/
  command.json
  train.log
  metrics.jsonl
  evaluation.json
  run.json
sb3/seed-100/
  ...
```

`summary.csv` reports mean and standard deviation across completed seeds for:

- training wall time;
- nominal collected transitions per second;
- deterministic evaluation reward;
- success rate;
- evaluation episode length;
- reward area under the learning curve, normalized by transition count.

The raw files matter. Inspect them before trusting the aggregate table, especially
when one run crashed, never reached a terminal state, or learned a high reward without
solving the task.

Use `--success-threshold 0.8` to also measure elapsed time to a logged 80% success
rate. This field is meaningful only when the scenario exposes a reliable success
event.

## Backend boundaries

The SB3 comparison adapter supports:

| Algorithm | Godot action space |
|---|---|
| DQN | discrete |
| PPO | discrete, continuous, or hybrid through latent-Box encoding |
| DDPG | continuous |
| TD3 | continuous |
| SAC | continuous |

For hybrid PPO, continuous components keep their bounds. A discrete component with
`N` choices becomes `N` latent values and is decoded with `argmax`. This permits a
useful experiment, but it is not mathematically identical to Metis PPO's categorical
head. Record this difference when publishing benchmark results.

Multi-agent parameter sharing is available with `--multi-agent`. Every agent
contributes one transition and uses the same SB3 policy. Agents in one Godot process
must terminate together because their physical world can only be reset as a unit. The
safe default raises on partial termination:

```text
--multi-agent --sb3-multi-agent-partial-done error
```

`reset-all` is available for exploratory runs. It truncates agents that were still
active and resets the entire world. This changes episode semantics, so it is not a
fair backend comparison unless Metis is configured to use the same rule.

Current-policy simultaneous self-play works when both teams share the policy and
terminate together. Historical opponent pools, independent multi-policy training,
demonstration prefill, BC variants, and asynchronous collectors remain native Metis
features.

SB3 saves `.zip` models and `.pkl` replay buffers. Metis saves TensorFlow checkpoints,
`.npz` replay buffers, and a portable Keras policy bundle. The benchmark evaluates
each format through its own backend and compares the resulting episode statistics;
it does not convert one model format into the other.

## Multi-agent comparison example

Pong has coordinated point termination, so it fits the safe grouped-lane model:

```bash
python/.venv/bin/python python/tools/benchmark_backends.py \
  --name pong_dqn_shared \
  --algorithm dqn \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/pong/pong_scenario.tscn \
  --total-timesteps 250000 \
  --evaluation-episodes 50 \
  --seeds 100 101 102 \
  --num-envs 4 \
  --multi-agent \
  --metis-python python/.venv/bin/python \
  --sb3-python python/.venv-sb3/bin/python
```

For Tanks, select PPO because the movement plus weapon contract is hybrid. The SB3
side will report `sb3_action=hybrid_box(...)` at startup so the encoding is visible in
the raw benchmark log.
