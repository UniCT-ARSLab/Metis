# Adding an RL algorithm to Metis

Use this guide when an algorithm needs genuinely different model, buffer, or update
semantics. If the change is only a loss term or an update rule, extending an existing
backend is usually cleaner than creating another training pipeline.

## 1. Write down the contract

Before touching code, decide:

- supported action types: discrete, continuous, or hybrid;
- on-policy or off-policy behavior;
- required networks and target networks;
- rollout or replay format;
- state that must survive a checkpoint;
- multi-agent transition semantics;
- asynchronous policy lag and synchronization rules;
- demonstration and opponent-pool support.

These choices shape the backend. Starting from a copied trainer and changing it until
it runs often leaves subtle mistakes around truncation, bootstrapping, or update
frequency.

## 2. Pick the closest implementation

- Discrete off-policy: `python/algorithms/dqn.py`.
- On-policy discrete, continuous, or hybrid: `python/algorithms/ppo.py`.
- Stochastic continuous control: `python/algorithms/sac.py`.
- Deterministic continuous control: `python/algorithms/common.py` and a DDPG/TD3 entry
  point.

Add a variant to shared code when the model and data flow are the same. Create a new
module when the buffer, rollout generation, or network contract changes.

## 3. Create the backend module

Place the file directly under `python/algorithms`:

```text
python/algorithms/my_algorithm.py
```

It must expose a no-argument `main()`. `python/train.py` selects the module and leaves
the command-line arguments in `sys.argv`.

```python
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    # Keep shared argument names consistent with the other backends.
    return parser.parse_args()


def main():
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
```

The `__main__` block is handy during development. The documented command remains
`python/train.py`.

## 4. Reuse the runtime

`core.training` provides:

- TensorFlow setup and GPU memory configuration;
- `ParallelEnvStepper` for synchronous collection;
- `AsyncCollectorPool`, async events, and `PolicySnapshot`;
- update schedulers;
- checkpoint and replay-resume helpers;
- `BestCheckpointTracker`;
- structured metrics and clean shutdown.

`envs` provides:

- `GodotProcessManager` for process lifecycle;
- `ScenarioGymEnv` for specification, reset, and step calls.

Do not copy the TCP protocol into an algorithm module, and do not start Godot with a
backend-specific `subprocess.Popen` call.

## 5. Validate the scenario specification

Reject unsupported spaces immediately after connecting the first environment:

```python
if env.action_type != "continuous":
    raise RuntimeError("my_algorithm requires continuous actions")
```

All environments in one run must agree on `obs_dim`, action specification, and agent
layout. In parameter-sharing runs, all learner agents must expose compatible
observation and action contracts.

Read dimensions and bounds from `env.obs_dim`, `env.action_type`, `env.action_size`,
and `env.action_space_spec`. No algorithm should contain scene-specific action names.

## 6. Preserve transition semantics

An off-policy transition needs the natural terminal flag independently of rollout
truncation:

```text
(obs, action, reward, next_obs, terminated)
```

A step limit ends collection but still bootstraps from `next_obs` when `terminated` is
false. An on-policy backend similarly computes the final bootstrap value for a
truncated rollout.

One multi-agent environment step may create several learner transitions. If additional
agents are meant to increase learner work, base update scheduling on transition count,
not only environment-step count.

## 7. Models and collector inference

Put a Keras factory in `core/models.py` only when several backends or `run.py` need it.
Keep algorithm-specific losses and update details in the algorithm module.

A policy snapshot published to collectors must accept the declared observation shape,
return the action format expected by `ScenarioGymEnv`, remain immutable during an
inference call, and carry a `policy_version`. Make it thread-safe or instantiate one
copy per worker.

For stochastic policies, keep training-time sampling separate from deterministic
evaluation behavior.

## 8. Asynchronous collection

Async support requires explicit answers to these questions:

1. When does a worker receive new weights?
2. How many transitions authorize an update?
3. How is queue pressure bounded?
4. Which counters are global and which belong to a worker?
5. How are workers, sockets, and Godot processes stopped on `Ctrl+C`?
6. Are opponent pools and multi-policy assignments actually connected?

Native Metis algorithms are expected to support independent multi-policy collection
in both modes. Async implementations should use `MultiPolicySnapshot` to publish the
complete policy group atomically, include `policy_id` in every transition or
trajectory, and use per-policy update credit through
`AsyncEventScheduler.ingest_by_policy()`. PPO-style on-policy learners also need a
generation barrier: never combine trajectories produced by different group versions.
Off-policy learners keep one replay buffer per policy and may mix older versions only
inside that policy's replay.

Reject unsupported combinations with a clear error. Silently accepting a flag is
worse than documenting a limitation.

## 9. Checkpoint and resume

Register every optimization variable in `tf.train.Checkpoint`: online and target
networks, optimizers, learned temperatures, episode/update counters, and schedules that
cannot be reconstructed from arguments.

Off-policy replay is saved separately. A resumed actor-critic may need a short critic
warmup with the actor frozen, as SAC does, to avoid damaging a restored policy before
the critic has adapted to loaded replay.

Every checkpoint also refreshes enough metadata for `python/run.py` to identify the
algorithm, observation dimension, and action contract.

## 10. Register the backend

In `python/train.py`:

1. Add the backend to `BACKENDS` and the `--algorithm` choices.
2. Decide whether `auto` should select it. Usually the difference between SAC, TD3,
   and DDPG cannot be inferred from a continuous action space, so specialized
   algorithms remain explicit.

In `python/run.py`:

1. Add the algorithm name.
2. Build or load the correct inference model.
3. Restore checkpoints and policy files.
4. Define deterministic and optional stochastic action selection.
5. Validate action type, size, bounds, and output decoding.

Reuse an existing loader when the inference architecture is compatible.

## 11. Useful logs

A backend should report completed episodes, collector mode, rewards, success metrics,
environment steps, transition count, replay or rollout size, actual update count,
primary losses, action statistics, and async throughput. A zero loss must be
distinguishable from an episode in which no update ran.

Send episode output through `core.training.print_episode_metrics()` and register
`add_dashboard_arguments()` plus `maybe_start_dashboard()` in the backend entry point.
This keeps the optional dashboard out of the learning loop while giving terminal logs
and monitoring the same source values.

Reuse shared metric keys when their meaning matches: `reward`, `progress`,
`critic_loss`, `actor_loss`, `env_steps_s`, `updates_s`, `alpha`, `finish`,
`collision`, and `stall`. New fields are preserved by the metrics API even when the
bundled page does not chart them yet.

## 12. Tests before documentation

Cover at least:

- target construction for `terminated` and `truncated` transitions;
- action shapes and bounds;
- one update that changes the expected weights;
- checkpoint save and restore;
- counter resume;
- multi-agent transition counts;
- async policy versions and worker shutdown;
- dispatch through `train.py`;
- loading through `run.py`.
- episode metrics reaching a registered sink without changing terminal formatting.

Run:

```bash
python/.venv/bin/python -m unittest discover -s python/tests
python/.venv/bin/python -m compileall -q python
```

Finish with a smoke test on a deliberately small scene. A complex task is a poor first
test of a new algorithm because scene bugs and learner bugs become hard to separate.
