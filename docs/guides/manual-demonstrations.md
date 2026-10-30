# Manual demonstrations

`python/recorder.py` lets a person control one Godot agent and stores the resulting transitions in a compressed `.npz` dataset. DQN can use discrete demonstrations; DDPG, SAC, TD3, and their demonstration variants can use continuous data. Standard single-policy PPO accepts expert observations and action labels for BC pretraining and optional auxiliary imitation, including discrete, continuous, multi-discrete and hybrid contracts.

## Choose the demonstration workflow

| Workflow | Supported path | What remains during training |
|---|---|---|
| Replay prefill | DQN and continuous off-policy trainers | Complete expert transitions enter ordinary replay and may be overwritten unless protected. |
| Protected replay | SAC with `--demo-protect`; DDPGfD's existing protected replay | Expert transitions remain available; protection alone does not preserve cloned behavior or require a fixed expert fraction. |
| PPO imitation | Standard single-policy PPO | A separate read-only label dataset supplies optional BC and `--ppo-demo-coef`; labels never enter on-policy rollouts or GAE. |
| Continuous online BC | DDPG+BC, TD3+BC, DDPGfD | A supervised term continues alongside the task objective, according to each variant's schedule. |

The Godot Train wizard exposes SAC's **Protect demonstrations** checkbox and PPO's **Expert dataset (NPZ)**, **BC pretraining epochs**, **Expert imitation weight**, and **Imitation decay updates** fields. Additional supported flags are available in **All options**. Demonstration support is not identical across algorithms or independent multi-policy modes; use the matching algorithm guide rather than copying every demo flag between trainers.

The current dataset contains:

```text
obs
actions
rewards
next_obs
dones
agent_ids
episode_indices
step_indices
action_names
action_type
obs_dim
action_size
num_actions
```

The current recorder saves a combined `dones` flag, true for either a natural terminal state or a truncation; it does not save separate `terminated` and `truncated` arrays. That recording format cannot reconstruct the distinction afterward. For replay tasks where timeout bootstrapping matters, use true-terminal recordings or prepare verified termination labels rather than assuming a timeout is a task failure. PPO's supervised observation/action labels do not depend on these flags. The recorder stores the action that Godot actually applied, not the string `"manual"` used to request manual control.

Replay requires complete, genuine transitions: the recorded action must have produced the stored next observation, reward and termination. Corrective action labels attached to another policy's trajectory belong in `--demo-bc-path`, not replay. PPO only reads expert observations and action labels and therefore does not require replay transitions; hybrid labels need the explicit head arrays described in [PPO demonstrations](../algorithms/ppo.md#demonstrations).

## Record one agent

Start Godot with a visible window and use the controls defined in `project.godot`:

```bash
python/.venv/bin/python python/recorder.py \
  --godot-bin "$GODOT_BIN" \
  --godot-project godot \
  --godot-scene res://scenarios/tanks/tanks_scenario.tscn \
  --output demos/tank_target_demo.npz \
  --agent-id Tank \
  --episodes 10 \
  --max-steps 500 \
  --step-delay 0.08 \
  --no-headless
```

Set `GODOT_BIN` to your Godot executable before using these examples. Recording mode disables automatic agent replication. Other agents are held inactive so their non-expert actions do not enter the dataset.

To collect another compatible agent into the same file, use `--append`:

```bash
python/.venv/bin/python python/recorder.py \
  --godot-bin "$GODOT_BIN" \
  --godot-project godot \
  --godot-scene res://scenarios/tanks/tanks_scenario.tscn \
  --output demos/tank_target_demo.npz \
  --append \
  --agent-id Tank2 \
  --episodes 10 \
  --max-steps 500 \
  --step-delay 0.08 \
  --no-headless
```

Only append agents with the same observation and action contract.

## Continuous driving example

```bash
python/.venv/bin/python python/recorder.py \
  --godot-bin "$GODOT_BIN" \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --output demos/cars_track_demo.npz \
  --agent-id Car \
  --episodes 10 \
  --max-steps 800 \
  --step-delay 0.04 \
  --no-headless
```

For Cars, each action is the applied vector:

```text
[move_input, rotation_input]
```

## Replay prefill

Use demonstrations as initial DQN replay without behavior-cloning pretraining:

```bash
python/.venv/bin/python python/train.py \
  --algorithm dqn \
  --godot-project godot \
  --godot-scene res://scenarios/tanks/tanks_scenario.tscn \
  --multi-agent \
  --demo-path demos/tank_target_demo.npz \
  --demo-bc-epochs 0 \
  --checkpoint-dir checkpoints/tanks_demo_prefill \
  --headless
```

## Behavior-cloning pretraining

```bash
python/.venv/bin/python python/train.py \
  --algorithm dqn \
  --godot-project godot \
  --godot-scene res://scenarios/tanks/tanks_scenario.tscn \
  --multi-agent \
  --demo-path demos/tank_target_demo.npz \
  --demo-bc-epochs 10 \
  --demo-bc-batch-size 128 \
  --checkpoint-dir checkpoints/tanks_demo_bc \
  --headless
```

SAC can use the same prefill and pretraining pattern:

```bash
python/.venv/bin/python python/train.py \
  --algorithm sac \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --num-envs 4 \
  --num-episodes 1500 \
  --max-steps-per-episode 800 \
  --replay-warmup 2000 \
  --demo-path demos/cars_track_demo.npz \
  --demo-bc-epochs 5 \
  --checkpoint-dir checkpoints/cars_sac_demo \
  --headless
```

Behavior cloning is skipped on resume by default. Use `--demo-bc-on-resume` only when repeating the supervised phase is intentional.

## Protecting SAC demonstrations

Add `--demo-protect` to the SAC example to reserve the complete prefilled expert transitions against online overwrite. Keep `--demo-prefill` enabled and leave at least one online slot beyond the selected expert dataset. All protected transitions must fit; Metis rejects insufficient capacity rather than silently dropping expert data.

Protection survives replay snapshot saves, resume and hard recovery. Saved replay protection remains authoritative on resume, but an already populated unprotected replay cannot be upgraded in place. Start a new replay with the expert dataset or restore a protected snapshot. A weights-only checkpoint does not contain replay data. The option currently supports single-policy SAC, not independent multi-policy demonstration routing.

This preserves data, not behavior: SAC still samples uniformly, with no guaranteed expert quota or persistent BC objective. Logs expose the protected count and the actual fraction of expert samples drawn by the learner. See [SAC protected demonstrations](../algorithms/sac.md#protected-demonstrations) for capacity, resume and telemetry details.

## PPO pretraining and auxiliary imitation

The following example assumes a compatible expert dataset for the continuous Cars scene. BC runs before ordinary PPO collection, and the auxiliary coefficient keeps expert labels in a separate supervised loss afterward:

```bash
python/.venv/bin/python python/train.py --algorithm ppo --godot-project godot --godot-scene res://scenarios/cars/cars_scenario.tscn --demo-path demos/cars_track_demo.npz --demo-bc-epochs 5 --ppo-demo-coef 0.1 --ppo-demo-decay-updates 100000 --ppo-demo-final-coef 0 --checkpoint-dir checkpoints/cars_ppo_demo --headless
```

The coefficient is a trial setting, not a tuned default. Standard single-policy sync and async collection are supported; residual and independent multi-policy demo training are rejected. `--demo-protect` and replay-prefill options do not apply to PPO. Add `--stop-after-bc` to inspect a clone before RL; PPO saves and stops without automatically evaluating that clone, so run it explicitly with `python/run.py` before continuing.

PPO uses categorical cross-entropy for discrete labels and action-mean MSE for continuous labels. Provide the expert files again on resume: they are not bundled in the checkpoint. Auxiliary imitation binds the data, action contract, seed, batch size and coefficient schedule to the saved run; an intentional objective change belongs in a new warm-start run. See [PPO demonstrations](../algorithms/ppo.md#demonstrations) for dataset shapes, validation, metrics and resume constraints.

## Online behavior cloning

`ddpg_bc` and `td3_bc` keep a behavior-cloning term active while collecting new experience. Demonstrations are sampled separately from online replay:

```bash
python/.venv/bin/python python/train.py \
  --algorithm td3_bc \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --num-envs 4 \
  --demo-path demos/cars_track_demo.npz \
  --demo-bc-weight-start 1.0 \
  --demo-bc-weight-end 0.05 \
  --demo-bc-decay-updates 100000 \
  --checkpoint-dir checkpoints/cars_td3_bc \
  --headless
```

Use `--no-demo-prefill` when the dataset should affect only the BC loss and should not enter replay.

## DDPG from Demonstrations

```bash
python/.venv/bin/python python/train.py \
  --algorithm ddpgfd \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --num-envs 4 \
  --demo-path demos/cars_track_demo.npz \
  --ddpgfd-pretrain-updates 1000 \
  --checkpoint-dir checkpoints/cars_ddpgfd \
  --headless
```

DDPGfD inserts demonstrations first, protects them from replay overwrite, and gives expert samples a priority bonus. Keep `--demo-prefill` enabled. This implementation uses prioritized one-step TD targets; it does not assume adjacent replay entries belong to one trajectory because several agents and Godot workers may interleave data.

## Before using a dataset

- Inspect a few episodes visually. A large mediocre dataset can anchor the policy to mediocre behavior.
- Check that actions cover the corrections the policy will need, not only successful steady motion.
- Record failures only when the algorithm is meant to learn from them.
- Start a new dataset whenever observation order, action structure, or normalization changes.
- Keep training and evaluation demonstrations separate if demonstrations are used to report offline metrics.
- Replay protection and imitation losses do not replace independent frozen evaluation. Optional [training stability controls](../training-stability.md) can qualify, independently test and retain a solved candidate without assuming that later training must remain equally good.
