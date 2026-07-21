# Manual demonstrations

`python/recorder.py` lets a person control one Godot agent and stores the resulting
transitions in a compressed `.npz` dataset. DQN can use discrete demonstrations;
DDPG, SAC, TD3, and their demonstration variants can use continuous data.

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

`dones` is true for either a natural terminal state or a truncation. The recorder also
stores the action that Godot actually applied, not the string `"manual"` used to
request manual control.

## Record one agent

Start Godot with a visible window and use the controls defined in `project.godot`:

```bash
python/.venv/bin/python python/recorder.py \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/tanks/tanks_scenario.tscn \
  --output demos/tank_target_demo.npz \
  --agent-id Tank \
  --episodes 10 \
  --max-steps 500 \
  --step-delay 0.08 \
  --no-headless
```

Recording mode disables automatic agent replication. Other agents are held inactive
so their non-expert actions do not enter the dataset.

To collect another compatible agent into the same file, use `--append`:

```bash
python/.venv/bin/python python/recorder.py \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
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
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
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

Behavior cloning is skipped on resume by default. Use `--demo-bc-on-resume` only when
repeating the supervised phase is intentional.

## Online behavior cloning

`ddpg_bc` and `td3_bc` keep a behavior-cloning term active while collecting new
experience. Demonstrations are sampled separately from online replay:

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

Use `--no-demo-prefill` when the dataset should affect only the BC loss and should not
enter replay.

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

DDPGfD inserts demonstrations first, protects them from replay overwrite, and gives
expert samples a priority bonus. Keep `--demo-prefill` enabled. This implementation
uses prioritized one-step TD targets; it does not assume adjacent replay entries belong
to one trajectory because several agents and Godot workers may interleave data.

## Before using a dataset

- Inspect a few episodes visually. A large mediocre dataset can anchor the policy to
  mediocre behavior.
- Check that actions cover the corrections the policy will need, not only successful
  steady motion.
- Record failures only when the algorithm is meant to learn from them.
- Start a new dataset whenever observation order, action structure, or normalization
  changes.
- Keep training and evaluation demonstrations separate if demonstrations are used to
  report offline metrics.
