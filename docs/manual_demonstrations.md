# Dimostrazioni Manuali per gli Agenti

Questa funzione permette di guidare manualmente un agente in Godot e salvare le transizioni in un dataset `.npz` riusabile dal trainer. Con azioni discrete viene usato dal DQN; con azioni continue viene usato dal DDPG.

Il formato salvato contiene:

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

## Registrare una Demo

Avvia Godot con finestra visibile e guida il tank con i comandi definiti in `project.godot`:

```bash
python/.venv/bin/python python/record_demonstrations.py \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --output demos/tank_target_demo.npz \
  --agent-id Tank \
  --episodes 10 \
  --max-steps 500 \
  --step-delay 0.08
```

Per registrare anche gli altri agenti, ripeti appendendo allo stesso file:

```bash
python/.venv/bin/python python/record_demonstrations.py \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --output demos/tank_target_demo.npz \
  --append \
  --agent-id Tank2 \
  --episodes 10 \
  --max-steps 500 \
  --step-delay 0.08
```

E poi:

```bash
python/.venv/bin/python python/record_demonstrations.py \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --output demos/tank_target_demo.npz \
  --append \
  --agent-id Tank3 \
  --episodes 10 \
  --max-steps 500 \
  --step-delay 0.08
```

## Registrare una Demo Cars

Per lo scenario Cars devi passare la scena e lasciare Godot visibile:

```bash
python/.venv/bin/python python/record_demonstrations.py \
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

Il recorder salva l'azione continua realmente applicata:

```text
[move_input, rotation_input]
```

## Usare le Demo nel Training

Solo replay-buffer prefill:

```bash
python/.venv/bin/python python/train_generic_dqn.py \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --multi-agent \
  --headless \
  --demo-path demos/tank_target_demo.npz \
  --demo-bc-epochs 0 \
  --checkpoint-dir checkpoints/generic_dqn_demo_prefill \
  --weights-path generic_dqn_demo_prefill.weights.h5
```

Behavior cloning più DQN:

```bash
python/.venv/bin/python python/train_generic_dqn.py \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --multi-agent \
  --headless \
  --demo-path demos/tank_target_demo.npz \
  --demo-bc-epochs 10 \
  --demo-bc-batch-size 128 \
  --checkpoint-dir checkpoints/generic_dqn_demo_bc \
  --weights-path generic_dqn_demo_bc.weights.h5
```

Per Cars/DDPG, puoi usare lo stesso entrypoint generico:

```bash
python/.venv/bin/python python/train_generic.py \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --num-envs 4 \
  --num-episodes 1500 \
  --max-steps-per-episode 800 \
  --replay-warmup 2000 \
  --demo-path demos/cars_track_demo.npz \
  --demo-bc-epochs 5 \
  --checkpoint-dir checkpoints/cars_track_demo \
  --actor-weights-path cars_actor_track_demo.weights.h5 \
  --critic-weights-path cars_critic_track_demo.weights.h5 \
  --headless
```

Per usare le demo solo per behavior cloning senza riempire il replay buffer:

```bash
--no-demo-prefill --demo-bc-epochs 10
```

## Note Pratiche

Il recorder registra l'azione realmente applicata da Godot, non la stringa `"manual"`. Per agenti discreti salva un action id; per agenti continui salva un vettore di valori.

Di default conviene registrare un agente alla volta. Gli altri agenti restano fermi, evitando di salvare azioni non esperte nel dataset.

Se cambi observation space o action space, le vecchie demo non saranno piu compatibili con il trainer.
