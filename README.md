# Godot Gymnasium Keras RL Framework

Piccolo framework sperimentale per addestrare agenti di reinforcement learning in scenari Godot usando Gymnasium e TensorFlow/Keras.

Godot gestisce simulazione, agenti, sensori, reward locali e bridge TCP. Python interroga lo scenario, legge observation/action space esposti dagli agenti e avvia il trainer più adatto.

Per creare un nuovo agente o scenario, vedi [Tutorial nuovo scenario/agente](docs/tutorial_nuovo_scenario_agente_rl.md).

## Architettura

- Una istanza Godot esegue uno scenario.
- `BridgeServer` espone reset, step, spec e configurazione via TCP.
- `ScenarioController` coordina agenti, reset, reward di scenario e terminal state.
- Ogni agente Godot espone observation space, action space, reward e done.
- `Agent/ActionSpace` dichiara azioni discrete, continue o ibride da Inspector.
- `Agent/ObservationSystem` registra observation source riusabili come metodi del corpo, raycast e sensori target.
- Python usa `ScenarioGymEnv` come wrapper Gymnasium generico.
- `train_generic.py` seleziona o inoltra al backend di training.

Il framework supporta scenari single-agent e multi-agent. In multi-agent il trainer salva transizioni per agente nel replay buffer, usando una policy condivisa quando gli agenti hanno observation/action space compatibili.

## Python Attuale

File principali:

- `python/train_generic.py`: entrypoint unico per il training.
- `python/train_generic_dqn.py`: azioni discrete.
- `python/train_generic_ddpg.py`: azioni continue.
- `python/train_generic_sac.py`: azioni continue con SAC.
- `python/train_generic_ppo.py`: azioni ibride.
- `python/run_generic_policy.py`: esecuzione di un modello addestrato.
- `python/record_demonstrations.py`: registrazione demo manuali.
- `python/random_scenario_rollout.py`: rollout casuale per validare uno scenario.
- `python/scenario_gym_env.py`: wrapper Gymnasium generico.
- `python/godot_process_manager.py`: avvio/stop istanze Godot.
- `python/models.py`: reti Keras condivise.
- `python/replay_buffer.py`: replay buffer.

La vecchia linea TeamBattle/DQN è archiviata in `python/legacy/team_battle/`.
La vecchia scena Godot TeamBattle è archiviata in `godot/legacy/team_battle/`.

## Installazione

```bash
cd python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Linux con GPU NVIDIA/CUDA:

```bash
pip install -r requirements-linux-cuda.txt
```

macOS Apple Silicon con Metal:

```bash
xcode-select --install
pip install -r requirements-macos-metal.txt
```

## Training Generico

Esempio SAC per lo scenario Cars:

```bash
python/.venv/bin/python python/train_generic.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --num-envs 4 \
  --num-episodes 2500 \
  --max-steps-per-episode 800 \
  --batch-size 128 \
  --replay-warmup 12000 \
  --random-exploration-episodes 40 \
  --action-smoothing 0.25 \
  --reset-progress-curriculum \
  --reset-progress-start-max 0.015 \
  --reset-progress-end-max 0.75 \
  --reset-progress-ramp-episodes 1600 \
  --checkpoint-dir checkpoints/cars_sac_rewards_v1 \
  --actor-weights-path cars_sac_rewards_actor_v1.weights.h5 \
  --critic1-weights-path cars_sac_rewards_critic1_v1.weights.h5 \
  --critic2-weights-path cars_sac_rewards_critic2_v1.weights.h5 \
  --multi-agent \
  --resume
```

Con `--algorithm auto`, il launcher sceglie:

- `dqn` per action space discreti.
- `ddpg` per action space continui.
- `ppo` per action space ibridi.

SAC va selezionato esplicitamente con `--algorithm sac`.

Per SAC, ogni checkpoint nuovo salva anche il replay buffer associato, per esempio
`ckpt-1050.*` e `replay-1050.npz`. `--resume` continua a caricare automaticamente
l'ultimo checkpoint dentro `--checkpoint-dir`. Per ripartire da uno stato preciso usa
invece `--resume-checkpoint checkpoints/nome_run/ckpt-1025`; la directory indicata con
`--checkpoint-dir` resta la destinazione dei checkpoint successivi e puo' quindi essere
diversa dalla directory sorgente. Se un vecchio checkpoint non ha il relativo replay,
il trainer ricostruisce il buffer e mantiene gli aggiornamenti disabilitati fino a
`--replay-warmup`.

Dopo un resume SAC, per default i primi 2000 gradient step aggiornano soltanto i
critic e i target critic. Actor e alpha restano congelati finche' le stime Q non si
sono riadattate al replay ripristinato. Il valore si configura con
`--critic-warmup-updates` e si disabilita impostandolo a `0`.

Tutti i trainer supportano `--log-format pretty` (default) e `--log-format compact`.
Con `Ctrl+C` salvano l'ultimo episodio completato, i pesi e, per SAC/DDPG/DQN,
anche il replay buffer; poi chiudono connessioni e processi Godot senza traceback.
Attendi il messaggio `Interrupted training saved` prima di chiudere il terminale.

Per azioni continue, la fase di esplorazione casuale iniziale usa eventuali limiti `exploration_low` e `exploration_high` dichiarati nello `action_space` dell'agente Godot. Se questi limiti non sono presenti, Python campiona uniformemente tra `low` e `high`.

## Reward

Le reward locali dell'agente stanno nel nodo `Agent/RewardSystem` e sono composte da figli `RewardComponent`, configurabili da Inspector.

Le reward che dipendono dallo scenario stanno in un nodo `ScenarioRewardSystem` collegato al `ScenarioController`. Per Cars, qui vivono progresso sul `Path3D`, penalita' di arretramento, target finale, stall e pace penalty. Il controller somma `local_term_rewards` e `scenario_reward`, poi invia a Python anche `scenario_terms` per debug.

## Validare Uno Scenario

Per provare lo scenario con azioni casuali:

```bash
python/.venv/bin/python python/random_scenario_rollout.py \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --multi-agent \
  --headless
```

## Eseguire Un Modello

Esempio SAC:

```bash
python/.venv/bin/python python/run_generic_policy.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --actor-weights-path cars_sac_rewards_actor_v1.weights.h5 \
  --multi-agent \
  --no-headless
```

Per osservare a lungo:

```bash
python/.venv/bin/python python/run_generic_policy.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --actor-weights-path cars_sac_rewards_actor_v1.weights.h5 \
  --multi-agent \
  --infinite \
  --no-time-limit \
  --no-headless
```

## Demo Manuali

Le demo si registrano con `record_demonstrations.py` e possono essere usate per prefill del replay buffer o behavior cloning. Vedi [Dimostrazioni Manuali](docs/manual_demonstrations.md).

## Note

Lo stato attuale punta a rendere Godot la fonte di verità per:

- observation space;
- action space;
- reward locali;
- terminal state;
- numero e identità degli agenti.

Python dovrebbe diventare sempre più automatico: dato uno scenario Godot valido, deve poter scegliere o ricevere l'algoritmo, costruire il modello compatibile e addestrare senza script specifici per scenario.
