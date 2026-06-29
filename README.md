# Godot 4 + Gymnasium + TensorFlow/Keras MARL starter

Questo progetto implementa una base **multi-agente 2v2** per iniziare con:
- **Godot 4** come simulatore 3D;
- **Gymnasium** come interfaccia environment Python;
- **TensorFlow/Keras** per il training;
- **parameter sharing** tra i due agenti trainabili di Team A.

Per creare un nuovo agente o un nuovo scenario RL, vedi:
[Tutorial nuovo scenario/agente](docs/tutorial_nuovo_scenario_agente_rl.md).

## Idea architetturale

L'ambiente naturale resta il **match completo** 2v2, ma il protocollo espone **canali logici separati per agente**.

Quindi:
- una istanza Godot = un match;
- ogni agente Team A ha il suo `id`, la sua osservazione, la sua reward e il suo flag di done;
- Python usa un wrapper Gymnasium unico per il match;
- il trainer salva le transizioni **per agente** nel replay buffer, ma con una policy Keras condivisa.

Questo consente di allenare i singoli agenti senza rompere la simulazione condivisa del mondo.

## Team

- `TeamA`: 2 agenti trainabili (`A0`, `A1`)
- `TeamB`: 2 agenti scripted baseline (`B0`, `B1`)

## Azioni per agente

8 azioni discrete per ciascun agente:
- `0`: idle
- `1`: forward
- `2`: turn_left
- `3`: turn_right
- `4`: forward_left
- `5`: forward_right
- `6`: shoot
- `7`: forward_shoot

L'action space Gymnasium dell'ambiente è:
```python
spaces.MultiDiscrete([8, 8])
```

## Osservazione per agente

Ogni agente Team A riceve 18 feature:
1. agent_id_norm
2. alive
3. hp_norm
4. reload_norm
5. speed_norm
6. self_x_norm
7. self_z_norm
8. nearest_enemy_local_x
9. nearest_enemy_local_z
10. nearest_enemy_dist
11. enemy_visible
12. nearest_ally_local_x
13. nearest_ally_local_z
14. ray_front
15. ray_front_left
16. ray_front_right
17. ray_left
18. ray_right

Osservazione totale del match:
- shape `(2, 18)`

I sensori non sono nodi `RayCast3D` nella scena: sono raycast programmatici eseguiti da `multi_agent_sensor_system.gd` con `PhysicsRayQueryParameters3D`. Le feature `ray_front`, `ray_front_left`, `ray_front_right`, `ray_left`, `ray_right` danno distanza normalizzata da ostacoli lungo direzioni locali del carro.

## Reward

Ogni agente Team A ha reward individuale, composta da:
- danno inflitto al nemico,
- danno subito,
- kill,
- morte,
- bonus vittoria/sconfitta di team,
- penalità se il carro forza il bordo arena,
- lieve step penalty.

L'environment Gymnasium restituisce come reward scalare la **media** delle reward individuali per compatibilità con la firma standard, ma in `info["per_agent_rewards"]` trovi le reward separate usate dal trainer custom.

## Protocollo TCP

### Richieste
```json
{"cmd":"hello","version":1}
{"cmd":"reset","seed":123}
{"cmd":"step","actions":{"A0":7,"A1":6}}
{"cmd":"close"}
```

### Risposta reset
```json
{
  "agents": [
    {"id":"A0","obs":[...]},
    {"id":"A1","obs":[...]}
  ],
  "info": {"episode_step":0}
}
```

### Risposta step
```json
{
  "agents": [
    {"id":"A0","obs":[...],"reward":0.12,"done":false,"alive":true},
    {"id":"A1","obs":[...],"reward":-0.03,"done":false,"alive":true}
  ],
  "terminated": false,
  "truncated": false,
  "info": {
    "episode_step": 14,
    "winner": -1,
    "team_a_alive": 2,
    "team_b_alive": 1
  }
}
```

## Contenuto del progetto

### Godot
- `godot/project.godot`
- `godot/scenes/main.tscn`
- `godot/scripts/main.gd`
- `godot/scripts/bridge_server.gd`
- `godot/scripts/tank_unit.gd`
- `godot/scripts/multi_agent_sensor_system.gd`
- `godot/scripts/multi_agent_reward_system.gd`
- `godot/scripts/team_battle_controller.gd`

### Python
- `python/requirements.txt`
- `python/team_battle_gym_env.py`
- `python/godot_process_manager.py`
- `python/replay_buffer.py`
- `python/models.py`
- `python/validate_env.py`
- `python/random_rollout.py`
- `python/train_shared_dqn.py`
- `python/train_self_play_dqn.py`
- `python/run_trained_policy.py`

## Avvio rapido

### 1. Godot
Apri `godot/` con Godot 4 ed esegui il progetto. La scena principale ascolta sulla porta 5555.

### 2. Python
```bash
cd python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python validate_env.py
python random_rollout.py
```

Su Linux con GPU NVIDIA/CUDA:
```bash
pip install -r requirements-linux-cuda.txt
```

Su macOS Apple Silicon con Metal:
```bash
xcode-select --install
pip install -r requirements-macos-metal.txt
```

Su Mac il backend Metal viene registrato da TensorFlow tramite `tensorflow-metal`; lo script stamperà i device GPU rilevati all'avvio.

### 3. Training parallelo
Imposta il path del binario Godot:
```bash
export GODOT_BIN=/percorso/al/binario/godot
python train_shared_dqn.py
```

Lo script lancerà più match Godot headless su porte diverse.

Esempio con argomenti:
```bash
python train_shared_dqn.py \
  --headless \
  --num-envs 4 \
  --num-episodes 2000 \
  --batch-size 128 \
  --learning-rate 0.0005 \
  --checkpoint-every 25
```

Per riprendere dall'ultimo checkpoint:
```bash
python train_shared_dqn.py --headless --resume
```

I checkpoint salvano modello, target model, optimizer, episodio ed epsilon in `checkpoints/shared_dqn` di default. Il replay buffer non viene salvato, quindi dopo un resume viene riempito di nuovo.

### 4. Self-play con curriculum
```bash
python train_self_play_dqn.py --headless
```

Questo trainer controlla tutti e quattro gli agenti (`A0`, `A1`, `B0`, `B1`) con una policy DQN condivisa. Godot restituisce osservazioni e reward simmetriche per entrambi i team, quindi Team B non usa la policy scripted durante il self-play.

Il curriculum è attivo di default e aumenta progressivamente la lunghezza massima degli episodi:
```bash
python train_self_play_dqn.py \
  --headless \
  --curriculum-start-steps 120 \
  --curriculum-ramp-episodes 600 \
  --max-steps-per-episode 400
```

Per disattivarlo:
```bash
python train_self_play_dqn.py --headless --no-curriculum
```

Per riprendere:
```bash
python train_self_play_dqn.py --headless --resume
```

### 5. Usare un modello addestrato

I file `*.weights.h5` salvano solo i pesi Keras: per usarli bisogna ricostruire la stessa rete e caricare i pesi. Lo script `run_trained_policy.py` lo fa automaticamente e pilota Godot via TCP.

Eseguire la policy self-play/all-agents da pesi finali:
```bash
python run_trained_policy.py \
  --mode self-play \
  --weights-path self_play_dqn_weights.weights.h5
```

Caricare l'ultimo checkpoint invece del file weights:
```bash
python run_trained_policy.py \
  --mode self-play \
  --load-from checkpoint \
  --checkpoint-dir checkpoints/self_play_dqn
```

Usare un modello Team A contro Team B scripted:
```bash
python run_trained_policy.py \
  --mode team-a \
  --weights-path shared_dqn_weights.weights.h5
```

Se Godot è già aperto e ascolta sulla porta desiderata:
```bash
python run_trained_policy.py --connect-only --port 5555
```

Per osservare la policy in modo continuo finché non premi `Ctrl+C`:
```bash
python run_trained_policy.py \
  --mode self-play \
  --load-from checkpoint \
  --checkpoint-dir checkpoints/self_play_dqn \
  --infinite \
  --no-time-limit \
  --no-headless
```

`--infinite` fa ripartire un nuovo episodio quando il match termina. `--no-time-limit` rimuove il troncamento artificiale a `max_steps`, quindi una partita finisce solo per una condizione terminale del gioco.

## Nota

Questa v1 usa:
- reward individuale,
- parameter sharing,
- Team B scripted.
- checkpoint/resume del trainer.
- self-play all-agents con policy condivisa.
- curriculum sulla durata degli episodi.

I passi successivi naturali sono:
- osservazioni più ricche,
- policy separate o central critic.
