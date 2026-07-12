# Tutorial: creare un nuovo agente e un nuovo scenario RL

Questa guida spiega come aggiungere un nuovo scenario di reinforcement learning al progetto, mantenendo la separazione attuale:

- **Godot** simula mondo, fisica, agenti, osservazioni e reward.
- **Python/Gymnasium** avvia gli ambienti, sceglie azioni, allena la rete e salva checkpoint.
- **TCP JSON** e' il contratto tra Godot e Python.

L'idea fondamentale e': se il nuovo scenario rispetta lo stesso protocollo, puoi riusare quasi tutto il codice Python.

---

## 1. Prima decisione: cosa vuoi addestrare?

Prima di scrivere codice, definisci queste cose su carta.

### Scenario

Esempi:

- carri che combattono in arena;
- droni che raccolgono risorse;
- robot che raggiungono un target;
- squadre che conquistano una zona;
- auto che completano un circuito;
- agenti che difendono una base.

### Agenti

Devi sapere:

- quanti agenti ci sono;
- quali sono controllati da Python;
- quali sono scripted;
- se il training e' cooperativo, competitivo o misto;
- se tutti condividono la stessa rete o se hanno policy diverse.

Esempio:

```text
Scenario: capture point
Agenti: A0, A1, B0, B1
Team controllati da Python: Team A e Team B
Obiettivo: stare nella zona centrale piu' a lungo degli avversari
Training: self-play competitivo
Policy: condivisa tra tutti gli agenti
```

### Azioni

Definisci un action space discreto.

Esempio:

```text
0 idle
1 forward
2 backward
3 turn_left
4 turn_right
5 forward_left
6 forward_right
7 interact
8 shoot
```

Questo significa:

```bash
--num-actions 9
```

### Osservazioni

Definisci cosa ogni agente vede.

Esempio:

```text
1. alive
2. hp_norm
3. self_x_norm
4. self_z_norm
5. target_x_local
6. target_z_local
7. target_distance
8. nearest_enemy_x_local
9. nearest_enemy_z_local
10. nearest_enemy_distance
11. nearest_ally_x_local
12. nearest_ally_z_local
13. ray_front
14. ray_left
15. ray_right
16. can_interact
17. reload_norm
18. speed_norm
```

Qui hai 18 feature:

```bash
--obs-dim 18
```

Se Godot restituisce 18 valori, Python deve costruire la rete con `obs_dim=18`.

---

## 2. Struttura consigliata dei file

Per un nuovo scenario, non modificare tutto il vecchio scenario. Crea file nuovi.

Esempio per uno scenario `capture_point`:

```text
godot/scenes/capture_point.tscn
godot/scripts/capture_point_controller.gd
godot/scripts/capture_point_sensor_system.gd
godot/scripts/capture_point_reward_system.gd
godot/scripts/capture_point_unit.gd
```

Puoi riusare:

```text
godot/scripts/bridge_server.gd
python/scenario_gym_env.py
python/train_generic.py
python/run_generic_policy.py
python/godot_process_manager.py
python/replay_buffer.py
python/models.py
```

I vecchi script TeamBattle sono in `python/legacy/team_battle/` e `godot/legacy/team_battle/`, e vanno usati solo come riferimento storico.

---

## 3. Contratto che Godot deve rispettare

Il `BridgeServer` parla con un controller Godot. Il controller del nuovo scenario deve implementare questi metodi:

```gdscript
func reset_episode(seed: int, observed_teams: Array = [0]) -> Dictionary

func step_episode(
	action_map: Dictionary,
	controlled_teams: Array = [0],
	observed_teams: Array = [0]
) -> Dictionary

func configure(config: Dictionary) -> Dictionary
```

### `reset_episode`

Resetta il mondo e restituisce osservazioni iniziali.

Formato atteso:

```gdscript
return {
	"agents": [
		{
			"id": "A0",
			"obs": [...]
		},
		{
			"id": "A1",
			"obs": [...]
		}
	],
	"info": {
		"episode_step": 0
	}
}
```

### `step_episode`

Riceve azioni da Python, avanza la simulazione e restituisce nuove osservazioni, reward e done.

Formato atteso:

```gdscript
return {
	"agents": [
		{
			"id": "A0",
			"obs": [...],
			"reward": 0.12,
			"done": false,
			"alive": true
		}
	],
	"terminated": false,
	"truncated": false,
	"info": {
		"episode_step": step_count,
		"winner": -1
	}
}
```

### `configure`

Serve per curriculum e parametri runtime.

Esempio:

```gdscript
func configure(config: Dictionary) -> Dictionary:
	if config.has("max_steps"):
		max_steps = int(config.max_steps)
	if config.has("obstacle_count"):
		obstacle_count = int(config.obstacle_count)
	if config.has("spawn_randomness"):
		spawn_randomness = float(config.spawn_randomness)

	return {
		"ok": true,
		"max_steps": max_steps,
		"obstacle_count": obstacle_count,
		"spawn_randomness": spawn_randomness
	}
```

---

## 4. Creare un nuovo agente Godot

Un agente deve avere almeno:

```gdscript
extends CharacterBody3D

@export var move_speed := 5.0
@export var turn_speed := 2.0
@export var max_hp := 100.0
@export var team_id := 0

var hp := 100.0
var alive := true
var current_action := 0

func reset_unit(new_transform: Transform3D, new_team_id: int) -> void:
	team_id = new_team_id
	global_transform = new_transform
	velocity = Vector3.ZERO
	hp = max_hp
	alive = true
	current_action = 0
	visible = true
	if has_node("CollisionShape3D"):
		$CollisionShape3D.disabled = false

func apply_action(action: int) -> void:
	current_action = action

func sim_step(delta: float) -> void:
	if not alive:
		return

	match current_action:
		0:
			velocity = Vector3.ZERO
		1:
			velocity = -global_transform.basis.z * move_speed
		2:
			velocity = global_transform.basis.z * move_speed
		3:
			rotate_y(-turn_speed * delta)
		4:
			rotate_y(turn_speed * delta)
		_:
			velocity = Vector3.ZERO

	move_and_slide()

func take_damage(amount: float) -> bool:
	if not alive:
		return false
	hp -= amount
	if hp <= 0.0:
		hp = 0.0
		alive = false
		visible = false
		velocity = Vector3.ZERO
		if has_node("CollisionShape3D"):
			$CollisionShape3D.disabled = true
		return true
	return false

func hp_norm() -> float:
	return hp / max_hp
```

Importante:

- `apply_action()` non deve simulare il mondo da sola: deve solo impostare intenzioni.
- `sim_step()` applica davvero movimento/fisica.
- Il controller chiama `sim_step()` per tutti gli agenti.

---

## 5. Creare il sensor system

Il sensor system trasforma lo stato Godot in vettori numerici.

Template:

```gdscript
extends Node3D

@export var controller_path: NodePath
@export var ray_length := 7.0

var controller

func _ready() -> void:
	controller = get_node(controller_path)

func get_team_channels(team_id: int) -> Array:
	var channels: Array = []
	var idx := 0
	for unit in controller.get_team_units(team_id):
		channels.append({
			"id": unit.name,
			"obs": _agent_observation(unit, team_id, idx)
		})
		idx += 1
	return channels

func get_teams_channels(team_ids: Array) -> Array:
	var channels: Array = []
	for team_id in team_ids:
		channels.append_array(get_team_channels(int(team_id)))
	return channels

func _agent_observation(agent, team_id: int, agent_index: int) -> Array:
	var obs: Array = []

	obs.append(float(agent_index))
	obs.append(1.0 if agent.alive else 0.0)
	obs.append(agent.hp_norm())
	obs.append(clampf(agent.global_position.x / 20.0, -1.0, 1.0))
	obs.append(clampf(agent.global_position.z / 20.0, -1.0, 1.0))

	var nearest_enemy = controller.nearest_enemy(agent, team_id)
	if nearest_enemy != null:
		var enemy_vec = nearest_enemy.global_position - agent.global_position
		var enemy_local = agent.global_transform.basis.inverse() * enemy_vec
		obs.append(clampf(enemy_local.x / 20.0, -1.0, 1.0))
		obs.append(clampf(enemy_local.z / 20.0, -1.0, 1.0))
		obs.append(clampf(enemy_vec.length() / 20.0, 0.0, 1.0))
	else:
		obs.append(0.0)
		obs.append(0.0)
		obs.append(1.0)

	obs.append(_ray_distance_normalized(agent, Vector3(0, 0, -1)))
	obs.append(_ray_distance_normalized(agent, Vector3(-1, 0, 0)))
	obs.append(_ray_distance_normalized(agent, Vector3(1, 0, 0)))

	return obs

func _ray_distance_normalized(agent, local_dir: Vector3) -> float:
	var space_state = get_world_3d().direct_space_state
	var from = agent.global_position + Vector3.UP * 0.6
	var world_dir = agent.global_transform.basis * local_dir.normalized()
	var to = from + world_dir * ray_length
	var query := PhysicsRayQueryParameters3D.create(from, to)
	query.exclude = [agent]
	var hit := space_state.intersect_ray(query)
	if hit.is_empty():
		return 1.0
	return clamp(from.distance_to(hit.position) / ray_length, 0.0, 1.0)
```

Regola d'oro: tutte le feature devono essere numeri ragionevolmente normalizzati, spesso tra `-1` e `1` oppure tra `0` e `1`.

---

## 6. Creare il reward system

Il reward system assegna un numero a ogni agente.

Template:

```gdscript
extends Node

@export var controller_path: NodePath

var controller
var prev_hp := {}
var prev_distance_to_target := {}

func _ready() -> void:
	controller = get_node(controller_path)
	reset_reward()

func reset_reward() -> void:
	prev_hp.clear()
	prev_distance_to_target.clear()
	for unit in controller.get_all_units():
		prev_hp[unit.name] = unit.hp
		prev_distance_to_target[unit.name] = controller.distance_to_objective(unit)

func compute_per_agent_rewards(step_events: Dictionary, winner: int) -> Dictionary:
	var rewards := {}

	for unit in controller.get_all_units():
		rewards[unit.name] = -0.002

		var old_dist := float(prev_distance_to_target.get(unit.name, controller.distance_to_objective(unit)))
		var new_dist := controller.distance_to_objective(unit)
		rewards[unit.name] += (old_dist - new_dist) * 0.01
		prev_distance_to_target[unit.name] = new_dist

		var old_hp := float(prev_hp.get(unit.name, unit.max_hp))
		if unit.hp < old_hp:
			rewards[unit.name] -= (old_hp - unit.hp) * 0.015
		prev_hp[unit.name] = unit.hp

	for agent_name in step_events.get("objective_ticks", {}).keys():
		rewards[agent_name] += float(step_events.objective_ticks[agent_name]) * 0.02

	for agent_name in step_events.get("kills", {}).keys():
		rewards[agent_name] += float(step_events.kills[agent_name]) * 1.0

	if winner != -1:
		for unit in controller.get_all_units():
			if unit.team_id == winner:
				rewards[unit.name] += 3.0
			else:
				rewards[unit.name] -= 3.0

	return rewards
```

Consigli:

- Usa reward dense all'inizio.
- Penalizza morte, bordi, stallo, collisioni inutili.
- Premia progressi intermedi, non solo vittoria finale.
- Tieni le reward in range moderati: valori enormi rendono DQN instabile.

---

## 7. Creare il controller dello scenario

Il controller coordina tutto.

Responsabilita':

- reset degli agenti;
- applicare azioni;
- avanzare la simulazione;
- calcolare eventi;
- calcolare reward;
- costruire risposta per Python.

Template:

```gdscript
extends Node

@export var agents_root_path: NodePath
@export var sensors_path: NodePath
@export var reward_system_path: NodePath
@export var max_steps := 400

var agents_root
var sensors
var reward_system
var step_count := 0

func _ready() -> void:
	agents_root = get_node(agents_root_path)
	sensors = get_node(sensors_path)
	reward_system = get_node(reward_system_path)

func get_team_units(team_id: int) -> Array:
	var result: Array = []
	for unit in agents_root.get_children():
		if unit.team_id == team_id:
			result.append(unit)
	return result

func get_all_units() -> Array:
	return agents_root.get_children()

func reset_episode(seed: int, observed_teams: Array = [0]) -> Dictionary:
	step_count = 0
	seed(seed)

	_reset_spawns()
	reward_system.reset_reward()

	return {
		"agents": sensors.get_teams_channels(observed_teams),
		"info": {
			"episode_step": step_count,
			"winner": -1
		}
	}

func configure(config: Dictionary) -> Dictionary:
	if config.has("max_steps"):
		max_steps = int(config.max_steps)
	return {"ok": true, "max_steps": max_steps}

func step_episode(action_map: Dictionary, controlled_teams: Array = [0], observed_teams: Array = [0]) -> Dictionary:
	step_count += 1

	for unit in get_all_units():
		if unit.alive and _team_is_controlled(unit.team_id, controlled_teams):
			unit.apply_action(int(action_map.get(unit.name, 0)))
		else:
			unit.apply_action(_scripted_action(unit))

	for _i in range(4):
		for unit in get_all_units():
			unit.sim_step(1.0 / 30.0)

	var step_events := _collect_step_events()
	var winner := _winner_team_id()
	var truncated := step_count >= max_steps
	var per_agent_rewards = reward_system.compute_per_agent_rewards(step_events, winner)

	var channels: Array = []
	for ch in sensors.get_teams_channels(observed_teams):
		var unit = _find_unit(ch.id)
		channels.append({
			"id": ch.id,
			"obs": ch.obs,
			"reward": float(per_agent_rewards.get(ch.id, 0.0)),
			"done": winner != -1 or truncated or not unit.alive,
			"alive": unit.alive
		})

	return {
		"agents": channels,
		"terminated": winner != -1,
		"truncated": truncated,
		"info": {
			"episode_step": step_count,
			"winner": winner
		}
	}

func _team_is_controlled(team_id: int, controlled_teams: Array) -> bool:
	for item in controlled_teams:
		if int(item) == team_id:
			return true
	return false
```

Questo e' lo scheletro. Poi devi implementare:

```gdscript
func _reset_spawns()
func _scripted_action(unit)
func _collect_step_events()
func _winner_team_id()
func _find_unit(unit_name: String)
```

---

## 8. Creare la scena Godot

Nella nuova scena devi collegare i NodePath.

Struttura consigliata:

```text
CapturePointMain
├── Agents
│   ├── A0
│   ├── A1
│   ├── B0
│   └── B1
├── ObjectiveArea
├── Obstacles
├── SensorSystem
├── RewardSystem
├── ScenarioController
└── BridgeServer
```

Nel `BridgeServer` imposta:

```text
controller_path = ../ScenarioController
```

Nel `SensorSystem` imposta:

```text
controller_path = ../ScenarioController
```

Nel `RewardSystem` imposta:

```text
controller_path = ../ScenarioController
```

Nel `ScenarioController` imposta:

```text
agents_root_path = ../Agents
sensors_path = ../SensorSystem
reward_system_path = ../RewardSystem
```

---

## 9. Avviare il trainer Python

Per nuovi scenari non creare un trainer specifico. Fai esporre agli agenti Godot `get_observations()` e `get_action_space()`, poi usa `train_generic.py`.

Esempio per azioni discrete:

```bash
python/.venv/bin/python python/train_generic.py \
  --algorithm dqn \
  --godot-bin /percorso/a/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/capture_point/capture_point.tscn \
  --num-envs 4 \
  --multi-agent \
  --headless
```

Esempio per azioni continue:

```bash
python/.venv/bin/python python/train_generic.py \
  --algorithm sac \
  --godot-bin /percorso/a/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --num-envs 4 \
  --multi-agent \
  --headless
```

Con `--algorithm auto`, Python legge lo scenario e sceglie DQN per azioni discrete, DDPG per azioni continue e PPO per azioni ibride. SAC va selezionato esplicitamente.

Per azioni continue puoi guidare l'esplorazione casuale iniziale direttamente da Godot:

```gdscript
func get_action_space() -> Dictionary:
	return {
		"move_input": {
			"size": 1,
			"action_type": "continuous",
			"low": 0.0,
			"high": 1.0,
			"exploration_low": 0.55,
			"exploration_high": 1.0
		},
		"rotation_input": {
			"size": 1,
			"action_type": "continuous",
			"low": -1.0,
			"high": 1.0,
			"exploration_low": -0.25,
			"exploration_high": 0.25
		}
	}
```

Se `exploration_low` e `exploration_high` non sono presenti, Python campiona casualmente tra `low` e `high`.

Nelle scene nuove, preferisci dichiarare questo sotto l'agente:

```text
Agent
  ActionSpace
    ContinuousAction move_input
    ContinuousAction rotation_input
  ObservationSystem
    MethodObservationSource
    RaycastObservationSource
    TargetRaycastObservationSource
  RewardSystem
    RewardComponent...
```

Il corpo concreto, ad esempio `Car` o `Tank`, dovrebbe restare responsabile soprattutto di fisica, input applicati e metodi domain-specific come `get_signed_forward_speed()` o `get_control_input()`.

---

## 9.1 Reward locali e reward di scenario

Usa due livelli:

- `Agent/RewardSystem`: reward che riguardano il singolo corpo o i suoi sensori, ad esempio collisione, velocita', input troppo bruschi, raycast vicini.
- `ScenarioController/ScenarioRewardSystem`: reward che richiedono conoscenza dello scenario, ad esempio progresso, target raggiunto, regole di episodio, stall o pace penalty.

I componenti di scenario gia' disponibili sono:

- `ProgressDeltaScenarioReward`: premia l'aumento del progresso e penalizza la sua diminuzione, senza conoscere come viene misurato.
- `EventScenarioReward`: assegna un bonus quando un evento configurabile diventa vero.
- `ProgressStallScenarioReward`: rileva agenti che non migliorano il proprio progresso.
- `ProgressPaceScenarioReward`: penalizza chi non copre una soglia minima di progresso entro una finestra di step.

Il controller invia a Python:

```text
reward = local_reward + scenario_reward
info.local_term_rewards
info.scenario_reward
info.scenario_terms
```

Quindi durante il debug puoi capire se l'agente sta guadagnando per comportamento locale o per avanzamento nello scenario.

Il progresso viene fornito dal nodo `ScenarioController/ProgressProvider`. Il controller e le reward vedono soltanto un valore numerico; la sorgente puo' essere sostituita dall'Inspector:

- `Path3DProgressProvider`: percentuale lungo una curva e, facoltativamente, spawn del curriculum sulla curva.
- `MethodProgressProvider`: chiama un metodo dell'agente o di un nodo dello scenario.
- uno script personalizzato derivato da `ProgressProvider`: puo' misurare distanza da un obiettivo, percentuale di oggetti raccolti, punteggio, fasi completate o qualsiasi altra metrica.

Per Cars viene usato `Path3DProgressProvider`, ma la progressione non entra nelle observations: serve soltanto a reward, diagnostica e curriculum. La policy continua quindi a guidare usando sensori e dinamica del veicolo, senza conoscere la posizione sulla pista.

Gli eventi dello scenario sono configurati nello stesso modo sotto `ScenarioEventSystem`:

- `AreaReachedEventSource`: attiva un evento quando un agente entra in una `Area3D` e puo' terminare l'episodio dell'agente.
- `ObservationThresholdEventSource`: attiva un evento quando una observation supera una soglia, con latch facoltativo.
- uno script derivato da `ScenarioEventSource`: puo' rappresentare raccolta oggetti, timer, punteggi, fasi o condizioni personalizzate.

Cars usa gli eventi `finish_reached` e `target_first_seen`. `EventScenarioReward` legge `finish_reached` dal contesto senza conoscere l'`Area3D` che lo ha generato.

---

## 10. Curriculum

Il curriculum serve a partire facile e aumentare difficolta'.

Esempi:

```text
Episodi 0-200: arena piccola, niente ostacoli
Episodi 200-500: arena media, pochi ostacoli
Episodi 500-1000: arena completa, ostacoli normali
Episodi 1000+: spawn random e match lunghi
```

In Python:

```python
def curriculum_config(episode, args):
	progress = min(1.0, episode / 1000.0)
	return {
		"max_steps": int(120 + progress * 280),
		"obstacle_count": int(progress * 8),
		"spawn_randomness": progress,
	}
```

Nel loop:

```python
config = curriculum_config(episode, args)
for env in envs:
	env.configure(**config)
```

In Godot:

```gdscript
func configure(config: Dictionary) -> Dictionary:
	if config.has("max_steps"):
		max_steps = int(config.max_steps)
	if config.has("obstacle_count"):
		obstacle_count = int(config.obstacle_count)
		_rebuild_obstacles()
	if config.has("spawn_randomness"):
		spawn_randomness = float(config.spawn_randomness)
	return {"ok": true}
```

---

## 11. Debug prima del training

Non partire subito col training. Prima fai tre test.

### Test 1: avvio Godot

Avvia la scena e controlla log:

```text
[BridgeServer] Listening on port 5555
```

### Test 2: reset e step random

Crea uno script tipo o usa `random_scenario_rollout.py`:

```python
from scenario_gym_env import ScenarioGymEnv

env = ScenarioGymEnv(
	port=5555,
	multi_agent=True,
)

obs, info = env.reset()
print(obs.shape, info)

for i in range(20):
	action = env.action_space.sample()
	obs, reward, terminated, truncated, info = env.step(action)
	print(i, reward, info["per_agent_rewards"], terminated, truncated)
	if terminated or truncated:
		break

env.close()
```

Verifica:

- `obs.shape` deve essere `(num_agents, obs_dim)`;
- le reward devono cambiare;
- `terminated` e `truncated` devono avere senso;
- nessun agente deve produrre `NaN`.

### Test 3: rollout con policy non addestrata

Usa lo script di run o un modello random. Serve solo a vedere che tutto si muove.

---

## 12. Training consigliato

Per scenario nuovo:

```bash
python train_capture_point_dqn.py \
  --headless \
  --num-envs 4 \
  --num-episodes 2000 \
  --batch-size 128 \
  --learning-rate 0.0005 \
  --replay-warmup 4000 \
  --epsilon-decay 0.997
```

Se il training e' instabile:

```bash
--learning-rate 0.00025
```

Se esplora troppo poco:

```bash
--epsilon-decay 0.999
```

Se non impara nulla:

- controlla che le osservazioni contengano davvero le informazioni necessarie;
- controlla che le reward siano dense;
- riduci la difficolta';
- guarda il comportamento con `--no-headless`;
- stampa reward separate per agente.

---

## 13. Usare il modello addestrato

Durante il training vengono salvati checkpoint:

```text
checkpoints/capture_point_dqn/
```

e pesi finali:

```text
capture_point_dqn_weights.weights.h5
```

Per usare i pesi devi ricostruire la stessa rete:

```python
from models import build_shared_q_network

model = build_shared_q_network(obs_dim=18, num_actions=9)
model.load_weights("capture_point_dqn_weights.weights.h5")
```

Per usare checkpoint:

```python
import tensorflow as tf

model = build_shared_q_network(obs_dim=18, num_actions=9)
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint("checkpoints/capture_point_dqn")).expect_partial()
```

Attenzione:

- se cambi `obs_dim`, i vecchi pesi non sono compatibili;
- se cambi `num_actions`, i vecchi pesi non sono compatibili;
- se cambi architettura della rete in `models.py`, i vecchi pesi possono non essere compatibili.

---

## 14. Errori tipici

### Il training parte ma non impara

Possibili cause:

- reward troppo sparse;
- osservazioni insufficienti;
- azioni non applicate correttamente;
- agenti bloccati da fisica/collisioni;
- epsilon cala troppo velocemente;
- learning rate troppo alto.

### `obs_dim` mismatch

Sintomo:

```text
ValueError: could not broadcast input array
```

Soluzione:

- conta le feature restituite da Godot;
- aggiorna `--obs-dim`;
- aggiorna default nel trainer.

### Agenti fermi

Controlla:

- `apply_action()` riceve azioni giuste?
- `sim_step()` viene chiamato?
- la fisica ha collisioni valide?
- i team controllati sono corretti?

### Godot si disconnette

Guarda:

```text
python/logs/godot_<porta>.log
```

Quasi sempre e' un errore GDScript runtime o parse.

### Reward sempre negative

Non e' sempre un problema. All'inizio e' normale. Diventa un problema se dopo molte migliaia di step:

- non aumenta il danno;
- non migliora la distanza dagli obiettivi;
- gli agenti ripetono azioni senza senso.

---

## 15. Checklist finale

Prima di lanciare un training lungo:

- [ ] La scena Godot parte.
- [ ] Il `BridgeServer` stampa `Listening`.
- [ ] `reset_episode()` restituisce osservazioni.
- [ ] `step_episode()` applica azioni.
- [ ] `obs_dim` in Python combacia con Godot.
- [ ] `num_actions` in Python combacia con `apply_action()`.
- [ ] Le reward cambiano durante un rollout random.
- [ ] `terminated` e `truncated` funzionano.
- [ ] Il curriculum non crea configurazioni impossibili.
- [ ] Il checkpoint viene salvato.
- [ ] `run_generic_policy.py` riesce a caricare il modello.

---

## 16. Regola pratica

Non cercare subito scenario complesso.

Progressione consigliata:

```text
1 agente, target fermo
1 agente, target random
1v1 senza ostacoli
1v1 con ostacoli
2v2 scripted
2v2 self-play
curriculum completo
```

Ogni passaggio deve essere osservabile e debuggabile. Se un agente non impara una versione semplice, quasi mai impara quella complessa.
