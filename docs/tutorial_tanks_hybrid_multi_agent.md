# Tutorial: Tanks 2v2 multi-agent con azioni ibride

Questo tutorial rifonda lo scenario Tanks come arena arcade 2v2. Due squadre, rossa e
blu, esplorano una mappa sconosciuta, evitano muri e ostacoli tramite sensori locali e
sparano missili quando trovano un avversario.

Non riutilizziamo il vecchio `scenarios/tanks/scenario_controller.gd`: contiene bridge,
reward e reset specifici ormai sostituiti dal framework generico. Il nuovo scenario
usa `ScenarioController.gd`, `Agent`, `ObservationSystem`, `ScenarioEventSystem` e
`ScenarioRewardSystem`.

## 1. Cosa deve imparare la policy

Ogni carro deve imparare contemporaneamente a:

1. accelerare, frenare o arretrare;
2. ruotare senza urtare gli ostacoli;
3. esplorare senza conoscere la mappa;
4. distinguere alleati e nemici visibili;
5. allineare il cannone con un nemico;
6. scegliere quando sparare;
7. contribuire alla vittoria della squadra.

Non forniamo Path3D, coordinate del nemico, waypoint o una griglia della mappa. Una
parete interrompe il RayCast: il carro non puo' vedere attraverso gli ostacoli.

## 2. Multi-agent e policy condivisa

La prima versione usa quattro agenti:

```text
RedTank1   team_id=0
RedTank2   team_id=0
BlueTank1  team_id=1
BlueTank2  team_id=1
```

Sono quattro istanze della stessa scena e usano una sola policy PPO condivisa. Il
colore e il numero della squadra non entrano nelle observation. I sensori restituiscono
soltanto relazioni `ally` o `enemy`, quindi la policy funziona allo stesso modo per
rosso e blu.

Con quattro env otteniamo sedici traiettorie contemporanee:

```text
4 env x 4 carri = 16 agenti, 1 modello condiviso
```

Questa e' una forma di self-play simultaneo. Con `--opponent-pool` il trainer mantiene
sempre una sola policy allenabile, ma l'altra squadra puo' eseguire una snapshot
storica congelata. Due policy diverse entrambe allenabili richiederebbero invece un
trainer multi-policy e non sono il comportamento del trainer generico.

## 3. Action space ibrido

L'action space contiene due componenti nominati:

```gdscript
{
	"movement": {
		"size": 2,
		"action_type": "continuous",
		"low": -1.0,
		"high": 1.0
	},
	"weapon": {
		"size": 2,
		"action_type": "discrete",
		"names": ["hold_fire", "fire"]
	}
}
```

Il payload ricevuto dal carro sara' simile a:

```gdscript
{
	"movement": [0.72, -0.18],
	"weapon": 1
}
```

`movement[0]` e' accelerazione; `movement[1]` e' rotazione. `weapon=1` richiede lo
sparo. La rete PPO ha quindi una testa gaussiana continua e una testa categorica
discreta, oltre al value estimator.

## 4. Observation locali

Una configurazione iniziale ragionevole e':

| Gruppo | Valori | Descrizione |
|---|---:|---|
| velocita' | 2 | avanti firmata e modulo normalizzato |
| input precedenti | 2 | accelerazione e rotazione correnti |
| stato | 3 | vita, ricarica pronta, vivo |
| ostacoli | 7 | distanze RayCast corte normalizzate |
| nemici | 4 | visibile, sinistra, centro, destra |
| alleati | 3 | sinistra, centro, destra |

Totale indicativo: 21 valori. Il numero effettivo dipende dai RayCast configurati.

Non includere:

- posizione globale;
- rotazione globale;
- coordinate degli avversari;
- indice del layout;
- `team_id`;
- progresso lungo un percorso.

Gli input precedenti aiutano PPO a interpretare inerzia e frenata. `reload_ready`
evita che debba dedurre il cooldown contando gli step.

## 5. Creare TankBattleAgent.tscn

Crea `res://agents/TankBattle/tank_battle_agent.tscn`:

```text
TankBattleAgent (CharacterBody3D) [tank_battle_agent.gd]
├── Mesh
├── CollisionShape3D
├── Muzzle (Marker3D)
├── ObstacleSensors (Node3D)
│   ├── FrontLeft (RayCast3D)
│   ├── Front (RayCast3D)
│   ├── FrontRight (RayCast3D)
│   ├── Left (RayCast3D)
│   ├── Right (RayCast3D)
│   ├── RearLeft (RayCast3D)
│   └── RearRight (RayCast3D)
├── VisionSensors (Node3D)
│   ├── VisionLeft (RayCast3D)
│   ├── VisionCenter (RayCast3D)
│   └── VisionRight (RayCast3D)
└── Agent [Agent.gd]
    ├── ActionSpace [ActionSpace.gd]
    │   ├── Movement [ContinuousAction.gd]
    │   └── Weapon [DiscreteActionSet.gd]
    ├── ObservationSystem [ObservationSystem.gd]
    │   ├── BodySpeed [BodySpeedObservationSource.gd]
    │   ├── Throttle [MethodObservationSource.gd]
    │   ├── Steering [MethodObservationSource.gd]
    │   ├── Health [MethodObservationSource.gd]
    │   ├── ReloadReady [MethodObservationSource.gd]
    │   ├── Alive [MethodObservationSource.gd]
    │   ├── Obstacles [RaycastObservationSource.gd]
    │   └── TeamVision [TeamRaycastObservationSource.gd]
    └── RewardSystem [RewardSystem.gd]
        ├── TimePenalty [StepPenaltyReward.gd]
        └── InvalidFirePenalty [EventReward.gd]
```

`TeamRaycastObservationSource` e' un nodo del framework. Cerca sul collider o sui suoi
genitori il metodo `get_team_id()` e confronta il risultato con quello del corpo che
osserva.

## 6. Configurare ActionSpace

Nel nodo `Movement`:

```text
action_name = "movement"
size = 2
low = -1.0
high = 1.0
```

Nel nodo `Weapon`:

```text
action_name = "weapon"
names = ["hold_fire", "fire"]
```

La presenza contemporanea di almeno una componente continua e una discreta fa
restituire automaticamente `action_type="hybrid"`. `train_generic.py --algorithm
auto` selezionera' PPO.

## 7. Configurare le observation

`BodySpeed`:

```text
include_forward_speed = true
include_absolute_speed = true
forward_speed_name = "forward_speed"
absolute_speed_name = "speed"
speed_scale = 12.0
```

I tre `MethodObservationSource` di stato:

```text
Throttle:
  observation_name = "throttle_input"
  method_name = "get_control_input"
  bind_string_arg = "throttle_input"

Steering:
  observation_name = "rotation_input"
  method_name = "get_control_input"
  bind_string_arg = "rotation_input"

Health:
  observation_name = "health"
  method_name = "get_health_observation"

ReloadReady:
  observation_name = "reload_ready"
  method_name = "get_reload_ready_observation"

Alive:
  observation_name = "alive"
  method_name = "get_alive_observation"
```

Per `Obstacles`, imposta:

```text
root_path = "../../../ObstacleSensors"
observation_prefix = "obstacle"
```

I RayCast ostacoli devono essere relativamente corti e collidere con muri, ostacoli e
carri. Restituiscono `1` quando liberi e valori vicini a `0` quando l'ostacolo e'
vicino.

Per `TeamVision` configura gli stessi tre RayCast sia come nemici sia come alleati:

```text
enemy_visible_observation_name = "enemy_visible"

enemy_signal_observations = {
  "enemy_left": "VisionSensors/VisionLeft",
  "enemy_center": "VisionSensors/VisionCenter",
  "enemy_right": "VisionSensors/VisionRight"
}

ally_signal_observations = {
  "ally_left": "VisionSensors/VisionLeft",
  "ally_center": "VisionSensors/VisionCenter",
  "ally_right": "VisionSensors/VisionRight"
}
```

I RayCast di visione devono essere piu' lunghi. Se incontrano prima un muro, tutte le
observation target di quel raggio valgono zero: questa e' l'occlusione desiderata.

## 8. Scrivere tank_battle_agent.gd

```gdscript
extends CharacterBody3D
class_name BattleTank

signal damage_received(victim:BattleTank, attacker:BattleTank, amount:float)
signal destroyed(victim:BattleTank, killer:BattleTank)

@export_category("Team")
@export var team_id := 0

@export_category("Movement")
@export var max_forward_speed := 12.0
@export var max_reverse_speed := 5.0
@export var acceleration := 18.0
@export var drag := 12.0
@export var turn_speed_degrees := 110.0

@export_category("Combat")
@export var max_health := 100.0
@export var fire_cooldown := 0.8
@export var projectile_scene: PackedScene
@export var projectile_parent: Node

@export_category("Control")
@export var manual_control := false

@onready var agent: Agent = $Agent
@onready var muzzle: Marker3D = $Muzzle

var _throttle_input := 0.0
var _rotation_input := 0.0
var _cooldown_left := 0.0
var _health := 100.0
var _alive := true
var _match_terminal := false
var _training_active := true
var _initial_collision_layer := 0


func _ready() -> void:
	_health = max_health
	_initial_collision_layer = collision_layer


func _physics_process(delta:float) -> void:
	_cooldown_left = maxf(_cooldown_left - delta, 0.0)
	if not _training_active or not _alive:
		return

	if manual_control:
		_throttle_input = Input.get_axis("move_back", "move_forward")
		_rotation_input = Input.get_axis("turn_left", "turn_right")

	rotate_y(-_rotation_input * deg_to_rad(turn_speed_degrees) * delta)

	var speed_limit := max_forward_speed if _throttle_input >= 0.0 else max_reverse_speed
	var desired_velocity := global_transform.basis.z * _throttle_input * speed_limit
	var flat_velocity := Vector3(velocity.x, 0.0, velocity.z)
	if absf(_throttle_input) > 0.05:
		flat_velocity = flat_velocity.move_toward(desired_velocity, acceleration * delta)
	else:
		flat_velocity = flat_velocity.move_toward(Vector3.ZERO, drag * delta)

	velocity.x = flat_velocity.x
	velocity.z = flat_velocity.z
	if not is_on_floor():
		velocity += get_gravity() * delta
	move_and_slide()


func apply_action(action:Variant) -> Variant:
	if typeof(action) == TYPE_STRING or typeof(action) == TYPE_STRING_NAME:
		if str(action) == "manual":
			return apply_manual_action()

	if not _alive or typeof(action) != TYPE_DICTIONARY:
		clear_inputs()
		return {"movement": [0.0, 0.0], "weapon": 0}

	var action_map: Dictionary = action
	var movement := agent.decode_continuous_action(action_map)
	_throttle_input = clampf(float(movement[0]), -1.0, 1.0)
	_rotation_input = clampf(float(movement[1]), -1.0, 1.0)
	var weapon_action := int(action_map.get("weapon", 0))
	if weapon_action == 1:
		try_fire()
	return {
		"movement": [_throttle_input, _rotation_input],
		"weapon": weapon_action
	}


func apply_manual_action() -> Dictionary:
	return apply_action({
		"movement": [
			Input.get_axis("move_back", "move_forward"),
			Input.get_axis("turn_left", "turn_right")
		],
		"weapon": 1 if Input.is_action_pressed("fire") else 0
	})


func try_fire() -> bool:
	if not _alive or _cooldown_left > 0.0 or projectile_scene == null:
		agent.add_reward_event("invalid_fire", 1.0)
		return false

	var projectile := projectile_scene.instantiate()
	var parent := projectile_parent if projectile_parent != null else get_tree().current_scene
	parent.add_child(projectile)
	projectile.global_transform = muzzle.global_transform
	projectile.launch(self)
	_cooldown_left = fire_cooldown
	return true


func take_damage(attacker:BattleTank, amount:float) -> void:
	if not _alive:
		return
	_health = maxf(_health - amount, 0.0)
	damage_received.emit(self, attacker, amount)
	if _health <= 0.0:
		_alive = false
		visible = false
		collision_layer = 0
		clear_inputs()
		velocity = Vector3.ZERO
		destroyed.emit(self, attacker)


func get_team_id() -> int:
	return team_id


func get_control_input(input_name:String) -> float:
	if input_name == "throttle_input":
		return _throttle_input
	if input_name == "rotation_input":
		return _rotation_input
	return 0.0


func get_health_observation() -> float:
	return clampf(_health / maxf(max_health, 0.001), 0.0, 1.0)


func get_reload_ready_observation() -> float:
	return 1.0 if _cooldown_left <= 0.0 else 0.0


func get_alive_observation() -> float:
	return 1.0 if _alive else 0.0


func is_alive() -> bool:
	return _alive


func set_match_terminal(value:bool) -> void:
	_match_terminal = value


func is_terminal() -> bool:
	return _match_terminal


func clear_inputs() -> void:
	_throttle_input = 0.0
	_rotation_input = 0.0


func reset_all(original_transform:Variant, reset_rewards := true) -> void:
	set_training_active(true)
	if typeof(original_transform) == TYPE_TRANSFORM3D:
		transform = original_transform
	_health = max_health
	_alive = true
	_match_terminal = false
	_cooldown_left = 0.0
	visible = true
	collision_layer = _initial_collision_layer
	clear_inputs()
	velocity = Vector3.ZERO
	if has_method("reset_physics_interpolation"):
		reset_physics_interpolation()
	agent.reset_observation_sources()
	if reset_rewards:
		agent.refresh_observation_sources()
		agent.reset_reward({"body": self})


func set_training_active(enabled:bool) -> void:
	_training_active = enabled
	set_physics_process(enabled)
	if not enabled:
		clear_inputs()
		velocity = Vector3.ZERO
```

Un carro distrutto non termina subito il proprio canale: rimane immobile e invisibile
fino alla fine del match. Cosi' riceve anche la reward finale di vittoria o sconfitta
della squadra. Tutti gli agenti diventano terminali insieme quando una squadra e'
eliminata.

## 9. Reward locali del carro

Configura `TimePenalty`:

```text
penalty = -0.001
term_name = "time"
```

Configura `InvalidFirePenalty`:

```text
event_name = "invalid_fire"
scale = -0.01
term_name = "invalid_fire"
```

Non aggiungere una reward elevata per `enemy_visible`: porterebbe i carri a guardare
un avversario senza sparare. Se vuoi shaping visivo, usa al massimo un valore molto
piccolo e controlla che non domini hit e vittorie.

## 10. Creare Missile.tscn

```text
Missile (CharacterBody3D) [missile.gd]
├── MeshInstance3D
└── CollisionShape3D
```

Assegna il gruppo `projectile` alla radice e usa:

```gdscript
extends CharacterBody3D
class_name TankMissile

@export var speed := 28.0
@export var damage := 34.0
@export var max_lifetime := 4.0

var shooter: BattleTank
var team_id := -1
var _lifetime := 0.0


func launch(owner_tank:BattleTank) -> void:
	shooter = owner_tank
	team_id = owner_tank.get_team_id()
	velocity = global_transform.basis.z.normalized() * speed
	add_collision_exception_with(owner_tank)


func _physics_process(delta:float) -> void:
	_lifetime += delta
	if _lifetime >= max_lifetime:
		queue_free()
		return

	var collision := move_and_collide(velocity * delta)
	if collision == null:
		return

	var collider := collision.get_collider()
	if collider is BattleTank:
		if collider.get_team_id() == team_id:
			add_collision_exception_with(collider)
			global_position += velocity.normalized() * 0.1
			return
		collider.take_damage(shooter, damage)
	queue_free()
```

Il missile non riceve observation e non e' un agente. E' soltanto una conseguenza
fisica dell'azione discreta.

## 11. Creare tank_battle_scenario.tscn

```text
TankBattleScenario (Node3D) [tank_battle_manager.gd]
├── BridgeServer [bridge_server.gd]
├── ScenarioController [ScenarioController.gd]
│   ├── ScenarioEventSystem [ScenarioEventSystem.gd]
│   │   ├── EnemyHit [ManualScenarioEventSource.gd]
│   │   ├── EnemyKill [ManualScenarioEventSource.gd]
│   │   ├── TeamKill [ManualScenarioEventSource.gd]
│   │   ├── DamageTaken [ManualScenarioEventSource.gd]
│   │   ├── Destroyed [ManualScenarioEventSource.gd]
│   │   ├── TeamWon [ManualScenarioEventSource.gd]
│   │   ├── TeamLost [ManualScenarioEventSource.gd]
│   │   └── MatchDraw [ManualScenarioEventSource.gd]
│   └── ScenarioRewardSystem [ScenarioRewardSystem.gd]
│       ├── HitReward [EventScenarioReward.gd]
│       ├── KillReward [EventScenarioReward.gd]
│       ├── TeamKillReward [EventScenarioReward.gd]
│       ├── DamagePenalty [EventScenarioReward.gd]
│       ├── DestroyedPenalty [EventScenarioReward.gd]
│       ├── WinReward [EventScenarioReward.gd]
│       └── LossPenalty [EventScenarioReward.gd]
├── Environment
│   ├── Floor
│   ├── Walls
│   └── Layouts
│       ├── EasyLayout
│       ├── MediumLayout
│       └── HardLayout
├── Agents
│   ├── RedTank1
│   ├── RedTank2
│   ├── BlueTank1
│   └── BlueTank2
└── Projectiles
```

Nel `ScenarioController` imposta manualmente tutti i carri:

```text
controlled_agents = [
  "../Agents/RedTank1",
  "../Agents/RedTank2",
  "../Agents/BlueTank1",
  "../Agents/BlueTank2"
]
number_of_replications = 0
randomize_reset = true
reset_position_jitter = (0.5, 0.0, 0.5)
reset_yaw_jitter_degrees = 15
max_steps = 800
```

Assegna `projectile_parent` di ogni carro al nodo `Projectiles`.

`TeamWon`, `TeamLost` e `MatchDraw` devono avere un `terminal_reason` non vuoto. Gli
altri eventi sono impulsi non terminali.

Configura gli event source con questi nomi:

| Nodo | event_name | terminal_reason |
|---|---|---|
| EnemyHit | `enemy_hit` | vuoto |
| EnemyKill | `enemy_kill` | vuoto |
| TeamKill | `team_enemy_killed` | vuoto |
| DamageTaken | `damage_taken` | vuoto |
| Destroyed | `destroyed` | vuoto |
| TeamWon | `team_won` | `team_won` |
| TeamLost | `team_lost` | `team_lost` |
| MatchDraw | `match_draw` | `match_draw` |

## 12. Configurare reward di scenario

Valori iniziali prudenti:

| Componente | Evento | Reward | only_once |
|---|---|---:|---|
| HitReward | `enemy_hit` | `+0.10` | false |
| KillReward | `enemy_kill` | `+0.80` | false |
| TeamKillReward | `team_enemy_killed` | `+0.20` | false |
| DamagePenalty | `damage_taken` | `-0.10` | false |
| DestroyedPenalty | `destroyed` | `-0.80` | true |
| WinReward | `team_won` | `+2.00` | true |
| LossPenalty | `team_lost` | `-2.00` | true |

Il killer riceve hit, kill e reward cooperativa; il compagno riceve la reward
cooperativa. La vittoria resta il segnale piu' importante.

Non usare una somma globale delle reward dei quattro carri. Ogni agente riceve i propri
eventi e soltanto le reward cooperative esplicitamente assegnate alla sua squadra.

## 13. Scrivere tank_battle_manager.gd

```gdscript
extends Node3D

@export var tanks: Array[BattleTank] = []
@export var layouts: Array[Node3D] = []

@onready var controller := $ScenarioController
@onready var enemy_hit: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/EnemyHit
@onready var enemy_kill: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/EnemyKill
@onready var team_kill: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/TeamKill
@onready var damage_taken: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/DamageTaken
@onready var destroyed_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/Destroyed
@onready var team_won: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/TeamWon
@onready var team_lost: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/TeamLost
@onready var match_draw: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/MatchDraw

var _training_episode := 0
var _match_finished := false


func _ready() -> void:
	controller.scenario_configured.connect(_on_scenario_configured)
	controller.episode_reset_started.connect(_reset_match)
	for tank in tanks:
		tank.damage_received.connect(_on_damage_received)
		tank.destroyed.connect(_on_tank_destroyed)


func _on_scenario_configured(config:Dictionary) -> void:
	_training_episode = int(config.get("training_episode", _training_episode))


func _reset_match(seed:int) -> void:
	_match_finished = false
	for projectile in get_tree().get_nodes_in_group("projectile"):
		projectile.queue_free()
	_select_layout(seed)
	for tank in tanks:
		tank.set_match_terminal(false)


func _on_damage_received(victim:BattleTank, attacker:BattleTank, _amount:float) -> void:
	if _match_finished or attacker == null or attacker.get_team_id() == victim.get_team_id():
		return
	enemy_hit.trigger(str(attacker.name))
	damage_taken.trigger(str(victim.name))


func _on_tank_destroyed(victim:BattleTank, killer:BattleTank) -> void:
	if _match_finished:
		return
	destroyed_event.trigger(str(victim.name))
	if killer != null and killer.get_team_id() != victim.get_team_id():
		enemy_kill.trigger(str(killer.name))
		for teammate in tanks:
			if teammate.get_team_id() == killer.get_team_id():
				team_kill.trigger(str(teammate.name))
	_check_match_finished()


func _check_match_finished() -> void:
	var red_alive := _alive_count(0)
	var blue_alive := _alive_count(1)
	if red_alive > 0 and blue_alive > 0:
		return

	_match_finished = true
	if red_alive == 0 and blue_alive == 0:
		for tank in tanks:
			match_draw.trigger(str(tank.name))
	else:
		var winner_team := 0 if red_alive > 0 else 1
		for tank in tanks:
			if tank.get_team_id() == winner_team:
				team_won.trigger(str(tank.name))
			else:
				team_lost.trigger(str(tank.name))

	for tank in tanks:
		tank.set_match_terminal(true)


func _alive_count(team:int) -> int:
	var result := 0
	for tank in tanks:
		if tank.get_team_id() == team and tank.is_alive():
			result += 1
	return result


func _select_layout(seed:int) -> void:
	if layouts.is_empty():
		return
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	var available_layouts := 1
	if _training_episode >= 500:
		available_layouts = mini(2, layouts.size())
	if _training_episode >= 1500:
		available_layouts = layouts.size()
	var selected := rng.randi_range(0, max(available_layouts - 1, 0))
	for idx in range(layouts.size()):
		layouts[idx].visible = idx == selected
		layouts[idx].process_mode = (
			Node.PROCESS_MODE_INHERIT if idx == selected
			else Node.PROCESS_MODE_DISABLED
		)
```

Ogni layout contiene corpi e collisioni completi. Disabilitare `process_mode` rimuove
anche la simulazione delle collisioni del layout non selezionato.

## 14. Collision layer

Una possibile configurazione:

| Oggetto | Layer | Mask |
|---|---:|---:|
| carri | 1 | 1, 2 |
| muri/ostacoli | 2 | 1, 4 |
| missili | 4 | 1, 2 |
| RayCast ostacoli | - | 1, 2 |
| RayCast visione | - | 1, 2 |

I colori non devono determinare le collisioni. La relazione amico/nemico viene letta
da `team_id`, così le due squadre possono usare la stessa scena e la stessa policy.

## 15. Curriculum

Il curriculum non deve rivelare la mappa. Cambia la distribuzione degli scenari:

| Episodi | Layout | Altre variazioni |
|---:|---|---|
| `0-499` | arena aperta | spawn quasi fissi, missili lenti |
| `500-1499` | ostacoli semplici | yaw e spawn leggermente casuali |
| `1500+` | piu' layout | spawn, ostacoli e velocita' variabili |

Mantieni sempre identici:

- numero e ordine delle observation;
- action space ibrido;
- significato di hit, kill e vittoria;
- numero di carri per env durante lo stesso run PPO.

Per passare da 1v1 a 2v2 e' preferibile usare due scene differenti ma con lo stesso
spec. Puoi inizializzare il modello 2v2 dai pesi 1v1, ma non riprendere ciecamente lo
stato dell'optimizer se reward e distribuzione cambiano molto.

## 16. Validare lo scenario

Prima usa azioni casuali:

```bash
python/.venv/bin/python python/random_scenario_rollout.py \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/tank_battle/tank_battle_scenario.tscn \
  --multi-agent \
  --steps 200 \
  --print-reward-terms \
  --no-headless
```

Lo spec deve mostrare:

```text
action_type=hybrid
agents=['RedTank1', 'RedTank2', 'BlueTank1', 'BlueTank2']
```

Controlla manualmente:

1. un muro azzera i segnali enemy dietro di esso;
2. un alleato produce `ally_*`, non `enemy_*`;
3. `weapon=1` crea al massimo un missile per cooldown;
4. un hit assegna reward soltanto agli agenti previsti;
5. un carro distrutto non si muove ma riceve l'esito finale;
6. tutti e quattro terminano quando una squadra viene eliminata;
7. il reset elimina i missili e ripristina vita, visibilita' e collisioni.

## 17. Avviare PPO hybrid multi-agent

```bash
python/.venv/bin/python python/train_generic.py \
  --algorithm auto \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/tank_battle/tank_battle_scenario.tscn \
  --num-envs 4 \
  --num-episodes 5000 \
  --max-steps-per-episode 800 \
  --batch-size 256 \
  --ppo-epochs 4 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-ratio 0.2 \
  --learning-rate 3e-4 \
  --entropy-coef 0.01 \
  --checkpoint-dir checkpoints/tank_battle_ppo_v1 \
  --weights-path tank_battle_ppo_v1.weights.h5 \
  --multi-agent \
  --collector-mode sync \
  --opponent-pool \
  --opponent-snapshot-every 100 \
  --opponent-pool-size 12 \
  --opponent-current-probability 0.2 \
  --headless
```

PPO e' on-policy: non usa replay buffer. `--batch-size` indica la dimensione dei
minibatch usati per aggiornare le traiettorie appena raccolte.

Per riprendere:

```bash
python/.venv/bin/python python/train_generic.py \
  --algorithm ppo \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/tank_battle/tank_battle_scenario.tscn \
  --num-envs 4 \
  --num-episodes 7000 \
  --checkpoint-dir checkpoints/tank_battle_ppo_v1 \
  --weights-path tank_battle_ppo_v1.weights.h5 \
  --multi-agent \
  --resume \
  --headless
```

## 18. Eseguire il modello ibrido

```bash
python/.venv/bin/python python/run_generic_policy.py \
  --algorithm ppo \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/tank_battle/tank_battle_scenario.tscn \
  --weights-path tank_battle_ppo_v1.weights.h5 \
  --multi-agent \
  --episodes 20 \
  --no-headless
```

In esecuzione, le azioni discrete usano `argmax` e quelle continue la media della
policy, senza rumore di esplorazione.

## 19. Metriche utili

La sola reward media non basta. Registra per squadra e per agente:

- vittorie, sconfitte e pareggi;
- hit rate: colpi a segno / missili sparati;
- danno inflitto e subito;
- kill e sopravvivenza;
- collisioni con ostacoli;
- percentuale di tempo con nemico visibile;
- distanza percorsa senza coordinate nella policy;
- frequenza dell'azione `fire` durante cooldown.

Una policy valida deve vincere su layout e seed non usati nel training. Poiche' rosso
e blu condividono il modello, valuta anche contro un bot semplice e contro checkpoint
precedenti: il 50% contro una copia identica non dimostra che sappia combattere.

## 20. Estensioni successive

Dopo aver validato il 2v2 puoi aggiungere, una cosa per volta:

- torretta indipendente con una terza azione continua;
- tipi di missile come seconda componente discreta;
- munizioni limitate e pickup;
- comunicazione locale tra alleati;
- fog of war con sensori diversi;
- league con rating e promozione selettiva delle snapshot del pool;
- team di dimensione variabile con masking.

Ogni nuova componente continua o discreta puo' essere aggiunta come figlio di
`ActionSpace`; PPO ricostruisce automaticamente le teste compatibili. Cambiare lo spec
invalida pero' i vecchi pesi, quindi completa prima una versione minima funzionante.
