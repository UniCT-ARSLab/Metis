# Tanks 2v2: multi-agent con azioni ibride

In questo tutorial costruiamo uno scenario arcade 3D con due squadre di carri armati.
Ogni carro esplora una mappa sconosciuta, evita gli ostacoli, riconosce alleati e nemici
attraverso RayCast e decide quando sparare.

L'esempio serve anche a mostrare il contratto ibrido di Metis:

- movimento e rotazione sono azioni continue;
- `hold_fire` e `fire` sono azioni discrete;
- tutti i carri condividono la stessa policy PPO;
- reward e terminalita' restano separate per agente;
- il `ScenarioController` e il bridge sono quelli generici del framework.

Il progetto contiene gia' l'implementazione descritta nel tutorial. I file reali sono:

```text
res://agents/Tank/tank.tscn
res://agents/Tank/tank.gd
res://agents/Tank/mesh.scn
res://scenarios/tanks/tanks_scenario.tscn
res://scenarios/tanks/tanks_scenario.gd
res://scenarios/tanks/tank_missile.tscn
res://scenarios/tanks/tank_missile.gd
res://scenarios/tanks/easy_map.tscn
res://scenarios/tanks/medium_map.tscn
res://scenarios/tanks/hard_map.tscn
```

Non serve piu' il vecchio `res://scenarios/tanks/scenario_controller.gd`: la scena usa
`res://scripts/agent/ScenarioController.gd` e mantiene in `tanks_scenario.gd` soltanto
le regole proprie della battaglia.


## 1. Definire il compito

Ogni carro deve imparare a:

1. accelerare, frenare e ruotare;
2. esplorare senza conoscere coordinate o layout;
3. evitare muri e altri carri;
4. distinguere alleati e nemici visibili;
5. sparare soltanto quando il tiro ha senso;
6. sopravvivere e contribuire alla vittoria della squadra.

La prima versione usa quattro agenti:

```text
RedTank1    team_id=0
RedTank2    team_id=0
BlueTank1   team_id=1
BlueTank2   team_id=1
```

Sono quattro istanze della stessa scena. Il colore e il numero della squadra non
entrano nelle observation: i sensori restituiscono relazioni locali come `ally` ed
`enemy`. La stessa policy puo' quindi controllare entrambi i lati.

Con quattro environment il learner raccoglie fino a sedici traiettorie:

```text
4 environment x 4 agenti = 16 canali, 1 policy condivisa
```

Questo e' parameter sharing con self-play simultaneo, non quattro modelli indipendenti.

## 2. Progettare l'action space ibrido

Lo spazio avra' due componenti:

```text
movement: continuous, size=2, range=[-1, 1]
weapon:   discrete,   actions=[hold_fire, fire]
```

Il payload che arriva al carro ha questa forma:

```gdscript
{
	"movement": [0.72, -0.18],
	"weapon": 1
}
```

`movement[0]` e' il throttle, `movement[1]` lo steering. L'indice `weapon=1`
corrisponde a `fire` perche' sara' il secondo figlio del set discreto.

Metis rileva automaticamente la presenza di componenti continue e discrete, dichiara
lo spazio `hybrid` e permette a PPO di creare una testa gaussiana e una categorica.

## 3. Progettare observation locali

Una prima configurazione ragionevole e':

| Gruppo | Valori | Contenuto |
|---|---:|---|
| velocita' | 2 | velocita' in avanti firmata e modulo |
| controllo precedente | 2 | throttle e steering correnti |
| stato | 3 | salute, ricarica pronta, vivo |
| ostacoli | 7 | distanza normalizzata dei RayCast corti |
| nemici | 4 | visibile, sinistra, centro, destra |
| alleati | 3 | sinistra, centro, destra |

Il totale e' 21, ma dipende dal numero effettivo di RayCast. Non inserire:

- posizione o rotazione globale;
- coordinate degli avversari;
- indice del layout;
- `team_id`;
- una rappresentazione completa della mappa.

Le pareti devono interrompere i RayCast di visione. Se un nemico e' dietro un muro,
l'observation deve indicare che non e' visibile.

## 4. La scena del carro

Apri `res://agents/Tank/tank.tscn`:

```text
TankBattleAgent                    CharacterBody3D, tank.gd
├── Mesh                           istanza di mesh.scn
├── CollisionShape3D
├── Muzzle                         Marker3D
├── ObstacleSensors                Node3D
│   ├── FrontLeft                  RayCast3D
│   ├── Front                      RayCast3D
│   ├── FrontRight                 RayCast3D
│   ├── Left                       RayCast3D
│   ├── Right                      RayCast3D
│   ├── RearLeft                   RayCast3D
│   └── RearRight                  RayCast3D
├── VisionSensors                  Node3D
│   ├── VisionLeft                 RayCast3D
│   ├── VisionCenter               RayCast3D
│   └── VisionRight                RayCast3D
└── Agent                          Agent.gd
    ├── ActionSpace                ActionSpace.gd
    │   ├── Movement               ContinuousAction.gd
    │   └── Weapon                 DiscreteActionSet.gd
    │       ├── HoldFire           DiscreteAction.gd
    │       └── Fire               DiscreteAction.gd
    ├── ObservationSystem          ObservationSystem.gd
    │   ├── BodySpeed              BodySpeedObservationSource.gd
    │   ├── Throttle               MethodObservationSource.gd
    │   ├── Steering               MethodObservationSource.gd
    │   ├── Health                 MethodObservationSource.gd
    │   ├── ReloadReady            MethodObservationSource.gd
    │   ├── Alive                  MethodObservationSource.gd
    │   ├── Obstacles              RaycastObservationSource.gd
    │   └── TeamVision             TeamRaycastObservationSource.gd
    └── RewardSystem               RewardSystem.gd
        ├── TimePenality           StepPenaltyReward.gd
        └── InvalidFirePenalty     EventReward.gd
```

I RayCast ostacoli possono essere corti. Quelli di visione devono essere piu' lunghi,
ma non devono ignorare muri e ostacoli.

## 5. Configurare le azioni dall'Inspector

Configura `Movement`:

```text
action_name = movement
size = 2
low = -1.0
high = 1.0
```

Configura `Weapon`:

```text
action_name = weapon
```

Le azioni reali sono i figli `DiscreteAction`:

| Nodo | `action_name` | `method_name` | `target_path` |
|---|---|---|---|
| HoldFire | `hold_fire` | `hold_fire` | vuoto |
| Fire | `fire` | `request_fire` | vuoto |

Con `target_path` vuoto, `DiscreteAction` invoca il metodo sul corpo `BattleTank`.
L'ordine dei figli determina gli indici e deve restare stabile per tutta la vita del
modello.

Non registrare nuovamente queste azioni con `agent.add_action()`: la scena deve avere
una sola fonte di verita'.

## 6. Il corpo del carro

Il seguente codice corrisponde a `res://agents/Tank/tank.gd`, collegato alla radice
`TankBattleAgent`:

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
var _training_active := true
var _initial_collision_layer := 0
var _initial_collision_mask := 0


func _ready() -> void:
	_health = max_health
	_initial_collision_layer = collision_layer
	_initial_collision_mask = collision_mask


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
	if (typeof(action) == TYPE_STRING or typeof(action) == TYPE_STRING_NAME) and str(action) == "manual":
		return apply_manual_action()

	manual_control = false
	clear_inputs()
	if not _alive or typeof(action) != TYPE_DICTIONARY:
		return _zero_action()

	var action_map: Dictionary = action
	var movement := agent.decode_continuous_action(action_map)
	_throttle_input = clampf(float(movement[0]), -1.0, 1.0) if movement.size() > 0 else 0.0
	_rotation_input = clampf(float(movement[1]), -1.0, 1.0) if movement.size() > 1 else 0.0

	var weapon_action := int(action_map.get("weapon", 0))
	if agent.act_discrete(weapon_action, "weapon") != OK:
		weapon_action = 0

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
		"weapon": 1 if Input.is_action_just_pressed("fire") else 0
	})


func hold_fire() -> void:
	pass


func request_fire() -> void:
	if not try_fire():
		agent.add_reward_event("invalid_fire", 1.0)


func try_fire() -> bool:
	if not _alive or _cooldown_left > 0.0 or projectile_scene == null:
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
	if _health > 0.0:
		return

	_alive = false
	visible = false
	collision_layer = 0
	collision_mask = 0
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


func is_terminal() -> bool:
	return false


func clear_inputs() -> void:
	_throttle_input = 0.0
	_rotation_input = 0.0


func reset_all(original_transform:Variant, reset_rewards := true) -> void:
	set_training_active(true)
	if typeof(original_transform) == TYPE_TRANSFORM3D:
		transform = original_transform
	_health = max_health
	_alive = true
	_cooldown_left = 0.0
	visible = true
	collision_layer = _initial_collision_layer
	collision_mask = _initial_collision_mask
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


func _zero_action() -> Dictionary:
	return {"movement": [0.0, 0.0], "weapon": 0}
```

`apply_action()` decodifica la parte continua tramite `Agent` e delega la parte
discreta al `DiscreteActionSet`. Il corpo non contiene nomi o indici duplicati.

Per provare il carro senza Python, abilita `manual_control` e configura queste azioni
in **Project Settings > Input Map**:

| Azione | Tasto suggerito |
|---|---|
| `move_forward` | W |
| `move_back` | S |
| `turn_left` | A |
| `turn_right` | D |
| `fire` | Spazio |

Nel progetto attuale le prime quattro sono gia' presenti; `fire` deve essere aggiunta.
Questa Input Map riguarda soltanto il controllo manuale: durante il training lo sparo
arriva dal componente discreto `weapon`.

Un carro distrutto non termina immediatamente il proprio canale: rimane inattivo fino
alla fine del match e puo' ricevere l'esito di squadra. L'observation `alive` permette
alla policy e al value estimator di distinguere questo stato.

## 7. Configurare le observation

Configura `BodySpeed`:

```text
include_forward_speed = true
include_absolute_speed = true
forward_speed_name = forward_speed
absolute_speed_name = speed
speed_scale = 12.0
```

Configura i `MethodObservationSource` lasciando vuoto `source_path`:

| Nodo | `observation_name` | `method_name` | `bind_string_arg` |
|---|---|---|---|
| Throttle | `throttle_input` | `get_control_input` | `throttle_input` |
| Steering | `rotation_input` | `get_control_input` | `rotation_input` |
| Health | `health` | `get_health_observation` | vuoto |
| ReloadReady | `reload_ready` | `get_reload_ready_observation` | vuoto |
| Alive | `alive` | `get_alive_observation` | vuoto |

Configura `Obstacles`:

```text
root_path = ../../../ObstacleSensors
observation_prefix = obstacle
```

La distanza vale `1` quando il raggio e' libero e si avvicina a `0` quando la
collisione e' vicina. I colori di debug vengono disabilitati automaticamente in
headless.

Configura `TeamVision`:

```text
team_method_name = get_team_id
enemy_visible_observation_name = enemy_visible

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

La source cerca `get_team_id()` sul collider o sui suoi genitori. Se il RayCast colpisce
prima un muro, non trova un proprietario di squadra e restituisce zero: e' proprio
l'occlusione desiderata.

Salva la scena e controlla lo spec con un rollout prima di assumere che `obs_dim` sia
21. Python usera' la dimensione dichiarata realmente da Godot.

## 8. Configurare le reward locali

`TimePenality`:

```text
term_name = time
penalty = -0.0005
weight = 1.0
```

`InvalidFirePenalty`:

```text
term_name = invalid_fire
event_name = invalid_fire
scale = -0.01
weight = 1.0
```

Questi termini appartengono al singolo corpo. Non premiare molto `enemy_visible`: una
policy potrebbe imparare a guardare un avversario senza combattere. Hit, danni, kill e
vittoria saranno eventi dello scenario.

Nel file `tank.tscn` attuale `TimePenality.penalty` vale `0.0`, quindi il termine e'
disabilitato. Impostalo a `-0.0005` soltanto se vuoi davvero attribuire un costo alla
durata: il valore mostrato sopra e' una proposta iniziale, non un requisito del bridge.

## 9. Il missile

Apri `res://scenarios/tanks/tank_missile.tscn`; il relativo script e'
`res://scenarios/tanks/tank_missile.gd`:

```text
TankMissile                      CharacterBody3D, tank_missile.gd
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

Il missile non e' un agente: non ha observation, reward o policy. E' una conseguenza
fisica dell'azione discreta.

## 10. Lo scenario

Apri `res://scenarios/tanks/tanks_scenario.tscn`:

```text
TanksScenario                      Node3D, tanks_scenario.gd
├── Environment
│   ├── Floor
│   ├── Walls
│   └── LayoutContainer            Node3D
├── Camera3D
├── Agents
│   ├── RedTank1                   BattleTank, team_id=0
│   ├── RedTank2                   BattleTank, team_id=0
│   ├── BlueTank1                  BattleTank, team_id=1
│   └── BlueTank2                  BattleTank, team_id=1
├── Projectiles                    Node3D
├── ScenarioController             ScenarioController.gd
│   ├── ScenarioEventSystem        ScenarioEventSystem.gd
│   │   ├── EnemyDamage            ManualScenarioEventSource.gd
│   │   ├── EnemyKill              ManualScenarioEventSource.gd
│   │   ├── TeamKillAssist         ManualScenarioEventSource.gd
│   │   ├── DamageTaken            ManualScenarioEventSource.gd
│   │   ├── Destroyed              ManualScenarioEventSource.gd
│   │   ├── TeamWon                ManualScenarioEventSource.gd
│   │   ├── TeamLost               ManualScenarioEventSource.gd
│   │   └── MatchDraw              ManualScenarioEventSource.gd
│   └── ScenarioRewardSystem       ScenarioRewardSystem.gd
│       ├── EnemyDamageReward      EventScenarioReward.gd
│       ├── EnemyKillReward        EventScenarioReward.gd
│       ├── TeamAssistReward       EventScenarioReward.gd
│       ├── DamageTakenPenalty     EventScenarioReward.gd
│       ├── DestroyedPenalty       EventScenarioReward.gd
│       ├── WinReward              EventScenarioReward.gd
│       ├── LossPenalty            EventScenarioReward.gd
│       └── DrawPenalty            EventScenarioReward.gd
└── BridgeServer                   bridge_server.gd
```

Non serve un `ProgressProvider`: questo scenario non possiede un progresso scalare
ordinato. Il percorso predefinito `ProgressProvider` non trova alcun nodo e viene
trattato come `null`; in alternativa puoi lasciare vuoto `progress_provider_path`.

L'array `layout_scenes` della radice contiene, in quest'ordine, `easy_map.tscn`,
`medium_map.tscn` e `hard_map.tscn`. `layout_container` punta a
`Environment/LayoutContainer`.

Configura `ScenarioController`:

```text
controlled_agents = [RedTank1, RedTank2, BlueTank1, BlueTank2]
scenario_reward_system_path = ScenarioRewardSystem
event_system_path = ScenarioEventSystem
max_steps = 1000
physics_frames_per_step = 1
randomize_reset = true
reset_position_jitter = (0.3, 0.0, 0.3)
reset_yaw_jitter_degrees = 15.0
deactivate_done_agents = true
number_of_replications = 0
```

Questi sono i valori salvati attualmente nella scena. Il parametro Python
`--max-steps-per-episode` puo' sovrascrivere `max_steps` quando il trainer configura
l'environment.

`BridgeServer.controller_path` e' `../ScenarioController`. Ogni carro usa
`tank_missile.tscn` come `projectile_scene` e punta `projectile_parent` a `Projectiles`.

Usa istanze di scene layout dentro `LayoutContainer`, non piu' layout sovrapposti resi
soltanto invisibili. Nascondere un `StaticBody3D` o disabilitarne il processing non
rimuove necessariamente le collisioni; rimuovere la scena inattiva dall'albero evita
ostacoli invisibili.

## 11. Tradurre il combattimento in eventi Metis

Configura le event source:

| Nodo | `event_name` | `terminal_reason` |
|---|---|---|
| EnemyDamage | `enemy_damage` | vuoto |
| EnemyKill | `enemy_kill` | vuoto |
| TeamKillAssist | `team_kill_assist` | vuoto |
| DamageTaken | `damage_taken` | vuoto |
| Destroyed | `destroyed` | vuoto |
| TeamWon | `team_won` | `team_won` |
| TeamLost | `team_lost` | `team_lost` |
| MatchDraw | `match_draw` | `match_draw` |

Le `ManualScenarioEventSource` sono componenti attuali del framework: lo scenario
chiama `trigger(agent_id, value)` quando avviene un fatto del dominio. Non calcolano
reward e non costruiscono la risposta Python.

Collega questo script alla radice dello scenario:

```gdscript
extends Node3D

@export var tanks: Array[BattleTank] = []
@export var layout_scenes: Array[PackedScene] = []
@export var layout_container: Node3D

@onready var controller: ScenarioController = $ScenarioController
@onready var enemy_damage: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/EnemyDamage
@onready var enemy_kill: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/EnemyKill
@onready var team_kill_assist: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/TeamKillAssist
@onready var damage_taken: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/DamageTaken
@onready var destroyed_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/Destroyed
@onready var team_won: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/TeamWon
@onready var team_lost: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/TeamLost
@onready var match_draw: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/MatchDraw

var _training_episode := 0
var _match_finished := false
var _current_layout: Node


func _ready() -> void:
	controller.scenario_configured.connect(_on_scenario_configured)
	controller.episode_reset_started.connect(_on_episode_reset_started)
	for tank in tanks:
		tank.damage_received.connect(_on_damage_received)
		tank.destroyed.connect(_on_tank_destroyed)


func _on_scenario_configured(config:Dictionary) -> void:
	_training_episode = int(config.get("training_episode", _training_episode))


func _on_episode_reset_started(episode_seed:int) -> void:
	_match_finished = false
	_clear_projectiles()
	_select_layout(episode_seed)


func _on_damage_received(victim:BattleTank, attacker:BattleTank, amount:float) -> void:
	if _match_finished or attacker == null:
		return
	if attacker.get_team_id() == victim.get_team_id():
		return
	var normalized_damage := amount / maxf(victim.max_health, 0.001)
	enemy_damage.trigger(str(attacker.name), normalized_damage)
	damage_taken.trigger(str(victim.name), normalized_damage)


func _on_tank_destroyed(victim:BattleTank, killer:BattleTank) -> void:
	if _match_finished:
		return
	destroyed_event.trigger(str(victim.name))
	if killer != null and killer.get_team_id() != victim.get_team_id():
		enemy_kill.trigger(str(killer.name))
		for teammate in tanks:
			if teammate.get_team_id() == killer.get_team_id():
				team_kill_assist.trigger(str(teammate.name))
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
		return

	var winner_team := 0 if red_alive > 0 else 1
	for tank in tanks:
		if tank.get_team_id() == winner_team:
			team_won.trigger(str(tank.name))
		else:
			team_lost.trigger(str(tank.name))


func _alive_count(team:int) -> int:
	var count := 0
	for tank in tanks:
		if tank.get_team_id() == team and tank.is_alive():
			count += 1
	return count


func _clear_projectiles() -> void:
	for projectile in get_tree().get_nodes_in_group("projectile"):
		var parent := projectile.get_parent()
		if parent != null:
			parent.remove_child(projectile)
		projectile.queue_free()


func _select_layout(episode_seed:int) -> void:
	if layout_scenes.is_empty() or layout_container == null:
		return
	if is_instance_valid(_current_layout):
		layout_container.remove_child(_current_layout)
		_current_layout.queue_free()

	var available := 1
	if _training_episode >= 500:
		available = mini(2, layout_scenes.size())
	if _training_episode >= 1500:
		available = layout_scenes.size()

	var rng := RandomNumberGenerator.new()
	rng.seed = episode_seed
	var selected := rng.randi_range(0, maxi(available - 1, 0))
	_current_layout = layout_scenes[selected].instantiate()
	layout_container.add_child(_current_layout)
```

Questo script contiene soltanto regole specifiche della battaglia: danni, squadre,
layout e proiettili. Reset dei componenti RL, step fisici, spec, reward aggregation e
terminalita' vengono ancora gestiti dai nodi Metis.

## 12. Configurare le reward di scenario

Configura gli `EventScenarioReward`:

| Componente | `event_name` | `reward` | `only_once` |
|---|---|---:|---|
| EnemyDamageReward | `enemy_damage` | `+0.25` | false |
| EnemyKillReward | `enemy_kill` | `+1.00` | false |
| TeamAssistReward | `team_kill_assist` | `+0.20` | false |
| DamageTakenPenalty | `damage_taken` | `-0.15` | false |
| DestroyedPenalty | `destroyed` | `-1.00` | true |
| WinReward | `team_won` | `+3.00` | true |
| LossPenalty | `team_lost` | `-3.00` | true |
| DrawPenalty | `match_draw` | `-0.25` | true |

`enemy_damage` e `damage_taken` ricevono una frazione della salute massima, quindi il
valore effettivo scala con il danno. Gli altri eventi valgono `1`.

Non sommare automaticamente le reward dei quattro carri. Il killer riceve il proprio
segnale, i compagni ricevono soltanto l'assist esplicito e ogni agente conserva danni,
distruzione ed esito finale personali.

Controlla i termini con `--print-reward-terms` prima di regolare i pesi. Il bonus di
vittoria deve restare il segnale dominante, senza rendere irrilevanti hit e kill.

## 13. Collision layer e reset

La scena usa questa configurazione:

| Oggetto | Layer Inspector | Mask Inspector |
|---|---:|---:|
| carri | 1 | 1, 2 |
| pavimento e muri esterni | 2 | 1, 4 |
| missili | 4 | 1, 2 |
| RayCast ostacoli | nessuno | 1, 2 |
| RayCast visione | nessuno | 1, 2 |

Gli `StaticBody3D` di `medium_map.tscn` e `hard_map.tscn` non impostano ancora
esplicitamente il layer e quindi ereditano il layer 1. Funzionano, ma per mantenere la
separazione indicata sopra e' preferibile assegnarli al layer 2.

La relazione amico/nemico dipende da `team_id`, non dal layer o dal colore.

Dopo ogni reset verifica che:

- vita, visibilita', layer e mask siano ripristinati;
- velocita' e input precedenti siano azzerati;
- tutti i missili del vecchio episodio siano rimossi;
- sia presente un solo layout collisionabile;
- RayCast e reward state leggano il nuovo episodio;
- il seed produca sempre la stessa selezione del layout.

## 14. Validare con azioni casuali

Prima del training:

```bash
python/.venv/bin/python python/tools/random_rollout.py \
  --godot-project godot \
  --godot-scene res://scenarios/tanks/tanks_scenario.tscn \
  --multi-agent \
  --steps 500 \
  --print-reward-terms \
  --no-headless
```

Lo spec deve indicare:

```text
agents=['RedTank1', 'RedTank2', 'BlueTank1', 'BlueTank2']
obs_shape=(4, 21)
action_type=hybrid
action components: movement, weapon
continuous size: 2
discrete weapon size: 2
```

Controlla manualmente:

1. il dizionario ibrido muove e fa sparare il carro corretto;
2. `hold_fire` non crea proiettili;
3. il cooldown impedisce spam e produce `invalid_fire`;
4. un muro nasconde il nemico ai sensori;
5. un alleato attiva `ally_*`, non `enemy_*`;
6. danni e kill premiano gli agenti previsti;
7. tutti i canali terminano quando una squadra viene eliminata;
8. un timeout e' `truncated`, non una vittoria o sconfitta.

Se lo spec o le reward non sono corretti, non iniziare ancora il training.

## 15. Primo training: self-play simultaneo

Inizia senza opponent pool. Il collector asincrono puo' usare tutti gli environment e
le due squadre eseguono la policy corrente:

```bash
python/.venv/bin/python python/train.py \
  --algorithm ppo \
  --godot-project godot \
  --godot-scene res://scenarios/tanks/tanks_scenario.tscn \
  --num-envs 4 \
  --num-episodes 5000 \
  --max-steps-per-episode 1000 \
  --physics-frames-per-step 1 \
  --batch-size 256 \
  --ppo-epochs 4 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-ratio 0.2 \
  --learning-rate 3e-4 \
  --entropy-coef 0.01 \
  --checkpoint-dir checkpoints/tank_battle_ppo_v1 \
  --multi-agent \
  --collector-mode async \
  --headless
```

PPO e' on-policy e non usa replay buffer. Non cambiare il numero di agenti nel mezzo
di un rollout o di un resume senza rivalutare la distribuzione dei dati.

Mantieni `physics_frames_per_step=1` finche' non hai validato sensori, proiettili e
cooldown. Aumentarlo riduce la frequenza delle decisioni e cambia la durata fisica di
tutti i termini espressi per step.

## 16. Seconda fase: opponent pool

Il self-play simultaneo puo' dimenticare strategie precedenti. Dopo che il combattimento
base funziona, puoi allenare una squadra contro snapshot storiche congelate:

```bash
python/.venv/bin/python python/train.py \
  --algorithm ppo \
  --godot-project godot \
  --godot-scene res://scenarios/tanks/tanks_scenario.tscn \
  --num-envs 4 \
  --num-episodes 8000 \
  --max-steps-per-episode 1000 \
  --policy-path checkpoints/tank_battle_ppo_v1 \
  --checkpoint-dir checkpoints/tank_battle_ppo_pool_v1 \
  --multi-agent \
  --collector-mode sync \
  --opponent-pool \
  --opponent-snapshot-every 100 \
  --opponent-pool-size 12 \
  --opponent-current-probability 0.2 \
  --headless
```

Per PPO l'opponent pool richiede il collector sincrono. Senza `--learner-team`, Metis
sceglie a ogni episodio quale squadra viene allenata; l'altra usa la snapshot. Con
`--learner-team 0` puoi fissare temporaneamente il lato learner per il debug.

`--policy-path` inizializza il nuovo run dalla policy del self-play base, ma crea un
optimizer e un opponent pool nuovi. Non e' un resume del vecchio run.

Non attivare il pool dall'episodio zero soltanto perche' e' disponibile. Prima valida
che una singola policy impari hit, sopravvivenza e vittoria contro se stessa.

## 17. Curriculum senza rivelare la mappa

Lo script dello scenario riceve `training_episode` tramite `scenario_configured` e usa
quel valore soltanto per scegliere la distribuzione dei layout:

| Episodi | Layout disponibili | Reset |
|---:|---|---|
| `0-499` | `easy_map.tscn` | arena aperta |
| `500-1499` | `easy_map.tscn`, `medium_map.tscn` | ostacoli semplici |
| `1500+` | tutti e tre i file | layout completi |

Il curriculum non entra nelle observation. Mantieni invariati:

- ordine e dimensione delle observation;
- componenti e ordine dell'action space;
- significato di hit, kill e vittoria;
- numero di carri durante lo stesso run PPO.

Per passare da 1v1 a 2v2 puoi usare scene differenti con lo stesso contratto e fare un
warm start dalla policy 1v1. Non e' automaticamente un resume equivalente: la
distribuzione delle traiettorie e delle reward e' cambiata.

## 18. Riprendere ed eseguire la policy

Per riprendere il training:

```bash
python/.venv/bin/python python/train.py \
  --algorithm ppo \
  --godot-project godot \
  --godot-scene res://scenarios/tanks/tanks_scenario.tscn \
  --num-envs 4 \
  --num-episodes 7000 \
  --max-steps-per-episode 1000 \
  --batch-size 256 \
  --ppo-epochs 4 \
  --checkpoint-dir checkpoints/tank_battle_ppo_v1 \
  --multi-agent \
  --collector-mode async \
  --resume \
  --headless
```

Riporta anche le opzioni strutturali del run originale, in particolare collector,
opponent pool e durata dell'episodio.

Per osservare il bundle Keras prodotto:

```bash
python/.venv/bin/python python/run.py \
  --policy-path checkpoints/tank_battle_ppo_v1 \
  --godot-project godot \
  --godot-scene res://scenarios/tanks/tanks_scenario.tscn \
  --multi-agent \
  --episodes 20 \
  --no-headless
```

`policy.json` contiene backend, algoritmo e contratto dell'action space, quindi non
serve ripetere `--algorithm ppo`. In esecuzione le categorie usano la scelta
deterministica e le continue la media della policy.

## 19. Metriche da osservare

La reward media non basta. Registra almeno:

- vittorie, sconfitte e pareggi per squadra;
- danno inflitto e subito;
- hit rate: colpi a segno diviso missili sparati;
- kill, morti e sopravvivenza;
- collisioni con muri e altri carri;
- tempo con un nemico realmente visibile;
- frequenza di `fire` durante cooldown;
- durata media dei match;
- risultati contro bot semplici e checkpoint precedenti.

Due copie identiche della stessa policy tenderanno al 50% anche se entrambe giocano
male. Valuta sempre su layout e seed separati e contro avversari con forza nota.

## Checklist

- [ ] Ogni carro ha lo stesso action e observation contract.
- [ ] `Weapon` usa figli `DiscreteAction`, non l'array legacy `names`.
- [ ] Il corpo delega l'azione discreta con `agent.act_discrete()`.
- [ ] Nessuna observation rivela coordinate o layout.
- [ ] I muri occludono i sensori di squadra.
- [ ] Le reward locali e di scenario restano separate.
- [ ] Gli eventi terminali vengono attivati per tutti gli agenti del match.
- [ ] I carri distrutti non si muovono ma ricevono l'esito finale.
- [ ] Un solo layout collisionabile e' presente nell'albero.
- [ ] Il rollout casuale passa prima del training.
- [ ] L'opponent pool viene aggiunto soltanto dopo il self-play base.
- [ ] `run.py` carica il bundle senza ricostruire manualmente la rete.

Quando questa versione funziona puoi aggiungere una torretta indipendente, tipi di
munizione, pickup o comunicazione locale. Ogni nuovo componente va dichiarato sotto
`ActionSpace`; se ne cambi ordine o dimensione, la vecchia policy non ha piu' lo stesso
contratto.
