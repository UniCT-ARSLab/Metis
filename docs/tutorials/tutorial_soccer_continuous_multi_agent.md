# Tutorial: Soccer 3D multi-agent con calcio continuo

Questo tutorial costruisce uno scenario di calcio arcade in 3D. Due
`CharacterBody3D`, uno rosso e uno blu, cercano di spingere e calciare una palla
`RigidBody3D` nella porta avversaria.

La prima versione e' 1v1, ma observation e action space sono progettati per passare a
2v2 o 3v3 usando lo stesso modello condiviso.

## 1. Scelta dell'action space

Usiamo tre azioni continue e SAC:

```gdscript
{
	"movement": {
		"size": 2,
		"action_type": "continuous",
		"low": -1.0,
		"high": 1.0
	},
	"kick_strength": {
		"size": 1,
		"action_type": "continuous",
		"low": 0.0,
		"high": 1.0
	}
}
```

Il vettore piatto ricevuto dal giocatore e':

```text
[accelerazione, rotazione, forza_calcio]
```

La forza continua permette di imparare:

- piccoli tocchi per controllare la palla;
- passaggi di media intensita';
- tiri forti;
- nessun calcio con valore vicino a zero.

Una versione discreta `none/kick` sarebbe piu' semplice, ma perderebbe la differenza
tra dribbling e tiro. Qui il calcio continuo rende l'esempio diverso da Tanks hybrid.

SAC tende comunque a produrre quasi sempre un valore non nullo: per questo
`min_kick_input`, `KickArea` e cooldown trasformano il valore continuo in un calcio
soltanto quando l'intenzione e' abbastanza forte e fisicamente valida. Se in seguito
aggiungi tackle, salto o cambio giocatore, quelle sono invece buone candidate per una
parte discreta hybrid. Cambiare action space rende incompatibili i pesi gia' allenati.

## 2. Multi-agent e squadre

Nel primo scenario:

```text
RedPlayer   team_id=0
BluePlayer  team_id=1
```

Entrambi sono istanze della stessa scena, hanno observation simmetriche e usano una
sola policy SAC condivisa. La policy non riceve il colore o il `team_id`; ogni istanza
riceve invece i riferimenti `own_goal` e `opponent_goal` corretti.

Con quattro env in 1v1:

```text
4 env x 2 giocatori = 8 agenti, 1 actor, 2 critic SAC
```

In 2v2 diventano sedici agenti, ma il modello allenabile rimane uno. Senza opponent
pool tutte le transizioni aggiornano la policy condivisa; con il pool vengono usate
soltanto quelle della squadra learner.

## 3. Observation agent-centriche

Ogni giocatore osserva valori locali e normalizzati:

| Gruppo | Dimensione | Descrizione |
|---|---:|---|
| velocita' | 2 | velocita' avanti e modulo |
| input precedenti | 3 | accelerazione, rotazione, calcio |
| palla | 4 | posizione relativa X/Z e velocita' locale X/Z |
| porta avversaria | 3 | direzione locale X/Z e distanza |
| porta propria | 3 | direzione locale X/Z e distanza |
| controllo calcio | 2 | palla in range e cooldown pronto |
| giocatori | 7 | nemico visibile, enemy L/C/R, ally L/C/R |

Totale indicativo: 24 valori.

Non includere coordinate globali, colore della squadra o indice del giocatore. I due
lati del campo devono apparire equivalenti alla rete.

La posizione della palla puo' essere letta direttamente per una prima versione. Se
vuoi una percezione piu' realistica, sostituiscila successivamente con RayCast o camera,
senza cambiare contemporaneamente fisica e reward.

## 4. Creare SoccerPlayer.tscn

```text
SoccerPlayer (CharacterBody3D) [soccer_player.gd]
├── Mesh
├── CollisionShape3D
├── KickArea (Area3D)
│   └── CollisionShape3D
├── VisionSensors (Node3D)
│   ├── VisionLeft (RayCast3D)
│   ├── VisionCenter (RayCast3D)
│   └── VisionRight (RayCast3D)
└── Agent [Agent.gd]
    ├── ActionSpace [ActionSpace.gd]
    │   ├── Movement [ContinuousAction.gd]
    │   └── KickStrength [ContinuousAction.gd]
    ├── ObservationSystem [ObservationSystem.gd]
    │   ├── BodySpeed [BodySpeedObservationSource.gd]
    │   ├── MoveInput [MethodObservationSource.gd]
    │   ├── RotationInput [MethodObservationSource.gd]
    │   ├── KickInput [MethodObservationSource.gd]
    │   ├── BallPosition [MethodObservationSource.gd]
    │   ├── BallVelocity [MethodObservationSource.gd]
    │   ├── OpponentGoalDirection [MethodObservationSource.gd]
    │   ├── OpponentGoalDistance [MethodObservationSource.gd]
    │   ├── OwnGoalDirection [MethodObservationSource.gd]
    │   ├── OwnGoalDistance [MethodObservationSource.gd]
    │   ├── BallInRange [MethodObservationSource.gd]
    │   ├── KickReady [MethodObservationSource.gd]
    │   └── TeamVision [TeamRaycastObservationSource.gd]
    └── RewardSystem [RewardSystem.gd]
        ├── TimePenalty [StepPenaltyReward.gd]
        └── WastedKickPenalty [EventReward.gd]
```

Posiziona `KickArea` davanti al giocatore, non intorno a tutto il corpo. Una forma
Box3D corta e larga rende il calcio leggibile: il giocatore deve prima orientarsi verso
la palla.

## 5. Configurare ActionSpace

`Movement`:

```text
action_name = "movement"
size = 2
low = -1.0
high = 1.0
use_custom_exploration_bounds = true
exploration_low = -0.6
exploration_high = 0.6
```

`KickStrength`:

```text
action_name = "kick_strength"
size = 1
low = 0.0
high = 1.0
use_custom_exploration_bounds = true
exploration_low = 0.0
exploration_high = 1.0
```

L'ordine dei nodi determina l'ordine del vettore continuo: i due valori di movement
prima della forza di calcio.

## 6. Configurare le observation

`BodySpeed`:

```text
forward_speed_name = "forward_speed"
absolute_speed_name = "speed"
speed_scale = 9.0
```

Input precedenti:

```text
MoveInput:
  observation_name = "move_input"
  method_name = "get_control_input"
  bind_string_arg = "move_input"

RotationInput:
  observation_name = "rotation_input"
  method_name = "get_control_input"
  bind_string_arg = "rotation_input"

KickInput:
  observation_name = "kick_input"
  method_name = "get_control_input"
  bind_string_arg = "kick_input"
```

Palla:

```text
BallPosition:
  observation_name = "ball_relative_position"
  method_name = "get_ball_relative_position_observation"

BallVelocity:
  observation_name = "ball_local_velocity"
  method_name = "get_ball_velocity_observation"
```

Porte, usando `bind_string_arg`:

```text
OpponentGoalDirection:
  observation_name = "opponent_goal_direction"
  method_name = "get_goal_direction_observation"
  bind_string_arg = "opponent"

OpponentGoalDistance:
  observation_name = "opponent_goal_distance"
  method_name = "get_goal_distance_observation"
  bind_string_arg = "opponent"

OwnGoalDirection:
  observation_name = "own_goal_direction"
  method_name = "get_goal_direction_observation"
  bind_string_arg = "own"

OwnGoalDistance:
  observation_name = "own_goal_distance"
  method_name = "get_goal_distance_observation"
  bind_string_arg = "own"
```

Controllo calcio:

```text
BallInRange:
  observation_name = "ball_in_kick_range"
  method_name = "get_ball_in_kick_range_observation"

KickReady:
  observation_name = "kick_ready"
  method_name = "get_kick_ready_observation"
```

`TeamVision` usa gli stessi tre RayCast per alleati e avversari:

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

In 1v1 gli `ally_*` valgono sempre zero, ma devono gia' esistere: così il passaggio a
2v2 non cambia `obs_dim` e i pesi restano compatibili.

## 7. Scrivere soccer_player.gd

```gdscript
extends CharacterBody3D
class_name SoccerPlayer

signal ball_kicked(player:SoccerPlayer, strength:float)

@export_category("Team")
@export var team_id := 0
@export var own_goal: Node3D
@export var opponent_goal: Node3D

@export_category("World")
@export var ball: RigidBody3D
@export var field_half_size := Vector2(18.0, 11.0)
@export var ball_speed_scale := 18.0

@export_category("Movement")
@export var max_forward_speed := 8.0
@export var max_reverse_speed := 4.0
@export var acceleration := 16.0
@export var drag := 12.0
@export var turn_speed_degrees := 150.0

@export_category("Kick")
@export var kick_cooldown := 0.35
@export var min_kick_input := 0.15
@export var min_kick_impulse := 2.0
@export var max_kick_impulse := 10.0
@export var kick_lift := 0.08

@export_category("Control")
@export var manual_control := false

@onready var agent: Agent = $Agent
@onready var kick_area: Area3D = $KickArea

var _move_input := 0.0
var _rotation_input := 0.0
var _kick_input := 0.0
var _kick_cooldown_left := 0.0
var _match_terminal := false
var _training_active := true


func _physics_process(delta:float) -> void:
	_kick_cooldown_left = maxf(_kick_cooldown_left - delta, 0.0)
	if not _training_active:
		return

	if manual_control:
		_move_input = Input.get_axis("move_back", "move_forward")
		_rotation_input = Input.get_axis("turn_left", "turn_right")
		_kick_input = 1.0 if Input.is_action_pressed("kick") else 0.0

	rotate_y(-_rotation_input * deg_to_rad(turn_speed_degrees) * delta)
	var speed_limit := max_forward_speed if _move_input >= 0.0 else max_reverse_speed
	var desired := global_transform.basis.z * _move_input * speed_limit
	var flat_velocity := Vector3(velocity.x, 0.0, velocity.z)
	if absf(_move_input) > 0.05:
		flat_velocity = flat_velocity.move_toward(desired, acceleration * delta)
	else:
		flat_velocity = flat_velocity.move_toward(Vector3.ZERO, drag * delta)
	velocity.x = flat_velocity.x
	velocity.z = flat_velocity.z
	if not is_on_floor():
		velocity += get_gravity() * delta
	move_and_slide()


func apply_action(action:Variant) -> Array:
	if typeof(action) == TYPE_STRING or typeof(action) == TYPE_STRING_NAME:
		if str(action) == "manual":
			return apply_manual_action()

	var values := agent.decode_continuous_action(action)
	_move_input = clampf(float(values[0]), -1.0, 1.0)
	_rotation_input = clampf(float(values[1]), -1.0, 1.0)
	_kick_input = clampf(float(values[2]), 0.0, 1.0)
	if _kick_input >= min_kick_input:
		try_kick(_kick_input)
	return [_move_input, _rotation_input, _kick_input]


func apply_manual_action() -> Array:
	return apply_action([
		Input.get_axis("move_back", "move_forward"),
		Input.get_axis("turn_left", "turn_right"),
		1.0 if Input.is_action_pressed("kick") else 0.0
	])


func try_kick(strength:float) -> bool:
	if ball == null or _kick_cooldown_left > 0.0 or not kick_area.overlaps_body(ball):
		agent.add_reward_event("wasted_kick", strength)
		return false

	var forward := global_transform.basis.z.normalized()
	var kick_direction := (forward + Vector3.UP * kick_lift).normalized()
	var normalized_strength := inverse_lerp(min_kick_input, 1.0, strength)
	var impulse := lerpf(min_kick_impulse, max_kick_impulse, normalized_strength)
	ball.apply_central_impulse(kick_direction * impulse)
	_kick_cooldown_left = kick_cooldown
	ball_kicked.emit(self, strength)
	return true


func get_team_id() -> int:
	return team_id


func get_control_input(input_name:String) -> float:
	if input_name == "move_input":
		return _move_input
	if input_name == "rotation_input":
		return _rotation_input
	if input_name == "kick_input":
		return _kick_input
	return 0.0


func get_ball_relative_position_observation() -> Vector2:
	if ball == null:
		return Vector2.ZERO
	var local: Vector3 = global_transform.basis.inverse() * (ball.global_position - global_position)
	return Vector2(
		clampf(local.x / maxf(field_half_size.x, 0.001), -1.0, 1.0),
		clampf(local.z / maxf(field_half_size.y, 0.001), -1.0, 1.0)
	)


func get_ball_velocity_observation() -> Vector2:
	if ball == null:
		return Vector2.ZERO
	var local_velocity: Vector3 = global_transform.basis.inverse() * ball.linear_velocity
	var normalized := Vector2(local_velocity.x, local_velocity.z) / maxf(ball_speed_scale, 0.001)
	return normalized.clamp(Vector2(-1.0, -1.0), Vector2(1.0, 1.0))


func get_goal_direction_observation(goal_name:String) -> Vector2:
	var goal := opponent_goal if goal_name == "opponent" else own_goal
	if goal == null:
		return Vector2.ZERO
	var local: Vector3 = global_transform.basis.inverse() * (goal.global_position - global_position)
	var flat := Vector2(local.x, local.z)
	return flat.normalized() if flat.length() > 0.001 else Vector2.ZERO


func get_goal_distance_observation(goal_name:String) -> float:
	var goal := opponent_goal if goal_name == "opponent" else own_goal
	if goal == null:
		return 0.0
	var distance := Vector2(
		goal.global_position.x - global_position.x,
		goal.global_position.z - global_position.z
	).length()
	return clampf(distance / maxf(field_half_size.length() * 2.0, 0.001), 0.0, 1.0)


func get_ball_in_kick_range_observation() -> float:
	return 1.0 if ball != null and kick_area.overlaps_body(ball) else 0.0


func get_kick_ready_observation() -> float:
	return 1.0 if _kick_cooldown_left <= 0.0 else 0.0


func set_match_terminal(value:bool) -> void:
	_match_terminal = value


func is_terminal() -> bool:
	return _match_terminal


func reset_all(original_transform:Variant, reset_rewards := true) -> void:
	set_training_active(true)
	if typeof(original_transform) == TYPE_TRANSFORM3D:
		transform = original_transform
	_move_input = 0.0
	_rotation_input = 0.0
	_kick_input = 0.0
	_kick_cooldown_left = 0.0
	_match_terminal = false
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
		_move_input = 0.0
		_rotation_input = 0.0
		_kick_input = 0.0
		velocity = Vector3.ZERO
```

Il calcio avviene durante `apply_action`, una volta per step del bridge. Il cooldown
impedisce che una forza alta applichi impulsi a ogni frame fisico.

## 8. Reward locali

`TimePenalty`:

```text
penalty = -0.0005
term_name = "time"
```

`WastedKickPenalty`:

```text
event_name = "wasted_kick"
scale = -0.002
term_name = "wasted_kick"
```

Il valore dell'evento e' la forza richiesta, quindi un tentativo debole viene punito
meno di un tiro massimo nel vuoto. Durante i primi test puoi disabilitare questa
penalita' per non bloccare l'esplorazione del calcio.

Non usare `ActionSmoothnessPenaltyReward` sul vettore completo: penalizzerebbe anche il
passaggio rapido da calcio zero a calcio forte. L'inerzia del corpo rende gia' fluido
il movimento.

## 9. Configurare la palla

La palla e' un `RigidBody3D`:

```text
Ball (RigidBody3D, gruppo "ball")
├── MeshInstance3D
└── CollisionShape3D (SphereShape3D)
```

Impostazioni iniziali:

```text
mass = 1.0
gravity_scale = 1.0
linear_damp = 0.4
angular_damp = 0.2
continuous_cd = true
contact_monitor = true
max_contacts_reported = 8
```

Assegna un `PhysicsMaterial` con bounce basso, circa `0.2`, e friction moderata. Bounce
troppo alto trasforma il gioco in flipper; damping troppo alto impedisce passaggi e
tiri lunghi.

## 10. Creare SoccerScenario.tscn

```text
SoccerScenario (Node3D) [soccer_manager.gd]
├── BridgeServer [bridge_server.gd]
├── ScenarioController [ScenarioController.gd]
│   ├── ScenarioEventSystem [ScenarioEventSystem.gd]
│   │   ├── BallTouched [ManualScenarioEventSource.gd]
│   │   ├── PersonalGoal [ManualScenarioEventSource.gd]
│   │   ├── TeamGoal [ManualScenarioEventSource.gd]
│   │   └── GoalConceded [ManualScenarioEventSource.gd]
│   └── ScenarioRewardSystem [ScenarioRewardSystem.gd]
│       ├── BallProgress [soccer_ball_progress_reward.gd]
│       ├── TouchReward [EventScenarioReward.gd]
│       ├── PersonalGoalReward [EventScenarioReward.gd]
│       ├── TeamGoalReward [EventScenarioReward.gd]
│       └── ConcededPenalty [EventScenarioReward.gd]
├── Field
│   ├── Floor
│   ├── Walls
│   ├── RedGoal (Area3D, difesa dal team 0)
│   └── BlueGoal (Area3D, difesa dal team 1)
├── Ball (RigidBody3D)
└── Players
    ├── RedPlayer
    └── BluePlayer
```

Nel `ScenarioController`:

```text
controlled_agents = ["../Players/RedPlayer", "../Players/BluePlayer"]
number_of_replications = 0
randomize_reset = true
reset_position_jitter = (0.5, 0.0, 1.5)
reset_yaw_jitter_degrees = 10.0
max_steps = 700
```

Configura le istanze:

```text
RedPlayer:
  team_id = 0
  own_goal = RedGoal
  opponent_goal = BlueGoal
  ball = Ball

BluePlayer:
  team_id = 1
  own_goal = BlueGoal
  opponent_goal = RedGoal
  ball = Ball
```

Entrambe usano la stessa scena e lo stesso spec.

## 11. Eventi e reward di scenario

Configura gli eventi:

| Nodo | event_name | terminal_reason |
|---|---|---|
| BallTouched | `ball_touched` | vuoto |
| PersonalGoal | `personal_goal` | vuoto |
| TeamGoal | `team_goal` | `team_goal` |
| GoalConceded | `goal_conceded` | `goal_conceded` |

Reward iniziali:

| Componente | Evento | Reward | only_once |
|---|---|---:|---|
| TouchReward | `ball_touched` | `+0.01` | false |
| PersonalGoalReward | `personal_goal` | `+0.50` | true |
| TeamGoalReward | `team_goal` | `+2.00` | true |
| ConcededPenalty | `goal_conceded` | `-2.00` | true |

Il goal di squadra viene assegnato a tutti i compagni. Il bonus personale va soltanto
all'ultimo giocatore della squadra vincente che ha toccato la palla. Un autogol non
riceve il bonus personale.

## 12. Reward densa di avanzamento della palla

Crea
`res://scenarios/soccer/rewards/soccer_ball_progress_reward.gd`:

```gdscript
extends "res://scripts/agent/scenario_reward_components/ScenarioRewardComponent.gd"
class_name SoccerBallProgressReward

@export var ball: RigidBody3D
@export var team_0_goal: Node3D
@export var team_1_goal: Node3D
@export var reward_scale := 0.02
@export var max_distance_delta := 1.0

var _previous_distance := {}


func reset_rewards() -> void:
	_previous_distance.clear()


func reset_agent(agent_id:String, context:Dictionary = {}) -> void:
	_previous_distance.erase(agent_id)


func compute_reward(agent:Node, context:Dictionary = {}) -> float:
	if ball == null or not agent.has_method("get_team_id"):
		return 0.0
	var agent_id := str(context.get("agent_id", agent.name))
	var current := _distance_to_opponent_goal(agent)
	var previous := float(_previous_distance.get(agent_id, current))
	_previous_distance[agent_id] = current
	var improvement := clampf(previous - current, -max_distance_delta, max_distance_delta)
	return improvement * reward_scale * weight


func _distance_to_opponent_goal(agent:Node) -> float:
	var target := team_1_goal if int(agent.get_team_id()) == 0 else team_0_goal
	if ball == null or target == null:
		return 0.0
	var delta := target.global_position - ball.global_position
	delta.y = 0.0
	return delta.length()
```

Nel contesto attuale il componente riceve il corpo agente in `compute_reward`, non in
`reset_agent`. Il reset elimina quindi il campione precedente; al primo compute il
fallback usa la distanza corrente e restituisce delta zero, evitando reward spurie.

Questa reward e' team-relative: quando la palla si avvicina alla porta blu favorisce i
rossi e sfavorisce i blu. Mantienila piccola rispetto al goal, altrimenti la policy puo'
imparare a far oscillare la palla senza segnare.

## 13. Scrivere soccer_manager.gd

```gdscript
extends Node3D

@export var players: Array[SoccerPlayer] = []
@export var ball: RigidBody3D
@export var red_goal: Area3D
@export var blue_goal: Area3D

@onready var controller := $ScenarioController
@onready var ball_touched: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/BallTouched
@onready var personal_goal: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/PersonalGoal
@onready var team_goal: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/TeamGoal
@onready var goal_conceded: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/GoalConceded

var _ball_start_transform: Transform3D
var _last_touch: SoccerPlayer
var _match_finished := false


func _ready() -> void:
	_ball_start_transform = ball.transform
	controller.episode_reset_started.connect(_reset_match)
	red_goal.body_entered.connect(_on_goal_entered.bind(0))
	blue_goal.body_entered.connect(_on_goal_entered.bind(1))
	for player in players:
		player.ball_kicked.connect(_on_ball_kicked)


func _reset_match(_seed:int) -> void:
	_match_finished = false
	_last_touch = null
	for player in players:
		player.set_match_terminal(false)
	ball.freeze = true
	ball.transform = _ball_start_transform
	ball.linear_velocity = Vector3.ZERO
	ball.angular_velocity = Vector3.ZERO
	ball.sleeping = false
	ball.freeze = false


func _on_ball_kicked(player:SoccerPlayer, _strength:float) -> void:
	if _match_finished:
		return
	_last_touch = player
	ball_touched.trigger(str(player.name))


func _on_goal_entered(body:Node, defending_team:int) -> void:
	if body != ball or _match_finished:
		return
	var scoring_team := 1 - defending_team
	_finish_with_goal(scoring_team)


func _finish_with_goal(scoring_team:int) -> void:
	_match_finished = true
	ball.freeze = true
	for player in players:
		if player.get_team_id() == scoring_team:
			team_goal.trigger(str(player.name))
		else:
			goal_conceded.trigger(str(player.name))
		player.set_match_terminal(true)

	if _last_touch != null and _last_touch.get_team_id() == scoring_team:
		personal_goal.trigger(str(_last_touch.name))
```

Il `defending_team` viene passato con `bind`: entrare nella porta rossa assegna il goal
al blu e viceversa.

Gli episodi terminati per `max_steps` sono pareggi per truncation. Se vuoi una reward
esplicita di pareggio, il manager deve ricevere un evento di timeout prima della
costruzione delle reward; per la prima versione e' meglio lasciare reward zero.

## 14. Collision layer

Configurazione possibile:

| Oggetto | Layer | Mask |
|---|---:|---:|
| giocatori | 1 | 1, 2, 4 |
| palla | 2 | 1, 4 |
| muri | 4 | 1, 2 |
| porte Area3D | 8 | 2 |
| KickArea | 16 | 2 |
| TeamVision RayCast | - | 1, 4 |

Il `KickArea` deve monitorare la palla ma non produrre collisione fisica. Le pareti
laterali possono contenere la palla; dietro le porte lascia spazio sufficiente affinche'
l'Area3D rilevi il goal.

## 15. Validare il 1v1

```bash
python/.venv/bin/python python/tools/random_rollout.py \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/soccer/soccer_1v1.tscn \
  --multi-agent \
  --steps 200 \
  --print-reward-terms \
  --no-headless
```

Lo spec atteso:

```text
action_type=continuous
action_size=3
agents=['RedPlayer', 'BluePlayer']
obs_shape=(2, 24)  # se TeamVision produce i sette valori descritti
```

Controlla:

1. i due giocatori ricevono direzioni opposte ma equivalenti delle porte;
2. calcio zero non applica impulsi;
3. un calcio debole muove la palla meno di uno forte;
4. il cooldown impedisce impulsi continui;
5. `wasted_kick` compare soltanto sopra la soglia;
6. un goal assegna reward opposte e termina entrambi;
7. reset ripristina palla, velocita' e stato terminale.

## 16. Training SAC 1v1

Non usare `--action-smoothing`: filtrerebbe anche `kick_strength`, trasformando un
impulso in un comando persistente.

```bash
python/.venv/bin/python python/train.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/soccer/soccer_1v1.tscn \
  --num-envs 4 \
  --num-episodes 4000 \
  --max-steps-per-episode 700 \
  --batch-size 128 \
  --replay-warmup 12000 \
  --random-exploration-episodes 50 \
  --action-smoothing 0.0 \
  --checkpoint-dir checkpoints/soccer_1v1_sac_v1 \
  --actor-weights-path soccer_1v1_actor_v1.weights.h5 \
  --critic1-weights-path soccer_1v1_critic1_v1.weights.h5 \
  --critic2-weights-path soccer_1v1_critic2_v1.weights.h5 \
  --multi-agent \
  --collector-mode sync \
  --opponent-pool \
  --opponent-snapshot-every 100 \
  --opponent-pool-size 12 \
  --opponent-current-probability 0.2 \
  --headless
```

SAC condivide actor e critic tra rosso e blu. Con il pool attivo, a ogni match una
squadra usa l'actor corrente e una usa una snapshot; solo le transizioni della squadra
learner aggiornano actor e critic. Negli episodi `current` entrambi i lati usano e
allenano la policy corrente.

L'opponent pool storico richiede `--collector-mode sync`; la combinazione con `async`
viene rifiutata per mantenere avversario e lato learner coerenti per tutto il match.
Senza pool, il parameter sharing current-vs-current puo' usare il collector asincrono.
Mantieni inizialmente `--physics-frames-per-step 1`: calcio, cooldown e contatti con la
palla sono sensibili alla frequenza delle decisioni. Se aumenti il valore, rivaluta
anche reward e finestre espresse in step.

## 17. Curriculum

Stadi consigliati:

### Stadio 1: controllo palla

- campo corto;
- porte larghe;
- palla inizialmente vicina a uno dei giocatori;
- `WastedKickPenalty` disabilitata;
- velocita' e forza massima ridotte.

### Stadio 2: 1v1 completo

- kickoff al centro;
- porte normali;
- spawn e yaw leggermente randomizzati;
- penalita' wasted kick molto piccola;
- reward personale e di squadra definitive.

### Stadio 3: generalizzazione

- posizione iniziale della palla casuale;
- massa e damping variati entro il 10%;
- dimensione della porta variata moderatamente;
- seed e spawn mai visti in valutazione.

Usa `controller.scenario_configured` e `training_episode` per selezionare lo stadio.
Non cambiare action space o observation durante il curriculum.

## 18. Passare da 1v1 a 2v2

Crea una seconda scena con:

```text
RedPlayer1, RedPlayer2
BluePlayer1, BluePlayer2
```

Tutti devono essere aggiunti a `controlled_agents` e `players`. Assegna correttamente
team e porte. Non usare la replica automatica: ogni istanza deve avere spawn e squadra
espliciti.

Le observation `ally_*` gia' presenti diventano informative senza cambiare `obs_dim`.
La reward `team_goal` viene assegnata a entrambi i compagni, favorendo collaborazione.

Puoi inizializzare il 2v2 dal checkpoint 1v1:

```bash
python/.venv/bin/python python/train.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/soccer/soccer_2v2.tscn \
  --num-envs 4 \
  --num-episodes 6000 \
  --max-steps-per-episode 900 \
  --batch-size 128 \
  --replay-warmup 12000 \
  --critic-warmup-updates 5000 \
  --policy-update-every 4 \
  --checkpoint-dir checkpoints/soccer_2v2_sac_v1 \
  --resume-checkpoint checkpoints/soccer_1v1_sac_v1/ckpt-4000 \
  --actor-weights-path soccer_2v2_actor_v1.weights.h5 \
  --critic1-weights-path soccer_2v2_critic1_v1.weights.h5 \
  --critic2-weights-path soccer_2v2_critic2_v1.weights.h5 \
  --multi-agent \
  --collector-mode sync \
  --opponent-pool \
  --opponent-snapshot-every 100 \
  --opponent-pool-size 12 \
  --opponent-current-probability 0.2 \
  --headless
```

Il checkpoint indicato deve esistere con il relativo replay buffer. Se il passaggio a
2v2 destabilizza il critic, usa un nuovo replay o aumenta il warmup; la distribuzione
delle esperienze e' cambiata anche se lo spec e' uguale.

## 19. Ruoli con modello condiviso

Una policy condivisa puo' sviluppare comportamenti diversi in base alle observation:

- il giocatore vicino alla palla attacca;
- quello vicino alla porta propria difende;
- un compagno libero puo' ricevere un passaggio.

Non serve un modello separato per attaccante e portiere. Se vuoi ruoli espliciti,
aggiungi una observation `role` mantenendo lo stesso modello, ma il ruolo deve essere
assegnato in modo coerente e restare disponibile anche in produzione.

Per incentivare i passaggi puoi aggiungere in seguito eventi `successful_pass`, ma
solo dopo che il sistema sa segnare. Reward premature per tocchi e passaggi possono
produrre palleggio infinito senza goal.

## 20. Eseguire il modello

```bash
python/.venv/bin/python python/run.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/soccer/soccer_1v1.tscn \
  --actor-weights-path soccer_1v1_actor_v1.weights.h5 \
  --multi-agent \
  --episodes 20 \
  --no-headless
```

In esecuzione SAC usa la media deterministica dell'actor, quindi `kick_strength` puo'
essere meno variabile rispetto al training stocastico.

Per eseguire il best checkpoint usa `--load-from checkpoint` insieme a
`--checkpoint-dir checkpoints/soccer_1v1_sac_v1/best` e ometti
`--actor-weights-path`. Riprendi invece il training dalla directory principale, che
conserva replay, optimizer e opponent pool.

## 21. Metriche

Oltre alla reward registra:

- goal segnati e subiti per squadra;
- tiri tentati, validi e sprecati;
- intensita' media dei calci validi;
- tocchi palla;
- possesso approssimato;
- distanza della palla dalla porta avversaria;
- tempo medio per segnare;
- pareggi per timeout;
- risultati contro checkpoint precedenti e un bot semplice.

Due copie della stessa policy tenderanno al 50% di vittorie anche se entrambe giocano
male. La valutazione contro bot e checkpoint congelati e' quindi indispensabile.

## 22. Estensioni successive

Dopo aver validato 1v1 e 2v2:

- calcio direzionale con quarta azione continua;
- sprint con consumo di energia;
- tackle come azione discreta, rendendo lo spazio hybrid;
- portiere con ruolo condizionato;
- passaggi e assist;
- fallo laterale e corner;
- league con rating e promozione selettiva delle snapshot del pool;
- numero variabile di giocatori con observation aggregate o attention.

La versione minima deve prima dimostrare tre competenze: raggiungere la palla,
orientarsi verso la porta avversaria e scegliere una forza di calcio utile.
