# Tutorial: creare Breakout come nuovo scenario RL

Questo tutorial costruisce un nuovo scenario 2D senza modificare Python. L'obiettivo
e' allenare un Paddle a muoversi a sinistra, restare fermo o muoversi a destra per
respingere una pallina e distruggere tutti i mattoni.

Il tutorial separa intenzionalmente tre responsabilita':

- il codice del gioco gestisce movimento, collisioni, pallina e mattoni;
- i nodi del framework dichiarano action space, observation, eventi e reward;
- Python legge automaticamente lo spec e sceglie DQN per le azioni discrete.

## 1. Il contratto minimo

Un corpo controllato dal framework deve avere un figlio `Agent` e implementare solo:

```gdscript
func apply_action(action:Variant) -> Variant
func reset_all(original_transform:Variant, reset_rewards := true) -> void
```

Puo' inoltre implementare, quando servono:

```gdscript
func is_terminal() -> bool
func set_training_active(enabled:bool) -> void
func set_training_optimized(enabled:bool) -> void
func apply_manual_action() -> Variant
```

Non deve implementare proxy come `get_observations()`, `get_reward()` o
`get_action_space()`: il `ScenarioController` trova direttamente il nodo `Agent`.

## 2. Progettare prima di scrivere

Per una prima versione di Breakout usiamo:

### Actions

Tre azioni discrete:

| ID | Nome | Effetto |
|---:|---|---|
| 0 | `idle` | Paddle fermo |
| 1 | `move_left` | Paddle a sinistra |
| 2 | `move_right` | Paddle a destra |

### Observations

Cinque valori normalizzati:

| Observation | Dimensione | Intervallo indicativo |
|---|---:|---:|
| `paddle_x` | 1 | `[-1, 1]` |
| `ball_relative_position` | 2 | `[-1, 1]` |
| `ball_velocity` | 2 | `[-1, 1]` |

La posizione relativa della pallina rende la policy indipendente dalla risoluzione
della finestra. Non diamo all'agente ID dei mattoni o coordinate assolute di ogni
mattone: per la prima versione non sono necessarie.

### Rewards

| Evento | Reward | Terminale |
|---|---:|---|
| ogni step | `-0.001` | no |
| mattone distrutto | `+1.0` | no |
| pallina persa | `-5.0` | si |
| tutti i mattoni distrutti | `+20.0` | si |

La piccola penalita' temporale evita una policy che si limita a tenere in vita la
pallina senza completare il livello.

### Perche' la pallina e' un RigidBody2D

Il Paddle e' l'agente e usa `CharacterBody2D`, perche' il suo movimento deve essere
controllato direttamente dalle azioni. La pallina usa invece `RigidBody2D`, perche'
deve partecipare alla simulazione fisica e rimbalzare contro corpi statici e mobili.

Questa scelta non significa lasciare la pallina completamente in balia della fisica.
Breakout e' un gioco arcade: gravita', attrito, damping e rotazione vanno disabilitati,
mentre velocita' e angoli troppo orizzontali vanno stabilizzati dal codice. In questo
modo manteniamo collisioni fisiche comode, ma episodi sufficientemente controllabili
e riproducibili per il reinforcement learning.

Una pallina `CharacterBody2D` sarebbe altrettanto valida, ma richiederebbe implementare
manualmente ogni rimbalzo con `move_and_collide()` e `Vector2.bounce()`. E' utile
quando si vuole una simulazione completamente deterministica; per questo primo
scenario `RigidBody2D` offre un compromesso piu' semplice.

## 3. Creare la scena Paddle

Crea `res://agents/BreakoutPaddle/breakout_paddle.tscn` con questo albero:

```text
BreakoutPaddle (CharacterBody2D) [breakout_paddle.gd]
├── Sprite2D
├── CollisionShape2D
└── Agent                 [Agent.gd]
    ├── ActionSpace       [ActionSpace.gd]
    │   └── Actions       [DiscreteActionSet.gd]
    │       ├── Idle      [DiscreteAction.gd]
    │       ├── MoveLeft  [DiscreteAction.gd]
    │       └── MoveRight [DiscreteAction.gd]
    ├── ObservationSystem [ObservationSystem.gd]
    │   ├── PaddleX       [MethodObservationSource.gd]
    │   ├── BallPosition  [MethodObservationSource.gd]
    │   └── BallVelocity  [MethodObservationSource.gd]
    └── RewardSystem      [RewardSystem.gd]
        └── TimePenalty   [StepPenaltyReward.gd]
```

`Actions` rappresenta una singola scelta categorica. I suoi figli descrivono sia i
nomi comunicati a Python sia i metodi eseguiti da Godot. Configurali dall'Inspector:

```text
Actions:
  action_name = "action"
  names = []  # Legacy: lascialo vuoto quando usi i figli DiscreteAction

Idle:
  action_name = "idle"
  target_path = ""
  method_name = "stop"

MoveLeft:
  action_name = "move_left"
  target_path = ""
  method_name = "move_left"

MoveRight:
  action_name = "move_right"
  target_path = ""
  method_name = "move_right"
```

Un `target_path` vuoto indica il corpo che possiede `Agent`, cioe' il Paddle. Un path
non vuoto viene risolto relativamente al Paddle e permette di invocare un metodo su un
suo componente. L'ordine dei figli e' l'ordine degli ID: `Idle = 0`, `MoveLeft = 1`,
`MoveRight = 2`.

Per costruire questa parte nell'editor:

1. aggiungi un normale nodo `Node` sotto `ActionSpace`, chiamalo `Actions` e assegna
   `DiscreteActionSet.gd`;
2. aggiungi tre nodi `Node` sotto `Actions` e assegna a ciascuno
   `DiscreteAction.gd`;
3. rinominali `Idle`, `MoveLeft` e `MoveRight`, poi configura i campi riportati sopra;
4. riordinali nell'albero se vuoi cambiare gli ID comunicati a Python.

Non compilare contemporaneamente `names` e i figli: quando esistono figli
`DiscreteAction`, questi sono la fonte di verita' e il vecchio array `names` viene
ignorato.

`arguments` e' facoltativo e in questo esempio resta vuoto. Permette anche di collegare
azioni diverse allo stesso metodo, per esempio `set_move_input(-1.0)` e
`set_move_input(1.0)`, senza creare un metodo distinto per ogni comando.

Non serve chiamare `agent.add_action()` in `_ready()`: quell'API rimane disponibile
per azioni registrate dinamicamente e per gli scenari legacy, mentre questo tutorial
usa la configurazione dichiarativa dall'Inspector come unica fonte di verita'.

La chiamata `agent.act(action_id)` dentro `apply_action()` rimane necessaria: non
registra nuovamente l'azione, ma chiede al framework di eseguire il figlio
`DiscreteAction` corrispondente all'ID ricevuto da Python.

Configura le observation:

```text
PaddleX:
  observation_name = "paddle_x"
  method_name = "get_paddle_x_observation"

BallPosition:
  observation_name = "ball_relative_position"
  method_name = "get_ball_relative_position_observation"

BallVelocity:
  observation_name = "ball_velocity"
  method_name = "get_ball_velocity_observation"
```

Configura `TimePenalty`:

```text
penalty = -0.0001
term_name = "time"
```

La penalita' vale `-0.15` ogni 1500 step: favorisce leggermente episodi efficienti
senza rendere sconveniente una partita lunga ma valida. Breakout puo' usare un episodio
senza time-limit perche' `life_lost` e `level_cleared` sono condizioni terminali.

## 4. Scrivere breakout_paddle.gd

Crea `res://agents/BreakoutPaddle/breakout_paddle.gd`:

```gdscript
extends CharacterBody2D
class_name BreakoutPaddle

@export var speed := 420.0
@export var horizontal_center := 0.0
@export var horizontal_limit := 540.0
@export var lock_vertical_position := true
@export var ball: RigidBody2D
@export var observation_position_scale := Vector2(540.0, 360.0)
@export var observation_ball_speed_scale := 500.0
@export var manual_control := false

@onready var agent: Agent = $Agent

var _move_input := 0.0
var _terminal := false
var _training_active := true
var _locked_y := 0.0


func _ready() -> void:
	_locked_y = global_position.y


func _physics_process(_delta:float) -> void:
	if not _training_active:
		return
	if manual_control:
		_move_input = Input.get_axis("move_left", "move_right")
	velocity = Vector2(_move_input * speed, 0.0)
	move_and_slide()
	if lock_vertical_position:
		global_position.y = _locked_y
		velocity.y = 0.0
	global_position.x = clampf(
		global_position.x,
		horizontal_center - horizontal_limit,
		horizontal_center + horizontal_limit
	)


func apply_action(action:Variant) -> Variant:
	if str(action) == "manual":
		return apply_manual_action()
	manual_control = false
	stop()
	var action_id := int(action)
	if agent.act(action_id) != OK:
		return 0
	return action_id


func apply_manual_action() -> int:
	var axis := Input.get_axis("move_left", "move_right")
	var action_id := 0
	if axis < -0.2:
		action_id = 1
	elif axis > 0.2:
		action_id = 2
	return apply_action(action_id)


func stop() -> void:
	_move_input = 0.0


func move_left() -> void:
	_move_input = -1.0


func move_right() -> void:
	_move_input = 1.0


func get_paddle_x_observation() -> float:
	return clampf(
		(global_position.x - horizontal_center) / maxf(horizontal_limit, 0.001),
		-1.0,
		1.0
	)


func get_ball_relative_position_observation() -> Vector2:
	if ball == null:
		return Vector2.ZERO
	var relative := ball.global_position - global_position
	return Vector2(
		clampf(relative.x / maxf(observation_position_scale.x, 0.001), -1.0, 1.0),
		clampf(relative.y / maxf(observation_position_scale.y, 0.001), -1.0, 1.0)
	)


func get_ball_velocity_observation() -> Vector2:
	if ball == null:
		return Vector2.ZERO
	return (ball.linear_velocity / maxf(observation_ball_speed_scale, 0.001)).clamp(
		Vector2(-1.0, -1.0),
		Vector2(1.0, 1.0)
	)


func set_episode_terminal(value:bool) -> void:
	_terminal = value


func is_terminal() -> bool:
	return _terminal


func reset_all(original_transform:Variant, reset_rewards := true) -> void:
	set_training_active(true)
	_terminal = false
	stop()
	velocity = Vector2.ZERO
	if typeof(original_transform) == TYPE_TRANSFORM2D:
		transform = original_transform
	_locked_y = global_position.y
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
		stop()
		velocity = Vector2.ZERO
```

`horizontal_center` deve coincidere con il centro X del campo. Se lavori in coordinate
da `0` a `1280`, usa circa `640`; se hai costruito il campo intorno all'origine, lascia
`0`. `horizontal_limit` e' la distanza massima dal centro raggiungibile dal Paddle,
tenendo conto di meta' della sua larghezza. Questi stessi valori normalizzano
`paddle_x`, che vale quindi circa `-1` a sinistra, `0` al centro e `+1` a destra.

`lock_vertical_position` deve restare attivo per Breakout. `move_and_slide()` puo'
separare due corpi sovrapposti anche lungo Y quando la pallina colpisce un bordo del
Paddle; dopo la collisione riportiamo quindi il Paddle alla quota memorizzata allo
spawn. Il reset aggiorna `_locked_y`, per cui la regola resta valida anche se in futuro
cambi la posizione iniziale tra gli episodi.

Questo script non calcola reward e non costruisce il vettore delle observation. Offre
solo i dati fisici che i nodi `MethodObservationSource` leggono.

## 5. Creare lo scenario

Crea `res://scenarios/breakout/breakout_scenario.tscn`:

```text
BreakoutScenario (Node2D) [breakout_scenario.gd]
├── BridgeServer          [bridge_server.gd]
├── ScenarioController    [ScenarioController.gd]
│   ├── ScenarioEventSystem [ScenarioEventSystem.gd]
│   │   ├── BrickDestroyed  [ManualScenarioEventSource.gd]
│   │   ├── LifeLost        [ManualScenarioEventSource.gd]
│   │   ├── LevelCleared    [ManualScenarioEventSource.gd]
│   │   └── BallHitEvent    [ManualScenarioEventSource.gd]
│   └── ScenarioRewardSystem [ScenarioRewardSystem.gd]
│       ├── BrickReward     [EventScenarioReward.gd]
│       ├── LifeLostPenalty [EventScenarioReward.gd]
│       ├── LevelReward     [EventScenarioReward.gd]
│       └── HitBall         [EventScenarioReward.gd]
├── BreakoutPaddle (istanza di breakout_paddle.tscn)
├── Ball (RigidBody2D) [breakout_ball.gd]
│   ├── Sprite2D
│   └── CollisionShape2D (CircleShape2D)
├── Walls (Node2D)
│   ├── LeftWall (StaticBody2D)
│   │   └── CollisionShape2D (RectangleShape2D)
│   ├── RightWall (StaticBody2D)
│   │   └── CollisionShape2D (RectangleShape2D)
│   └── Ceiling (StaticBody2D)
│       └── CollisionShape2D (RectangleShape2D)
├── Bricks (Node2D)
│   ├── Brick01 (StaticBody2D, gruppo "brick")
│   │   ├── Sprite2D
│   │   └── CollisionShape2D (RectangleShape2D)
│   ├── Brick02 (StaticBody2D, gruppo "brick")
│   └── ...
└── LossZone (Area2D)
    └── CollisionShape2D (RectangleShape2D)
```

Nel `BridgeServer` imposta:

```text
controller_path = "../ScenarioController"
lockstep_enabled = true
```

Con il lockstep attivo Godot mette in pausa lo scenario mentre Python sceglie l'azione
o aggiorna la rete. La fisica avanza soltanto durante `reset` e `step`: il tempo di
calcolo di TensorFlow non cambia quindi la durata dell'azione precedente. La modalita'
manuale dall'editor resta normale finche' nessun client Python e' connesso.

Nel `ScenarioController`:

```text
controlled_agents = ["../BreakoutPaddle"]
max_steps = 0
physics_frames_per_step = 1
randomize_reset = false
reset_position_jitter = Vector3(0, 0, 0)
reset_yaw_jitter_degrees = 0.0
number_of_replications = 0
```

La Paddle parte inizialmente al centro. La casualita' viene applicata alla Ball dallo
script dello scenario e cresce con il curriculum: in questo modo i primi lanci sono
raggiungibili, senza addestrare sempre la stessa traiettoria.

Non aggiungere un `ProgressProvider`: in Breakout il progresso lungo un percorso non
esiste. Il provider e' facoltativo.

### Configurare correttamente Ball

Seleziona `Ball` e imposta dall'Inspector:

```text
mass = 1.0
gravity_scale = 0.0
linear_damp_mode = Replace
linear_damp = 0.0
angular_damp_mode = Replace
angular_damp = 0.0
lock_rotation = true
can_sleep = false
continuous_cd = Cast Shape
contact_monitor = true
max_contacts_reported = 8
collision_layer = 1
collision_mask = 1
```

`gravity_scale = 0` impedisce alla traiettoria di curvare verso il basso. I due damping
a zero evitano che la pallina perda energia nel tempo. `continuous_cd` riduce il
rischio che una pallina veloce attraversi un mattone sottile tra due frame fisici.
`contact_monitor` e `max_contacts_reported` sono necessari per ricevere il segnale
`body_entered` usato dallo scenario.

In `Physics Material Override` crea un nuovo `PhysicsMaterial`:

```text
friction = 0.0
bounce = 1.0
rough = false
absorbent = false
```

Assegna una `CircleShape2D` leggermente piu' piccola della parte visibile della
pallina. Collisioni rettangolari o una shape piu' grande dello sprite producono
rimbalzi poco intuitivi. Le pareti e i mattoni possono invece usare
`RectangleShape2D`.

Il valore `paddle_half_width` nello script deve corrispondere circa alla meta' della
larghezza della `CollisionShape2D` del Paddle, non necessariamente a quella dello
sprite. Lascia aperto il fondo del campo e posiziona `LossZone` poco sotto il Paddle:
se aggiungi anche una parete inferiore, la pallina non potra' mai essere persa.

Le tre pareti sono `StaticBody2D`: sinistra, destra e soffitto. Estendi le loro shape
leggermente oltre gli angoli, così la pallina non trova fessure tra due collisioni.
`LossZone` deve attraversare tutta la larghezza del campo e avere `monitoring = true`.
Il suo segnale `body_entered` non va collegato dall'Inspector, perche' `_ready()` lo
collega gia' via codice; un doppio collegamento produrrebbe due notifiche della stessa
pallina persa.

Il materiale con bounce massimo conserva quasi tutta l'energia, ma una collisione con
il Paddle in movimento puo' comunque alterare velocita' e direzione. Lo script dello
scenario correggera' questi valori dopo ogni impatto: e' una regola arcade deliberata,
non una reward e non un'azione nascosta dell'agente.

### Scrivere breakout_ball.gd

Crea `res://scenarios/breakout/breakout_ball.gd` e assegnalo al nodo `Ball`:

```gdscript
extends RigidBody2D
class_name BreakoutBall

var _reset_pending := false
var _pending_transform := Transform2D.IDENTITY
var _pending_velocity := Vector2.ZERO


func reset_ball(reset_transform:Transform2D, launch_velocity:Vector2) -> void:
	_pending_transform = reset_transform
	_pending_velocity = launch_velocity
	_reset_pending = true
	freeze = false
	sleeping = false


func stop_ball() -> void:
	_reset_pending = false
	set_deferred("freeze", true)
	call_deferred("reset_physics_interpolation")


func _integrate_forces(state:PhysicsDirectBodyState2D) -> void:
	if not _reset_pending:
		return

	state.transform = _pending_transform
	state.linear_velocity = _pending_velocity
	state.angular_velocity = 0.0
	_reset_pending = false
	call_deferred("_finish_reset_interpolation")


func _finish_reset_interpolation() -> void:
	reset_physics_interpolation()
```

`RigidBody2D` conserva il proprio stato nel server fisico. Cambiare `transform`,
velocita', `freeze` e poi `unfreeze` nello stesso callback puo' mostrare la posizione
nuova per un frame e lasciare che il server ripristini subito il vecchio stato.
`_integrate_forces()` applica invece transform e velocita' insieme, nel punto
autorevole della simulazione. Il metodo `reset_ball()` si limita a preparare il reset
che verra' consumato al frame fisico successivo.

## 6. Configurare eventi e reward di scenario

Configura i quattro `ManualScenarioEventSource`:

```text
BrickDestroyed:
  event_name = "brick_destroyed"

LifeLost:
  event_name = "life_lost"
  terminal_reason = "life_lost"

LevelCleared:
  event_name = "level_cleared"
  terminal_reason = "level_cleared"

BallHit:
  event_name = "ball_hit"
```

Configura i componenti del `ScenarioRewardSystem`:

```text
BrickReward:
  event_name = "brick_destroyed"
  reward = 1.0
  only_once = false

LifeLostPenalty:
  event_name = "life_lost"
  reward = -10.0
  only_once = true

LevelReward:
  event_name = "level_cleared"
  reward = 20.0
  only_once = true

HitBall:
  event_name = "ball_hit"
  reward = 0.2
  only_once = false
```

Il codice del gioco emette soltanto eventi semantici. I numeri delle reward rimangono
visibili e modificabili dall'Inspector.

## 7. Scrivere breakout_scenario.gd

Crea `res://scenarios/breakout/breakout_scenario.gd` e assegnalo alla radice:

```gdscript
extends Node2D

@export var ball_speed := 420.0
@export var paddle_half_width := 64.0
@export_range(0.05, 0.50) var min_ball_vertical_ratio := 0.20
@export_category("Reset Randomization")
@export_range(0.0, 600.0, 1.0) var ball_position_jitter_x := 120.0
@export_range(0.0, 1.0, 0.01) var launch_horizontal_range := 0.35
@export_range(0.0, 0.25, 0.01) var ball_speed_jitter_ratio := 0.05
@export_category("Reset Curriculum")
@export var reset_curriculum_enabled := true
@export_range(0.0, 600.0, 1.0) var curriculum_final_ball_jitter_x := 420.0
@export_range(1, 10000, 1) var curriculum_ramp_episodes := 800
@export_category("Manual Test")
@export var auto_start_manual := true
@export var manual_seed := 1

@onready var controller := $ScenarioController
@onready var paddle: BreakoutPaddle = $BreakoutPaddle
@onready var paddle_agent: Agent = $BreakoutPaddle/Agent
@onready var ball: BreakoutBall = $Ball
@onready var loss_zone: Area2D = $LossZone
@onready var brick_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/BrickDestroyed
@onready var lost_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/LifeLost
@onready var cleared_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/LevelCleared
@onready var ball_hit_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/BallHitEvent

var _ball_start_transform: Transform2D
var _bricks: Array[Node2D] = []
var _remaining_bricks := 0
var _episode_ball_speed := 0.0
var _training_episode := 0


func _ready() -> void:
	_ball_start_transform = ball.global_transform
	_episode_ball_speed = ball_speed
	for child in $Bricks.get_children():
		if child is Node2D:
			_bricks.append(child)
	controller.episode_reset_started.connect(_reset_game)
	controller.scenario_configured.connect(_on_scenario_configured)
	ball.body_entered.connect(_on_ball_body_entered)
	loss_zone.body_entered.connect(_on_loss_zone_body_entered)
	if auto_start_manual and paddle.manual_control:
		call_deferred("_reset_game", manual_seed)


func _reset_game(seed:int) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	paddle.set_episode_terminal(false)
	_remaining_bricks = _bricks.size()
	for brick in _bricks:
		brick.visible = true
		brick.process_mode = Node.PROCESS_MODE_INHERIT
		if brick is CollisionObject2D:
			brick.collision_layer = 1

	var reset_transform := _ball_start_transform
	var position_jitter := _current_ball_position_jitter_x()
	reset_transform.origin.x += rng.randf_range(-position_jitter, position_jitter)
	var launch_direction := Vector2(
		rng.randf_range(-launch_horizontal_range, launch_horizontal_range),
		1.0
	).normalized()
	var speed_scale := rng.randf_range(
		1.0 - ball_speed_jitter_ratio,
		1.0 + ball_speed_jitter_ratio
	)
	_episode_ball_speed = ball_speed * speed_scale
	ball.reset_ball(reset_transform, launch_direction * _episode_ball_speed)


func _on_scenario_configured(config:Dictionary) -> void:
	_training_episode = maxi(0, int(config.get("training_episode", _training_episode)))


func _current_ball_position_jitter_x() -> float:
	if not reset_curriculum_enabled:
		return ball_position_jitter_x
	var ratio := clampf(
		float(_training_episode) / float(maxi(curriculum_ramp_episodes, 1)),
		0.0,
		1.0
	)
	return lerpf(ball_position_jitter_x, curriculum_final_ball_jitter_x, ratio)


func _on_ball_body_entered(body:Node) -> void:
	if body == paddle:
		# Il punto di impatto permette all'agente di controllare il rimbalzo.
		var horizontal_offset := clampf(
			(ball.global_position.x - paddle.global_position.x) / maxf(paddle_half_width, 0.001),
			-1.0,
			1.0
		)
		call_deferred("_apply_paddle_bounce", horizontal_offset)
		return

	# Aspetta la risposta fisica di Godot, poi ripristina la velocita' arcade.
	call_deferred("_stabilize_ball_velocity")
	var brick := body as Node2D
	if brick == null or not brick.is_in_group("brick") or not brick.visible:
		return
	brick.visible = false
	brick.set_deferred("process_mode", Node.PROCESS_MODE_DISABLED)
	if brick is CollisionObject2D:
		brick.set_deferred("collision_layer", 0)
	_remaining_bricks -= 1
	brick_event.trigger(str(paddle.name))
	if _remaining_bricks <= 0:
		ball.stop_ball()
		cleared_event.trigger(str(paddle.name))
		paddle.set_episode_terminal(true)


func _on_loss_zone_body_entered(body:Node) -> void:
	if body != ball:
		return
	ball.stop_ball()
	lost_event.trigger(str(paddle.name))
	paddle.set_episode_terminal(true)


func _stabilize_ball_velocity() -> void:
	if ball.freeze or ball.linear_velocity.is_zero_approx():
		return
	_set_ball_direction(ball.linear_velocity.normalized())


func _apply_paddle_bounce(horizontal_offset:float) -> void:
	_set_ball_direction(Vector2(horizontal_offset, -1.0))
	ball_hit_event.trigger(str(paddle.name))


func _set_ball_direction(direction:Vector2) -> void:
	if direction.is_zero_approx():
		direction = Vector2(0.0, -1.0)
	var vertical_sign := signf(direction.y)
	if is_zero_approx(vertical_sign):
		vertical_sign = -1.0
	if absf(direction.y) < min_ball_vertical_ratio:
		direction.y = vertical_sign * min_ball_vertical_ratio
	ball.linear_velocity = direction.normalized() * _episode_ball_speed
	ball.angular_velocity = 0.0
```

Il reset della pallina viene richiesto a `BreakoutBall` e applicato atomicamente nel
frame fisico successivo. Posizione e velocita' iniziali cambiano in modo riproducibile
in base al seed. Dopo l'impatto con il Paddle, la coordinata orizzontale del
punto di contatto determina la nuova direzione: colpire vicino al bordo produce un
tiro piu' diagonale. `min_ball_vertical_ratio` evita traiettorie quasi orizzontali che
potrebbero rimbalzare tra le pareti per centinaia di step senza raggiungere ne' Paddle
ne' mattoni.

Non impostare la posizione di un `RigidBody2D` continuamente durante il gioco. Qui lo
facciamo soltanto tra due episodi e attraverso `PhysicsDirectBodyState2D`. Durante la
partita la posizione resta interamente sotto il controllo del motore fisico.

### Avvio manuale e avvio dal trainer

Durante il training, Python invia il comando `reset` al bridge e il
`ScenarioController` emette `episode_reset_started`: e' questo segnale a chiamare
`_reset_game()` e lanciare la pallina. Avviando direttamente la scena dall'editor non
esiste ancora un client Python, quindi il blocco `auto_start_manual` esegue un primo
reset differito quando il Paddle ha `manual_control = true`.

Per una prova manuale:

```text
BreakoutPaddle.manual_control = true
BreakoutScenario.auto_start_manual = true
BreakoutScenario.manual_seed = 1
```

Prima del training e' comunque consigliato salvare:

```text
BreakoutPaddle.manual_control = false
```

Con `manual_control` attivo, `_physics_process()` legge la tastiera a ogni frame e
sovrascriverebbe l'azione ricevuta da Python. Come protezione, `apply_action()` lo
disattiva automaticamente alla prima azione non manuale. `auto_start_manual` da solo
non interferisce con il training, perché viene usato soltanto quando anche il Paddle
e' manuale.

## 8. Collision layer consigliati

Una configurazione semplice:

| Oggetto | Layer | Mask |
|---|---:|---:|
| Ball | 1 | 1 |
| Paddle | 1 | 1 |
| Walls | 1 | 1 |
| Bricks | 1 | 1 |
| LossZone | 2 | 1 |

Metti ogni mattone nel gruppo `brick` dal pannello Node > Groups.

## 9. Validare prima del training

Avvia azioni casuali:

```bash
python/.venv/bin/python python/random_scenario_rollout.py \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/breakout/breakout_scenario.tscn \
  --steps 100 \
  --print-reward-terms \
  --no-headless
```

Controlla che lo spec riporti:

```text
action_type=discrete
obs_shape=(5,)
num_actions=3
```

Verifica poi questi casi manualmente:

1. la Paddle parte al centro; lo stesso seed riproduce la posizione della Ball e seed diversi la spostano;
2. un mattone produce una sola reward `brick_destroyed` per collisione;
3. la LossZone termina l'episodio con `terminal_reason=life_lost`;
4. l'ultimo mattone termina con `terminal_reason=level_cleared`;
5. la pallina non cade per gravita' e mantiene circa `ball_speed` dopo i rimbalzi;
6. colpire lati diversi del Paddle cambia l'angolo della traiettoria;
7. la pallina non rimane intrappolata in una traiettoria quasi orizzontale;
8. nessuna observation contiene `NaN` o valori enormi.

Il rollout puo' essere eseguito anche con `--delay 2`: tra due richieste la Ball deve
restare ferma. Ogni nuova riga deve mostrare soltanto l'avanzamento prodotto dallo
`step` corrente. In headless il process manager usa `--fixed-fps 60`, che conserva un
delta fisico di 1/60 s ma esegue i tick senza attendere il tempo reale.

### Osservare il training con piu' env

Il collector asincrono e' il default: ogni env avanza indipendentemente e il learner
consuma la coda senza imporre una barriera fra le istanze. Senza `--headless`, tutti
gli env sono visibili. Per mostrare una sola preview e lasciare gli altri worker
headless usa:

```text
--collector-mode async --render-env-count 1
```

La preview non e' un env speciale: usa la stessa policy, le stesse observation e le
stesse reward degli altri. Per mostrare tutte le finestre ometti `--render-env-count`,
ma il rendering concorrente le rende sensibilmente piu' lente. Per confrontare il
comportamento sincronizzato usa `--collector-mode sync`; in quel caso
`--no-parallel-env-steps` abilita anche il vecchio stepping seriale, utile soltanto per
debug.

Con `--collector-mode sync`, il lockstep mette ancora in pausa la preview mentre
TensorFlow aggiorna il modello. Puoi invece separare raccolta e learner con:

```text
--collector-mode async \
--async-queue-capacity 256 \
--async-update-every 4 \
--async-drain-max-events 64 \
--render-env-count 1
```

In questa modalita' ogni env possiede un collector indipendente e una copia locale
della policy; TensorFlow allena il modello principale mentre Godot continua a
raccogliere. Il lockstep rimane attivo dentro ogni transizione, quindi non vengono
persi tick fisici. La preview e' normalmente molto piu' fluida, ma non e' garantita a
60 FPS se la coda si riempie o inferenza e socket sono troppo lenti.

## 10. Avviare il training

`--algorithm auto` rileva lo spazio discreto e seleziona DQN:

```bash
python/.venv/bin/python python/train_generic.py \
  --algorithm auto \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/breakout/breakout_scenario.tscn \
  --num-envs 4 \
  --num-episodes 2000 \
  --max-steps-per-episode 0 \
  --batch-size 128 \
  --replay-warmup 5000 \
  --epsilon-start 1.0 \
  --epsilon-min 0.05 \
  --epsilon-decay 0.997 \
  --collector-mode async \
  --async-update-every 4 \
  --async-drain-max-events 64 \
  --log-action-every 10 \
  --checkpoint-dir checkpoints/breakout_lockstep_v1 \
  --weights-path breakout_lockstep_v1.weights.h5 \
  --headless
```

Per il primo training dopo l'introduzione del lockstep usa una directory nuova e non
aggiungere `--resume` o `--initial-weights-path`. I vecchi replay contengono transizioni
nelle quali Godot poteva avanzare durante l'aggiornamento TensorFlow; anche i pesi
derivati da quelle transizioni non sono una base affidabile per questa validazione.

Il log DQN ora riporta diagnostica indipendente dal tipo di scenario. Per Breakout
leggerai, per esempio:

```text
outcome   reward=[...]  success=1/4 (25.00%)  steps=mean:184.5 range:[91,302]
          events={'ball_hit': 6, 'brick_destroyed': 14, 'level_cleared': 1, 'life_lost': 3}
          terminal={'level_cleared': 1, 'life_lost': 3}
```

`success` conta `level_cleared`, non `target_reached`. `events` somma gli eventi
avvenuti nell'episodio su tutti gli env; `terminal` spiega come sono terminati; `steps`
permette di riconoscere subito episodi insolitamente brevi.

Il valore `0` viene inviato da Python al `ScenarioController` e disabilita la sola
troncatura per numero di step. L'episodio continua finche' la palla viene persa oppure
sono distrutti tutti i blocchi. Non usare questa modalita' in uno scenario privo di una
condizione terminale affidabile.

Per riprendere successivamente il training completo:

```bash
python/.venv/bin/python python/train_generic.py \
  --algorithm dqn \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/breakout/breakout_scenario.tscn \
  --num-envs 4 \
  --batch-size 128 \
  --checkpoint-dir checkpoints/breakout_lockstep_v1 \
  --weights-path breakout_lockstep_v1.weights.h5 \
  --max-steps-per-episode 0 \
  --num-episodes 3000 \
  --resume \
  --headless
```

## 11. Eseguire il modello addestrato

Al termine del training non serve uno script specifico per Breakout. Usa
`run_generic_policy.py`, che legge lo stesso action space e lo stesso observation
space esposti dallo scenario.

Per caricare automaticamente l'ultimo checkpoint disponibile:

```bash
python/.venv/bin/python python/run_generic_policy.py \
  --algorithm dqn \
  --load-from checkpoint \
  --checkpoint-dir checkpoints/breakout_lockstep_v1 \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/breakout/breakout_scenario.tscn \
  --episodes 20 \
  --max-steps 1500 \
  --epsilon 0.0 \
  --delay 0.02 \
  --no-headless
```

`--epsilon 0.0` disabilita le azioni casuali: il Paddle usa sempre l'azione con il
Q-value maggiore. `--delay 0.02` rallenta soltanto la visualizzazione e non modifica
la fisica dello scenario.

Se vuoi caricare direttamente i pesi finali invece di un checkpoint:

```bash
python/.venv/bin/python python/run_generic_policy.py \
  --algorithm dqn \
  --load-from weights \
  --weights-path breakout_lockstep_v1.weights.h5 \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/breakout/breakout_scenario.tscn \
  --episodes 20 \
  --max-steps 1500 \
  --epsilon 0.0 \
  --delay 0.02 \
  --no-headless
```

Per lasciare la policy in esecuzione finche' non premi `Ctrl+C`:

```bash
python/.venv/bin/python python/run_generic_policy.py \
  --algorithm dqn \
  --load-from checkpoint \
  --checkpoint-dir checkpoints/breakout_lockstep_v1 \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/breakout/breakout_scenario.tscn \
  --infinite \
  --no-time-limit \
  --epsilon 0.0 \
  --delay 0.02 \
  --no-headless
```

Su macOS sostituisci il valore di `--godot-bin` con:

```text
/Applications/Godot.app/Contents/MacOS/Godot
```

Il modello puo' essere caricato solo se action space e dimensione delle observation
sono uguali a quelli usati durante il training. Se modifichi le observation del
Paddle, devi usare un modello addestrato con la nuova dimensione.

## 12. Curriculum automatico

Prima di ogni reset, tutti i trainer comunicano allo scenario il valore
`training_episode`. Il valore proviene dal checkpoint, quindi non riparte da zero
dopo `--resume`.

Il curriculum di reset e' gia' presente nello script completo:

```gdscript
ball_position_jitter_x = 120.0
curriculum_final_ball_jitter_x = 420.0
curriculum_ramp_episodes = 800
reset_curriculum_enabled = true
```

All'episodio 0 la Ball nasce al massimo a 120 pixel dal centro. La distanza massima
cresce linearmente fino a 420 pixel all'episodio 800. Paddle, action space,
observation e reward non cambiano, percio' la rete attraversa il curriculum senza
dover essere ricreata.

Un secondo curriculum, facoltativo, puo' modificare numero di mattoni, velocita' e
angolo iniziale. Aggiungi soltanto queste proprieta', riusando `_training_episode` e
`_on_scenario_configured()` gia' presenti:

```gdscript
@export var gameplay_curriculum_enabled := false
@export var stage_2_episode := 250
@export var stage_3_episode := 700

func _apply_gameplay_curriculum() -> void:
	if not gameplay_curriculum_enabled:
		return
	if _training_episode < stage_2_episode:
		ball_speed = 320.0
		launch_horizontal_range = 0.20
	elif _training_episode < stage_3_episode:
		ball_speed = 380.0
		launch_horizontal_range = 0.35
	else:
		ball_speed = 420.0
		launch_horizontal_range = 0.50
```

Chiama `_apply_gameplay_curriculum()` all'inizio di `_reset_game()`, prima di calcolare
`speed_scale`. Lascialo disabilitato nel primo esperimento: il curriculum di posizione
e' sufficiente per validare il framework senza cambiare troppe variabili insieme.

Le soglie per episodio sono un punto di partenza riproducibile. Una versione piu'
avanzata puo' passare allo stadio successivo dopo, per esempio, l'80% di livelli
completati negli ultimi 100 episodi. In quel caso salva lo stadio nel checkpoint o
rendilo derivabile da `training_episode`, per evitare regressioni dopo un resume.

## 13. Domain randomization

Dopo che l'agente completa stabilmente il livello, randomizza con moderazione:

- posizione orizzontale iniziale della pallina;
- angolo di lancio, evitando angoli quasi orizzontali;
- velocita' della pallina entro circa il 10%;
- larghezza del Paddle entro circa il 10%;
- righe e disposizione dei mattoni.

Usa sempre il `seed` ricevuto da `_reset_game(seed)`. In questo modo ogni ambiente ha
partite differenti, ma una partita rimane riproducibile conoscendo il seed.

Non randomizzare contemporaneamente colori, collisioni, velocita' e layout durante i
primi episodi. La varieta' aiuta la generalizzazione solo dopo che esiste una strategia
di base.

## 14. Valutazione separata

Ogni 100-200 episodi esegui il modello senza esplorazione e senza aggiornamenti. Non
valutare la policy guardando soltanto la reward del training, perche' epsilon produce
ancora azioni casuali.

Registra almeno:

- percentuale di livelli completati;
- mattoni medi distrutti;
- durata media dell'episodio;
- percentuale di palline perse;
- risultato per ciascuno stadio del curriculum.

Conserva una scena o configurazione di valutazione con curriculum disabilitato e tutti
i mattoni attivi. Il modello migliore e' quello che generalizza sul livello completo,
non necessariamente l'ultimo checkpoint.

## 15. Quando aggiungere complessita'

Prima fai imparare una singola riga di mattoni. Poi aumenta gradualmente:

1. tre righe di mattoni;
2. angolo iniziale della pallina piu' casuale;
3. velocita' crescente;
4. dimensione Paddle ridotta;
5. layout diversi dei mattoni.

Il curriculum resta nel gioco: Python continua a usare lo stesso trainer generico.
Non cambiare contemporaneamente observation, reward e fisica; altrimenti diventa
difficile capire quale modifica ha aiutato o rotto il training.

## 16. Regola pratica per nuovi scenari

Per ogni nuovo progetto chiediti nell'ordine:

1. quali decisioni deve prendere l'agente;
2. quali informazioni avrebbe davvero a disposizione;
3. quali eventi definiscono successo e fallimento;
4. quali reward intermedie sono indispensabili;
5. quale condizione impedisce episodi infiniti.

Il codice del corpo risponde ai punti 1 e 2. I nodi `Agent`, `ObservationSystem`,
`RewardSystem`, `ScenarioEventSystem` e `ScenarioRewardSystem` configurano tutto il
resto dall'Inspector.
