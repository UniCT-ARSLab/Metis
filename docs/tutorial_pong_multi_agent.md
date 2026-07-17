# Tutorial: Pong con due agenti e policy condivisa

Questo tutorial costruisce Pong in 2D con due Paddle controllati contemporaneamente.
I Paddle sono due istanze della stessa scena, espongono lo stesso action space e lo
stesso vettore di observation. Python usa una sola policy allenabile condivisa. Nel
self-play simultaneo il replay riceve le esperienze di entrambi; con opponent pool
riceve soltanto quelle della squadra controllata dalla policy corrente.

## 1. Multi-agent, multi-env e self-play

Sono concetti distinti:

- **multi-agent**: una partita contiene `LeftPaddle` e `RightPaddle`;
- **multi-env**: Python esegue piu' partite Godot indipendenti su porte diverse;
- **policy condivisa**: entrambi i Paddle usano la stessa rete neurale;
- **self-play simultaneo**: la policy corrente gioca contro un'altra copia di se stessa;
- **opponent pool**: una squadra gioca usando una vecchia snapshot congelata.

Con due Paddle e quattro env, il self-play simultaneo puo' produrre fino a otto
transizioni per step. Contro una snapshot ne produce invece quattro: le azioni del
vecchio avversario servono alla partita, ma non devono essere attribuite alla policy
corrente. Non viene scelto il Paddle migliore e non esistono due policy allenabili.

Questo e' adatto a Pong soltanto se le observation sono espresse dal punto di vista
dell'agente. Se il Paddle destro vedesse coordinate globali non trasformate, la rete
dovrebbe imparare due problemi differenti.

Il framework supporta sia self-play simultaneo sia un pool di vecchie policy. Nel
secondo caso una squadra usa la policy corrente e l'altra una snapshot congelata;
soltanto le transizioni della squadra corrente allenano il modello.

## 2. Progettare il problema

### Actions

Ogni Paddle ha tre azioni discrete:

| ID | Nome | Effetto |
|---:|---|---|
| 0 | `idle` | fermo |
| 1 | `move_up` | sale |
| 2 | `move_down` | scende |

### Observations agent-centriche

Ogni Paddle riceve sei valori:

| Observation | Dimensione | Significato |
|---|---:|---|
| `paddle_y` | 1 | posizione verticale propria |
| `ball_relative_position` | 2 | pallina rispetto al Paddle |
| `ball_velocity` | 2 | velocita' vista nella direzione dell'avversario |
| `opponent_delta_y` | 1 | differenza verticale dell'avversario |

Per il Paddle destro invertiamo soltanto l'asse X di posizione e velocita'. In questo
modo, per entrambi, X positiva significa "verso l'avversario".

### Reward zero-sum

| Evento | Vincitore | Perdente | Terminale |
|---|---:|---:|---|
| tocco della pallina | `+0.02` | `0` | no |
| punto segnato | `+1.0` | `-1.0` | entrambi |
| ogni step | `-0.0005` | `-0.0005` | no |

La reward per il tocco e' piccola: aiuta all'inizio, ma non deve diventare piu'
conveniente del segnare. Non premiare semplicemente il movimento del Paddle, altrimenti
la policy puo' imparare a oscillare senza seguire la pallina.

## 3. Creare PongPaddle.tscn

Crea `res://agents/PongPaddle/pong_paddle.tscn`:

```text
PongPaddle (CharacterBody2D) [pong_paddle.gd]
├── Sprite2D
├── CollisionShape2D
└── Agent                    [Agent.gd]
    ├── ActionSpace          [ActionSpace.gd]
    │   └── Actions          [DiscreteActionSet.gd]
    ├── ObservationSystem    [ObservationSystem.gd]
    │   ├── PaddleY          [MethodObservationSource.gd]
    │   ├── BallPosition     [MethodObservationSource.gd]
    │   ├── BallVelocity     [MethodObservationSource.gd]
    │   └── OpponentY        [MethodObservationSource.gd]
    └── RewardSystem         [RewardSystem.gd]
        └── TimePenalty      [StepPenaltyReward.gd]
```

Configura `Actions`:

```text
action_name = "action"
names = ["idle", "move_up", "move_down"]
```

Configura le observation nell'ordine indicato:

```text
PaddleY:
  observation_name = "paddle_y"
  method_name = "get_paddle_y_observation"

BallPosition:
  observation_name = "ball_relative_position"
  method_name = "get_ball_relative_position_observation"

BallVelocity:
  observation_name = "ball_velocity"
  method_name = "get_ball_velocity_observation"

OpponentY:
  observation_name = "opponent_delta_y"
  method_name = "get_opponent_delta_y_observation"
```

Configura `TimePenalty.penalty = -0.0005` e assegna il gruppo `paddle` alla radice.

## 4. Scrivere pong_paddle.gd

```gdscript
extends CharacterBody2D
class_name PongPaddle

@export var speed := 420.0
@export var vertical_limit := 300.0
@export var ball_speed_scale := 520.0
@export var view_direction := 1.0
@export var team_id := 0
@export var ball: CharacterBody2D
@export var opponent: PongPaddle
@export var manual_control := false

@onready var agent: Agent = $Agent

var _move_input := 0.0
var _terminal := false
var _training_active := true


func _ready() -> void:
	agent.add_action("idle", Callable(self, "stop"))
	agent.add_action("move_up", Callable(self, "move_up"))
	agent.add_action("move_down", Callable(self, "move_down"))


func get_team_id() -> int:
	return team_id


func _physics_process(_delta:float) -> void:
	if not _training_active:
		return
	if manual_control:
		_move_input = Input.get_axis("move_up", "move_down")
	velocity = Vector2(0.0, _move_input * speed)
	move_and_slide()
	global_position.y = clampf(global_position.y, -vertical_limit, vertical_limit)


func apply_action(action:Variant) -> Variant:
	if str(action) == "manual":
		return apply_manual_action()
	stop()
	var action_id := int(action)
	if agent.act(action_id) != OK:
		return 0
	return action_id


func apply_manual_action() -> int:
	var axis := Input.get_axis("move_up", "move_down")
	var action_id := 0
	if axis < -0.2:
		action_id = 1
	elif axis > 0.2:
		action_id = 2
	return apply_action(action_id)


func stop() -> void:
	_move_input = 0.0


func move_up() -> void:
	_move_input = -1.0


func move_down() -> void:
	_move_input = 1.0


func get_paddle_y_observation() -> float:
	return clampf(global_position.y / maxf(vertical_limit, 0.001), -1.0, 1.0)


func get_ball_relative_position_observation() -> Vector2:
	if ball == null:
		return Vector2.ZERO
	var relative := ball.global_position - global_position
	relative.x *= view_direction
	return Vector2(
		clampf(relative.x / 640.0, -1.0, 1.0),
		clampf(relative.y / maxf(vertical_limit, 0.001), -1.0, 1.0)
	)


func get_ball_velocity_observation() -> Vector2:
	if ball == null:
		return Vector2.ZERO
	var perceived_velocity := ball.velocity
	perceived_velocity.x *= view_direction
	return (perceived_velocity / maxf(ball_speed_scale, 0.001)).clamp(
		Vector2(-1.0, -1.0),
		Vector2(1.0, 1.0)
	)


func get_opponent_delta_y_observation() -> float:
	if opponent == null:
		return 0.0
	return clampf(
		(opponent.global_position.y - global_position.y) / maxf(vertical_limit * 2.0, 0.001),
		-1.0,
		1.0
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

Nello scenario imposta `view_direction = 1` sul Paddle sinistro e `-1` sul destro.
Entrambi continuano a usare `move_up` e `move_down` nello stesso sistema di coordinate.

## 5. Creare PongBall.tscn

Usa un `CharacterBody2D`, non un `RigidBody2D`, per mantenere velocita' costante e
rimbalzi riproducibili:

```text
PongBall (CharacterBody2D) [pong_ball.gd]
├── Sprite2D
└── CollisionShape2D
```

`pong_ball.gd`:

```gdscript
extends CharacterBody2D
class_name PongBall

signal paddle_hit(paddle:Node)

@export var speed := 420.0


func _physics_process(delta:float) -> void:
	var collision := move_and_collide(velocity * delta)
	if collision == null:
		return
	velocity = velocity.bounce(collision.get_normal()).normalized() * speed
	var collider := collision.get_collider()
	if collider is Node and collider.is_in_group("paddle"):
		paddle_hit.emit(collider)


func launch(direction:float, vertical_component:float) -> void:
	velocity = Vector2(direction, vertical_component).normalized() * speed
```

## 6. Creare pong_scenario.tscn

```text
PongScenario (Node2D) [pong_game.gd]
├── BridgeServer
├── ScenarioController
│   ├── ScenarioEventSystem
│   │   ├── RallyHit   [ManualScenarioEventSource.gd]
│   │   ├── PointWon   [ManualScenarioEventSource.gd]
│   │   └── PointLost  [ManualScenarioEventSource.gd]
│   └── ScenarioRewardSystem
│       ├── RallyReward [EventScenarioReward.gd]
│       ├── WinReward   [EventScenarioReward.gd]
│       └── LossPenalty [EventScenarioReward.gd]
├── LeftPaddle  (istanza PongPaddle)
├── RightPaddle (istanza PongPaddle)
├── Ball        (istanza PongBall)
├── TopWall     (StaticBody2D)
├── BottomWall  (StaticBody2D)
├── LeftGoal    (Area2D)
└── RightGoal   (Area2D)
```

Nel `ScenarioController`:

```text
controlled_agents = ["../LeftPaddle", "../RightPaddle"]
number_of_replications = 0
randomize_reset = false
max_steps = 1200
```

Non usare la replica automatica: qui ogni istanza deve avere posizione, avversario e
`view_direction` espliciti.

Assegna su entrambi i Paddle il riferimento a `Ball`. Poi assegna:

```text
LeftPaddle.opponent = RightPaddle
LeftPaddle.view_direction = 1.0
LeftPaddle.team_id = 0

RightPaddle.opponent = LeftPaddle
RightPaddle.view_direction = -1.0
RightPaddle.team_id = 1
```

## 7. Configurare eventi e reward

```text
RallyHit:
  event_name = "rally_hit"

PointWon:
  event_name = "point_won"
  terminal_reason = "point_won"

PointLost:
  event_name = "point_lost"
  terminal_reason = "point_lost"
```

Reward:

```text
RallyReward:
  event_name = "rally_hit"
  reward = 0.02
  only_once = false

WinReward:
  event_name = "point_won"
  reward = 1.0
  only_once = true

LossPenalty:
  event_name = "point_lost"
  reward = -1.0
  only_once = true
```

## 8. Scrivere pong_game.gd

```gdscript
extends Node2D

@onready var controller := $ScenarioController
@onready var left_paddle: PongPaddle = $LeftPaddle
@onready var right_paddle: PongPaddle = $RightPaddle
@onready var ball: PongBall = $Ball
@onready var left_goal: Area2D = $LeftGoal
@onready var right_goal: Area2D = $RightGoal
@onready var rally_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/RallyHit
@onready var win_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/PointWon
@onready var loss_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/PointLost

var _ball_start_transform: Transform2D
var _training_episode := 0
var _point_finished := false


func _ready() -> void:
	_ball_start_transform = ball.transform
	controller.scenario_configured.connect(_on_scenario_configured)
	controller.episode_reset_started.connect(_reset_point)
	ball.paddle_hit.connect(_on_paddle_hit)
	left_goal.body_entered.connect(_on_left_goal_entered)
	right_goal.body_entered.connect(_on_right_goal_entered)


func _on_scenario_configured(config:Dictionary) -> void:
	_training_episode = int(config.get("training_episode", _training_episode))


func _reset_point(seed:int) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	_point_finished = false
	left_paddle.set_episode_terminal(false)
	right_paddle.set_episode_terminal(false)
	ball.transform = _ball_start_transform
	ball.speed = _curriculum_ball_speed()
	var serve_direction := -1.0 if rng.randi() % 2 == 0 else 1.0
	ball.launch(serve_direction, rng.randf_range(-_curriculum_angle(), _curriculum_angle()))


func _on_paddle_hit(paddle:Node) -> void:
	if not _point_finished:
		rally_event.trigger(str(paddle.name))


func _on_left_goal_entered(body:Node) -> void:
	if body == ball:
		_finish_point(right_paddle, left_paddle)


func _on_right_goal_entered(body:Node) -> void:
	if body == ball:
		_finish_point(left_paddle, right_paddle)


func _finish_point(winner:PongPaddle, loser:PongPaddle) -> void:
	if _point_finished:
		return
	_point_finished = true
	ball.velocity = Vector2.ZERO
	win_event.trigger(str(winner.name))
	loss_event.trigger(str(loser.name))
	winner.set_episode_terminal(true)
	loser.set_episode_terminal(true)


func _curriculum_ball_speed() -> float:
	if _training_episode < 300:
		return 300.0
	if _training_episode < 800:
		return 380.0
	return 460.0


func _curriculum_angle() -> float:
	if _training_episode < 300:
		return 0.25
	if _training_episode < 800:
		return 0.50
	return 0.80
```

## 9. Perche' entrambi possono usare lo stesso modello

La policy riceve sempre una situazione nel proprio riferimento:

- pallina davanti: X relativa positiva;
- pallina dietro: X relativa negativa;
- avversario sopra o sotto: stessa convenzione per entrambi;
- azioni verticali: identiche.

Sebbene TensorFlow produca un batch di azioni per tutti gli agenti, ogni riga dipende
dalla relativa observation. I Paddle non fanno necessariamente la stessa mossa.

Nel self-play simultaneo il replay contiene transizioni del tipo:

```text
(obs LeftPaddle, azione LeftPaddle, reward LeftPaddle, next_obs LeftPaddle)
(obs RightPaddle, azione RightPaddle, reward RightPaddle, next_obs RightPaddle)
```

Entrambe aggiornano la stessa Q-network. Contro una snapshot, invece, Python crea una
maschera in base a `team_id`. Se in un env il learner e' il team `0`, viene conservata
solo la prima transizione; in un altro env puo' essere sorteggiato il team `1`.

La snapshot usa una seconda istanza della stessa architettura Q-network, caricata con
pesi storici e non aggiornata. La policy corrente continua a usare l'epsilon del
training; l'avversario storico sceglie invece l'azione greedy con epsilon zero.

## 10. Validare il multi-agent

```bash
python/.venv/bin/python python/random_scenario_rollout.py \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/pong/pong_scenario.tscn \
  --multi-agent \
  --steps 200 \
  --print-reward-terms \
  --no-headless
```

Lo spec atteso e':

```text
agents=['LeftPaddle', 'RightPaddle']
action_type=discrete
obs_shape=(2, 6)
num_actions=3
teams=[0, 1]
unassigned=0
```

Controlla inoltre:

1. i due vettori observation sono diversi quando i Paddle sono in posizioni diverse;
2. un tocco assegna `rally_hit` soltanto al Paddle che ha colpito;
3. un punto assegna `+1` al vincitore e `-1` al perdente;
4. entrambi risultano terminali nello stesso step;
5. il reset riposiziona Paddle e pallina;
6. all'avvio del trainer compaiono `teams=[0, 1]` e `unassigned=0`.

## 11. Avviare il training

```bash
python/.venv/bin/python python/train_generic.py \
  --algorithm dqn \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/pong/pong_scenario.tscn \
  --num-envs 4 \
  --num-episodes 2500 \
  --max-steps-per-episode 1200 \
  --batch-size 128 \
  --replay-warmup 8000 \
  --epsilon-start 1.0 \
  --epsilon-min 0.05 \
  --epsilon-decay 0.997 \
  --target-update-every 20 \
  --checkpoint-dir checkpoints/pong_shared_dqn_v1 \
  --weights-path pong_shared_dqn_v1.weights.h5 \
  --multi-agent \
  --collector-mode async \
  --opponent-pool \
  --opponent-snapshot-every 100 \
  --opponent-pool-size 10 \
  --opponent-current-probability 0.2 \
  --opponent-sampling uniform \
  --headless
```

Con quattro env vengono controllati otto Paddle. Esiste una sola policy allenabile;
il pool conserva fino a dieci copie congelate dei suoi pesi. Per ogni episodio viene
scelta una snapshot e, in ogni env, viene sorteggiato quale `team_id` apprende. Nel 20%
degli episodi entrambi i team usano invece la policy corrente.

Il trainer DQN supporta l'opponent pool anche con `--collector-mode async`. Ogni worker
seleziona a inizio episodio la propria snapshot e il proprio lato learner, conserva
questa associazione fino al terminale e usa un modello avversario privato. Le transizioni
della snapshot congelata restano escluse dal replay. Gli altri backend supportano il
collector async normale, ma al momento richiedono ancora `sync` quando usano
`--opponent-pool`.

Pong richiede reazioni rapide: conserva `--physics-frames-per-step 1` come base. Un
frame skip maggiore puo' aumentare il throughput, ma riduce la frequenza con cui il
Paddle puo' correggere la direzione e cambia la durata delle reward per step.

### Significato dei parametri del pool

| Parametro | Effetto |
|---|---|
| `--opponent-pool` | attiva il self-play contro policy storiche |
| `--opponent-snapshot-every 100` | salva una copia ogni 100 episodi completati |
| `--opponent-pool-size 10` | conserva al massimo dieci snapshot |
| `--opponent-current-probability 0.2` | usa current-vs-current nel 20% degli episodi |
| `--opponent-sampling uniform` | sceglie uniformemente tra le snapshot non future |
| `--learner-team 0` | opzionale: allena sempre il lato 0 invece di alternarlo |

Lascia normalmente `--learner-team` non specificato. La rotazione per env evita che il
modello associ una strategia al lato sinistro o destro. `latest` e' utile durante una
fase di debug o un curriculum molto rapido; `uniform` protegge meglio dalla perdita di
strategie gia' apprese.

Poiche' gli episodi contro snapshot inseriscono meta' transizioni, il replay cresce
piu' lentamente rispetto al self-play simultaneo. Quattro env sono un buon punto di
partenza; non abbassare `--replay-warmup` soltanto per far iniziare prima gli update.

## 12. Curriculum consigliato

Il codice precedente usa tre stadi:

| Episodi | Velocita' | Angolo verticale |
|---:|---:|---:|
| `0-299` | 300 | ridotto |
| `300-799` | 380 | medio |
| `800+` | 460 | completo |

Puoi aggiungere anche:

- Paddle inizialmente piu' alto e poi piu' corto;
- piccola casualita' nella posizione iniziale dei Paddle;
- variazione della velocita' entro il 5-10%;
- arena leggermente piu' alta nella valutazione finale.

Action space e observation devono mantenere la stessa dimensione in tutti gli stadi.

Le snapshot non conservano una copia della scena: una vecchia policy viene eseguita
nelle condizioni fisiche dell'episodio corrente. Per esempio, una snapshot del primo
stadio puo' affrontare la pallina veloce del terzo. Questo e' utile per misurare
robustezza, ma durante passaggi di curriculum molto bruschi puo' rendere il match troppo
facile o troppo difficile. In quel caso usa temporaneamente `--opponent-sampling latest`
e torna a `uniform` quando la policy si e' stabilizzata.

## 13. Opponent pool e suoi limiti

Nel self-play simultaneo entrambi usano la policy corrente, quindi anche l'avversario
cambia mentre la rete impara. Questo puo' produrre cicli: una strategia sfrutta quella
corrente, poi viene sostituita da un'altra che dimentica difese precedenti.

Le opzioni usate sopra implementano la prima versione avanzata:

1. salvare periodicamente snapshot della policy;
2. scegliere talvolta uno snapshot come avversario congelato;
3. alternare la squadra allenabile per evitare un bias di lato;
4. escludere dal replay le transizioni prodotte dalla policy congelata.

Il manifest e i pesi sono salvati in
`checkpoints/pong_shared_dqn_v1/opponents/`, quindi `--resume` riutilizza anche il pool.
`--opponent-sampling uniform` allena contro epoche diverse; `latest` usa sempre la
snapshot piu' recente. `--learner-team 0` puo' fissare il lato allenabile, ma normalmente
e' meglio lasciare il sorteggio automatico.

La directory assume questa forma:

```text
checkpoints/pong_shared_dqn_v1/
├── checkpoint
├── ckpt-100.*
├── replay-100.npz
└── opponents/
    ├── manifest.json
    ├── policy-00000000.weights.h5
    ├── policy-00000100.weights.h5
    └── policy-00000200.weights.h5
```

Il manifest registra algoritmo, dimensione delle observation, numero di azioni ed
episodio di ogni snapshot. Se cambi `obs_dim` o action space, il pool precedente viene
rifiutato: usa un nuovo `--checkpoint-dir` oppure un nuovo `--opponent-pool-dir`.

Quando il limite viene superato, il framework elimina dal pool la snapshot piu'
vecchia. I normali checkpoint TensorFlow seguono invece `--keep-checkpoints` e hanno un
ciclo di conservazione separato.

### Bootstrap facoltativo

E' valido attivare il pool dall'episodio zero, come nel comando precedente. La prima
snapshot rappresenta la rete iniziale e fornisce un avversario debole e stabile.

In alternativa puoi eseguire 200-300 episodi di self-play simultaneo con
`--no-opponent-pool`, poi riprendere lo stesso checkpoint aggiungendo
`--collector-mode async --opponent-pool`. Al primo episodio ripreso, il framework crea la prima snapshot dalla
policy gia' parzialmente allenata. Questa variante puo' aiutare se all'inizio gli scambi
finiscono immediatamente e il replay contiene quasi soltanto servizi falliti.

Non e' ancora presente una league con rating e promozione condizionata a una soglia di
vittorie. Il pool limita i cicli e la dimenticanza, ma la valutazione contro snapshot
e bot di riferimento resta necessaria.

## 14. Riprendere il training

Per continuare la stessa sessione usa la stessa directory e ripeti anche le opzioni del
pool:

```bash
python/.venv/bin/python python/train_generic.py \
  --algorithm dqn \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/pong/pong_scenario.tscn \
  --num-envs 4 \
  --num-episodes 4000 \
  --max-steps-per-episode 1200 \
  --batch-size 128 \
  --replay-warmup 8000 \
  --epsilon-min 0.05 \
  --epsilon-decay 0.997 \
  --target-update-every 20 \
  --checkpoint-dir checkpoints/pong_shared_dqn_v1 \
  --weights-path pong_shared_dqn_v1.weights.h5 \
  --multi-agent \
  --collector-mode async \
  --opponent-pool \
  --opponent-snapshot-every 100 \
  --opponent-pool-size 10 \
  --opponent-current-probability 0.2 \
  --opponent-sampling uniform \
  --resume \
  --headless
```

`--num-episodes 4000` indica l'episodio finale complessivo, non altri 4000 episodi. Il
resume ripristina rete, target network, optimizer, epsilon e replay buffer. Il pool
viene riaperto dal suo manifest. Caricare soltanto `--initial-weights-path` e' invece un
nuovo training con optimizer, epsilon e replay nuovi.

Se riprendi deliberatamente un checkpoint piu' vecchio della prima snapshot rimasta
nel pool, il log mostra `current:no_eligible_snapshot`: il trainer evita di usare una
policy proveniente dal futuro e torna temporaneamente a current-vs-current.

## 15. Leggere i log

Nel blocco `mode` troverai uno di questi valori:

| Valore | Significato |
|---|---|
| `opponent=snapshot:500` | il team avversario usa i pesi salvati all'episodio 500 |
| `opponent=current` | entrambi i team usano la policy corrente |
| `opponent=current:no_eligible_snapshot` | non esiste ancora una snapshot abbastanza vecchia |
| `opponent=disabled` | il pool non e' attivo |

`epsilon` riguarda soltanto la policy learner; la snapshot DQN usa sempre l'azione
greedy. `replay` conta esclusivamente le transizioni valide per allenare la policy
corrente. Le reward mostrate restano separate per env e agente, comprese quelle
dell'avversario, così puoi verificare che punto vinto e perso siano opposti.

Il trainer DQN generico stampa anche `success_rate` e `seen_rate`, metriche pensate per
scenari con eventi target. Pong non usa quegli eventi e tali valori possono restare a
zero: per Pong osserva reward, durata degli scambi e conteggio dei punti.

## 16. Eseguire il modello allenato

Per provare la policy corrente usando l'ultimo checkpoint:

```bash
python/.venv/bin/python python/run_generic_policy.py \
  --algorithm dqn \
  --load-from checkpoint \
  --checkpoint-dir checkpoints/pong_shared_dqn_v1 \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/pong/pong_scenario.tscn \
  --multi-agent \
  --episodes 20 \
  --max-steps 1200 \
  --epsilon 0.0 \
  --no-headless
```

Con la finestra visibile il runner usa automaticamente la modalita' realtime. Per
provare la policy selezionata dalle valutazioni automatiche sostituisci la directory con
`checkpoints/pong_shared_dqn_v1/best`. I resume devono continuare a usare la directory
principale, che conserva replay, optimizer e opponent pool.

Questo comando usa la stessa policy corrente per entrambi i Paddle. Per osservare una
singola snapshot storica su entrambi i lati puoi usare, per esempio:

```bash
python/.venv/bin/python python/run_generic_policy.py \
  --algorithm dqn \
  --load-from weights \
  --weights-path checkpoints/pong_shared_dqn_v1/opponents/policy-00002000.weights.h5 \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/pong/pong_scenario.tscn \
  --multi-agent \
  --episodes 20 \
  --epsilon 0.0 \
  --no-headless
```

`run_generic_policy.py` non mette ancora due file di pesi diversi nello stesso match:
la scelta current-vs-snapshot e' disponibile durante il training, mentre una vera suite
di torneo tra policy resta un'estensione separata.

## 17. Valutazione

Non valutare una policy Pong soltanto contro se stessa: due copie con lo stesso difetto
possono sembrare equilibrate. Misura separatamente:

- vittorie contro un Paddle che segue direttamente la Y della pallina;
- vittorie contro checkpoint precedenti;
- durata media degli scambi;
- errori su servizi verso sinistra e verso destra;
- prestazioni con velocita' non viste durante il training.

Una policy utile deve essere robusta, non soltanto ottenere circa il 50% contro una
copia identica di se stessa.

Usa seed di valutazione distinti da quelli del training e prova entrambi i lati. Una
valutazione utile puo' essere organizzata così:

| Avversario | Cosa misura |
|---|---|
| policy corrente | simmetria e stabilita' visiva, non forza assoluta |
| bot che segue la Y della pallina | capacita' minima di costruire uno scambio |
| snapshot recente | progresso locale |
| snapshot vecchie | dimenticanza di strategie precedenti |
| velocita' e angoli non visti | generalizzazione |

## 18. Errori comuni

- `missing team_id`: uno dei due corpi non espone `get_team_id()` o `team_id`.
- `requires exactly two teams`: i due Paddle hanno lo stesso ID oppure sono presenti
  piu' di due squadre.
- `model metadata does not match`: observation o action space sono cambiati rispetto
  al manifest; avvia una nuova famiglia di checkpoint/pool.
- Replay che cresce piu' lentamente: e' previsto nei match contro snapshot, perché le
  transizioni dell'avversario vengono escluse.
- Reward media vicina a zero: in un gioco zero-sum puo' essere normale anche quando la
  qualita' migliora; misura punti, rally e risultati contro riferimenti fissi.
- Circa 50% contro se stessa: non dimostra che la policy sappia giocare, indica solo
  che i due lati sono equivalenti.
