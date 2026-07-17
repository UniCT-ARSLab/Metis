# Tutorial: guida autonoma con e senza conoscenza del percorso

Questo tutorial costruisce due esperimenti di guida autonoma usando lo stesso veicolo,
la stessa fisica e le stesse azioni continue:

1. **Path-aware**: il veicolo conosce il Path3D, la propria progressione, l'offset
   laterale e la direzione dei prossimi tratti;
2. **Sensor-only**: il veicolo non riceve Path3D, coordinate o progresso e guida usando
   soltanto velocita', input precedenti e RayCast.

Il confronto e' utile soltanto se tutto il resto rimane uguale. Usa la stessa pista,
le stesse reward e gli stessi seed per capire quanto la conoscenza privilegiata del
percorso aiuti davvero.

## 1. Una distinzione fondamentale

Un Path3D puo' essere usato in due modi indipendenti:

### Path3D usato dal trainer

Il `ProgressProvider` misura quanto il veicolo avanza. Serve per:

- reward di progresso;
- penalita' di arretramento;
- stall e pace;
- spawn casuali lungo la pista;
- metriche e curriculum.

Questo non significa che il modello conosca il percorso. Il dato rimane nel sistema di
training e non entra nel vettore delle observation.

### Path3D usato dalla policy

Un `Path3DNavigationObservationSource` aggiunge al vettore:

- percentuale di percorso;
- offset laterale;
- heading rispetto alla tangente;
- direzioni locali di punti futuri.

In questo caso il modello possiede davvero informazioni sul percorso e in produzione
deve continuare ad avere una mappa e una stima della propria posizione.

## 2. Cosa significa "guidare su un percorso qualsiasi"

La variante sensor-only impara **lane/track following locale**: resta nello spazio
percorribile, evita muri e segue una strada non ramificata.

Non puo' scegliere magicamente la strada giusta in una rete con incroci. Se a una
biforcazione entrambe le direzioni sono libere e non esiste un obiettivo osservabile,
le due situazioni sono indistinguibili. Nessun algoritmo puo' dedurre una destinazione
che non compare nelle observation.

Per reti stradali con destinazione servirebbe almeno uno tra:

- comando di navigazione locale, per esempio sinistra/dritto/destra;
- direzione relativa del target;
- mappa e localizzazione;
- memoria ricorrente e indizi osservabili precedenti.

Questo tutorial usa piste continue senza biforcazioni, così "seguire la strada" e'
un compito ben definito.

## 3. Action space comune

Entrambi gli agenti hanno due azioni continue:

```gdscript
{
	"move_input": {
		"size": 1,
		"action_type": "continuous",
		"low": -1.0,
		"high": 1.0
	},
	"rotation_input": {
		"size": 1,
		"action_type": "continuous",
		"low": -1.0,
		"high": 1.0
	}
}
```

Nel primo esempio `move_input < 0` agisce da freno. Puoi introdurre la retromarcia in
seguito, ma iniziare senza evita che la policy scopra strategie basate su continui
cambi avanti/indietro.

## 4. Observation comuni

La scena base fornisce:

| Observation | Dimensione | Origine |
|---|---:|---|
| `forward_speed` | 1 | velocita' firmata |
| `speed` | 1 | modulo della velocita' |
| `forward_clearance` | 1 | spazio libero davanti |
| `move_input` | 1 | input applicato precedentemente |
| `rotation_input` | 1 | input applicato precedentemente |
| `ray_*` | N | distanze normalizzate dei RayCast |

Con nove RayCast, come nell'attuale Cars, otteniamo 14 valori.

Non inserire la velocita' non normalizzata. I valori molto piu' grandi delle altre
observation rendono l'ottimizzazione inutilmente difficile.

## 5. Creare AutonomousVehicle.tscn

Puoi partire dalla scena Cars ripulita oppure creare:

```text
AutonomousVehicle (CharacterBody3D) [autonomous_vehicle.gd]
├── Mesh
├── CollisionShape3D
├── ViewSensors (Node3D)
│   ├── Front (RayCast3D)
│   ├── Front15Left (RayCast3D)
│   ├── Front30Left (RayCast3D)
│   ├── Left (RayCast3D)
│   ├── Front15Right (RayCast3D)
│   ├── Front30Right (RayCast3D)
│   ├── Right (RayCast3D)
│   ├── RearLeft (RayCast3D)
│   └── RearRight (RayCast3D)
└── Agent [Agent.gd]
    ├── ActionSpace [ActionSpace.gd]
    │   ├── MoveInput [ContinuousAction.gd]
    │   └── RotationInput [ContinuousAction.gd]
    ├── ObservationSystem [ObservationSystem.gd]
    │   ├── BodySpeed [BodySpeedObservationSource.gd]
    │   ├── ForwardClearance [RaycastClearanceObservationSource.gd]
    │   ├── MoveInput [MethodObservationSource.gd]
    │   ├── RotationInput [MethodObservationSource.gd]
    │   └── Raycasts [RaycastObservationSource.gd]
    └── RewardSystem [RewardSystem.gd]
        ├── ForwardVelocity [ForwardVelocityReward.gd]
        ├── TimePenalty [StepPenaltyReward.gd]
        ├── SmoothnessPenalty [ActionSmoothnessPenaltyReward.gd]
        └── CollisionPenalty [EventReward.gd]
```

## 6. Configurare ActionSpace

`MoveInput`:

```text
action_name = "move_input"
size = 1
low = -1.0
high = 1.0
use_custom_exploration_bounds = true
exploration_low = 0.35
exploration_high = 1.0
```

`RotationInput`:

```text
action_name = "rotation_input"
size = 1
low = -1.0
high = 1.0
use_custom_exploration_bounds = true
exploration_low = -0.35
exploration_high = 0.35
```

I limiti di esplorazione iniziale incoraggiano movimento in avanti e sterzate moderate.
Non sono una regola hard della policy allenata.

## 7. Configurare le observation comuni

`BodySpeed`:

```text
forward_speed_name = "forward_speed"
absolute_speed_name = "speed"
speed_scale = 20.0
```

`ForwardClearance`:

```text
observation_name = "forward_clearance"
raycast_source_path = "../Raycasts"
forward_dot_threshold = 0.86
```

I due `MethodObservationSource`:

```text
MoveInput:
  observation_name = "move_input"
  method_name = "get_control_input"
  bind_string_arg = "move_input"

RotationInput:
  observation_name = "rotation_input"
  method_name = "get_control_input"
  bind_string_arg = "rotation_input"
```

`Raycasts`:

```text
root_path = "../../.."
observation_prefix = "ray"
```

La distanza normalizzata vale `1` a raggio libero e si avvicina a `0` quando una
collisione e' vicina. La lunghezza fisica dei RayCast determina quanto in anticipo il
veicolo puo' reagire.

## 8. Script minimo autonomous_vehicle.gd

```gdscript
extends CharacterBody3D
class_name AutonomousVehicle

@export var acceleration := 35.0
@export var brake_strength := 8.0
@export var friction := 3.0
@export var max_speed := 20.0
@export var steering_speed_degrees := 100.0
@export var min_speed_for_steering := 0.5
@export var crash_floor_normal_threshold := 0.5
@export var manual_control := false

@onready var agent: Agent = $Agent

var _move_input := 0.0
var _rotation_input := 0.0
var _crashed := false
var _training_active := true


func _physics_process(delta:float) -> void:
	if not _training_active or _crashed:
		return
	if manual_control:
		_move_input = Input.get_axis("move_back", "move_forward")
		_rotation_input = Input.get_axis("turn_left", "turn_right")

	var forward := global_transform.basis.z.normalized()
	var forward_speed := absf(get_signed_forward_speed())
	var steering_authority := clampf(
		forward_speed / maxf(min_speed_for_steering * 4.0, 0.001),
		0.0,
		1.0
	)
	rotate_y(-_rotation_input * steering_authority * deg_to_rad(steering_speed_degrees) * delta)

	if _move_input >= 0.0:
		velocity += forward * _move_input * acceleration * delta
	else:
		velocity = velocity.lerp(Vector3.ZERO, minf(-_move_input * brake_strength * delta, 1.0))
	velocity = velocity.lerp(Vector3.ZERO, minf(friction * delta, 1.0))
	velocity = velocity.limit_length(max_speed)
	if not is_on_floor():
		velocity += get_gravity() * delta
	move_and_slide()
	_update_crash_state()


func apply_action(action:Variant) -> Array:
	if typeof(action) == TYPE_STRING or typeof(action) == TYPE_STRING_NAME:
		if str(action) == "manual":
			return apply_manual_action()
	var values := agent.decode_continuous_action(action)
	_move_input = clampf(float(values[0]), -1.0, 1.0)
	_rotation_input = clampf(float(values[1]), -1.0, 1.0)
	return [_move_input, _rotation_input]


func apply_manual_action() -> Array:
	return apply_action([
		Input.get_axis("move_back", "move_forward"),
		Input.get_axis("turn_left", "turn_right")
	])


func get_control_input(input_name:String) -> float:
	if input_name == "move_input":
		return _move_input
	if input_name == "rotation_input":
		return _rotation_input
	return 0.0


func get_signed_forward_speed() -> float:
	var forward := global_transform.basis.z.normalized()
	return Vector3(velocity.x, 0.0, velocity.z).dot(forward)


func is_crashed() -> bool:
	return _crashed


func is_terminal() -> bool:
	return _crashed


func reset_all(original_transform:Variant, reset_rewards := true) -> void:
	set_training_active(true)
	if typeof(original_transform) == TYPE_TRANSFORM3D:
		transform = original_transform
	_crashed = false
	_move_input = 0.0
	_rotation_input = 0.0
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
		velocity = Vector3.ZERO


func _update_crash_state() -> void:
	if _crashed:
		return
	for idx in range(get_slide_collision_count()):
		var collision := get_slide_collision(idx)
		if collision != null and collision.get_normal().dot(Vector3.UP) < crash_floor_normal_threshold:
			_crashed = true
			agent.add_reward_event("collision", 1.0)
			return
```

La fisica puo' essere sostituita con `VehicleBody3D` o un modello piu' realistico. Il
contratto RL non cambia finche' action e observation mantengono stesso ordine e scala.

## 9. Reward locali comuni

Valori iniziali moderati:

```text
ForwardVelocity:
  reward = 0.02
  reference_speed = 15.0
  term_name = "forward_velocity"

TimePenalty:
  penalty = -0.001
  term_name = "time"

SmoothnessPenalty:
  penalty_scale = -0.01
  term_name = "action_smoothness"

CollisionPenalty:
  event_name = "collision"
  scale = -20.0
  term_name = "collision"
```

La reward di velocita' deve essere piu' piccola della reward di progresso. Altrimenti
il veicolo puo' imparare a muoversi velocemente in una direzione sbagliata.

## 10. Creare lo scenario di guida

```text
DrivingScenario (Node3D)
├── BridgeServer [bridge_server.gd]
├── ScenarioController [ScenarioController.gd]
│   ├── ProgressProvider [Path3DProgressProvider.gd]
│   ├── ScenarioEventSystem [ScenarioEventSystem.gd]
│   │   └── FinishReached [AreaReachedEventSource.gd]
│   └── ScenarioRewardSystem [ScenarioRewardSystem.gd]
│       ├── ProgressReward [ProgressDeltaScenarioReward.gd]
│       ├── FinishReward [EventScenarioReward.gd]
│       ├── StallPenalty [ProgressStallScenarioReward.gd]
│       └── PacePenalty [ProgressPaceScenarioReward.gd]
├── Environment
│   ├── Road
│   ├── Walls
│   └── FinishArea (Area3D)
├── NavigationPath (Path3D, gruppo "navigation_path")
└── Agents
    └── Vehicle
```

Nel `ScenarioController`:

```text
controlled_agents = ["../Agents/Vehicle"]
randomize_reset = true
reset_use_progress_provider = true
reset_progress_min = 0.0
reset_progress_max = 0.03
reset_lateral_jitter = 0.20
reset_yaw_jitter_degrees = 5.0
reset_align_to_progress = true
max_steps = 1000
```

Nel `ProgressProvider`, assegna `NavigationPath`. Nel `FinishReached`, assegna
`FinishArea`, usa `event_name="finish_reached"` e
`terminal_reason="finish_reached"`.

## 11. Reward di scenario comuni

Un punto di partenza meno aggressivo dell'esempio Cars sperimentale:

```text
ProgressReward:
  progress_reward_scale = 100.0
  backward_penalty_scale = 100.0

FinishReward:
  event_name = "finish_reached"
  reward = 50.0
  only_once = true

StallPenalty:
  terminate_on_stalled_progress = true
  stalled_progress_window_steps = 150
  stalled_progress_min_delta = 0.005
  stalled_progress_penalty = -8.0

PacePenalty:
  progress_pace_bucket_size = 0.01
  progress_pace_steps_per_bucket = 130
  progress_pace_penalty = -2.0
  terminate_on_progress_pace_failure = false
```

All'inizio usa una sola terminazione per mancato progresso. Attivare insieme stall e
pace molto severi puo' chiudere episodi utili prima che la policy impari a sterzare.

## 12. Esempio A: veicolo path-aware

Aggiungi sotto `Agent/ObservationSystem`:

```text
PathNavigation [Path3DNavigationObservationSource.gd]
```

Configuralo:

```text
path_group = "navigation_path"
progress_observation_name = "path_progress"
lateral_observation_name = "path_lateral_offset"
heading_observation_name = "path_heading"
lookahead_observation_prefix = "path_lookahead"
lookahead_distances = [5.0, 12.0, 25.0]
lateral_scale = 6.0
loop_path = false
```

Il nodo aggiunge dieci valori:

```text
path_progress                 1
path_lateral_offset           1
path_heading                  2
path_lookahead_0              2
path_lookahead_1              2
path_lookahead_2              2
```

Con nove RayCast, `obs_dim` passa da 14 a 24.

`path_heading` e look-ahead sono espressi nel riferimento locale del veicolo. Questo
e' preferibile a fornire coordinate mondiali: la stessa curva a una posizione o
rotazione diversa produce indicazioni equivalenti.

### Vantaggi

- apprendimento piu' rapido;
- curve anticipate prima che entrino nei RayCast;
- controllo preciso dell'offset laterale;
- gestione migliore di piste larghe o senza muri vicini.

### Svantaggi

- richiede mappa e localizzazione anche in produzione;
- `path_progress` puo' favorire memorizzazione della pista;
- errori di localizzazione producono observation sbagliate;
- non rappresenta ostacoli temporanei, quindi i RayCast restano necessari.

Per una variante route-aware ma meno legata alla pista, lascia heading e look-ahead ma
imposta `progress_observation_name` a stringa vuota. Le direzioni locali generalizzano
meglio della percentuale assoluta.

## 13. Esempio B: veicolo sensor-only

Non aggiungere `PathNavigation`. Il veicolo mantiene soltanto le 14 observation comuni.

Il `ProgressProvider` resta nello scenario, ma viene letto soltanto da controller,
reward e curriculum. Il modello TensorFlow non riceve mai `progress`, offset o
look-ahead.

### Vantaggi

- puo' essere trasferito su piste geometricamente diverse;
- non richiede GPS, mappa o localizzazione;
- e' piu' vicino a un piccolo veicolo reale con sensori di distanza;
- impara una reazione locale tra spazi liberi, velocita' e sterzata.

### Svantaggi

- apprendimento piu' lento;
- curve oltre la portata dei sensori non sono prevedibili;
- situazioni localmente identiche producono la stessa decisione;
- incroci e obiettivi remoti richiedono informazioni aggiuntive;
- una rete feed-forward non conserva memoria degli step precedenti.

Non chiamarlo "completamente generico" dopo averlo allenato su una sola pista. Anche
senza coordinate, la policy puo' riconoscere combinazioni di RayCast specifiche. La
generalizzazione va misurata su layout non visti.

## 14. Due scene o un nodo abilitabile

La soluzione piu' chiara e' creare:

```text
vehicle_sensor_only.tscn       obs_dim=14
vehicle_path_aware.tscn        obs_dim=24
```

La seconda puo' ereditare dalla prima e aggiungere soltanto `PathNavigation`.

Non attivare o disattivare il nodo durante lo stesso training: cambiare `obs_dim`
invalida il modello e il replay buffer. Usa directory e pesi separati.

## 15. Curriculum uguale per entrambi

Per confrontarli usa lo stesso curriculum:

1. spawn vicino alla partenza e yaw quasi corretto;
2. spawn su una porzione crescente della pista;
3. maggiore jitter laterale;
4. piu' layout con curve differenti;
5. valutazione sempre dalla partenza completa.

I trainer comunicano `training_episode` allo scenario. Un manager puo' quindi scegliere
layout e difficolta' senza cambiare Python.

Comando SAC path-aware:

```bash
python/.venv/bin/python python/train_generic.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/driving/path_aware_scenario.tscn \
  --num-envs 4 \
  --num-episodes 3000 \
  --max-steps-per-episode 1000 \
  --batch-size 128 \
  --replay-warmup 12000 \
  --random-exploration-episodes 40 \
  --action-smoothing 0.25 \
  --reset-progress-curriculum \
  --reset-progress-start-max 0.02 \
  --reset-progress-end-max 0.60 \
  --reset-progress-ramp-episodes 1600 \
  --checkpoint-dir checkpoints/driving_path_aware_sac_v1 \
  --actor-weights-path driving_path_aware_actor_v1.weights.h5 \
  --critic1-weights-path driving_path_aware_critic1_v1.weights.h5 \
  --critic2-weights-path driving_path_aware_critic2_v1.weights.h5 \
  --headless
```

Comando sensor-only:

```bash
python/.venv/bin/python python/train_generic.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/driving/sensor_only_scenario.tscn \
  --num-envs 4 \
  --num-episodes 3000 \
  --max-steps-per-episode 1000 \
  --batch-size 128 \
  --replay-warmup 12000 \
  --random-exploration-episodes 40 \
  --action-smoothing 0.25 \
  --reset-progress-curriculum \
  --reset-progress-start-max 0.02 \
  --reset-progress-end-max 0.60 \
  --reset-progress-ramp-episodes 1600 \
  --checkpoint-dir checkpoints/driving_sensor_only_sac_v1 \
  --actor-weights-path driving_sensor_only_actor_v1.weights.h5 \
  --critic1-weights-path driving_sensor_only_critic1_v1.weights.h5 \
  --critic2-weights-path driving_sensor_only_critic2_v1.weights.h5 \
  --headless
```

Non usare `--multi-agent` finche' non hai validato un singolo veicolo. In seguito puoi
replicare agenti che non collidono tra loro per raccogliere piu' esperienza nello
stesso env.

Il training e' headless e asincrono per default. Per osservare un solo environment usa
`--no-headless --render-env-count 1`. Parti con `--physics-frames-per-step 1`: valori
maggiori riducono inferenze e traffico socket, ma mantengono acceleratore e sterzo per
piu' tick fisici. Di conseguenza cambiano reattivita', durata delle finestre di stall e
tempo fisico rappresentato da `--max-steps-per-episode`.

## 16. Allenare su piu' piste

Per generalizzare davvero, usa piu' curve e geometrie. Una soluzione pulita mantiene
un solo nodo `NavigationPath` e cambia la sua `Curve3D` durante il reset, insieme al
layout stradale corrispondente.

Esempio di manager:

```gdscript
extends Node3D

@export var navigation_path: Path3D
@export var finish_area: Area3D
@export var layouts: Array[Node3D] = []
@export var curves: Array[Curve3D] = []

@onready var controller := $ScenarioController

var _training_episode := 0


func _ready() -> void:
	controller.scenario_configured.connect(_on_configured)
	controller.episode_reset_started.connect(_on_reset_started)


func _on_configured(config:Dictionary) -> void:
	_training_episode = int(config.get("training_episode", _training_episode))


func _on_reset_started(seed:int) -> void:
	if layouts.is_empty() or curves.size() != layouts.size():
		return
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	var available := 1
	if _training_episode >= 600:
		available = mini(2, layouts.size())
	if _training_episode >= 1400:
		available = layouts.size()
	var selected := rng.randi_range(0, available - 1)
	for idx in range(layouts.size()):
		layouts[idx].visible = idx == selected
		layouts[idx].process_mode = (
			Node.PROCESS_MODE_INHERIT if idx == selected
			else Node.PROCESS_MODE_DISABLED
		)
	navigation_path.curve = curves[selected]
	var end_local := navigation_path.curve.sample_baked(navigation_path.curve.get_baked_length())
	finish_area.global_position = navigation_path.to_global(end_local)
```

Il manager cambia la pista prima che `ProgressProvider` e agenti vengano resettati.
Entrambe le varianti ricevono la stessa pista; soltanto la path-aware ne osserva la
geometria.

## 17. Validare prima del training

Path-aware:

```bash
python/.venv/bin/python python/random_scenario_rollout.py \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/driving/path_aware_scenario.tscn \
  --steps 20 \
  --print-reward-terms \
  --no-headless
```

Controlla `obs_shape=(24,)` con nove RayCast.

Sensor-only:

```bash
python/.venv/bin/python python/random_scenario_rollout.py \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/driving/sensor_only_scenario.tscn \
  --steps 20 \
  --print-reward-terms \
  --no-headless
```

Controlla `obs_shape=(14,)` e verifica che nessuna chiave inizi con `path_`.

## 18. Confronto corretto

Valuta entrambi senza esplorazione su quattro categorie:

| Test | Path-aware | Sensor-only |
|---|---|---|
| pista di training | velocita' di completamento | velocita' di completamento |
| stessa pista, spawn nuovi | robustezza localizzazione | robustezza sensori |
| pista mai vista | trasferimento look-ahead | generalizzazione locale |
| ostacolo temporaneo | fusione path/sensori | reazione sensori |

Registra:

- finish rate;
- collision rate;
- tempo medio;
- velocita' media;
- progresso massimo e medio;
- steering e action delta;
- risultato per ogni layout, non soltanto la media globale.

Il path-aware dovrebbe imparare prima. Il sensor-only dovrebbe degradare meno quando
cambia la geometria, a condizione di essere stato allenato su sufficiente varieta'.

## 19. Eseguire i modelli

Path-aware:

```bash
python/.venv/bin/python python/run_generic_policy.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/driving/path_aware_scenario.tscn \
  --actor-weights-path driving_path_aware_actor_v1.weights.h5 \
  --episodes 20 \
  --no-headless
```

Sensor-only:

```bash
python/.venv/bin/python python/run_generic_policy.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/driving/sensor_only_scenario.tscn \
  --actor-weights-path driving_sensor_only_actor_v1.weights.h5 \
  --episodes 20 \
  --no-headless
```

I pesi non sono intercambiabili: i due modelli hanno input dimension differenti.

Per eseguire la policy migliore invece dei pesi finali, sostituisci
`--actor-weights-path` con `--load-from checkpoint` e
`--checkpoint-dir checkpoints/driving_path_aware_sac_v1/best`, oppure usa la
corrispondente directory `driving_sensor_only_sac_v1/best`. I resume usano sempre la
directory principale con il replay buffer cronologico.

## 20. Scelta pratica

Scegli path-aware quando:

- il percorso e' noto;
- disponi di localizzazione affidabile;
- vuoi prestazioni e apprendimento rapidi;
- il veicolo deve seguire una route specifica.

Scegli sensor-only quando:

- la geometria cambia;
- non hai una mappa;
- vuoi trasferire la policy su un veicolo reale con sensori locali;
- il compito e' seguire una pista non ramificata, non raggiungere una destinazione
  arbitraria.

Una terza soluzione spesso migliore combina sensori e un comando di navigazione
locale. Il planner conosce la route e invia soltanto `left`, `straight` o `right`; la
policy resta responsabile di velocita', sterzata ed evitamento ostacoli. In questo modo
separi pianificazione globale e controllo reattivo senza consegnare l'intera mappa alla
rete neurale.
