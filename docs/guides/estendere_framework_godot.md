# Estendere Metis in Godot

Questa guida mostra come aggiungere componenti riusabili e configurabili da Inspector.
Gli script del framework stanno in `godot/scripts/agent/`; il codice specifico di un
gioco resta in `godot/scenarios` o nel corpo dell'agente.

## Principio di scelta

Prima di creare un nodo, individua il livello corretto:

| Esigenza | Estensione |
| --- | --- |
| Leggere stato locale o sensori | `ObservationSource` |
| Valutare comportamento locale | `RewardComponent` |
| Valutare stato globale/percorso | `ScenarioRewardComponent` |
| Pubblicare collisioni o fatti | `ScenarioEventSource` |
| Misurare avanzamento | `ProgressProvider` |
| Dichiarare una nuova action | figlio di `ActionSpace` |

Una collisione non dovrebbe essere contemporaneamente rilevata dentro tre reward. Crea
un evento, poi fai dipendere da quell'evento reward e terminalita'.

## Nuova observation source

Prima di scrivere codice, verifica se basta uno dei due adattatori generici:

```text
MethodObservationSource:
  source_path = ""              # Vuoto: corpo agente
  method_name = Select...        # Solo metodi compatibili

PropertyObservationSource:
  observation_name = "health"
  source_path = ""              # Oppure un figlio relativo al corpo
  property_path = Select...      # Per esempio health o velocity:x
  normalize_numeric = true
  input_min = 0.0
  input_max = 100.0
  output_min = 0.0
  output_max = 1.0
```

Il campo accanto a `Select...` resta editabile, quindi puoi indicare manualmente una
sottoproprieta' che il menu delle property dirette non mostra. Usa una source nuova
soltanto quando la observation richiede calcoli, query fisiche o stato proprio.

Esempio: distanza normalizzata da un target.

```gdscript
extends "res://scripts/agent/observations/ObservationSource.gd"
class_name TargetDistanceObservationSource

@export var observation_name := "target_distance"
@export var target: Node3D
@export var max_distance := 20.0

var _body: Node3D


func register_observations(agent:Agent, body:Node) -> void:
	_body = body as Node3D
	agent.add_observation(observation_name, Callable(self, "_read_distance"))


func _read_distance() -> float:
	if _body == null or target == null:
		return 1.0
	return clampf(_body.global_position.distance_to(target.global_position) / maxf(max_distance, 0.000001), 0.0, 1.0)
```

Salvalo in `godot/scripts/agent/observations/`, aggiungilo come figlio di
`ObservationSystem` e imposta i campi da Inspector.

Regole:

- registra sempre lo stesso numero di valori;
- restituisci valori finiti;
- usa `reset_source()` per cache episodiche;
- usa `refresh_source()` per forzare RayCast o query dopo il reset;
- non usare una posizione globale se vuoi una policy indipendente dalla mappa.

## Nuova reward locale

Esempio: premiare la velocita' soltanto quando un segnale e' attivo.

```gdscript
extends "res://scripts/agent/reward_components/RewardComponent.gd"
class_name GatedSpeedReward

@export var gate_observation := "path_clear"
@export var speed_observation := "forward_speed"
@export var gate_threshold := 0.5
@export var reward_scale := 0.02


func compute_reward(context:Dictionary) -> float:
	var observations: Dictionary = context.get("observations", {})
	if float(observations.get(gate_observation, 0.0)) < gate_threshold:
		return 0.0
	var speed := maxf(float(observations.get(speed_observation, 0.0)), 0.0)
	return speed * reward_scale * weight
```

Come figlio di `RewardSystem`, il componente riceve `agent`, `body`, `observations`,
`events` e i dati aggiunti dal controller. Usa `term_name` per ottenere log leggibili.

Se mantieni memoria tra step, implementa:

```gdscript
func reset_reward(_context:Dictionary = {}) -> void:
	# Azzera contatori e valori del vecchio episodio.
	pass
```

Il metodo `compute_reward()` deve restituire il valore gia' moltiplicato per `weight`.

## Nuova reward di scenario

Usala quando il valore dipende da progress, squadra, goal o oggetti esterni al corpo.

```gdscript
extends "res://scripts/agent/scenario_reward_components/ScenarioRewardComponent.gd"
class_name PossessionScenarioReward

@export var event_name := "has_possession"
@export var reward_per_step := 0.005


func compute_reward(_agent:Node, context:Dictionary = {}) -> float:
	return reward_per_step * weight if bool(context.get(event_name, false)) else 0.0
```

Gli eventi di scenario vengono uniti nel context usando direttamente il loro nome.
Per stato per agente usa dictionary indicizzati da `agent_id` e azzerali sia in
`reset_rewards()` sia in `reset_agent(agent_id, context)`.

Puoi inoltre implementare:

- `get_terminal_reason(agent_id)` per terminare il canale;
- `is_agent_stalled(agent_id)` per diagnostica e stop;
- `get_agent_terms(agent_id)` per metriche aggiuntive;
- `is_episode_end_only()` se il valore deve essere calcolato solo alla fine.

## Nuova event source

Esempio: evento impostato quando la salute scende a zero.

```gdscript
extends "res://scripts/agent/events/ScenarioEventSource.gd"
class_name HealthDepletedEventSource

@export var health_property := "health"

var _values := {}


func reset_events() -> void:
	_values.clear()


func reset_agent(agent:Node, _context:Dictionary = {}) -> void:
	_values[str(agent.name)] = false


func update_agent(agent:Node, _context:Dictionary = {}) -> void:
	_values[str(agent.name)] = float(agent.get(health_property)) <= 0.0


func get_agent_events(agent_id:String) -> Dictionary:
	return {event_name: bool(_values.get(agent_id, false))}
```

Imposta `event_name` e, se deve terminare l'episodio, `terminal_reason` da Inspector.
Per agent id personalizzati usa la stessa funzione di identificazione del controller,
non assumere il nome del nodo in un progetto che lo sovrascrive.

## Nuovo progress provider

Il progress puo' essere qualunque misura monotona o quasi monotona:

```gdscript
extends "res://scripts/agent/progress/ProgressProvider.gd"
class_name BrickProgressProvider

@export var bricks_container: Node

var _initial_count := 0


func reset_provider() -> void:
	_initial_count = bricks_container.get_child_count() if bricks_container != null else 0


func measure_progress(_agent:Node, _context:Dictionary = {}) -> float:
	if bricks_container == null or _initial_count <= 0:
		return 0.0
	var remaining := bricks_container.get_child_count()
	return 1.0 - float(remaining) / float(_initial_count)
```

Il provider va assegnato a `ScenarioController.progress_provider_path`. Le reward di
progress e il curriculum lo useranno senza conoscere il tipo concreto.

Non aggiungere automaticamente il progress alle observation: reward/curriculum e input
della policy sono responsabilita' separate.

## Nuovo componente di action space

Per la maggior parte dei casi bastano `ContinuousAction` e `DiscreteActionSet`. Un nuovo
figlio deve almeno esporre:

```gdscript
func get_action_spec() -> Dictionary:
	return {
		"name": "ability",
		"size": 2,
		"action_type": "discrete",
		"names": ["none", "activate"]
	}
```

Per l'esecuzione automatica discreta implementa anche
`execute_action(action, default_target)`. Se crei un tipo diverso da `discrete` o
`continuous`, devi estendere parser Gymnasium, algoritmi e runner: non e' una modifica
soltanto Godot.

## Collegamento da Inspector

1. Crea lo script nella cartella della sua famiglia.
2. Usa `class_name` per renderlo disponibile nel menu Add Node.
3. Aggiungi il nodo sotto il sistema aggregatore corretto.
4. Imposta `term_name`, nomi observation/evento e NodePath.
5. Verifica la spec con il rollout casuale.

```bash
python/.venv/bin/python python/tools/random_rollout.py \
  --godot-bin /percorso/a/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/mio_scenario.tscn \
  --steps 200
```

## Checklist di qualita'

- Il componente funziona con piu' agenti senza condividere stato accidentale.
- Ogni stato episodico viene azzerato al reset.
- Il seed controlla tutte le randomizzazioni.
- Headless non esegue camera, UI o debug inutili.
- Observation e action mantengono ordine e dimensione.
- Reward terms hanno nomi univoci e scala confrontabile.
- Gli eventi one-shot non vengono emessi a ogni step.
- Un agente terminato non continua a modificare la scena.
- Il componente generico non contiene riferimenti a Cars, Tanks, Paddle o nomi di scena.

Quando una logica non e' realmente riusabile, lasciala nello scenario: generalizzare un
caso singolo rende l'Inspector piu' difficile senza migliorare il framework.
