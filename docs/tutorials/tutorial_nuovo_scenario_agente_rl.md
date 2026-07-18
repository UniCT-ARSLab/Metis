# Creare un nuovo agente e un nuovo scenario con Metis

Questo tutorial mostra il percorso normale per aggiungere un ambiente di reinforcement
learning a Metis. Non costruiremo un bridge o un trainer dedicato: useremo i nodi del
framework e scriveremo soltanto la fisica e le regole che appartengono al nostro gioco.

Come esempio realizzeremo un piccolo agente 3D che deve raggiungere un target. E' un
compito semplice, ma contiene tutti i pezzi che ritornano negli scenari piu' grandi:

- action space dichiarato in Godot;
- observation componibili;
- reward locale e reward di scenario;
- evento terminale;
- progress indipendente dal modo in cui viene misurato;
- validazione, training, checkpoint ed esecuzione della policy.

## Il modello mentale

Metis divide il lavoro in questo modo:

```text
Godot                                Python
---------------------------------    --------------------------------
fisica e collisioni                  selezione dell'algoritmo
azioni disponibili                   raccolta delle transizioni
costruzione delle observation        learner TensorFlow/Keras
reward ed eventi terminali           replay o rollout
reset e randomizzazione              checkpoint e valutazione
```

Il `BridgeServer` collega i due lati. Il `ScenarioController` standard implementa gia'
`spec`, `configure`, `reset` e `step`: in un nuovo scenario non devi riscrivere questo
protocollo.

## 1. Definisci prima il problema

Prima di aprire l'editor, scrivi quattro cose.

### Obiettivo

Una frase osservabile, per esempio:

> L'agente deve entrare nell'area del target senza impiegare passi inutili.

Evita obiettivi vaghi come "deve muoversi bene". Una condizione precisa rende piu'
semplice progettare reward e terminalita'.

### Azioni

Per il primo prototipo useremo quattro azioni discrete:

```text
0 idle
1 forward
2 forward_left
3 forward_right
```

L'ordine sara' determinato dai nodi figli di `DiscreteActionSet`, non da una costante
scritta in Python.

### Observation

L'agente vedra':

```text
target_local_x       [-1, 1]
target_local_forward [-1, 1]
target_distance      [ 0, 1]
speed                [ 0, 1]
```

Le coordinate sono relative all'agente. In questo modo la policy non deve imparare una
mappa di coordinate assolute e puo' essere riutilizzata dopo aver spostato lo scenario.

### Reward e fine episodio

Useremo:

- una piccola penalita' per decision step;
- una reward proporzionale al progresso verso il target;
- un bonus quando il target viene raggiunto;
- `terminated` quando l'agente entra nell'area;
- `truncated` se supera il limite di step.

Il bonus descrive il risultato. Il progresso aiuta l'esplorazione. La penalita' rende
preferibile una soluzione breve, ma non deve essere tanto forte da rendere conveniente
non provare.

## 2. Prepara file e scena dell'agente

Una struttura ordinata puo' essere:

```text
godot/agents/TargetSeeker/
    target_seeker.gd
    target_seeker.tscn

godot/scenarios/target_seeker/
    target_scenario.gd
    target_scenario.tscn
```

Crea `target_seeker.tscn` con questo albero:

```text
TargetSeeker                 CharacterBody3D
├── MeshInstance3D
├── CollisionShape3D
└── Agent                    script Agent.gd
    ├── ActionSpace          script ActionSpace.gd
    ├── ObservationSystem    script ObservationSystem.gd
    └── RewardSystem         script RewardSystem.gd
```

I nomi predefiniti sono utili: il nodo `Agent` cerca proprio `ActionSpace`,
`ObservationSystem` e `RewardSystem` se non imposti altri `NodePath`.

## 3. Scrivi soltanto il comportamento del corpo

Collega questo script al `CharacterBody3D`:

```gdscript
extends CharacterBody3D
class_name TargetSeeker

@export var move_speed := 5.0
@export var turn_speed := 2.5
@export var target: Node3D
@export var target_distance_scale := 20.0

@onready var agent: Agent = $Agent

var _drive_input := 0.0
var _turn_input := 0.0
var _training_active := true


func _physics_process(delta:float) -> void:
	if not _training_active:
		return

	rotate_y(_turn_input * turn_speed * delta)
	var forward := -global_transform.basis.z
	var vertical_speed := velocity.y
	velocity = forward * (_drive_input * move_speed)
	velocity.y = vertical_speed
	if not is_on_floor():
		velocity += get_gravity() * delta
	move_and_slide()


func apply_action(action:Variant) -> Variant:
	clear_control()
	var action_id := int(action)
	if agent.act(action_id) != OK:
		return 0
	return action_id


func set_control(drive:float, turn:float) -> void:
	_drive_input = clampf(drive, -1.0, 1.0)
	_turn_input = clampf(turn, -1.0, 1.0)


func clear_control() -> void:
	_drive_input = 0.0
	_turn_input = 0.0


func get_target_local_observation() -> Vector2:
	if target == null:
		return Vector2.ZERO
	var local_offset := global_transform.basis.inverse() * (target.global_position - global_position)
	var scale := maxf(target_distance_scale, 0.001)
	return Vector2(
		clampf(local_offset.x / scale, -1.0, 1.0),
		clampf(-local_offset.z / scale, -1.0, 1.0)
	)


func get_target_distance_observation() -> float:
	if target == null:
		return 1.0
	return clampf(global_position.distance_to(target.global_position) / maxf(target_distance_scale, 0.001), 0.0, 1.0)


func get_speed_observation() -> float:
	var horizontal_velocity := Vector2(velocity.x, velocity.z)
	return clampf(horizontal_velocity.length() / maxf(move_speed, 0.001), 0.0, 1.0)


func reset_all(original_transform:Variant, reset_rewards := true) -> void:
	set_training_active(true)
	clear_control()
	velocity = Vector3.ZERO
	if typeof(original_transform) == TYPE_TRANSFORM3D:
		transform = original_transform
	if has_method("reset_physics_interpolation"):
		reset_physics_interpolation()
	agent.reset_observation_sources()
	if reset_rewards:
		agent.refresh_observation_sources()
		agent.reset_reward({"body": self})


func is_terminal() -> bool:
	return false


func set_training_active(enabled:bool) -> void:
	_training_active = enabled
	set_physics_process(enabled)
	if not enabled:
		clear_control()
		velocity = Vector3.ZERO
```

Il corpo implementa le parti concrete richieste dal controller:

- `apply_action()` applica l'azione ricevuta;
- `reset_all()` ripristina fisica e componenti RL;
- `is_terminal()` puo' segnalare una fine locale, come una collisione fatale;
- `set_training_active()` permette di fermare un agente gia' concluso.

Non abbiamo scritto `get_action_space()` o assemblato manualmente il vettore delle
observation: se ne occupa il figlio `Agent` leggendo i propri componenti.

## 4. Dichiara le azioni dall'Inspector

Sotto `Agent/ActionSpace` aggiungi:

```text
Movement                    DiscreteActionSet.gd
├── Idle                    DiscreteAction.gd
├── Forward                 DiscreteAction.gd
├── ForwardLeft             DiscreteAction.gd
└── ForwardRight            DiscreteAction.gd
```

Configura `Movement.action_name = "movement"`. Per ogni `DiscreteAction` lascia vuoto
`target_path`, cosi' il target predefinito sara' il corpo `TargetSeeker`, e imposta:

| Nodo | `action_name` | `method_name` | `arguments` |
|---|---|---|---|
| Idle | `idle` | `set_control` | `[0.0, 0.0]` |
| Forward | `forward` | `set_control` | `[1.0, 0.0]` |
| ForwardLeft | `forward_left` | `set_control` | `[1.0, 1.0]` |
| ForwardRight | `forward_right` | `set_control` | `[1.0, -1.0]` |

`DiscreteActionSet.names` e' il campo legacy: lascialo vuoto quando usi i figli
`DiscreteAction`. Non chiamare anche `agent.add_action()` per le stesse azioni, altrimenti
avresti due fonti di configurazione.

## 5. Componi le observation

Sotto `Agent/ObservationSystem` aggiungi tre `MethodObservationSource`:

| Nodo | `observation_name` | `method_name` | Dimensione |
|---|---|---|---|
| TargetLocal | `target_local` | `get_target_local_observation` | 2 |
| TargetDistance | `target_distance` | `get_target_distance_observation` | 1 |
| Speed | `speed` | `get_speed_observation` | 1 |

Lascia vuoto `source_path`: i metodi si trovano sul corpo padre dell'`Agent`.

Il plugin Metis Inspector aggiunge `Select...` ai campi dei metodi e delle property. Se
un metodo appena scritto non compare, salva lo script e la scena, assicurati che non ci
siano errori di parsing e riseleziona il nodo. Il nome puo' comunque essere inserito a
mano.

Un `Vector2` viene appiattito automaticamente. La spec risultante avra' quindi
`obs_dim=4`; non devi passare `--obs-dim` a Python. L'ordine e' quello dei figli dentro
`ObservationSystem`, quindi non riordinarli quando vuoi riprendere un modello esistente.

Per altri scenari puoi combinare anche:

- `PropertyObservationSource` per leggere una property e normalizzarla;
- `BodySpeedObservationSource` e `BodyKinematicsObservationSource`;
- `RaycastObservationSource` e `RaycastClearanceObservationSource`;
- source per target, team e navigazione `Path3D`;
- un nuovo `ObservationSource` quando la misura e' davvero specifica.

Inserisci solo informazioni disponibili all'agente nel mondo reale o simulato. Il
progresso usato per una reward non deve diventare automaticamente una observation.

## 6. Aggiungi la reward locale

Sotto `Agent/RewardSystem` aggiungi uno `StepPenaltyReward`:

```text
term_name = time
penalty = -0.001
weight = 1.0
```

Questa e' una reward locale perche' dipende soltanto dal passare di un decision step.
Non aggiungiamo una reward generica per il movimento: in questo compito muoversi in
cerchio non e' progresso e non merita un premio.

## 7. Costruisci lo scenario con i sistemi standard

Crea `target_scenario.tscn`:

```text
TargetScenario                     Node3D, script target_scenario.gd
├── World
│   ├── Floor                      StaticBody3D
│   └── Target                     Area3D
│       ├── MeshInstance3D
│       └── CollisionShape3D
├── Agents
│   └── TargetSeeker               istanza di target_seeker.tscn
├── ScenarioController             script ScenarioController.gd
│   ├── ScenarioRewardSystem       script ScenarioRewardSystem.gd
│   │   ├── Progress               ProgressDeltaScenarioReward.gd
│   │   └── Goal                   EventScenarioReward.gd
│   ├── ProgressProvider           MethodProgressProvider.gd
│   └── ScenarioEventSystem        script ScenarioEventSystem.gd
│       └── GoalReached            AreaReachedEventSource.gd
└── BridgeServer                   script bridge_server.gd
```

Collega nell'Inspector:

- `TargetSeeker.target` a `World/Target`;
- `ScenarioController.controlled_agents` a `Agents/TargetSeeker`;
- `BridgeServer.controller_path` a `../ScenarioController`;
- `GoalReached.area` a `World/Target`;
- `ProgressProvider.source_path` al nodo radice `TargetScenario`.

I path di reward, progress ed eventi del controller hanno gia' i nomi mostrati
nell'albero. Se cambi quei nomi, aggiorna i relativi `NodePath` esportati.

Controlla collision layer e mask dell'`Area3D`: il segnale `body_entered` deve vedere il
`CharacterBody3D`.

## 8. Definisci progress, evento e reward di scenario

Collega questo script al nodo radice:

```gdscript
extends Node3D

@export var target: Node3D
@export var max_target_distance := 20.0


func get_progress(agent:Node) -> float:
	if target == null or not agent is Node3D:
		return 0.0
	var distance := (agent as Node3D).global_position.distance_to(target.global_position)
	return 1.0 - clampf(distance / maxf(max_target_distance, 0.001), 0.0, 1.0)
```

Nel `MethodProgressProvider` configura:

```text
method_name = get_progress
pass_agent_to_source = true
```

Il provider espone un numero fra `0` e `1`. Il controller e le reward non hanno bisogno
di sapere se quel numero rappresenta distanza, percentuale di pista, blocchi distrutti
o salute di un boss.

Configura `ProgressDeltaScenarioReward`:

```text
term_name = progress
progress_reward_scale = 2.0
backward_penalty_scale = 2.0
```

Configura `GoalReached`:

```text
event_name = goal_reached
terminal_reason = goal_reached
only_once = true
```

Configura `EventScenarioReward`:

```text
term_name = goal
event_name = goal_reached
reward = 5.0
only_once = true
```

L'evento descrive il fatto; la reward decide quanto vale. `terminal_reason` fa terminare
il canale dell'agente senza dover aggiungere logica al `BridgeServer`.

Il totale inviato a Python sara':

```text
local RewardSystem + ScenarioRewardSystem
```

I termini restano separati in `info.local_term_rewards` e `info.scenario_terms`, che e'
molto piu' utile di un unico numero durante il debug.

## 9. Configura reset e durata

Sul `ScenarioController` parti con:

```text
max_steps = 500
physics_frames_per_step = 1
randomize_reset = false
deactivate_done_agents = true
```

`max_steps` produce una truncation, non una terminalita' naturale. Impostarlo a `0`
disabilita il limite, ma fallo soltanto quando lo scenario possiede una conclusione o
una stall detection affidabile.

`physics_frames_per_step` indica per quanti tick fisici resta attiva la stessa azione.
A `60 Hz`, un valore `4` fa decidere la policy a `15 Hz`. Cambiarlo modifica anche la
durata fisica di penalita' per step, timeout e finestre di stall.

Per randomizzare il target a ogni episodio puoi usare il segnale del controller invece
di riscrivere `reset_episode()`:

```gdscript
@export var target_spawn_half_extent := Vector2(7.0, 7.0)
@onready var scenario_controller: ScenarioController = $ScenarioController


func _ready() -> void:
	scenario_controller.episode_reset_started.connect(_on_episode_reset_started)


func _on_episode_reset_started(episode_seed:int) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = episode_seed
	target.position.x = rng.randf_range(-target_spawn_half_extent.x, target_spawn_half_extent.x)
	target.position.z = rng.randf_range(-target_spawn_half_extent.y, target_spawn_half_extent.y)
```

Usa il seed ricevuto: due run con lo stesso seed devono poter ricreare lo stesso reset.
Prima verifica il problema con target fisso, poi abilita la randomizzazione.

## 10. Valida prima di allenare

Esporta il percorso di Godot una volta:

```bash
export GODOT_BIN=/percorso/del/eseguibile/Godot
```

Poi esegui un rollout casuale:

```bash
python/.venv/bin/python python/tools/random_rollout.py \
  --godot-project godot \
  --godot-scene res://scenarios/target_seeker/target_scenario.tscn \
  --steps 500 \
  --print-reward-terms \
  --no-headless
```

Prima del training verifica che:

- la spec riporti un agente, `obs_dim=4`, action type `discrete` e quattro azioni;
- ogni azione produca il movimento atteso;
- le observation siano finite e cambino mentre il corpo si muove;
- `progress` cresca avvicinandosi e diminuisca allontanandosi;
- `goal_reached` assegni il bonus una volta sola;
- l'episodio termini entrando nell'area;
- il reset azzeri velocita', input e reward state.

Se Godot si disconnette, guarda `.runtime/godot_logs/godot_<porta>.log`: quasi sempre
troverai li' l'errore GDScript reale.

## 11. Avvia il training generico

Non servono `--obs-dim`, `--num-actions` o un file Python per questo scenario. Metis
legge la spec da Godot:

```bash
python/.venv/bin/python python/train.py \
  --algorithm auto \
  --godot-project godot \
  --godot-scene res://scenarios/target_seeker/target_scenario.tscn \
  --num-envs 4 \
  --num-episodes 1500 \
  --max-steps-per-episode 500 \
  --checkpoint-dir checkpoints/target_seeker_dqn_v1 \
  --headless
```

Con questo action space, `auto` sceglie DQN. Per le azioni continue sceglie DDPG; per
le ibride sceglie PPO. SAC, TD3 e le varianti con dimostrazioni si selezionano
esplicitamente.

Il training usa collector asincrono e modalita' headless per default. Per osservare una
sola istanza senza renderizzare tutte le altre:

```text
--no-headless --render-env-count 1 --render-mode light-gpu
```

Non giudicare il training soltanto dalla loss. Guarda reward, successo, durata degli
episodi e comportamento reale con una valutazione senza esplorazione.

## 12. Multi-agent senza cambiare trainer

Per allenare piu' copie dello stesso agente:

1. aggiungi altre istanze sotto `Agents`, oppure usa la replica del controller;
2. assegna nomi unici;
3. inseriscile in `controlled_agents`;
4. mantieni observation e action space compatibili;
5. aggiungi `--multi-agent` al rollout, al training e al runner.

```bash
python/.venv/bin/python python/train.py \
  --algorithm dqn \
  --godot-project godot \
  --godot-scene res://scenarios/target_seeker/target_scenario.tscn \
  --num-envs 4 \
  --multi-agent \
  --checkpoint-dir checkpoints/target_seeker_multi_v1
```

Le copie producono transizioni separate ma condividono la stessa policy. Non vengono
creati quattro modelli e non viene scelto il migliore fra gli agenti. Reward,
terminalita' e diagnostica restano personali.

Se servono policy indipendenti per ruoli diversi, il flusso multi-policy non e' ancora
automatico: non nascondere ruoli incompatibili dentro un unico parameter sharing.

## 13. Passare ad azioni continue o ibride

Per un controllo continuo sostituisci `DiscreteActionSet` con, per esempio:

```text
ActionSpace
├── Drive       ContinuousAction, size=1, low=-1, high=1
└── Steering    ContinuousAction, size=1, low=-1, high=1
```

Nel corpo, `apply_action()` usa:

```gdscript
var values := agent.decode_continuous_action(action)
_drive_input = float(values[0])
_turn_input = float(values[1])
return values
```

Per un agente ibrido mantieni sia i `ContinuousAction` sia un `DiscreteActionSet`, per
esempio movimento continuo e `shoot/reload` discreti. La spec diventera' `hybrid` e PPO
ricevera' il dictionary dei componenti. Il corpo deve applicare entrambi senza assumere
che l'azione sia un singolo intero.

## 14. Curriculum e dimostrazioni

Un curriculum dovrebbe cambiare la difficolta', non aggiungere informazioni segrete
alla policy. Puoi usare:

- `scenario_configured` e `training_episode` per ostacoli, velocita' o dimensioni;
- reset progressivo con un provider che implementa `build_reset_transform()`;
- `Path3DProgressProvider` per spawn lungo un percorso;
- randomizzazione crescente di target, spawn e fisica.

Avanza la difficolta' soltanto dopo aver validato la fase precedente. Se cambi
continuamente distribuzione prima che la policy impari, il curriculum diventa rumore.

Per raccogliere dimostrazioni il corpo deve supportare l'azione speciale `"manual"` e
restituire l'azione effettivamente applicata. Poi usa:

```bash
python/.venv/bin/python python/recorder.py \
  --godot-project godot \
  --godot-scene res://scenarios/target_seeker/target_scenario.tscn \
  --output demonstrations/target_seeker_demo.npz \
  --episodes 20 \
  --no-headless
```

La guida [Dimostrazioni manuali](../guides/manual_demonstrations.md) spiega prefill,
behavior cloning, DDPGfD e TD3+BC.

## 15. Esegui la policy addestrata

Per osservare il bundle Keras finale:

```bash
python/.venv/bin/python python/run.py \
  --policy-path checkpoints/target_seeker_dqn_v1 \
  --godot-project godot \
  --godot-scene res://scenarios/target_seeker/target_scenario.tscn \
  --episodes 20 \
  --epsilon 0.0 \
  --no-headless
```

Con `policy.json` non devi ripetere l'algoritmo: il runner legge il contratto salvato.
Per riprendere davvero il training usa invece la directory cronologica:

```text
--checkpoint-dir checkpoints/target_seeker_dqn_v1 --resume
```

Un warm start da `policy.keras` carica la policy, ma non ripristina optimizer, target
network, contatori e replay. Resume e warm start non sono la stessa cosa.

## 16. Quando un vecchio training resta compatibile

| Modifica | Pesi | Replay | Indicazione pratica |
|---|---|---|---|
| ordine/dimensione observation | incompatibili | incompatibile | riparti da zero |
| significato o scala observation | caricabili ma incoerenti | incoerente | normalmente riparti |
| dimensione o ordine azioni | incompatibili | incompatibile | riparti da zero |
| reward soltanto | caricabili | contiene vecchi ritorni | evita il vecchio replay |
| fisica o dinamica moderate | caricabili | distribuzione precedente | valuta un warm start prudente |
| soli elementi visivi | compatibili | compatibile | puoi riprendere |

Il fatto che un file venga caricato senza errore non significa che il suo contenuto sia
ancora semanticamente corretto.

## Checklist finale

- [ ] Il corpo ha un figlio `Agent` configurato.
- [ ] Le azioni sono dichiarate una sola volta nell'`ActionSpace`.
- [ ] Le observation sono finite, normalizzate e di dimensione stabile.
- [ ] Il `RewardSystem` contiene soltanto termini locali.
- [ ] Eventi e reward globali stanno nei sistemi dello scenario.
- [ ] `controlled_agents` contiene tutti e soli i corpi allenati.
- [ ] Il bridge punta al `ScenarioController` standard.
- [ ] Reset casuali usano il seed ricevuto.
- [ ] `terminated` e `truncated` rappresentano cause diverse.
- [ ] Il rollout casuale passa prima del training.
- [ ] `run.py` riesce a caricare il bundle prodotto.

Quando qualcosa non impara, riduci il problema: un agente, un obiettivo fisso, poche
azioni e reward leggibili. Aggiungi randomizzazione, multi-agent e curriculum soltanto
dopo che quella versione funziona. E' piu' veloce correggere un contratto piccolo che
interpretare migliaia di episodi di un ambiente ambiguo.
