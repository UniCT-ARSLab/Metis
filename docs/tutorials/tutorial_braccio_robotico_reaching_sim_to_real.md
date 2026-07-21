# Braccio robotico: reaching, ostacoli e sim-to-real

Questo tutorial costruisce il digital twin di una cella robotica in cui un braccio
raggiunge target noti evitando ostacoli fissi. Geometria, frame, limiti e controllo
devono corrispondere alla cella reale; la policy non dipende da sensori di prossimita'
che il robot non possiede.

Il risultato finale usa:

- un braccio con `N` giunti revolute, per esempio 3, 7 oppure 8;
- azioni continue di velocita' articolare oppure cartesiana con IK;
- propriocezione, target relativo e geometria nota della cella;
- reward dense per il progresso e sparse per successo e collisione;
- una catena di `Node3D`, uno `Skeleton3D` oppure una scena generata da URDF;
- TD3+BC / DDPG+BC da traiettorie esperte, con SAC come confronto;
- una safety layer esterna alla policy per l'esecuzione sul robot.

Il tutorial si ferma al reaching. Presa, gripper e contatti con oggetti sono un secondo
problema: richiedono una simulazione dinamica e un modello di contatto molto piu'
accurati.

## 1. E' fattibile con Metis?

Si', con due distinzioni importanti.

Metis possiede gia' il necessario per addestrare la policy:

- action space continui;
- observation componibili;
- reward locali e di scenario;
- eventi terminali, curriculum e reset deterministici;
- piu' istanze Godot e collector asincrono;
- SAC, DDPG, TD3 e varianti con behavior cloning;
- recorder, checkpoint, policy Keras ed export TFLite/ONNX.

Godot e' adeguato per un primo task di reaching e obstacle avoidance con controllo di
posizione o velocita'. Non e' invece, allo stato attuale di Metis, un sostituto diretto
di MuJoCo o robosuite per torque control, contatti fini, presa e identificazione
dinamica del robot.

La prima versione usera' quindi un modello cinematico comandato in velocita'. E' una
scelta coerente con un robot reale che possiede gia' servo e controller articolari: la
policy decide le velocita' desiderate, mentre un livello piu' basso applica limiti e
stabilizza i giunti.

Inoltre, una cella statica e completamente nota e' prima di tutto un problema di motion
planning. Un planner geometrico puo' generare traiettorie collision-free e costituisce
la baseline da battere, oltre che l'esperto per il behavior cloning. RL diventa utile
per approssimare velocemente il planner e ottimizzare durata, fluidita' o consumo, ma
non dimostra da solo che una traiettoria sia sempre globalmente ottima o sicura.

Senza sensori, il sistema non puo' osservare una persona o un oggetto aggiunto alla
cella dopo la calibrazione. L'area reale deve quindi essere controllata e interdetta,
oppure servono sensori esterni. Questa limitazione non puo' essere risolta dal training.

## 2. Cosa fa davvero l'esempio OpenArmX

`openarmx_robosuite_RL` e' un buon riferimento, ma non risolve esattamente questo
scenario. Il progetto:

1. registra un modello OpenArmX in robosuite/MuJoCo;
2. costruisce un task di presa di un cubo;
3. genera dimostrazioni con una logica esperta assistita dalla visione;
4. salva le transizioni in dataset `.npz`;
5. addestra un actor con DDPG piu' behavior cloning;
6. valuta la policy risultante.

La demo stabile usa un solo braccio, una regione di spawn del cubo piuttosto limitata
e una presa assistita. Il risultato dichiarato dal repository riguarda quella
configurazione, non reaching arbitrario con ostacoli sconosciuti.

L'idea utile da portare in Metis e' la pipeline:

```text
modello del robot -> dimostrazioni -> pretraining BC -> RL online -> valutazione
```

Per il nostro task conviene validare prima modello, collision checker e planner. Quando
il contratto di observation e action e' stabile, le traiettorie del planner diventano
dimostrazioni per `td3_bc`, `ddpg_bc` o `ddpgfd`. SAC da zero resta un confronto utile,
non la prima scelta per una cella fissa gia' modellata.

## 3. Prima decisione: cosa controlla la policy

Le alternative principali sono tre.

| Azione | Vantaggio | Limite |
|---|---|---|
| torque per giunto | controllo dinamico completo | difficile e rischioso nel sim-to-real |
| delta cartesiano dell'end-effector | apprendimento piu' semplice | l'IK decide la configurazione del gomito |
| velocita' per giunto | trasferibile e permette di aggirare ostacoli | action space piu' grande |

Per questo tutorial scegliamo un vettore di velocita' articolari:

```text
action = [dq_0, dq_1, ..., dq_(N-1)] in [-1, 1]
```

Ogni valore viene moltiplicato per la velocita' massima del giunto. Per un braccio con
`N` giunti, `action_size=N`: vale allo stesso modo per 3, 7 o 8 giunti.

La sezione 8.1 costruisce anche la variante cartesiana: la policy emette tre velocita'
del TCP e un solver IK le trasforma in comandi articolari.

Non comandare direttamente i motori reali con l'output della rete. Il runtime hardware
deve ancora applicare limiti di posizione, velocita', accelerazione, watchdog e arresto
di emergenza.

## 4. Il contratto di observation senza sensori di prossimita'

La policy ricevera' sempre questo ordine:

```text
joint_positions       N valori in [-1, 1]
joint_velocities      N valori in [-1, 1]
target_error_base     3 valori in [-1, 1]
previous_action       N valori articolari oppure 3 valori IK
```

Indichiamo con `N` il numero di giunti controllati. Con azioni articolari:

```text
obs_dim = N posizioni + N velocita' + 3 target error + N previous action
obs_dim = 3*N + 3
```

Nella variante IK posizionale, l'azione precedente ha sempre tre valori:

```text
obs_dim = N posizioni + N velocita' + 3 target error + 3 previous action
obs_dim = 2*N + 6
```

| Giunti `N` | Action articolare | Obs articolare | Action IK | Obs IK |
|---:|---:|---:|---:|---:|
| 3 | 3 | 12 | 3 | 12 |
| 7 | 7 | 24 | 3 | 20 |
| 8 | 8 | 27 | 3 | 22 |

Non includiamo posizione assoluta nel mondo o un finto sensore. Target e stato
articolare sono espressi nel frame della base. Se ostacoli e base non cambiano, la loro
geometria viene appresa implicitamente nei pesi della policy.

Questa observation e' trasferibile perche' il robot reale puo' ricostruirla con:

- encoder per angoli e velocita' articolari;
- target espresso nel frame calibrato della base;
- ultimo comando realmente applicato.

Collisioni e clearance possono comunque essere calcolate dal digital twin come reward
privilegiate durante il training: le reward non sono necessarie in inferenza. Sul robot
reale lo stesso URDF e la stessa scena geometrica devono vivere in un supervisore, per
esempio una planning scene, che rifiuta il comando successivo se avvicina il robot a
una collisione.

Se vuoi una sola policy per piu' layout, allora la geometria non puo' rimanere implicita.
Occorre aggiungere all'observation una descrizione nota del layout, come pose e dimensioni
dei primi `K` ostacoli oppure distanze per link calcolate dalla planning scene. Non sono
sensori di prossimita': sono query sul modello calibrato. Se il layout reale non viene
aggiornato, anche quelle query saranno sbagliate.

## 5. Costruisci un digital twin misurabile

Per un robot esistente non stimare a occhio la geometria. Servono:

- origine e asse di ogni giunto;
- ordine e segno delle articolazioni;
- lunghezze dei link;
- limiti minimi e massimi;
- posizione zero e posa home;
- mesh visuali e collisioni semplificate;
- velocita' ammissibili dal controller reale.

Per la cella servono inoltre:

- CAD o misure di tavolo, pareti, supporti e attrezzature;
- trasformazione rigida tra base robot e mondo;
- posa TCP rispetto all'ultimo link;
- collision shape conservative, leggermente piu' grandi del reale;
- tempo di controllo, latenza e limiti di accelerazione misurati;
- un elenco esplicito degli oggetti che non possono cambiare durante l'esecuzione.

OpenArm pubblica descrizioni URDF/xacro, mesh, limiti, cinematica e parametri inerziali.
Genera prima un URDF senza macro e scegli una delle tre strade supportate dal tutorial:

1. ricostruisci una catena di pivot `Node3D` usando origin e axis URDF;
2. importa le mesh come glTF con uno `Skeleton3D` e associa i joint ai bone;
3. usa un importer URDF che produca una scena Godot e poi collega i joint importati.

Per Godot 4.6 esiste gia' l'add-on community
[Godot URDF](https://godotengine.org/asset-library/asset/5127), che genera visual e
collisioni trascinando un `.urdf` nella scena oppure usando `urdf_loader.gd`. Prima di
scrivere un importer Metis parallelo conviene integrare e verificare questo add-on.

Un eventuale importer nativo Metis deve usare `XMLParser` o una libreria URDF e produrre
sempre lo stesso contratto:

```text
RobotRoot
├── pivot o bone per ogni joint mobile
├── visual per ogni link
├── collision shape per ogni link
├── attachment dell'end-effector
└── metadata/resource con:
    joint_names, parent, axis, origin, lower, upper, home, max_velocity
```

L'importer puo' scegliere `Node3D` o `Skeleton3D`; l'`Agent` non deve dipendere da tale
scelta. Gli URDF `continuous`, `revolute`, `prismatic` e `fixed`, i percorsi
`package://`, le scale delle mesh e la conversione dei frame devono essere gestiti
esplicitamente. Il primo scenario puo' limitarsi a joint `revolute` e `fixed`, ma deve
rifiutare con un errore i tipi non supportati anziche' importarli in modo ambiguo.

Usa una convenzione metrica:

```text
1 unita' Godot = 1 metro
angoli interni = radianti
```

ROS/URDF usa normalmente un frame diverso da Godot. Definisci una sola trasformazione
calibrata `T_godot_from_robot` e riusala per target, collisioni e valutazione. Non
correggere assi e segni in punti sparsi del codice.

Un digital twin non sara' mai perfettamente identico. L'obiettivo e' misurare l'errore,
modellarlo e rendere conservative le collisioni. Texture fotorealistiche contano poco
per una policy state-based; millimetri, frame, TCP, ritardi e limiti contano moltissimo.

Definisci criteri di accettazione prima del training:

| Verifica | Metodo |
|---|---|
| posa TCP | confronta simulazione e robot su almeno 20 configurazioni misurate |
| assi e segni | applica un piccolo comando positivo a un giunto per volta |
| ostacoli | misura la distanza link-ostacolo in pose vicine al margine |
| velocita' | confronta risposta a gradino e tempo per piccoli movimenti |
| latenza | misura da invio comando a variazione encoder |

Registra errore medio e massimo. Il margine delle collision shape e gli intervalli di
randomizzazione devono coprire quell'errore, non una stima arbitraria.

## 6. File e struttura delle scene

Crea:

```text
godot/agents/RobotArm/
├── robot_arm.gd
└── robot_arm.tscn

godot/agents/RobotArms/
└── urdf_robot_arm_agent.gd       adapter pronto per GodotRobot

godot/scenarios/robot_arm_reaching/
├── robot_arm_reaching_scenario.gd
└── robot_arm_reaching_scenario.tscn
```

Con una catena di pivot `Node3D` la scena dell'agente puo' avere questa struttura:

```text
RobotArm                              Node3D, script robot_arm.gd
├── Joint0                           Node3D, pivot
│   ├── Link0                        MeshInstance3D
│   ├── Link0Safety                  Area3D + CollisionShape3D
│   └── Joint1                       Node3D
│       ├── Link1                    MeshInstance3D
│       ├── Link1Safety              Area3D + CollisionShape3D
│       └── ...
│           ├── ToolSafety           Area3D + CollisionShape3D
│           └── EndEffector          Marker3D
└── Agent                            Agent.gd
    ├── ActionSpace                  ActionSpace.gd
    │   └── JointVelocity            ContinuousAction.gd
    ├── ObservationSystem            ObservationSystem.gd
    │   ├── JointPositions           MethodObservationSource.gd
    │   ├── JointVelocities          MethodObservationSource.gd
    │   ├── TargetError              MethodObservationSource.gd
    │   └── PreviousAction           MethodObservationSource.gd
    └── RewardSystem                 RewardSystem.gd
        ├── JointMotion              FunctionRewardComponent.gd
        ├── JointLimit               FunctionRewardComponent.gd
        ├── Smoothness               ActionSmoothnessPenaltyReward.gd
        └── Time                     StepPenaltyReward.gd
```

Con uno `Skeleton3D`, usa invece `BoneAttachment3D` per TCP e collisioni:

```text
RobotArm                              Node3D, script robot_arm.gd
├── Armature                         Node3D
│   └── Skeleton3D                  Skeleton3D importato
│       ├── RobotMesh              MeshInstance3D con Skin
│       ├── Link0Attachment        BoneAttachment3D
│       │   └── Link0Safety       Area3D + CollisionShape3D
│       ├── ...
│       └── TcpAttachment          BoneAttachment3D
│           └── EndEffector          Marker3D
└── Agent                            stessa struttura precedente
```

I `BoneAttachment3D` devono puntare ai bone corretti. Disabilita eventuali
`AnimationPlayer` o `AnimationTree` che scrivono sugli stessi bone: durante il task e'
`RobotArmBody` a controllarne la posa.

Con una scena generata dall'add-on `godot_urdf`, mantieni invece il `GodotRobot`
importato come figlio diretto del corpo agente:

```text
XArmAgent                              Node3D, script urdf_robot_arm_agent.gd
├── xarm                              GodotRobot, istanza del file .urdf
├── EndEffector                       Marker3D aggiornato dal TCP link
└── Agent                             Agent.gd
    ├── ActionSpace                   ActionSpace.gd
    │   └── JointVelocity             ContinuousAction.gd
    ├── ObservationSystem             ObservationSystem.gd
    │   ├── JointPositions            MethodObservationSource.gd
    │   ├── JointVelocities           MethodObservationSource.gd
    │   ├── TargetError               MethodObservationSource.gd
    │   └── PreviousAction            MethodObservationSource.gd
    └── RewardSystem                  RewardSystem.gd
```

In entrambi i backend, `joint_axes` e' espresso nel frame locale del pivot o del bone.
Per uno Skeleton importato non copiare alla cieca l'asse URDF: applica prima la
conversione di frame scelta e verifica il verso con un piccolo angolo positivo.

Le collision shape dei link sono primitive semplici, per esempio capsule e box. Le
mesh dettagliate sono adatte al rendering, non al rilevamento rapido delle collisioni.
Espandi le shape di sicurezza del margine richiesto, per esempio 5-10 mm: cosi' una
collisione simulata rappresenta l'ingresso nella fascia vietata, non il contatto reale.

### Collegare una scena importata da URDF

1. Esporta lo xacro in un `.urdf` risolto e porta nel progetto le mesh referenziate.
2. Installa e abilita un importer compatibile con la versione di Godot. Per Godot 4.6
   puoi partire dall'add-on [Godot URDF](https://godotengine.org/asset-library/asset/5127).
3. Importa il file, salva il risultato come scena e controlla scala, assi e limiti un
   giunto alla volta.
4. Se il file usa `<mimic>`, verifica che i follower non siano elencati tra gli
   attuatori indipendenti. L'add-on incluso in Metis conserva `joint`, `multiplier` e
   `offset` e sincronizza i target dei relativi `Generic6DOFJoint3D`.
5. Se l'importer genera pivot controllabili, assegnali a `joints` in ordine cinematico
   e scegli `NODE_CHAIN`.
6. Se genera uno `Skeleton3D`, assegna `skeleton`, compila `joint_bone_names` nello
   stesso ordine e scegli `SKELETON_3D`.
7. Aggiungi o genera i `BoneAttachment3D` per TCP e volumi di sicurezza.

Non tutti gli importer producono la stessa gerarchia. Per essere compatibile con questo
tutorial l'output deve esporre i joint come pivot modificabili oppure come bone e deve
conservare `axis`, `lower`, `upper`, `home` e `velocity`. Se genera `HingeJoint3D` o
corpi fisici, crea un adapter specifico invece di trattarli come semplici pivot.

La variante `GodotRobot` inclusa nel progetto espone:

```gdscript
var joint_names := robot.get_actuated_joint_names()
robot.set_joint_target_velocity("shoulder_pan", 0.25)
robot.set_joint_target_position("grip_left", -0.5, 20.0, 2.0)
```

`get_actuated_joint_names()` esclude automaticamente `fixed` e `mimic`. Non inviare
azioni direttamente a un follower: per una relazione URDF
`q_follower = multiplier * q_source + offset`, il comando del source viene propagato
alla velocita' e al target del servo del follower. Dopo aver modificato il file URDF o
l'add-on, usa `Reimport` sul file `.urdf` prima di aggiornare la scena agente.

### Estendere o sostituire l'importer URDF

L'add-on incluso copre il percorso basato su `Generic6DOFJoint3D`, compresi i joint
mimic. Realizza un importer differente soltanto se serve un output a pivot `Node3D` o
`Skeleton3D`, oppure se il robot richiede tag e trasmissioni non rappresentabili dal
backend fisico corrente. Non mettere comunque il parser dentro `robot_arm.gd`: un
add-on riusabile puo' avere questa
struttura:

```text
godot/addons/metis_urdf/
├── plugin.cfg
├── plugin.gd                     EditorPlugin
├── urdf_import_plugin.gd         EditorImportPlugin
├── urdf_parser.gd                XML -> descrizione intermedia
├── urdf_robot_description.gd     link, joint, geometry, limits
├── node_chain_builder.gd          descrizione -> pivot Node3D
└── skeleton_builder.gd            descrizione -> Skeleton3D
```

`urdf_import_plugin.gd` deve riconoscere `.urdf`, invocare il parser, scegliere il
builder da un'opzione di importazione, creare una `PackedScene` e salvarla nel path
generato dall'editor. Godot espone questo flusso tramite `EditorImportPlugin`; il parser
XML disponibile nel motore e' volutamente low-level, quindi conviene trasformare prima
il documento in una risorsa intermedia indipendente dalla scena.

Implementa l'MVP in questo ordine:

1. analizza tutti i `<link>` e indicizzali per nome;
2. analizza i `<joint>` con parent, child, origin `xyz/rpy`, axis e limit;
3. trova il link radice e ordina la catena a partire da esso;
4. risolvi mesh relative, `package://`, scala e conversione degli assi una sola volta;
5. crea prima i pivot/bone e poi visual, collisioni e TCP attachment;
6. salva metadata con nomi e ordine dei joint usati da `RobotArmBody`;
7. fallisci esplicitamente su tipi o geometrie non supportati.

Per il builder a pivot, ogni joint mobile e' un `Node3D` posizionato con l'`origin`
URDF; il link figlio e' sotto quel pivot e ruota attorno ad `axis`. Per il builder
Skeleton, crea un bone per link, assegna parent e rest dalla stessa trasformazione e
usa la pose rotation del bone per il joint. Lo `Skeleton3D` deforma le mesh, ma non
crea automaticamente collisioni robotiche: servono comunque `BoneAttachment3D` con
`Area3D` o un builder equivalente.

Il primo MVP puo' supportare `fixed` e `revolute`. Aggiungi in seguito `continuous`,
`prismatic`, `mimic`, materiali, inertial e alberi ramificati. Prima di dichiararlo
compatibile, confronta per almeno 20 configurazioni casuali la posa di ogni link e del
TCP con una libreria URDF/ROS di riferimento. Un importer che mostra bene la mesh ma
sbaglia di pochi gradi un asse non e' utilizzabile per sim-to-real.

## 7. Script completo del corpo articolato

Questa prima implementazione e' destinata ai backend `NODE_CHAIN` e `SKELETON_3D`.
Se il nodo importato e' un `GodotRobot`, usa invece l'adapter della sezione 7.1: un
`Generic6DOFJoint3D` non e' un pivot da ruotare direttamente.

Collega questo script a `RobotArm`. I valori esportati permettono di usare lo stesso
codice con bracci differenti.

```gdscript
extends Node3D
class_name RobotArmBody

enum ArmatureBackend {
    NODE_CHAIN,
    SKELETON_3D,
}

signal target_reached
signal obstacle_collision

@export_category("Armature")
@export_enum("Node chain", "Skeleton3D") var armature_backend := ArmatureBackend.NODE_CHAIN
@export var joints: Array[Node3D] = []
@export var skeleton: Skeleton3D
@export var joint_bone_names := PackedStringArray()

@export_category("Kinematics")
@export var joint_axes: Array[Vector3] = []
@export var joint_min_degrees := PackedFloat32Array()
@export var joint_max_degrees := PackedFloat32Array()
@export var joint_home_degrees := PackedFloat32Array()
@export var joint_max_speed_degrees := PackedFloat32Array()
@export var end_effector: Node3D

@export_category("Task")
@export var target: Node3D
@export var workspace_scale := 1.2
@export var success_distance := 0.05
@export var success_hold_physics_frames := 10

@export_category("Digital twin safety")
@export var safety_volumes: Array[Area3D] = []
@export var obstacle_group := "robot_obstacle"

@export_category("Control")
@export var manual_control := false

@onready var agent: Agent = $Agent

var _rest_bases: Array[Basis] = []
var _bone_indices: Array[int] = []
var _joint_angles: Array[float] = []
var _joint_velocities: Array[float] = []
var _commands: Array[float] = []
var _previous_action: Array[float] = []
var _pending_reset_offsets: Array[float] = []
var _terminal := false
var _succeeded := false
var _collided := false
var _success_frames := 0
var _training_active := true


func _ready() -> void:
    if not _configuration_is_valid():
        set_physics_process(false)
        return

    if not _prepare_armature_backend():
        set_physics_process(false)
        return
    _resize_state()
    for area in safety_volumes:
        area.monitoring = true
        if not area.body_entered.is_connected(_on_safety_body_entered):
            area.body_entered.connect(_on_safety_body_entered)
        var area_callable := Callable(self, "_on_safety_area_entered").bind(area)
        if not area.area_entered.is_connected(area_callable):
            area.area_entered.connect(area_callable)


func _physics_process(delta:float) -> void:
    if not _training_active or _terminal:
        return

    for index in range(get_joint_count()):
        var old_angle := _joint_angles[index]
        var max_speed := deg_to_rad(float(joint_max_speed_degrees[index]))
        var requested_speed := _commands[index] * max_speed
        var min_angle := deg_to_rad(float(joint_min_degrees[index]))
        var max_angle := deg_to_rad(float(joint_max_degrees[index]))
        _joint_angles[index] = clampf(old_angle + requested_speed * delta, min_angle, max_angle)
        _joint_velocities[index] = (_joint_angles[index] - old_angle) / maxf(delta, 0.000001)

    _apply_joint_pose()
    _update_success_state()


func apply_action(action:Variant) -> Variant:
    if (typeof(action) == TYPE_STRING or typeof(action) == TYPE_STRING_NAME) and str(action) == "manual":
        return apply_manual_action()

    manual_control = false
    var values := agent.decode_continuous_action(action)
    for index in range(get_joint_count()):
        _commands[index] = clampf(float(values[index]), -1.0, 1.0) if index < values.size() else 0.0
        _previous_action[index] = _commands[index]
    return _previous_action.duplicate()


func apply_manual_action() -> Array:
    var values: Array = []
    values.resize(get_joint_count())
    for index in range(get_joint_count()):
        var negative := StringName("joint_%d_negative" % index)
        var positive := StringName("joint_%d_positive" % index)
        values[index] = Input.get_action_strength(positive) - Input.get_action_strength(negative)
    return apply_action(values)


func reset_all(original_transform:Variant, reset_rewards := true) -> void:
    set_training_active(true)
    if typeof(original_transform) == TYPE_TRANSFORM3D:
        transform = original_transform

    _terminal = false
    _succeeded = false
    _collided = false
    _success_frames = 0
    for index in range(get_joint_count()):
        var offset := _pending_reset_offsets[index] if index < _pending_reset_offsets.size() else 0.0
        _joint_angles[index] = clampf(
            deg_to_rad(float(joint_home_degrees[index])) + offset,
            deg_to_rad(float(joint_min_degrees[index])),
            deg_to_rad(float(joint_max_degrees[index]))
        )
        _joint_velocities[index] = 0.0
        _commands[index] = 0.0
        _previous_action[index] = 0.0
    _pending_reset_offsets.clear()
    _apply_joint_pose()

    if has_method("reset_physics_interpolation"):
        reset_physics_interpolation()
    agent.reset_observation_sources()
    if reset_rewards:
        agent.refresh_observation_sources()
        agent.reset_reward({"body": self})


func set_reset_joint_offsets(offsets_radians:Array) -> void:
    _pending_reset_offsets = offsets_radians.duplicate()


func set_training_active(enabled:bool) -> void:
    _training_active = enabled
    set_physics_process(enabled)
    if not enabled:
        for index in range(_commands.size()):
            _commands[index] = 0.0


func is_terminal() -> bool:
    return _terminal


func get_progress() -> float:
    return 1.0 - clampf(_target_distance() / maxf(workspace_scale, 0.001), 0.0, 1.0)


func get_joint_position_observation() -> Array:
    var result: Array = []
    for index in range(get_joint_count()):
        var low := deg_to_rad(float(joint_min_degrees[index]))
        var high := deg_to_rad(float(joint_max_degrees[index]))
        result.append(remap(_joint_angles[index], low, high, -1.0, 1.0))
    return result


func get_joint_velocity_observation() -> Array:
    var result: Array = []
    for index in range(get_joint_count()):
        var max_speed := maxf(deg_to_rad(float(joint_max_speed_degrees[index])), 0.000001)
        result.append(clampf(_joint_velocities[index] / max_speed, -1.0, 1.0))
    return result


func get_target_error_observation() -> Vector3:
    if target == null or end_effector == null:
        return Vector3.ZERO
    var error_world := target.global_position - end_effector.global_position
    var error_base := global_transform.basis.inverse() * error_world
    var scale := maxf(workspace_scale, 0.001)
    return Vector3(
        clampf(error_base.x / scale, -1.0, 1.0),
        clampf(error_base.y / scale, -1.0, 1.0),
        clampf(error_base.z / scale, -1.0, 1.0)
    )


func get_previous_action_observation() -> Array:
    return _previous_action.duplicate()


func get_joint_motion_penalty() -> float:
    var effort := 0.0
    for command in _commands:
        effort += absf(command)
    return -effort / maxf(float(_commands.size()), 1.0)


func get_joint_limit_penalty() -> float:
    var penalty := 0.0
    var margin := 0.10
    for index in range(get_joint_count()):
        var low := deg_to_rad(float(joint_min_degrees[index]))
        var high := deg_to_rad(float(joint_max_degrees[index]))
        var ratio := clampf(inverse_lerp(low, high, _joint_angles[index]), 0.0, 1.0)
        var edge_distance := minf(ratio, 1.0 - ratio)
        if edge_distance < margin:
            var error := (margin - edge_distance) / margin
            penalty -= error * error
    return penalty / maxf(float(get_joint_count()), 1.0)


func get_control_input(input_name:String) -> float:
    if not input_name.begins_with("joint_velocity_"):
        return 0.0
    var index := int(input_name.trim_prefix("joint_velocity_"))
    return _commands[index] if index >= 0 and index < _commands.size() else 0.0


func _apply_joint_pose() -> void:
    for index in range(get_joint_count()):
        var rotation := Quaternion(joint_axes[index].normalized(), _joint_angles[index])
        if armature_backend == ArmatureBackend.SKELETON_3D:
            skeleton.set_bone_pose_rotation(_bone_indices[index], rotation)
        else:
            joints[index].transform.basis = _rest_bases[index] * Basis(rotation)


func _update_success_state() -> void:
    if _target_distance() <= success_distance:
        _success_frames += 1
    else:
        _success_frames = 0
    if _success_frames < success_hold_physics_frames:
        return
    _succeeded = true
    _terminal = true
    target_reached.emit()


func _on_safety_body_entered(body:Node) -> void:
    if _terminal or not body.is_in_group(obstacle_group):
        return
    _register_collision()


func _on_safety_area_entered(other:Area3D, source:Area3D) -> void:
    if _terminal:
        return
    var source_index := safety_volumes.find(source)
    var other_index := safety_volumes.find(other)
    if source_index < 0 or other_index < 0:
        return
    # Volumi consecutivi appartengono a link adiacenti e possono toccarsi normalmente.
    if absi(source_index - other_index) <= 1:
        return
    _register_collision()


func _register_collision() -> void:
    if _terminal:
        return
    _collided = true
    _terminal = true
    obstacle_collision.emit()


func _target_distance() -> float:
    if target == null or end_effector == null:
        return workspace_scale
    return end_effector.global_position.distance_to(target.global_position)


func get_joint_count() -> int:
    if armature_backend == ArmatureBackend.SKELETON_3D:
        return joint_bone_names.size()
    return joints.size()


func _prepare_armature_backend() -> bool:
    if armature_backend == ArmatureBackend.NODE_CHAIN:
        for joint in joints:
            _rest_bases.append(joint.transform.basis)
        return true

    skeleton.reset_bone_poses()
    for bone_name in joint_bone_names:
        var bone_index := skeleton.find_bone(bone_name)
        if bone_index < 0:
            push_error("RobotArm: bone '%s' not found" % bone_name)
            return false
        _bone_indices.append(bone_index)
    return true


func _resize_state() -> void:
    var count := get_joint_count()
    _joint_angles.resize(count)
    _joint_velocities.resize(count)
    _commands.resize(count)
    _previous_action.resize(count)
    for index in range(count):
        _joint_angles[index] = deg_to_rad(float(joint_home_degrees[index]))
        _joint_velocities[index] = 0.0
        _commands[index] = 0.0
        _previous_action[index] = 0.0
    _apply_joint_pose()


func _configuration_is_valid() -> bool:
    var count := get_joint_count()
    var valid := count > 0
    if armature_backend == ArmatureBackend.SKELETON_3D:
        valid = valid and skeleton != null
    else:
        valid = valid and joints.size() == count
    valid = valid and joint_axes.size() == count
    valid = valid and joint_min_degrees.size() == count
    valid = valid and joint_max_degrees.size() == count
    valid = valid and joint_home_degrees.size() == count
    valid = valid and joint_max_speed_degrees.size() == count
    valid = valid and end_effector != null
    if not valid:
        push_error("RobotArm: armature, arrays and end_effector must describe the same DOF count")
    return valid
```

Questo non e' un solver fisico a torque. I pivot o le pose dei bone producono la stessa
cinematica comandata; le `Area3D` rilevano l'ingresso nella geometria vietata. E'
intenzionale per il primo esperimento: prima validiamo policy, geometria e sim-to-real
del comando; la dinamica puo' essere aggiunta in una fase separata.

### 7.1 Corpo agente per `GodotRobot`

Metis include l'adapter completo in:

```text
res://agents/RobotArms/urdf_robot_arm_agent.gd
```

Collegalo alla radice della scena agente, non al nodo `GodotRobot`. L'adapter ricava
limiti e velocita' dal file URDF, integra i comandi articolari, aggiorna tutti i link
con la forward kinematics URDF e applica automaticamente le relazioni `mimic`.

Il nucleo del contratto Metis e':

```gdscript
@onready var agent: Agent = $Agent

var _robot: GodotRobot
var _joint_names := PackedStringArray()


func _ready() -> void:
    _robot = get_node(robot_path) as GodotRobot
    _robot.control_mode = GodotRobot.ControlMode.KINEMATIC
    _joint_names = (
        controlled_joint_names
        if not controlled_joint_names.is_empty()
        else _robot.get_actuated_joint_names()
    )


func apply_action(action:Variant) -> Variant:
    var values := agent.decode_continuous_action(action)
    for index in range(_joint_names.size()):
        var normalized := clampf(float(values[index]), -1.0, 1.0)
        var joint: URDFJoint = _robot.urdf.get_joint(_joint_names[index])
        _robot.set_joint_target_velocity(
            _joint_names[index], normalized * joint.limit.velocity)
    return values


func get_joint_position_observation() -> Array:
    var result: Array = []
    for joint_name in _joint_names:
        var joint: URDFJoint = _robot.urdf.get_joint(joint_name)
        var lower := minf(joint.limit.lower, joint.limit.upper)
        var upper := maxf(joint.limit.lower, joint.limit.upper)
        result.append(remap(
            _robot.get_joint_position(joint_name),
            lower, upper, -1.0, 1.0))
    return result


func reset_all(original_transform:Variant, reset_rewards := true) -> void:
    if typeof(original_transform) == TYPE_TRANSFORM3D:
        transform = original_transform
    _robot.reset_joint_positions(_build_home_positions())
    agent.reset_observation_sources()
    if reset_rewards:
        agent.refresh_observation_sources()
        agent.reset_reward({"body": self})
```

Il file fornito aggiunge anche observation di velocita', errore TCP e azione
precedente, posizione home, offset di curriculum, successo mantenuto per piu' frame,
collisioni esterne ricavate automaticamente dalle `CollisionShape3D` importate,
`safety_volumes` opzionali, helper per le reward e dimensionamento automatico di
`JointVelocity`.

Per lo xArm presente nel progetto imposta:

```text
robot_path = xarm
controlled_joint_names = [
    xarm_6_joint,
    xarm_5_joint,
    xarm_4_joint,
    xarm_3_joint,
    xarm_2_joint,
    wrist_roll
]
use_kinematic_control = true
tcp_link_name = hand_link
end_effector_path = EndEffector
workspace_scale = 0.5
success_distance = 0.02
```

Questi sono sei comandi per il reaching. `grip_left` e' il settimo attuatore
indipendente e va aggiunto soltanto in un task di presa. `grip_right`, tendini e dita
non entrano nell'action space: sono follower `mimic` di `grip_left`.

La modalita' cinematica e' la baseline raccomandata: la posa letta nelle observation
coincide con quella applicata e il reset e' deterministico. Le collision shape
importate vengono mosse insieme ai link congelati e l'adapter esegue query fisiche
contro i nodi del gruppo `robot_obstacle`. La modalita' `PHYSICS_MOTORS` resta
disponibile, ma per usarla come digital twin dinamico servono masse, inerzie, guadagni
e lettura degli angoli fisici calibrati.

## 8. Configura azioni e observation nell'Inspector

### Imposta il numero di giunti `N`

Conta soltanto i giunti mobili e controllati dalla policy. I joint URDF `fixed` fanno
parte della gerarchia, ma non dell'action space. Lo script di questo tutorial gestisce
joint `revolute`; i `prismatic` richiedono una colonna Jacobiana e un'integrazione
lineare dedicate.

Nel backend `NODE_CHAIN`, `N` corrisponde a `joints.size()`. Nel backend
`SKELETON_3D`, corrisponde a `joint_bone_names.size()`. Per entrambi, configura
esattamente `N` elementi in ciascuna proprieta':

```text
joint_axes
joint_min_degrees
joint_max_degrees
joint_home_degrees
joint_max_speed_degrees
```

L'indice deve descrivere sempre lo stesso giunto in tutte le liste:

```text
index 0 -> primo giunto mobile dalla base
index 1 -> secondo giunto mobile
...
index N-1 -> ultimo giunto controllato prima del TCP
```

Esempi:

| Robot | `joints` o `joint_bone_names` | Lunghezza di ogni array | Action size articolare |
|---|---:|---:|---:|
| braccio 3-DOF | 3 | 3 | 3 |
| braccio 7-DOF | 7 | 7 | 7 |
| braccio 8-DOF | 8 | 8 | 8 |

`RobotArmBody._configuration_is_valid()` blocca la simulazione se le lunghezze non
coincidono. L'`ActionSpace`, invece, e' un nodo separato: devi impostarne manualmente
`size=N` e verificare con `random_rollout.py` che `action_size` coincida con
`RobotArmBody.get_joint_count()`.

Con `URDFRobotArmAgentBody`, limiti e velocita' arrivano invece dal file URDF e
`JointVelocity.size` viene aggiornato automaticamente in base a
`controlled_joint_names`. Controlla comunque il valore dichiarato dal bridge prima
del training.

Su `Agent/ActionSpace/JointVelocity` imposta:

```text
action_name = joint_velocity
size = N                         # inserisci il numero: 3, 7, 8, ...
low = -1.0
high = 1.0
```

Configura i quattro `MethodObservationSource`:

| Nodo | observation_name | method_name |
|---|---|---|
| JointPositions | `joint_positions` | `get_joint_position_observation` |
| JointVelocities | `joint_velocities` | `get_joint_velocity_observation` |
| TargetError | `target_error_base` | `get_target_error_observation` |
| PreviousAction | `previous_action` | `get_previous_action_observation` |

Lascia `source_path` vuoto: la source usera' il corpo padre dell'`Agent`.

`auto_detect_environment_collisions` usa le shape URDF per gli ostacoli esterni ed
esclude intenzionalmente tutti i corpi del robot dalla query. Per la self-collision
servono ancora `SafetyVolumes` dedicati oppure un collision checker che conosca le
coppie di link adiacenti da ignorare.

La policy riceve soltanto stato articolare, target e azione precedente. In una cella
fissa la geometria degli ostacoli e' implicita nei dati di training. Il manifest
`policy.json` salva nomi e dimensioni delle observation, ma la rete Keras riceve
soltanto il vettore numerico.

Gli script sono generici in `N`, il modello addestrato no. Cambiare il numero di giunti
cambia `obs_dim` e, nel controllo articolare, anche `action_size`. Crea quindi directory
di checkpoint e dataset separate, per esempio:

```text
checkpoints/robot_arm_3dof_joint_v1
checkpoints/robot_arm_7dof_joint_v1
checkpoints/robot_arm_8dof_ik_v1
```

Non usare `--resume` dopo aver cambiato `N` e non riutilizzare demo con una diversa
morfologia o un diverso ordine dei giunti.

## 8.1 Variante completa: controllo cartesiano con IK

La variante precedente lascia alla policy il controllo di ogni giunto. Con IK la
policy sceglie invece la velocita' cartesiana desiderata del TCP:

```text
action = [vx, vy, vz] in [-1, 1]
```

Un controller di inverse kinematics converte questa velocita' in `dq` articolari. E'
una buona scelta quando vuoi semplificare l'esplorazione e sul robot reale esiste gia'
un controller cartesiano affidabile.

L'IK non evita automaticamente gli ostacoli. Risolve il problema "come muovo i giunti
per spostare il TCP in questa direzione", non "qual e' la traiettoria collision-free".
La policy deve ancora imparare i waypoint cartesiani, mentre collision checker e safety
supervisor restano obbligatori.

### Quando conviene

| Situazione | Controllo consigliato |
|---|---|
| reaching con spazio ampio | IK cartesiana a 3 azioni |
| molti DOF ma traiettoria TCP semplice | IK cartesiana |
| ostacoli che richiedono una specifica posa del gomito | velocita' articolari |
| robot ridondante con solver dotato di obiettivo secondario | IK + posture/null-space |
| orientamento TCP indispensabile | IK 6D o planner esterno |

Con una action puramente 3D la policy non controlla la configurazione ridondante del
gomito. Se due configurazioni raggiungono lo stesso TCP ma una collide, il solver deve
avere un criterio secondario deterministico oppure devi aggiungere un comando di
postura. Non nascondere questo problema aumentando soltanto la collision penalty.

Con 3 giunti, limita i `TargetSpawns` al workspace realmente raggiungibile e non
richiedere un orientamento TCP che il robot non puo' controllare. Con 7 o 8 giunti la
ridondanza aumenta: la DLS restituisce una soluzione locale a norma minima, ma per
scegliere deliberatamente gomito e postura serve un termine di null-space oppure il
controllo articolare della sezione precedente.

### IK differenziale riusabile con entrambi i backend

Per il tutorial usiamo una damped least-squares IK posizionale. Per ogni giunto
revolute, la colonna del Jacobiano lineare e':

```text
J_i = axis_world_i x (tcp_world - joint_origin_world)
```

e il comando articolare e':

```text
dq = J^T * inverse(J * J^T + lambda^2 * I) * velocity_tcp
```

Questa forma richiede l'inversione di una matrice `3 x 3`, indipendentemente dal numero
di giunti. Funziona con pivot `Node3D` e con bone `Skeleton3D`, a condizione che il bone
origin generato dall'importer coincida con l'origine del joint URDF.

Crea `godot/agents/RobotArm/robot_arm_ik.gd`:

```gdscript
extends RobotArmBody
class_name RobotArmIKBody

@export_category("Cartesian IK")
@export var max_cartesian_speed_m_s := 0.12
@export_range(0.001, 0.5, 0.001) var damping := 0.04
@export_range(0.05, 1.0, 0.05) var joint_speed_ratio := 0.8

var _cartesian_action := Vector3.ZERO
var _previous_cartesian_action: Array[float] = [0.0, 0.0, 0.0]
var _ik_residual_m_s := 0.0


func apply_action(action:Variant) -> Variant:
    if (typeof(action) == TYPE_STRING or typeof(action) == TYPE_STRING_NAME) and str(action) == "manual":
        return apply_manual_action()

    manual_control = false
    var values := agent.decode_continuous_action(action)
    _cartesian_action = Vector3(
        clampf(float(values[0]), -1.0, 1.0) if values.size() > 0 else 0.0,
        clampf(float(values[1]), -1.0, 1.0) if values.size() > 1 else 0.0,
        clampf(float(values[2]), -1.0, 1.0) if values.size() > 2 else 0.0
    )
    _previous_cartesian_action = [
        _cartesian_action.x,
        _cartesian_action.y,
        _cartesian_action.z,
    ]
    return _previous_cartesian_action.duplicate()


func apply_manual_action() -> Array:
    return apply_action([
        Input.get_axis("tcp_x_negative", "tcp_x_positive"),
        Input.get_axis("tcp_y_negative", "tcp_y_positive"),
        Input.get_axis("tcp_z_negative", "tcp_z_positive"),
    ])


func reset_all(original_transform:Variant, reset_rewards := true) -> void:
    _cartesian_action = Vector3.ZERO
    _previous_cartesian_action = [0.0, 0.0, 0.0]
    _ik_residual_m_s = 0.0
    super.reset_all(original_transform, reset_rewards)


func get_previous_action_observation() -> Array:
    return _previous_cartesian_action.duplicate()


func get_ik_residual_penalty() -> float:
    var scale := maxf(max_cartesian_speed_m_s, 0.000001)
    return -clampf(_ik_residual_m_s / scale, 0.0, 1.0)


func get_control_input(input_name:String) -> float:
    if input_name.begins_with("tcp_velocity_"):
        var index := int(input_name.trim_prefix("tcp_velocity_"))
        if index >= 0 and index < _previous_cartesian_action.size():
            return _previous_cartesian_action[index]
        return 0.0
    return super.get_control_input(input_name)


func _physics_process(delta:float) -> void:
    if not _training_active or _terminal:
        return

    var velocity_base := _cartesian_action * max_cartesian_speed_m_s
    var velocity_world := global_transform.basis.orthonormalized() * velocity_base
    var joint_speeds := _solve_damped_least_squares(velocity_world)

    for index in range(get_joint_count()):
        var old_angle := _joint_angles[index]
        var max_speed := deg_to_rad(float(joint_max_speed_degrees[index]))
        var requested_speed := joint_speeds[index] if index < joint_speeds.size() else 0.0
        requested_speed = clampf(requested_speed, -max_speed, max_speed)
        var min_angle := deg_to_rad(float(joint_min_degrees[index]))
        var max_angle := deg_to_rad(float(joint_max_degrees[index]))
        _joint_angles[index] = clampf(old_angle + requested_speed * delta, min_angle, max_angle)
        _joint_velocities[index] = (_joint_angles[index] - old_angle) / maxf(delta, 0.000001)
        _commands[index] = clampf(requested_speed / maxf(max_speed, 0.000001), -1.0, 1.0)

    _apply_joint_pose()
    _update_success_state()


func _solve_damped_least_squares(velocity_world:Vector3) -> Array[float]:
    var columns: Array[Vector3] = []
    var tcp_position := end_effector.global_position

    for index in range(get_joint_count()):
        var joint_transform := _joint_world_transform(index)
        var axis_world := (joint_transform.basis * joint_axes[index]).normalized()
        columns.append(axis_world.cross(tcp_position - joint_transform.origin))

    var lambda_squared := damping * damping
    var matrix_x := Vector3(lambda_squared, 0.0, 0.0)
    var matrix_y := Vector3(0.0, lambda_squared, 0.0)
    var matrix_z := Vector3(0.0, 0.0, lambda_squared)
    for column in columns:
        matrix_x += column * column.x
        matrix_y += column * column.y
        matrix_z += column * column.z

    var system := Basis(matrix_x, matrix_y, matrix_z)
    var correction := system.inverse() * velocity_world
    var speeds: Array[float] = []
    var achieved_velocity := Vector3.ZERO

    for index in range(columns.size()):
        var speed_limit := (
            deg_to_rad(float(joint_max_speed_degrees[index])) * joint_speed_ratio
        )
        var speed := clampf(columns[index].dot(correction), -speed_limit, speed_limit)
        speeds.append(speed)
        achieved_velocity += columns[index] * speed

    _ik_residual_m_s = (velocity_world - achieved_velocity).length()
    return speeds


func _joint_world_transform(index:int) -> Transform3D:
    if armature_backend == ArmatureBackend.SKELETON_3D:
        var bone_pose := skeleton.get_bone_global_pose(_bone_indices[index])
        return skeleton.global_transform * bone_pose
    return joints[index].global_transform
```

Duplica `robot_arm.tscn` come `robot_arm_ik.tscn` e collega `robot_arm_ik.gd` al nodo
radice `RobotArm`. Non aggiungere lo script IK come secondo nodo di controllo: deve
sostituire lo script della radice, ereditandone il comportamento.

Lo script eredita reset, observation articolari, collisioni, limiti, target e reward da
`RobotArmBody`. Il damping evita velocita' enormi vicino alle singolarita'; non elimina
pero' singolarita', target irraggiungibili o collisioni.

### Configurazione Metis per IK

Sostituisci `JointVelocity` con un solo `ContinuousAction`:

```text
node = TcpVelocity
action_name = tcp_velocity
size = 3
low = -1.0
high = 1.0
```

Mantieni le stesse quattro observation source. Cambia soltanto la dimensione di
`previous_action`, che ora contiene tre valori indipendentemente da `N`:

```text
obs_dim = N joint positions + N joint velocities + 3 target error + 3 previous action
obs_dim = 2*N + 6
```

Quindi ottieni `12` observation con 3 giunti, `20` con 7 giunti e `22` con 8 giunti.
L'action space IK resta sempre di dimensione 3.

Python non richiede modifiche: il bridge descrive automaticamente l'action space
continuo a tre valori. Lo scenario deve pero' istanziare `robot_arm_ik.tscn` al posto di
`robot_arm.tscn`.

Nel nodo `Smoothness` usa:

```text
input_names = [tcp_velocity_0, tcp_velocity_1, tcp_velocity_2]
free_delta = 0.08
penalty_scale = -0.005
```

Puoi aggiungere una reward diagnostica `FunctionRewardComponent`:

```text
term_name = ik_residual
node_caller = RobotArm
method_name = get_ik_residual_penalty
weight = 0.01
```

Tienila piccola: una policy non deve preferire lo zero, che ha residuo perfetto, al
raggiungimento del target. Collisione, successo, progresso, tempo e stall restano quelli
delle sezioni successive.

Per la guida manuale aggiungi all'Input Map:

```text
tcp_x_negative / tcp_x_positive
tcp_y_negative / tcp_y_positive
tcp_z_negative / tcp_z_positive
```

### Solver IK nativi di Godot 4.6

Se usi uno `Skeleton3D`, Godot 4.6 include `CCDIK3D`, `FABRIK3D` e `JacobianIK3D` come
`SkeletonModifier3D`. Il vecchio `SkeletonIK3D` e' deprecato e non e' consigliato per
una nuova implementazione.

- `FABRIK3D` e' preciso per catene semplici senza limiti complessi;
- `CCDIK3D` e' una scelta migliore quando servono assi e limitazioni per joint;
- `JacobianIK3D` produce movimenti morbidi, ma non e' un collision planner.

Per provarli, aggiungi il solver come figlio dello `Skeleton3D`, configura una setting
con root bone, end bone e `IKTarget`, imposta l'asse di rotazione di ogni joint, abilita
`deterministic` e assegna limitazioni coerenti con l'URDF. Non lasciare che il modifier
e `RobotArmBody` scrivano contemporaneamente sugli stessi bone.

Per collegare un solver nativo a Metis serve un adapter che, dopo ogni soluzione:

1. estrae dal bone pose gli angoli articolari nello stesso ordine URDF;
2. applica limiti hard e velocita' massime;
3. aggiorna `_joint_angles` e `_joint_velocities` usati dalle observation;
4. rifiuta la soluzione se i `SafetyVolumes` entrano nella fascia vietata.

La DLS proposta sopra evita questa doppia autorita' e rimane utilizzabile anche con una
catena `Node3D`. Sul robot reale puoi sostituirla con MoveIt, Pinocchio, KDL o il solver
del produttore, mantenendo invariato il contratto cartesiano della policy.

### Training della variante IK

Non riutilizzare checkpoint, replay o demo della variante articolare: `action_size` e
`obs_dim` sono cambiati. Avvia un run nuovo:

```bash
python/.venv/bin/python python/train.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/robotarms/XarmScenario.tscn \
  --num-envs 8 \
  --num-episodes 4000 \
  --max-steps-per-episode 300 \
  --physics-frames-per-step 3 \
  --batch-size 256 \
  --replay-warmup 20000 \
  --checkpoint-dir checkpoints/robot_arm_reaching_ik_sac_v1 \
  --actor-weights-path robot_arm_reaching_ik_sac_actor_v1.weights.h5 \
  --critic1-weights-path robot_arm_reaching_ik_sac_critic1_v1.weights.h5 \
  --critic2-weights-path robot_arm_reaching_ik_sac_critic2_v1.weights.h5 \
  --headless
```

Le dimostrazioni TD3+BC restano consigliate, ma devono registrare le tre velocita'
cartesiane realmente applicate. Valuta anche `ik_residual`, frequenza di saturazione dei
giunti, distanza dai limiti e collisioni del gomito: il solo errore TCP non basta.

## 9. Configura le reward locali

Le reward locali descrivono qualita' del controllo che appartengono al braccio.

`JointMotion`, `FunctionRewardComponent`:

```text
term_name = joint_motion
node_caller = RobotArm
method_name = get_joint_motion_penalty
weight = 0.003
```

`JointLimit`, `FunctionRewardComponent`:

```text
term_name = joint_limit
node_caller = RobotArm
method_name = get_joint_limit_penalty
weight = 0.02
```

`Smoothness`, `ActionSmoothnessPenaltyReward`:

```text
input_names = [joint_velocity_0, ..., joint_velocity_(N-1)]
free_delta = 0.08
penalty_scale = -0.005
normalize_by_input_count = true
```

Nell'Inspector inserisci i nomi reali, senza la notazione `(N-1)`. Per esempio:

```text
N=3 -> [joint_velocity_0, joint_velocity_1, joint_velocity_2]
N=8 -> [joint_velocity_0, ..., joint_velocity_7]
```

`Time`, `StepPenaltyReward`:

```text
penalty = -0.002
```

`JointMotion` scoraggia traiettorie inutilmente lunghe, `Smoothness` evita inversioni
brusche e `Time` impedisce alla policy di restare ferma. Tieni questi pesi piccoli:
raggiungere il target senza collisioni deve dominare. La distanza dagli ostacoli non
viene premiata continuamente; il margine e' gia' rappresentato dalle collision shape
conservative. Se aggiungerai un collision checker capace di calcolare distanze esatte,
potrai usarle come reward privilegiata senza inserirle nelle observation.

## 10. Costruisci lo scenario

Usa questo albero:

```text
RobotArmReachingScenario              Node3D, script scenario
├── BridgeServer                      bridge_server.gd
├── ScenarioController                ScenarioController.gd
│   ├── ProgressProvider              MethodProgressProvider.gd
│   ├── ScenarioEventSystem           ScenarioEventSystem.gd
│   │   ├── GoalReached               ManualScenarioEventSource.gd
│   │   └── Collision                 ManualScenarioEventSource.gd
│   └── ScenarioRewardSystem          ScenarioRewardSystem.gd
│       ├── Progress                  ProgressDeltaScenarioReward.gd
│       ├── GoalReward                EventScenarioReward.gd
│       ├── CollisionPenalty          EventScenarioReward.gd
│       └── NoProgress                ProgressStallScenarioReward.gd
├── RobotArm                          istanza di robot_arm.tscn
├── Target                            Marker3D o MeshInstance3D
├── Workcell                          Node3D con geometria calibrata
│   ├── Table                      StaticBody3D, gruppo robot_obstacle
│   └── Fixtures                   StaticBody3D, gruppo robot_obstacle
├── TargetSpawns                      Node3D con Marker3D validati
├── Camera3D
└── WorldEnvironment
```

Configura:

```text
BridgeServer.controller_path = ../ScenarioController
ScenarioController.controlled_agents = [../RobotArm]
ScenarioController.physics_frames_per_step = 3
ScenarioController.max_steps = 300
ScenarioController.deactivate_done_agents = true
```

A 60 Hz, tre frame fisici per step producono una frequenza decisionale di 20 Hz. E'
un buon primo valore per un controller articolare. Il comando Python puo' sovrascriverlo
con `--physics-frames-per-step`.

`ProgressProvider`:

```text
source_path = vuoto
method_name = get_progress
pass_agent_to_source = true
```

Quando `source_path` e' vuoto, il provider chiama `get_progress()` direttamente sul
corpo controllato.

Eventi:

```text
GoalReached.event_name = target_reached
GoalReached.terminal_reason = target_reached

Collision.event_name = collision
Collision.terminal_reason = collision
```

Reward di scenario:

```text
Progress.term_name = target_progress
Progress.progress_reward_scale = 10.0
Progress.backward_penalty_scale = 12.0

GoalReward.event_name = target_reached
GoalReward.reward = 30.0

CollisionPenalty.event_name = collision
CollisionPenalty.reward = -20.0

NoProgress.terminate_on_stalled_progress = true
NoProgress.stalled_progress_window_steps = 120
NoProgress.stalled_progress_min_delta = 0.01
NoProgress.stalled_progress_penalty = -5.0
```

Il totale per un agente diventa:

```text
10 * progresso verso il target
- 12 * regresso
- vicinanza ai limiti articolari
- movimento articolare non necessario
- cambi bruschi di comando
- costo temporale
+ 30 al successo
- 20 alla collisione
- 5 allo stall
```

## 11. Reset e curriculum nella cella reale

Inizia con target pre-validati da un planner o da prove manuali. Gli ostacoli della
cella non cambiano tra le fasi: rimuoverli nei primi episodi insegnerebbe scorciatoie
che nel mondo reale non esistono. Il curriculum amplia invece il set di target, riduce
la tolleranza finale e introduce piccoli offset nella posa iniziale.

Collega questo script al nodo radice:

```gdscript
extends Node3D

@export var target_spawns: Array[Marker3D] = []
@export var easy_target_count := 4

@onready var controller: ScenarioController = $ScenarioController
# Entrambi i backend implementano lo stesso contratto, ma non ereditano dalla
# stessa classe GDScript.
@onready var arm = $RobotArm
@onready var target: Node3D = $Target
@onready var goal_event = $ScenarioController/ScenarioEventSystem/GoalReached
@onready var collision_event = $ScenarioController/ScenarioEventSystem/Collision

var _training_episode := 0


func _ready() -> void:
    arm.target = target
    arm.target_reached.connect(_on_target_reached)
    arm.obstacle_collision.connect(_on_obstacle_collision)
    controller.scenario_configured.connect(_on_scenario_configured)
    controller.episode_reset_started.connect(_on_episode_reset_started)


func _on_scenario_configured(config:Dictionary) -> void:
    _training_episode = int(config.get("training_episode", _training_episode))


func _on_episode_reset_started(seed:int) -> void:
    if target_spawns.is_empty():
        push_error("RobotArmReachingScenario requires at least one TargetSpawn")
        return
    var rng := RandomNumberGenerator.new()
    rng.seed = seed

    var target_pool_size := target_spawns.size()
    var joint_jitter_degrees := 0.0

    if _training_episode < 300:
        target_pool_size = maxi(1, mini(easy_target_count, target_spawns.size()))
        arm.success_distance = 0.08
    elif _training_episode < 800:
        arm.success_distance = 0.05
    elif _training_episode < 1500:
        joint_jitter_degrees = 2.0
        arm.success_distance = 0.04
    else:
        joint_jitter_degrees = 5.0
        arm.success_distance = 0.03

    var target_index := rng.randi_range(0, maxi(target_pool_size - 1, 0))
    target.global_transform = target_spawns[target_index].global_transform
    arm.set_reset_joint_offsets(_sample_joint_offsets(rng, joint_jitter_degrees))


func _sample_joint_offsets(rng:RandomNumberGenerator, max_degrees:float) -> Array[float]:
    var offsets: Array[float] = []
    offsets.resize(arm.get_joint_count())
    for index in range(offsets.size()):
        offsets[index] = deg_to_rad(rng.randf_range(-max_degrees, max_degrees))
    return offsets


func _on_target_reached() -> void:
    goal_event.trigger(str(arm.name))


func _on_obstacle_collision() -> void:
    collision_event.trigger(str(arm.name))
```

Ogni ostacolo deve essere uno `StaticBody3D` o `Area3D`, appartenere al gruppo
`robot_obstacle` e usare un layer incluso in `environment_collision_mask`. Le sue
collision shape devono provenire da CAD o misure reali, poi essere ingrandite del
margine di sicurezza stabilito. Per un URDF con collision shape valide non devi
compilare manualmente `safety_volumes` per rilevare gli ostacoli esterni.

Mantieni un secondo insieme di target mai usati dal curriculum. Servira' per misurare
interpolazione nello stesso spazio operativo. Se la disposizione fisica della cella
cambia, aggiornala anche nel modello geometrico: senza sensori il robot non puo'
dedurla da solo.

## 12. Valida la scena prima del training

Avvia un rollout casuale:

```bash
python/.venv/bin/python python/tools/random_rollout.py \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/robot_arm_reaching/robot_arm_reaching_scenario.tscn \
  --steps 400 \
  --print-reward-terms \
  --no-headless
```

Prima di allenare verifica:

- `action_type=continuous`;
- `action_size=N` per il controllo articolare oppure `action_size=3` per IK posizionale;
- `obs_dim=3*N+3` per il controllo articolare oppure `obs_dim=2*N+6` per IK;
- nessun giunto salta al primo step;
- segni e assi delle azioni corrispondono al robot reale;
- collisione produce `terminal_reason=collision` e reward negativa;
- successo richiede di restare sul target e produce `target_reached`;
- il reset non lascia collisioni nello stato precedente;
- target e posa iniziale sono deterministici a seed uguale;
- una stessa configurazione produce lo stesso TCP con backend `NODE_CHAIN` e
  `SKELETON_3D`;
- almeno 10 configurazioni note coincidono con la forward kinematics URDF entro la
  tolleranza scelta.

Prova anche azioni costanti su un solo giunto. E' il modo piu' rapido per scoprire un
ordine o un segno sbagliato prima che la rete impari a compensarlo.

## 13. Baseline online con SAC

SAC e' una buona baseline per verificare che contratto, reward e terminalita' siano
corretti. Gestisce naturalmente azioni continue e non richiede demo, ma esplorare da
zero vicino a ostacoli e' inefficiente. Esegui prima una prova breve; per il training
principale usa la pipeline con planner della sezione successiva.

```bash
python/.venv/bin/python python/train.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/robot_arm_reaching/robot_arm_reaching_scenario.tscn \
  --num-envs 8 \
  --num-episodes 4000 \
  --max-steps-per-episode 300 \
  --physics-frames-per-step 3 \
  --batch-size 256 \
  --replay-warmup 20000 \
  --replay-capacity 500000 \
  --random-exploration-episodes 40 \
  --checkpoint-every 25 \
  --checkpoint-dir checkpoints/robot_arm_reaching_sac_v1 \
  --actor-weights-path robot_arm_reaching_sac_actor_v1.weights.h5 \
  --critic1-weights-path robot_arm_reaching_sac_critic1_v1.weights.h5 \
  --critic2-weights-path robot_arm_reaching_sac_critic2_v1.weights.h5 \
  --headless
```

Non usare `--multi-agent`: qui c'e' un braccio per environment. Gli otto processi Godot
raccolgono gia' esperienza in parallelo. Replicare molti bracci nella stessa scena ha
senso soltanto se sono completamente indipendenti e non condividono ostacoli.
`--num-envs 8` indica otto simulatori paralleli e non ha alcun rapporto con il numero
di giunti del robot.

I primi criteri da osservare non sono soltanto la reward media:

```text
success_rate
collision_rate
progress medio e massimo
step medi al successo
violazioni del margine di collisione
percentuale di episodi terminati per stall
```

La reward puo' salire anche per una policy prudente che non completa il task. La
percentuale di successo sui target tenuti fuori e' la metrica principale.

## 14. Percorso consigliato: planner, dimostrazioni e TD3+BC

Un planner collision-aware deve essere la baseline e l'esperto. A parita' di URDF e
geometria della cella, genera una traiettoria articolare per ogni coppia
`configurazione iniziale -> target`, poi eseguila dentro Godot alla stessa frequenza di
controllo della policy. In questo modo Metis calcola observation, reward, collisioni e
`next_obs` con la sua implementazione e il dataset non contiene preprocessing diversi.

Per ogni traiettoria:

1. il planner trova una sequenza `q[0..T]` senza collisioni nella planning scene;
2. limita velocita' e accelerazioni come sul controller reale;
3. ricava l'azione normalizzata `a_t = (q[t+1] - q[t]) / (dt * max_speed)`;
4. riproduce `a_t` nello scenario Godot;
5. salva le transizioni Metis `obs, actions, rewards, next_obs, dones`.

Per la variante IK, il punto 3 cambia: ricava
`a_t = (tcp[t+1] - tcp[t]) / (dt * max_cartesian_speed)` e lascia che sia
`RobotArmIKBody` a produrre le velocita' articolari. Non salvare `dq` in un dataset che
dichiara `action_size=3`, altrimenti il behavior cloning impara un contratto sbagliato.

MoveIt puo' fornire planning scene, self-collision e controllo geometrico delle
traiettorie. Il planner resta anche il riferimento con cui confrontare durata,
lunghezza articolare e tasso di successo. Una policy RL puo' approssimarlo piu'
velocemente e migliorare il costo scelto, ma non offre una prova matematica di
ottimalita' globale.

Se non hai ancora il collegamento al planner, puoi creare un primo dataset manuale.

Per imitare la pipeline OpenArmX, aggiungi nell'Input Map:

```text
joint_0_negative / joint_0_positive
...
joint_(N-1)_negative / joint_(N-1)_positive
```

La notazione `(N-1)` e' descrittiva: per 3 giunti l'ultima coppia e' `joint_2_*`, per
7 giunti e' `joint_6_*`, per 8 giunti e' `joint_7_*`.

Collega tasti, gamepad o un dispositivo di teleoperazione. `apply_manual_action()`
restituisce il vettore realmente applicato, quindi `recorder.py` puo' salvarlo.
Nella variante IK usa gli input `tcp_*` definiti nella sezione 8.1; il recorder
riconoscera' automaticamente l'action space continuo a tre valori.

```bash
python/.venv/bin/python python/recorder.py \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/robot_arm_reaching/robot_arm_reaching_scenario.tscn \
  --output demos/robot_arm_reaching_v1.npz \
  --agent-id RobotArm \
  --episodes 50 \
  --max-steps 300 \
  --step-delay 0.05 \
  --no-headless
```

Poi avvia TD3+BC:

```bash
python/.venv/bin/python python/train.py \
  --algorithm td3_bc \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/robot_arm_reaching/robot_arm_reaching_scenario.tscn \
  --num-envs 8 \
  --num-episodes 4000 \
  --max-steps-per-episode 300 \
  --physics-frames-per-step 3 \
  --batch-size 256 \
  --replay-warmup 5000 \
  --replay-capacity 500000 \
  --demo-path demos/robot_arm_reaching_v1.npz \
  --demo-bc-epochs 20 \
  --demo-bc-weight-start 1.0 \
  --demo-bc-weight-end 0.05 \
  --demo-bc-decay-updates 150000 \
  --action-smoothing 0.0 \
  --checkpoint-dir checkpoints/robot_arm_reaching_td3_bc_v1 \
  --actor-weights-path robot_arm_reaching_td3_bc_actor_v1.weights.h5 \
  --critic-weights-path robot_arm_reaching_td3_bc_critic1_v1.weights.h5 \
  --critic2-weights-path robot_arm_reaching_td3_bc_critic2_v1.weights.h5 \
  --headless
```

Le demo devono coprire piu' target, configurazioni iniziali e soluzioni del gomito nella
stessa cella. Cinquanta traiettorie quasi identiche accelerano l'overfitting invece del
task. Conserva nel dataset anche traiettorie del planner non ottime ma sicure: TD3+BC
potra' migliorarne tempo e fluidita' tramite reward.

Il dataset deve rispettare esattamente il formato descritto in
[`manual_demonstrations.md`](../guides/manual_demonstrations.md).

## 15. Esegui e valuta la policy

Il trainer salva `policy.keras` e `policy.json` nella directory dei checkpoint.

```bash
python/.venv/bin/python python/run.py \
  --algorithm auto \
  --load-from policy \
  --policy-path checkpoints/robot_arm_reaching_td3_bc_v1 \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/robot_arm_reaching/robot_arm_reaching_scenario.tscn \
  --episodes 50 \
  --max-steps 300 \
  --execution-mode realtime \
  --no-headless
```

Il comando precedente e' una valutazione: dopo ogni successo o fallimento avvia un nuovo
episodio. Per spostare il target a mano e verificare che la policy continui a inseguirlo senza
resettare il robot, usa `--continue-after-success`, `--infinite` e `--no-time-limit`:

```bash
python/.venv/bin/python python/run.py \
  --algorithm auto \
  --load-from policy \
  --policy-path checkpoints/robot_arm_reaching_td3_bc_v1 \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/robotarms/XarmScenario.tscn \
  --execution-mode realtime \
  --continue-after-success \
  --infinite \
  --no-time-limit \
  --no-initial-reset \
  --no-reset \
  --no-headless
```

In questa modalita' `target_reached` produce ancora l'evento di successo, ma non termina
l'episodio. Quando il target viene spostato oltre la distanza di riarmo, il rilevatore torna
attivo e il braccio continua a ricevere osservazioni e azioni. Collisioni e condizioni di
sicurezza rimangono terminali. Durante il training, dove l'opzione non viene passata,
`target_reached` continua a terminare normalmente l'episodio.

`--no-initial-reset` evita anche il reset fisico iniziale: observation, reward, eventi e
progress vengono inizializzati rispetto alla posa corrente, senza spostare il braccio o il
target. `--no-reset` conserva la posa anche ai terminali successivi. Una collisione chiude
ancora l'episodio logico e viene registrata nei risultati, ma con `--infinite` Metis ripulisce
il terminale e continua subito l'inferenza dalla stessa configurazione fisica. Se il robot
rimane a contatto con l'ostacolo, la collisione puo' naturalmente essere rilevata di nuovo.

Valuta almeno quattro suite:

1. seed di training, per controllare che il task sia stato appreso;
2. target tenuti fuori dal training ma appartenenti alla stessa cella;
3. parametri perturbati entro l'incertezza misurata di latenza, velocita' e calibrazione;
4. shadow mode con stato articolare e target provenienti dal robot reale.

Non scegliere la policy solo dalla reward. Promuovila se rispetta soglie separate, per
esempio successo maggiore del 95%, nessuna collisione in una suite estesa e distanza
minima verificata dal collision checker sopra il margine definito.

## 16. Domain randomization per il sim-to-real

Dopo che il task funziona senza randomizzazione, varia gradualmente:

- zero articolare e segno della calibrazione entro l'errore misurato;
- velocita' massima, guadagno e deadband degli attuatori;
- ritardo di comando e una piccola probabilita' di action hold;
- rumore e ritardo degli encoder;
- errore misurato di lunghezza dei link e posa TCP;
- tolleranza misurata nella posa di base e ostacoli;
- posizione del target;
- durata del timestep entro un intervallo realistico.

Randomizzare valori irrealistici non rende automaticamente robusta la policy. Misura
prima il robot reale e usa intervalli che comprendano l'incertezza osservata.

Non spostare gli ostacoli oltre la loro tolleranza reale soltanto per rendere il task
piu' difficile: in assenza di observation sulla geometria creeresti piu' ambienti
indistinguibili che richiedono azioni diverse. Se la cella ha configurazioni note e
selezionabili, aggiungi alla observation un identificatore o descrittore esplicito del
layout e addestra su ciascuna configurazione.

## 17. Portare la policy sul robot vero

Il runtime reale deve ricostruire lo stesso vettore del training:

```text
q encoder
-> normalizzazione con gli stessi limiti
dq encoder
-> divisione per le stesse velocita' massime
target reale
-> trasformazione nel frame base e divisione per workspace_scale
azione precedente
-> ultimo comando normalizzato realmente applicato, articolare o cartesiano
```

Poi:

```text
observation float32
-> policy.keras oppure policy.tflite
-> azione normalizzata
-> joint scaling, se la policy e' articolare
   oppure IK validata, se la policy e' cartesiana
-> safety supervisor
-> controller articolare
```

Una policy IK esportata non contiene il solver: produce soltanto `[vx, vy, vz]`. Sul
robot devi eseguire un IK equivalente e deterministico, con gli stessi frame, limiti e
criteri di postura usati durante il training. Cambiare solver puo' cambiare il gomito e
quindi trasformare una traiettoria TCP valida in una collisione.

Il file Keras contiene la rete che trasforma un vettore numerico in azioni. Non contiene
il codice che legge gli encoder, ordina i giunti, trasforma il target nel frame base o
applica scaling e limiti. `policy.json` conserva il contratto e deve essere distribuito
insieme al modello e al codice di preprocessing.

Per esportare TFLite:

```bash
python/.venv/bin/python python/export.py \
  --policy checkpoints/robot_arm_reaching_td3_bc_v1 \
  --format tflite \
  --tflite-quantization float16
```

Prima di comandare il robot esegui tre passaggi:

1. **shadow mode**: calcola azioni ma non inviarle, confrontandole con un operatore;
2. **bassa velocita'**: workspace vuoto, limiti ridotti, e-stop pronto;
3. **ostacoli morbidi e supervisione**: aumenta difficolta' soltanto dopo test ripetuti.

La safety layer deve essere indipendente dalla policy e almeno:

- troncare comandi fuori dai limiti articolari;
- limitare velocita' e accelerazione;
- fermare il robot su timeout della policy o degli encoder;
- verificare collisioni previste sul comando successivo usando lo stesso URDF e la
  planning scene calibrata;
- imporre una workspace fence;
- offrire e-stop hardware e arresto da supervisore.

RL decide come muoversi. Non deve essere l'unico componente che decide se un movimento
e' sicuro. Senza sensori di prossimita', la cella reale deve essere chiusa o interlocked:
un ostacolo inatteso non compare ne' nelle observation ne' nella planning scene.

## 18. Quando passare a una simulazione piu' robotica

Resta su questa versione Godot se il task usa:

- reaching senza contatto intenzionale;
- comando di posizione o velocita';
- ostacoli statici, noti e calibrati;
- stato articolare disponibile dal controller.

Valuta MuJoCo, robosuite o un secondo backend se vuoi:

- torque control;
- presa e manipolazione con attrito;
- contatti ripetuti e forze;
- identificazione dinamica accurata;
- controller operational-space gia' validati.

Metis puo' comunque restare il livello di organizzazione dell'esperimento, ma non e'
utile forzare Godot a essere il motore migliore per ogni problema.

## 19. Ordine di lavoro consigliato

Procedi in questo ordine e non saltare direttamente al training lungo:

1. importa un solo braccio e valida cinematiche, assi e limiti;
2. confronta backend `Node3D` o `Skeleton3D` con la forward kinematics URDF;
3. calibra base, TCP, collision shape e ostacoli della cella;
4. muovi ogni giunto manualmente entro limiti ridotti;
5. misura una baseline IK/planner collision-aware;
6. scegli e congela il contratto tra azioni articolari e azioni cartesiane IK;
7. genera e riproduci in Godot dimostrazioni del planner;
8. valida reward e terminalita' con un breve SAC;
9. addestra TD3+BC e confrontalo con planner e SAC;
10. amplia gradualmente target e pose iniziali, senza rimuovere gli ostacoli reali;
11. aggiungi domain randomization misurata;
12. valida in shadow mode sul robot;
13. abilita movimento reale a bassa velocita' con safety supervisor.

## Riferimenti

- [OpenArmX robosuite RL](https://github.com/openarmx/openarmx_robosuite_RL)
- [Documentazione OpenArm](https://docs.openarm.dev/)
- [Descrizione URDF/xacro OpenArm](https://docs.openarm.dev/api-reference/description/)
- [robosuite](https://robosuite.ai/docs/overview.html)
- [Skeleton3D in Godot](https://docs.godotengine.org/en/stable/classes/class_skeleton3d.html)
- [CCDIK3D in Godot](https://docs.godotengine.org/en/stable/classes/class_ccdik3d.html)
- [FABRIK3D in Godot](https://docs.godotengine.org/en/stable/classes/class_fabrik3d.html)
- [JacobianIK3D in Godot](https://docs.godotengine.org/en/stable/classes/class_jacobianik3d.html)
- [Plugin di importazione Godot](https://docs.godotengine.org/en/stable/tutorials/plugins/editor/import_plugins.html)
- [MoveIt planning scene](https://moveit.picknik.ai/main/doc/examples/planning_scene/planning_scene_tutorial.html)
- [Dimostrazioni manuali in Metis](../guides/manual_demonstrations.md)
- [Esportare una policy Metis](../guides/esportare_policy.md)
