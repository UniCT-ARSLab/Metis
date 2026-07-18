# Metis

**Modular Environment for Training Intelligent Systems**

Metis e' un piccolo framework di reinforcement learning che mette insieme Godot,
Gymnasium e TensorFlow/Keras.

Nasce da un'esigenza pratica: poter costruire un agente e il suo mondo in Godot, usando
scene, fisica e Inspector, senza dover scrivere un nuovo programma Python per ogni
esperimento. Lo scenario descrive cosa osserva l'agente, quali azioni puo' compiere,
come viene premiato e quando termina un episodio. Python legge questo contratto e si
occupa del training.

Il progetto e' ancora in evoluzione, ma non e' piu' soltanto una demo: comprende
training discreto, continuo e ibrido, scenari con piu' agenti, raccolta asincrona,
checkpoint, dimostrazioni manuali, self-play ed esportazione dei modelli.

La documentazione completa si trova in [docs/README.md](docs/README.md). Per costruire
subito qualcosa, il punto di partenza migliore e'
[Creare un nuovo scenario e agente](docs/tutorials/tutorial_nuovo_scenario_agente_rl.md).

## Cosa offre oggi

| Area | Supporto |
|---|---|
| Simulazione | Godot 4, scene 2D e 3D, fisica e reset deterministici |
| API ambiente | wrapper Gymnasium generico e bridge TCP JSON con `TCP_NODELAY` |
| Action space | discreti, continui, multi-discreti e ibridi |
| Observation | metodi, property, cinematica, velocita', RayCast, target, team e Path3D |
| Reward | componenti locali per agente e componenti globali di scenario |
| Algoritmi | DQN, PPO, DDPG, DDPG+BC, DDPGfD, TD3, TD3+BC e SAC |
| Raccolta | uno o piu' environment, collector sincrono o asincrono |
| Multi-agent | transizioni separate per agente e parameter sharing |
| Competizione | self-play simultaneo e opponent pool con policy storiche |
| Dimostrazioni | registrazione manuale, replay prefill e behavior cloning |
| Salvataggio | checkpoint completi, replay buffer, best checkpoint e policy Keras |
| Esecuzione | lockstep riproducibile oppure realtime per osservare la policy |
| Esportazione | Keras, TensorFlow Lite e ONNX opzionale |
| Piattaforme | Linux CPU/CUDA e macOS Apple Silicon con TensorFlow Metal |

Metis usa trainer TensorFlow/Keras propri. Gymnasium definisce l'interfaccia
dell'ambiente, ma il progetto non dipende da Stable-Baselines3.

## Come e' diviso il lavoro

Godot e Python hanno responsabilita' diverse e abbastanza nette.

**Godot possiede il mondo:**

- fisica, collisioni, oggetti e regole del gioco;
- corpi controllati e sensori;
- observation e action space;
- reward, eventi e condizioni terminali;
- reset, randomizzazione e curriculum dello scenario.

**Python possiede l'apprendimento:**

- avvio delle istanze Godot;
- adattamento Gymnasium;
- raccolta di replay o rollout;
- reti Keras, optimizer e aggiornamenti;
- checkpoint, valutazione e inferenza.

Tra i due c'e' `BridgeServer`, che espone `spec`, `configure`, `reset` e `step` via
TCP. Il server vive in Godot perche' e' Godot ad avere lo stato autorevole della
simulazione. Python resta il client che orchestra gli environment e il learner.

```text
python/train.py
    |
    +-- GodotProcessManager -> uno o piu' processi Godot
    +-- ScenarioGymEnv      -> ambiente Gymnasium
    +-- algorithms/*        -> learner Keras
                                |
Godot                           |
    BridgeServer <--------------+
        ScenarioController
            Agent[]
            ScenarioEventSystem
            ScenarioRewardSystem
            ProgressProvider
```

## Installazione

Il progetto e' sviluppato con Godot 4.6.2. Versioni Godot 4 vicine possono funzionare,
ma la 4.6.2 e' quella verificata durante lo sviluppo.

Da root del repository:

```bash
python3 -m venv python/.venv
python/.venv/bin/python -m pip install --upgrade pip
python/.venv/bin/python -m pip install -r python/requirements.txt
```

Per una GPU NVIDIA su Linux:

```bash
python/.venv/bin/python -m pip install -r python/requirements-linux-cuda.txt
```

Per Apple Silicon e Metal:

```bash
xcode-select --install
python/.venv/bin/python -m pip install -r python/requirements-macos-metal.txt
```

Il file macOS usa intenzionalmente `tensorflow==2.18.1` e
`tensorflow-metal==1.2.0`. Il plugin Metal attuale non e' compatibile con le ABI delle
versioni TensorFlow piu' recenti.

Puoi evitare di ripetere il percorso di Godot impostando:

```bash
export GODOT_BIN=/percorso/del/eseguibile/Godot
```

## Primo giro completo

Prima di allenare conviene verificare che lo scenario rispetti il contratto. Questo
comando avvia Pong, legge la spec e applica azioni casuali:

```bash
python/.venv/bin/python python/tools/random_rollout.py \
  --godot-project godot \
  --godot-scene res://scenarios/pong/pong_scenario.tscn \
  --multi-agent \
  --steps 300 \
  --print-reward-terms \
  --no-headless
```

Un training DQN per Breakout puo' partire cosi:

```bash
python/.venv/bin/python python/train.py \
  --algorithm dqn \
  --godot-project godot \
  --godot-scene res://scenarios/breakout/breakout_scenario.tscn \
  --num-envs 4 \
  --num-episodes 2000 \
  --max-steps-per-episode 0 \
  --checkpoint-dir checkpoints/breakout_dqn_v1 \
  --headless
```

Durante i checkpoint Metis salva anche `policy.keras` e `policy.json`. Per guardare la
policy, senza dover ricordare l'architettura del modello:

```bash
python/.venv/bin/python python/run.py \
  --checkpoint-dir checkpoints/breakout_dqn_v1 \
  --godot-project godot \
  --godot-scene res://scenarios/breakout/breakout_scenario.tscn \
  --episodes 20 \
  --no-headless
```

`run.py` legge algoritmo, observation e action contract da `policy.json`.

## Costruire un agente in Godot

L'albero consigliato e' questo:

```text
AgentBody
└── Agent
    ├── ActionSpace
    ├── ObservationSystem
    └── RewardSystem
```

Il corpo rimane un normale `CharacterBody`, `RigidBody` o nodo scelto dal progetto.
Continua a contenere movimento, collisioni, animazioni e logica concreta. Il figlio
`Agent` descrive soltanto l'interfaccia RL.

### Azioni

`ActionSpace` puo' contenere:

- `DiscreteActionSet`, con figli `DiscreteAction` collegati a metodi Godot;
- `ContinuousAction`, con nome, dimensione, limiti e limiti di esplorazione;
- entrambi, quando l'agente ha uno spazio ibrido.

Un tank, per esempio, puo' usare accelerazione e sterzo continui insieme a uno sparo
discreto. Python riceve la struttura completa dalla scena; non contiene nomi speciali
come `accelerate`, `steer` o `shoot`.

### Observation

`ObservationSystem` conserva ordine e dimensione delle observation. Le source incluse
coprono:

- property e metodi del corpo o di un altro nodo;
- posizione, velocita' e cinematica;
- distanza e stato di uno o piu' RayCast;
- target visibili e appartenenza a team;
- informazioni locali rispetto a un Path3D.

Il plugin editor **Metis Inspector** aggiunge menu per scegliere metodi e property
compatibili senza doverli digitare a memoria. Le scene restano normali file Godot e il
runtime non dipende dal plugin.

Le observation devono avere ordine e dimensione stabili. Cambiare una source, la sua
normalizzazione o il suo significato rende normalmente incompatibile un modello gia'
allenato.

### Reward, eventi e progresso

Le reward locali stanno sotto `Agent/RewardSystem`. Sono adatte a movimento, velocita',
input, sensori, penalita' per step e funzioni personalizzate.

Le reward che appartengono al compito stanno in `ScenarioRewardSystem`: punti, goal,
progresso, vittorie, collisioni globali o assenza di avanzamento. Ogni agente riceve il
proprio totale; in multi-agent le reward non vengono sommate automaticamente tra tutti.

`ScenarioEventSystem` separa il fatto dal suo valore. Una collisione puo' emettere
`ball_hit`; un componente reward decide se vale `0.02`, `1.0` o niente. Questa divisione
rende piu' semplice ritoccare lo shaping senza riscrivere la logica del gioco.

`ProgressProvider` non significa necessariamente percorso. Puo' rappresentare metri
percorsi, mattoni distrutti, salute rimanente o una fase del compito. Metis include un
provider Path3D e uno che chiama un metodo personalizzato.

La reference completa dei nodi e' in [Metis in Godot](docs/reference/godot.md).

## Algoritmi

`python/train.py` e' l'unico entrypoint pubblico del training.

| Algoritmo | Action space | Quando usarlo |
|---|---|---|
| `dqn` | discreto | baseline semplice per poche azioni categoriche |
| `ppo` | discreto, continuo, ibrido | rollout on-policy e spazi composti |
| `ddpg` | continuo | actor-critic deterministico essenziale |
| `sac` | continuo | esplorazione entropica e policy stocastica |
| `td3` | continuo | DDPG con twin critics e aggiornamenti ritardati |
| `ddpg_bc` | continuo + demo | DDPG regolarizzato con behavior cloning |
| `ddpgfd` | continuo + demo | demo protette in replay e pretraining |
| `td3_bc` | continuo + demo | TD3 con loss BC adattiva |

Con `--algorithm auto`, Metis sceglie DQN per spazi discreti, DDPG per continui e PPO
per ibridi. Gli altri algoritmi vanno richiesti esplicitamente: la scelta dipende dal
problema, non soltanto dalla forma delle azioni.

## Episodi e frequenza delle decisioni

`--max-steps-per-episode` e' comunicato anche al `ScenarioController`, quindi Python e
Godot condividono lo stesso limite. Con `0` il limite e' disabilitato e l'episodio
termina soltanto per una condizione dello scenario:

```text
--max-steps-per-episode 0
```

Va usato solo quando esiste una conclusione affidabile o un controllo di stall.

`--physics-frames-per-step N` mantiene la stessa azione per `N` tick fisici. Con fisica
a 60 Hz e `N=4`, la policy decide a 15 Hz. Reward per step, timeout e finestre di stall
sono misurati in decision step, quindi cambiare questo valore cambia anche la loro
durata fisica.

## Uno o molti environment

`--num-envs` avvia processi Godot indipendenti. Il collector predefinito e' asincrono:
un environment lento non ferma gli altri e il learner continua a consumare transizioni
dalla coda.

```text
--num-envs 4 --collector-mode async
```

DQN, DDPG, TD3 e SAC usano replay buffer e copie CPU della policy nei collector. PPO
usa rollout congelati per generazione: gli environment sono asincroni durante la
raccolta, ma l'update avviene soltanto quando i dati appartengono alla stessa versione
della policy.

Il learner resta unico. Su una singola GPU e' in genere piu' efficiente raccogliere da
piu' simulatori e addestrare una rete centrale che far competere piu' learner per la
stessa GPU. Le reti piccole possono comunque mostrare un utilizzo GPU basso: spesso il
collo di bottiglia e' nella simulazione, nelle socket o nei batch ridotti.

Per una raccolta piu' facile da riprodurre puoi usare:

```text
--collector-mode sync --parallel-env-steps
```

Il rendering e' indipendente dal collector:

```text
--headless                                      # nessuna finestra
--num-envs 4 --no-headless                     # quattro finestre
--num-envs 4 --no-headless --render-env-count 1 # una preview
```

## Multi-agent e self-play

Con `--multi-agent`, ogni step puo' produrre una transizione per ciascun agente attivo.
Se observation e action space sono compatibili, gli agenti condividono la stessa rete
ma conservano reward, terminalita' e diagnostica separate. Aggiungere agenti aumenta la
raccolta di esperienza; non crea automaticamente modelli diversi e non sceglie il
"migliore" tra gli agenti.

Negli scenari a due squadre puoi aggiungere un opponent pool:

```text
--multi-agent \
--opponent-pool \
--opponent-snapshot-every 100 \
--opponent-pool-size 10 \
--opponent-current-probability 0.2
```

Una squadra usa la policy corrente, l'altra una snapshot storica congelata. Le
transizioni dell'avversario non entrano nel batch del learner. DQN supporta opponent
pool anche con collector asincrono; PPO, SAC e la famiglia DDPG/TD3 richiedono per ora
`--collector-mode sync` quando il pool e' attivo.

Il parameter sharing non equivale a un sistema generico multi-policy. Allenare nello
stesso scenario piu' policy indipendenti, con reti e optimizer distinti, richiede ancora
un'estensione esplicita del trainer.

## Curriculum e randomizzazione

Python comunica a Godot `training_episode`, seed dell'environment e seed per agente.
Lo scenario puo' usarli per aumentare difficolta', velocita', variabilita' dello spawn o
porzione di percorso disponibile. Il curriculum resta quindi parte dello scenario e
viene ripreso in modo deterministico da un checkpoint.

`ScenarioController` include anche replica degli agenti, randomizzazione di posizione e
rotazione, spawn tramite progress provider e disattivazione di camera/UI in headless.
Durante una registrazione manuale la replica viene disabilitata per lasciare un solo
agente controllabile.

## Dimostrazioni manuali

`python/recorder.py` salva dataset `.npz` con observation, azione applicata, reward,
stato successivo, terminalita', agente, episodio e step.

```bash
python/.venv/bin/python python/recorder.py \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --output demos/cars_demo.npz \
  --agent-id Car \
  --episodes 10 \
  --max-steps 800 \
  --no-headless
```

DQN, SAC e la famiglia DDPG/TD3 possono usare le demo per prefill e behavior cloning.
Le varianti `ddpg_bc`, `ddpgfd` e `td3_bc` richiedono almeno un `--demo-path`.

La guida [Dimostrazioni manuali](docs/guides/manual_demonstrations.md) spiega formato,
append, pretraining e uso nel replay.

## Checkpoint, policy e resume

Metis salva due tipi di artefatto, con scopi diversi:

- `ckpt-*` e `replay-*.npz` servono a riprendere il training;
- `policy.keras` e `policy.json` servono a eseguire o esportare la policy.

`--resume` riprende l'ultimo checkpoint della directory. `--resume-checkpoint` ne sceglie
uno preciso. `--policy-path` fa invece un warm start della sola rete, con optimizer,
critic, replay ed episodio nuovi. Accetta bundle Metis, `.keras`, modelli completi `.h5`
e file `.weights.h5`.

Ogni trainer puo' valutare periodicamente checkpoint congelati e mantenere il migliore
in `CHECKPOINT_DIR/best/`. Il criterio automatico privilegia il tasso di successo e usa
la reward media per gli spareggi o quando lo scenario non espone successi.

Con `Ctrl+C` il trainer salva l'ultimo stato consistente e chiude processi e socket.
Conviene aspettare il messaggio di conferma prima di chiudere il terminale.

## Eseguire ed esportare una policy

`python/run.py` usa lockstep nelle valutazioni headless e realtime quando la finestra e'
visibile. Le modalita' si possono forzare con `--execution-mode`.

Per eseguire un file specifico:

```bash
python/.venv/bin/python python/run.py \
  --policy-path exports/mia_policy/policy.keras \
  --godot-project godot \
  --godot-scene res://scenarios/mio_scenario.tscn \
  --infinite \
  --no-time-limit \
  --no-headless
```

Con un bundle dotato di `policy.json` non serve indicare l'algoritmo. Per un vecchio
`.h5` senza manifest puo' essere necessario specificarlo.

TFLite e ONNX si producono con:

```bash
python/.venv/bin/python python/export.py \
  --policy checkpoints/nome_run \
  --format all
```

TFLite e' disponibile con TensorFlow. ONNX richiede prima
`python/requirements-export.txt`. Il manifest descrive ordine delle observation,
action space e decoder, ma l'applicazione finale deve comunque ricostruire le stesse
observation e normalizzazioni usate durante il training.

## Esempi e tutorial

- [Breakout](docs/tutorials/tutorial_breakout_da_zero.md): single-agent discreto,
  eventi, reward sparse e shaping opzionale.
- [Pong](docs/tutorials/tutorial_pong_multi_agent.md): due agenti, policy condivisa,
  reward personali e opponent pool.
- [Tanks](docs/tutorials/tutorial_tanks_hybrid_multi_agent.md): squadre 3D, sensori
  locali e azioni ibride.
- [Guida autonoma](docs/tutorials/tutorial_guida_autonoma_path_vs_sensori.md): confronto
  tra percorso noto e guida basata soltanto sui sensori.
- [Soccer](docs/tutorials/tutorial_soccer_continuous_multi_agent.md): gioco di squadra
  3D con palla fisica e controllo continuo.

## Limiti da conoscere

Metis prova ad automatizzare la parte ripetitiva, non a nascondere le scelte di RL.
Oggi restano questi confini:

- agenti che condividono una policy devono avere lo stesso contratto;
- il multi-policy indipendente non e' ancora un flusso generico;
- l'opponent pool asincrono e' disponibile soltanto per DQN;
- cambiare observation, azioni, reward o fisica invalida spesso replay e checkpoint;
- episodi senza limite richiedono terminalita' o stall detection affidabili;
- un modello Keras non incorpora da solo sensori e preprocessing Godot;
- gli algoritmi inclusi sono implementazioni del progetto, non wrapper SB3 certificati.

Queste limitazioni non impediscono di estendere il framework. Le guide
[Aggiungere un algoritmo RL](docs/guides/aggiungere_algoritmo_rl.md) e
[Estendere Metis in Godot](docs/guides/estendere_framework_godot.md) descrivono i punti
di estensione e i test attesi.

## Struttura del repository

```text
godot/
  agents/       scene degli agenti
  scenarios/    ambienti e regole dei task
  scripts/      bridge e componenti Metis
  addons/       plugin Metis Inspector

python/
  train.py      training
  run.py        inferenza e valutazione
  recorder.py   dimostrazioni manuali
  export.py     TFLite e ONNX
  algorithms/   backend RL
  core/         replay, modelli, checkpoint, async e opponent pool
  envs/         Gymnasium e process manager Godot
  tools/        diagnostica
  tests/        test automatici

docs/
  tutorials/    scenari completi costruiti passo passo
  guides/       procedure riusabili ed estensioni
  reference/    contratti e architettura
```

Per eseguire i test Python:

```bash
cd python
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

La direzione del progetto resta semplice: Godot descrive il problema, Python impara a
risolverlo, e il confine tra i due deve rimanere abbastanza pulito da poter cambiare
scenario senza ricominciare ogni volta dal framework.
