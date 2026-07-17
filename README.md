# Godot Gymnasium Keras RL Framework

Piccolo framework sperimentale per addestrare agenti di reinforcement learning in scenari Godot usando Gymnasium e TensorFlow/Keras.

Godot gestisce simulazione, agenti, sensori, reward locali e bridge TCP. Python interroga lo scenario, legge observation/action space esposti dagli agenti e avvia il trainer più adatto.

Per creare un nuovo agente o scenario, vedi [Tutorial nuovo scenario/agente](docs/tutorial_nuovo_scenario_agente_rl.md).

## Architettura

- Una istanza Godot esegue uno scenario.
- `BridgeServer` espone reset, step, spec e configurazione via TCP.
- `ScenarioController` coordina agenti, reset, reward di scenario e terminal state.
- Il bridge usa lockstep di default: con Python connesso, la fisica avanza solo durante
  `reset` e `step`; in headless Godot usa timestep fisso senza attesa del tempo reale.
- Con `--collector-mode async` (default), DQN, DDPG, TD3 e SAC eseguono un collector indipendente per
  ogni env, ciascuno con una copia CPU della policy. Il learner aggiorna il modello in
  parallelo e pubblica periodicamente nuovi pesi ai collector.
- Il training usa un solo learner centrale: replay buffer, optimizer e modello trainabile
  non vengono duplicati. Su una singola GPU questo evita gradienti concorrenti e costose
  sincronizzazioni tra learner.
- PPO raccoglie in async un rollout completo per env con una policy congelata. Esegue
  l'update soltanto quando tutti i rollout della generazione hanno la stessa versione,
  quindi resta correttamente on-policy senza una barriera a ogni step.
- Con `--collector-mode sync`, i trainer inviano gli step dei vari env in parallelo ma
  attendono tutte le risposte prima di aggiornare il modello.
- Il training e' headless per default. Usa `--no-headless` per mostrare le istanze;
  insieme a `--render-env-count 1` viene renderizzata una sola preview e gli altri
  worker restano headless.
- Ogni agente Godot espone observation space, action space, reward e done.
- `Agent/ActionSpace` dichiara azioni discrete, continue o ibride da Inspector.
- `Agent/ObservationSystem` registra observation source riusabili come metodi del corpo, raycast e sensori target.
- Python usa `ScenarioGymEnv` come wrapper Gymnasium generico.
- `train_generic.py` seleziona o inoltra al backend di training.

Il framework supporta scenari single-agent e multi-agent. In multi-agent il trainer salva transizioni per agente nel replay buffer, usando una policy condivisa quando gli agenti hanno observation/action space compatibili.

Negli scenari competitivi a due squadre, tutti i backend supportano anche self-play con
opponent pool. Ogni corpo agente deve esporre `get_team_id()` (oppure una proprieta'
`team_id`) e il training va avviato con `--multi-agent --opponent-pool`. Python salva
snapshot storiche in `CHECKPOINT_DIR/opponents`, assegna a rotazione una squadra alla
policy corrente e usa una snapshot congelata per l'altra. Le esperienze dell'avversario
non vengono inserite nel replay o nel batch on-policy.
L'opponent pool storico funziona in async con DQN; PPO, SAC e la famiglia DDPG/TD3
richiedono ancora `--collector-mode sync` quando il pool e' attivo.

## Python Attuale

File principali:

- `python/train_generic.py`: entrypoint unico per il training.
- `python/train_generic_dqn.py`: azioni discrete.
- `python/train_generic_ddpg.py`: DDPG per azioni continue.
- `python/train_generic_ddpg_bc.py`: DDPG con behavior cloning.
- `python/train_generic_ddpgfd.py`: DDPG from Demonstrations.
- `python/train_generic_td3.py`: TD3 per azioni continue.
- `python/train_generic_td3_bc.py`: TD3 con behavior cloning.
- `python/deterministic_training.py`: infrastruttura condivisa dai cinque trainer deterministici.
- `python/train_generic_sac.py`: azioni continue con SAC.
- `python/train_generic_ppo.py`: azioni ibride.
- `python/run_generic_policy.py`: esecuzione di un modello addestrato.
- `python/record_demonstrations.py`: registrazione demo manuali.
- `python/random_scenario_rollout.py`: rollout casuale per validare uno scenario.
- `python/scenario_gym_env.py`: wrapper Gymnasium generico.
- `python/godot_process_manager.py`: avvio/stop istanze Godot.
- `python/models.py`: reti Keras condivise.
- `python/replay_buffer.py`: replay buffer.
- `python/opponent_pool.py`: snapshot storiche, sampling e maschere learner per self-play.

La vecchia linea TeamBattle/DQN è archiviata in `python/legacy/team_battle/`.
La vecchia scena Godot TeamBattle è archiviata in `godot/legacy/team_battle/`.

## Installazione

```bash
cd python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Linux con GPU NVIDIA/CUDA:

```bash
pip install -r requirements-linux-cuda.txt
```

macOS Apple Silicon con Metal:

```bash
xcode-select --install
python -m pip install -r requirements-macos-metal.txt
```

Il file macOS mantiene `tensorflow==2.18.1` insieme a
`tensorflow-metal==1.2.0`: versioni TensorFlow piu' recenti non sono
compatibili con l'ABI dell'attuale plugin Metal e falliscono durante
`import tensorflow`.

## Training Generico

Esempio SAC per lo scenario Cars:

```bash
python/.venv/bin/python python/train_generic.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --num-envs 4 \
  --num-episodes 2500 \
  --max-steps-per-episode 800 \
  --batch-size 128 \
  --replay-warmup 12000 \
  --random-exploration-episodes 40 \
  --action-smoothing 0.25 \
  --reset-progress-curriculum \
  --reset-progress-start-max 0.015 \
  --reset-progress-end-max 0.75 \
  --reset-progress-ramp-episodes 1600 \
  --checkpoint-dir checkpoints/cars_sac_rewards_v1 \
  --actor-weights-path cars_sac_rewards_actor_v1.weights.h5 \
  --critic1-weights-path cars_sac_rewards_critic1_v1.weights.h5 \
  --critic2-weights-path cars_sac_rewards_critic2_v1.weights.h5 \
  --multi-agent \
  --resume
```

Con `--algorithm auto`, il launcher sceglie:

- `dqn` per action space discreti.
- `ddpg` per action space continui.
- `ppo` per action space ibridi.

SAC va selezionato esplicitamente con `--algorithm sac`.

Per SAC, ogni checkpoint nuovo salva anche il replay buffer associato, per esempio
`ckpt-1050.*` e `replay-1050.npz`. `--resume` continua a caricare automaticamente
l'ultimo checkpoint dentro `--checkpoint-dir`. Per ripartire da uno stato preciso usa
invece `--resume-checkpoint checkpoints/nome_run/ckpt-1025`; la directory indicata con
`--checkpoint-dir` resta la destinazione dei checkpoint successivi e puo' quindi essere
diversa dalla directory sorgente. Se un vecchio checkpoint non ha il relativo replay,
il trainer ricostruisce il buffer e mantiene gli aggiornamenti disabilitati fino a
`--replay-warmup`.

Dopo un resume SAC, per default i primi 2000 gradient step aggiornano soltanto i
critic e i target critic. Actor e alpha restano congelati finche' le stime Q non si
sono riadattate al replay ripristinato. Il valore si configura con
`--critic-warmup-updates` e si disabilita impostandolo a `0`.

Terminato il warmup, SAC aggiorna actor e alpha ogni due aggiornamenti dei critic
(`--policy-update-every 2`). Nei resume usa inoltre learning rate piu' prudenti,
configurabili con `--resume-actor-learning-rate` e
`--resume-alpha-learning-rate`; i learning rate normali restano validi per i run
avviati da zero.

Tutti i trainer mantengono per default anche checkpoint selezionati in
`CHECKPOINT_DIR/best/`. Ogni 100 episodi una policy congelata viene valutata in una
istanza Godot headless separata, senza esplorazione, usando 20 episodi, seed fissi e
il livello finale del curriculum (`training_episode=num_episodes`). In modalita'
`auto` il confronto privilegia il tasso di successo e usa la reward media a parita'
di successo; per scenari senza successi raggiunti la reward distingue comunque i
candidati. Le metriche e il checkpoint scelto sono registrati in
`CHECKPOINT_DIR/best/best_metrics.json`.

La valutazione avviene in background su una copia esatta del checkpoint candidato,
senza fermare il learner. Se una valutazione precedente e' ancora attiva, il candidato
successivo viene saltato invece di accumulare processi e code. Alla fine del training il
trainer attende per default fino a 120 secondi l'ultima valutazione
(`--best-final-drain-timeout`); con `Ctrl+C` la annulla per garantire una chiusura pulita.

La frequenza e il campione si configurano con `--best-evaluation-every` e
`--best-evaluation-episodes`; `--best-metric reward_mean` rende invece la reward il
criterio primario. `--best-evaluation-training-episode` consente di fissare
esplicitamente la difficolta' di valutazione e `--no-best-checkpoint` disabilita la
funzione. Il valutatore usa la CPU per default per non creare un secondo runtime
TensorFlow sulla GPU del learner; si puo' scegliere il dispositivo automatico con
`--best-evaluation-device auto`. Quando il training non ha un limite di step, le
valutazioni usano comunque un watchdog di 10000 step per classificare come fallite
le policy entrate in cicli senza terminale; il valore si cambia con
`--best-evaluation-max-steps`.

I checkpoint cronologici restano la fonte consigliata per `--resume`, perche'
conservano anche il replay buffer. I checkpoint `best` conservano modello,
optimizer e stato del trainer ma non duplicano il replay buffer: sono destinati
soprattutto alla valutazione e all'esecuzione della policy migliore.

Per eseguire direttamente la policy migliore usa `--load-from checkpoint` insieme a
`--checkpoint-dir CHECKPOINT_DIR/best`. Per riprendere il training usa invece la
directory cronologica principale, oppure `--resume-checkpoint` con un checkpoint
preciso dotato del replay associato.

Tutti i trainer supportano `--log-format pretty` (default) e `--log-format compact`.
Con `Ctrl+C` salvano l'ultimo episodio completato, i pesi e, per SAC/DDPG/TD3/DQN,
anche il replay buffer; poi chiudono connessioni e processi Godot senza traceback.
Attendi il messaggio `Interrupted training saved` prima di chiudere il terminale.

Per azioni continue, la fase di esplorazione casuale iniziale usa eventuali limiti `exploration_low` e `exploration_high` dichiarati nello `action_space` dell'agente Godot. Se questi limiti non sono presenti, Python campiona uniformemente tra `low` e `high`.

### Durata degli episodi

`--max-steps-per-episode` viene applicato sia dal trainer Python sia dal
`ScenarioController` Godot, quindi non esistono due limiti indipendenti. Un valore
positivo produce `truncated=true` all'ultimo step. Il valore `0` disabilita il limite:

```text
--max-steps-per-episode 0
```

In questa modalita' l'episodio termina soltanto quando lo scenario emette una condizione
terminale, per esempio `life_lost`, `level_cleared`, collisione o goal. Usala soltanto
quando ogni episodio possiede una condizione terminale affidabile: un episodio bloccato
non aggiorna i contatori per episodio e puo' trattenere indefinitamente un collector.
Il default resta finito per proteggere scenari nuovi o configurati in modo incompleto.

`terminated=true` indica una conclusione reale dello scenario; `truncated=true` indica
un taglio esterno, come il limite di step. I trainer mantengono il bootstrap del valore
sulle transizioni troncate e lo azzerano soltanto sui terminali reali.

### Frequenza delle azioni

`--physics-frames-per-step N` mantiene la stessa azione per `N` tick fisici Godot prima
di restituire una nuova observation. Per esempio, con fisica a 60 Hz e `N=4`, la policy
decide a 15 Hz. Questo riduce round trip socket e inferenze senza cambiare `time_scale`
o il significato di `delta`; il default e' `1`.

Il parametro va scelto in base alla dinamica dello scenario. Reward per step, finestre
di stall, cooldown espressi in step e `--max-steps-per-episode` misurano decision step,
quindi modificare `N` cambia la loro durata fisica e puo' richiedere di scalarli.

### Collector sincrono e asincrono

La modalita' predefinita e' asincrona:

```text
--collector-mode async
```

Per configurare la separazione fra raccolta e training negli algoritmi off-policy usa:

```text
--collector-mode async \
--async-queue-capacity 256 \
--async-policy-sync-steps 100 \
--async-policy-publish-updates 100 \
--async-update-basis transitions \
--async-update-every 4 \
--async-max-updates-per-env-step 1 \
--async-drain-max-events 64 \
--async-updates-per-step 1
```

In `async`, ogni istanza Godot avanza senza aspettare gli altri env e senza aspettare
gli aggiornamenti TensorFlow, finche' la coda non raggiunge la capacita' massima. Il
log mostra `queue`, `policy_version`, `env_steps_s`, `transitions_s` e `updates_s`: una coda spesso vicina al limite indica che il
learner non riesce a consumare alla velocita' dei collector. Aumentare la coda assorbe
picchi brevi ma non risolve un learner stabilmente piu' lento. Per default il learner
ingerisce fino a 64 eventi per burst e richiede un update ogni quattro transizioni dei
singoli agenti. Questo mantiene il rapporto fra esperienza e apprendimento anche quando
uno step multi-agent produce molte transizioni. Il limite
`--async-max-updates-per-env-step 1` impedisce pero' che molti agenti generino un numero
illimitato di update TensorFlow: gli update eccedenti vengono intenzionalmente limitati
e compaiono nel log come `updates_throttled`. Il log mostra anche
`transitions_step`, cioe' quanti agenti attivi hanno mediamente prodotto esperienza per
step Godot.

Puoi modificare l'intervallo con `--async-update-every`; il vecchio nome
`--async-update-every-steps` resta un alias compatibile. Per ripristinare esattamente il
comportamento precedente usa `--async-update-basis env_steps`. Il numero di update per
intervallo e' `--async-updates-per-step`; usa
`--async-max-updates-per-env-step 0` soltanto se vuoi disabilitare il limite di sicurezza.

In DQN, DDPG, TD3 e SAC le policy locali sono sincronizzate ogni
`--async-policy-sync-steps` control step e il
learner pubblica una snapshot ogni `--async-policy-publish-updates` aggiornamenti della
policy. Valori piu' bassi riducono il ritardo della policy ma aumentano copie e
contesa CPU. Il replay buffer, gli optimizer e i checkpoint appartengono soltanto al
learner. I checkpoint memorizzano gli episodi gia' consumati dal learner; dopo
un'interruzione un episodio parziale puo' essere ripetuto, evitando di saltare
esperienza che era ancora in coda.

DQN, DDPG, TD3 e SAC usano un replay buffer NumPy circolare preallocato: il costo del
sampling non cresce con la dimensione del buffer. In async, `--async-replay-save`
(default) copia uno snapshot consistente e comprime il file `.npz` in background.
La chiusura finale attende comunque il completamento del file. Usa
`--no-async-replay-save` soltanto per tornare al salvataggio bloccante.

Il learner DQN compila per default gli update in un grafo TensorFlow e raggruppa gli
update consecutivi prima di tornare a Python. Questo riduce le sincronizzazioni CPU/GPU
senza cambiare il numero di gradient step o la frequenza delle snapshot. Per diagnosi o
compatibilita' puoi ripristinare il percorso eager con `--no-tf-compile-learner`.

Le copie di policy dei collector eseguono inferenza su CPU. La GPU e' usata dal learner
per forward pass, target e backpropagation; con reti piccole l'utilizzo GPU istantaneo
puo' quindi restare basso anche quando il training funziona correttamente. Spostare ogni
collector sulla GPU produrrebbe molte inferenze minuscole e concorrenti; servirebbe un
inference server centrale con batching per renderlo vantaggioso.

Su Linux/CUDA `--gpu-memory-growth` e' attivo per default e lascia crescere la memoria
TensorFlow in base al bisogno, invece di riservare quasi tutta la VRAM all'avvio. Usa
`--no-gpu-memory-growth` per ripristinare il comportamento TensorFlow standard. Il flag
non modifica il backend Metal di macOS.

PPO usa una variante async on-policy: gli env avanzano indipendentemente durante il
rollout, poi attendono tutti il PPO update della generazione prima di ripartire con la
nuova rete e il nuovo `log_std`. Nei log compare `collector=async_on_policy`. Non usa
policy lag e non scarta rollout.

Tutti i backend supportano `--collector-mode async` con single-agent o multi-agent e
parameter sharing. Anche l'opponent pool storico DQN supporta async: ogni worker conserva
la propria snapshot congelata e il proprio lato learner per l'intero episodio. Negli
altri backend l'opponent pool richiede ancora `--collector-mode sync`.

Per ripristinare il comportamento precedente usa esplicitamente:

```text
--collector-mode sync --parallel-env-steps
```

La modalita' async rimuove le pause causate dal learner e rende la preview molto piu'
fluida, ma non garantisce da sola 60 FPS esatti. Se socket, inferenza o coda introducono
backpressure, Godot resta correttamente in lockstep invece di inventare tick non
registrati nel replay.

Il rendering e' indipendente dal collector mode:

```text
# Tutti gli env visibili
--num-envs 4 --no-headless

# Solo un env visibile, gli altri tre headless
--num-envs 4 --no-headless --render-env-count 1

# Tutti gli env headless
--num-envs 4 --headless
```

### Self-play con opponent pool

Le opzioni sono comuni a DQN, DDPG, SAC e PPO:

```text
--opponent-pool
--opponent-snapshot-every 100
--opponent-pool-size 10
--opponent-current-probability 0.2
--opponent-sampling uniform
```

Con DQN puoi aggiungere `--collector-mode async`; con PPO, SAC, DDPG, TD3 e relative
varianti usa ancora `--collector-mode sync` quando il pool e' attivo.

`uniform` campiona tutte le snapshot conservate, mentre `latest` usa sempre la piu'
recente. Nel 20% degli episodi dell'esempio entrambi i team usano la policy corrente.
Il lato learner viene sorteggiato separatamente per ogni env; `--learner-team 0` lo
rende fisso. Il manifest del pool viene riaperto automaticamente con `--resume`.

## Reward

Le reward locali dell'agente stanno nel nodo `Agent/RewardSystem` e sono composte da figli `RewardComponent`, configurabili da Inspector.

Le reward che dipendono dallo scenario stanno in un nodo `ScenarioRewardSystem` collegato al `ScenarioController`. Per Cars, qui vivono progresso sul `Path3D`, penalita' di arretramento, target finale, stall e pace penalty. Il controller somma `local_term_rewards` e `scenario_reward`, poi invia a Python anche `scenario_terms` per debug.

## Validare Uno Scenario

Per provare lo scenario con azioni casuali:

```bash
python/.venv/bin/python python/random_scenario_rollout.py \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --multi-agent \
  --headless
```

## Eseguire Un Modello

Esempio SAC:

```bash
python/.venv/bin/python python/run_generic_policy.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --actor-weights-path cars_sac_rewards_actor_v1.weights.h5 \
  --multi-agent \
  --execution-mode realtime \
  --realtime-action-hz 60 \
  --realtime-simulation-fps 60 \
  --no-headless
```

Per osservare a lungo:

```bash
python/.venv/bin/python python/run_generic_policy.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --actor-weights-path cars_sac_rewards_actor_v1.weights.h5 \
  --multi-agent \
  --infinite \
  --no-time-limit \
  --execution-mode realtime \
  --realtime-action-hz 60 \
  --realtime-simulation-fps 60 \
  --no-headless
```

`run_generic_policy.py` usa `--execution-mode auto`: seleziona `lockstep` quando e'
headless e `realtime` quando la finestra Godot e' visibile. In lockstep Godot esegue
un decision step per richiesta Python, soluzione deterministica adatta a confronti e
test. In realtime Godot continua la simulazione tra due inferenze mantenendo l'ultima
azione ricevuta. Le due modalita' possono sempre essere forzate esplicitamente.
`--realtime-action-hz` limita la frequenza di aggiornamento delle azioni in tempo
reale (60 Hz per default, `0` senza pacing), mentre
`--realtime-simulation-fps` limita il loop Godot a 60 FPS anche in headless o senza
VSync. Il parametro storico `--delay` viene applicato soltanto in lockstep.

## Demo Manuali

Le demo si registrano con `record_demonstrations.py` e possono essere usate per prefill del replay buffer o behavior cloning. Vedi [Dimostrazioni Manuali](docs/manual_demonstrations.md).

Per action space continui sono disponibili cinque trainer deterministici espliciti:

| `--algorithm` | Uso |
| --- | --- |
| `ddpg` | Un actor e un critic; non richiede dimostrazioni. E' anche il default di `auto` per azioni continue. |
| `ddpg_bc` | DDPG con una loss di behavior cloning calcolata soltanto su vere dimostrazioni. |
| `ddpgfd` | DDPG from Demonstrations con pretraining actor/critic, replay prioritizzato e transizioni demo protette. |
| `td3` | Twin critics, target policy smoothing e aggiornamento ritardato dell'actor. |
| `td3_bc` | TD3 con regolarizzazione BC adattiva su un dataset esperto separato. |

Le varianti `ddpg_bc`, `ddpgfd` e `td3_bc` richiedono almeno un
`--demo-path`. DDPGfD mantiene le demo nel replay senza permettere alle transizioni
online di sovrascriverle. TD3 possiede due critic, quindi non puo' riprendere un
checkpoint DDPG; ogni variante verifica l'identita' dell'algoritmo salvata nel checkpoint.

`td3_bc` e' una variante online: il critic continua a imparare dal replay del training,
mentre l'actor riceve anche la loss BC da un batch esperto separato. DDPGfD usa un target
TD a un passo. Il ritorno n-step ausiliario del paper non viene ricostruito attraversando
il replay, perche' negli scenari con piu' env e agenti le transizioni adiacenti possono
appartenere a traiettorie differenti.

## Tutorial Nuovi Scenari

Per costruire passo passo un agente 2D con azioni discrete, observation configurate
dall'Inspector, eventi e reward di scenario, vedi
[Breakout da zero](docs/tutorial_breakout_da_zero.md).

Per uno scenario competitivo con due istanze dello stesso agente, policy condivisa,
observation simmetriche e self-play simultaneo, vedi
[Pong multi-agent](docs/tutorial_pong_multi_agent.md).

Per un'arena 3D a squadre con mappa sconosciuta, sensori locali, missili, reward
cooperative e action space ibrido continuo/discreto, vedi
[Tanks 2v2 hybrid multi-agent](docs/tutorial_tanks_hybrid_multi_agent.md).

Per confrontare guida continua con percorso noto e guida generalizzabile basata
soltanto su sensori locali, vedi
[Guida autonoma: Path3D vs sensor-only](docs/tutorial_guida_autonoma_path_vs_sensori.md).

Per costruire un gioco di calcio arcade 3D con `CharacterBody3D`, palla fisica,
calcio a intensita' continua, squadre e policy SAC condivisa, vedi
[Soccer 3D multi-agent](docs/tutorial_soccer_continuous_multi_agent.md).

## Note

Lo stato attuale punta a rendere Godot la fonte di verità per:

- observation space;
- action space;
- reward locali;
- terminal state;
- numero e identità degli agenti.

Python dovrebbe diventare sempre più automatico: dato uno scenario Godot valido, deve poter scegliere o ricevere l'algoritmo, costruire il modello compatibile e addestrare senza script specifici per scenario.
