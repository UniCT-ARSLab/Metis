# Metis Python Runtime

Il runtime Python di Metis scopre il contratto dichiarato dallo scenario Godot e lo
collega al backend RL selezionato, senza introdurre trainer specifici per scenario.

## Entry point pubblici

### `python/train.py`

Legge gli argomenti comuni, rileva eventualmente lo spazio di azione e carica in modo
lazy un modulo in `python/algorithms`. Con `--algorithm auto` sceglie DQN per discrete,
DDPG per continue e PPO per ibride. Gli algoritmi alternativi si selezionano per nome.
`--policy-path` esegue un warm start della sola policy da `.keras`, modello `.h5` o
`.weights.h5`; non sostituisce il resume dello stato completo.

### `python/run.py`

Carica per default `policy.keras` e `policy.json`, oppure un modello `.h5`, i vecchi pesi/checkpoint,
e applica la policy senza training. Supporta esecuzione lockstep per valutazioni riproducibili e
realtime per osservare il comportamento a velocita' naturale.

### `python/export.py`

Converte il bundle Keras prodotto dai trainer in TensorFlow Lite e/o ONNX. TFLite usa
TensorFlow; ONNX richiede le dipendenze opzionali in `requirements-export.txt`.

### `python/recorder.py`

Avvia uno scenario in controllo manuale e salva tuple `obs`, `actions`, `rewards`,
`next_obs`, `terminated` e `truncated` in un dataset `.npz`.

## Package interni

### `python/algorithms/`

- `dqn.py`: Q-learning per azioni discrete;
- `ppo.py`: actor-critic on-policy per discrete e ibride;
- `sac.py`: actor-critic entropico per continue;
- `ddpg.py`, `td3.py`: entrypoint deterministici;
- `ddpg_bc.py`, `td3_bc.py`: behavior cloning insieme al learning online;
- `ddpgfd.py`: replay dimostrativo prioritario;
- `common.py`: implementazione condivisa dalla famiglia DDPG/TD3.

Ogni backend espone `main()` e possiede parser, ciclo sync/async, checkpoint e log.

### `python/core/`

- `models.py`: factory Keras e funzioni di inferenza compilate;
- `policy_artifact.py`: salvataggio atomico di `policy.keras` e relativo manifest;
- `replay_buffer.py`: replay uniforme/prioritizzato, demo protette e snapshot;
- `opponent_pool.py`: snapshot storiche e sampling degli avversari;
- `training.py`: collector async, parallel stepper, TensorFlow runtime, best checkpoint,
  stampa metriche e utility condivise.

Il package core non deve conoscere scene come Cars, Tanks o Breakout.

### `python/envs/`

- `scenario.py`: wrapper Gymnasium del protocollo TCP;
- `process_manager.py`: avvio, readiness, log e chiusura dei processi Godot.

Questa e' la frontiera I/O. Gli algoritmi non devono aprire socket o costruire comandi
Godot direttamente.

### `python/tools/`

Contiene rollout casuale e benchmark del bridge. Sono strumenti diagnostici, non backend
di training.

### `python/tests/`

I test coprono buffer, target e truncation, action packing, async collector, opponent
pool, checkpoint e dispatcher. Un nuovo backend deve aggiungere test della propria
semantica, non soltanto un test di import.

## Dipendenze consentite

```text
CLI -> algorithms -> core
 |         |          |
 +---------+--------> envs
```

`core` non importa algoritmi concreti. `envs` non importa TensorFlow. Godot non dipende
da una classe Python specifica. Mantenere queste direzioni evita dipendenze circolari e
permette ai collector di funzionare senza caricare il learner.

## Matrice algoritmi

| Algoritmo | Azioni | Off/on-policy | Replay | Demo |
| --- | --- | --- | --- | --- |
| DQN | discrete | off-policy | si | prefill |
| PPO | discrete/hybrid | on-policy | no | no |
| DDPG | continuous | off-policy | si | prefill |
| DDPG+BC | continuous | off-policy | si | obbligatorie |
| DDPGfD | continuous | off-policy | prioritizzato | obbligatorie |
| TD3 | continuous | off-policy | si | prefill |
| TD3+BC | continuous | off-policy | si | obbligatorie |
| SAC | continuous | off-policy | si | prefill |

La compatibilita' con async e multi-agent e' una responsabilita' del backend. Non va
dedotta solo dal fatto che il modello accetti batch.

## Convenzioni

- Gli argomenti comuni devono mantenere lo stesso nome tra backend.
- `0` per `max_steps` significa nessun limite di step, se supportato dallo scenario.
- Ogni checkpoint deve aggiornare il bundle Keras portabile della policy.
- Un checkpoint deve poter essere caricato da `run.py` senza dipendere dal replay.
- Un resume di training off-policy deve poter ripristinare anche il replay.
- Le metriche multi-agent contano agenti e transizioni, non soltanto step di ambiente.
- La valutazione automatica del best checkpoint usa per default un solo thread CPU, per
  non sottrarre risorse ai collector e al learner. Il limite si regola con
  `--best-evaluation-cpu-threads`; `--no-best-checkpoint` disattiva la valutazione.
- I moduli interni non sono nuovi entrypoint pubblici: i comandi documentati usano le
  quattro CLI nella radice di `python/`: `train.py`, `run.py`, `recorder.py` ed
  `export.py`.
