# Aggiungere un algoritmo RL a Metis

Questa guida descrive come integrare un backend senza creare una seconda pipeline di
socket, processi Godot o checkpoint. Prima di iniziare, verifica che l'algoritmo non sia
gia' una variante configurabile di DQN, PPO, SAC o della famiglia DDPG/TD3.

## 1. Definire il contratto

Scrivi prima una scheda breve:

- action type supportati: discrete, continuous, hybrid;
- on-policy oppure off-policy;
- reti richieste e relativi target;
- formato del buffer;
- stato necessario nel checkpoint;
- compatibilita' multi-agent;
- strategia async e policy lag accettabile;
- supporto a demo e opponent pool.

Queste decisioni determinano il backend. Non partire copiando un trainer casuale e
adattando finche' non gira: e' facile conservare una semantica sbagliata di terminalita'
o aggiornamenti.

## 2. Scegliere il punto di partenza

- Discrete off-policy: `python/algorithms/dqn.py`.
- On-policy discrete/hybrid: `python/algorithms/ppo.py`.
- Continuous stochastic: `python/algorithms/sac.py`.
- Continuous deterministic: `python/algorithms/common.py` e uno degli entrypoint
  DDPG/TD3.

Se la differenza e' una loss o una regola di update, aggiungi una variante al motore
condiviso. Se cambiano buffer, rollout o modello, crea un modulo autonomo.

## 3. Creare il modulo

Il file va direttamente in `python/algorithms`, per esempio:

```text
python/algorithms/my_algorithm.py
```

Deve esporre un `main()` senza argomenti, perche' `python/train.py` inoltra gli argomenti
tramite `sys.argv`:

```python
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    # Aggiungere gli stessi argomenti condivisi degli altri backend.
    return parser.parse_args()


def main():
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
```

Il blocco `__main__` e' utile per debug come modulo, ma l'entrypoint documentato resta
`python/train.py`.

## 4. Riutilizzare l'infrastruttura

Importa da `core.training`:

- configurazione TensorFlow e memory growth;
- `ParallelEnvStepper` per il collector sync;
- `AsyncCollectorPool`, eventi async e `PolicySnapshot`;
- scheduler degli update;
- risoluzione resume e replay snapshot;
- `BestCheckpointTracker`;
- log strutturati e chiusura pulita.

Importa da `envs`:

- `GodotProcessManager` per avviare/chiudere processi;
- `ScenarioGymEnv` per spec, reset e step.

Non copiare il protocollo TCP nel backend e non chiamare direttamente `subprocess.Popen`.

## 5. Validare la spec

Dopo aver connesso il primo environment, controlla esplicitamente:

```python
if env.action_type not in {"continuous"}:
    raise RuntimeError("my_algorithm requires continuous actions")
```

Verifica inoltre che tutti gli environment abbiano stessa `obs_dim`, action spec e
lista degli agenti. In multi-agent verifica la compatibilita' tra i corpi che condividono
la policy.

Non dedurre nomi o dimensioni dallo scenario. Usa:

- `env.obs_dim`;
- `env.action_type`;
- `env.action_size`;
- `env.action_space_spec`;
- limiti continuous ricevuti dalla spec.

## 6. Semantica delle transizioni

Una transizione off-policy deve conservare separatamente terminalita' e truncation:

```text
(obs, action, reward, next_obs, terminated)
```

Il limite di step interrompe il rollout, ma il target continua il bootstrap se
`terminated` e' falso. Per on-policy, calcola il bootstrap value dell'ultima observation
quando il rollout e' troncato.

In multi-agent uno step produce una transizione per ogni agente learner attivo. Lo
scheduler async dovrebbe basare gli update sulle transizioni quando vuoi che piu' agenti
aumentino anche il lavoro del learner.

## 7. Modello e inferenza collector

Metti factory Keras riusabili in `core/models.py` solo se servono a piu' backend o al
runner. Mantieni nel modulo algoritmo loss e dettagli specifici.

La funzione pubblicata ai collector deve:

- accettare una observation con forma stabile;
- restituire il formato previsto da `ScenarioGymEnv`;
- essere thread-safe oppure istanziata per worker;
- non aggiornare pesi durante un'inferenza;
- associare ogni snapshot a una `policy_version`.

Per policy stocastiche separa sampling di training e azione deterministica di run.

## 8. Collector asincrono

Supportare async non significa soltanto avviare thread. Devi definire:

1. quando un worker riceve nuovi pesi;
2. quante transizioni autorizzano un update;
3. come limitare la coda e la pressione sul replay;
4. quali contatori sono globali e quali per worker;
5. come interrompere worker e processi con Ctrl+C;
6. se opponent pool e multi-policy sono realmente cablati.

Se una funzione non e' supportata, rifiutala con un errore chiaro. Un flag accettato ma
ignorato e' peggiore di un limite esplicito.

## 9. Checkpoint e resume

Registra in `tf.train.Checkpoint` tutto lo stato che cambia l'ottimizzazione:

- reti online e target;
- optimizer;
- temperatura o coefficienti appresi;
- episode/update counter;
- schedule che non puo' essere ricostruita dagli argomenti.

Per off-policy salva il replay separatamente. Dopo il resume, evita update distruttivi
immediati: se necessario usa un critic warmup con actor congelato, come SAC.

Il checkpoint deve includere metadata sufficienti a `python/run.py` per identificare
algoritmo, observation dimension e action spec.

## 10. Registrare il backend

In `python/train.py`:

1. aggiungi il nome a `BACKENDS`;
2. aggiungilo alle scelte di `--algorithm`;
3. decidi se `auto` deve selezionarlo. In genere un action type non basta per scegliere
   tra SAC, TD3 e DDPG, quindi i nuovi algoritmi restano espliciti.

In `python/run.py`:

1. aggiungi il nome alle scelte;
2. costruisci il modello di inferenza corretto;
3. ripristina checkpoint o pesi;
4. implementa l'azione deterministica/stocastica desiderata;
5. valida che action type e dimensioni coincidano.

Se il modello e' compatibile con un backend esistente, riusa il loader invece di
duplicarlo.

## 11. Log minimi

Ogni backend deve mostrare:

- episodio completato e modalita' collector;
- reward e successo;
- step/transizioni raccolte;
- dimensione replay o rollout;
- numero reale di update;
- loss principali;
- statistica delle action;
- policy version e throughput in async.

Una loss a zero deve distinguere tra "nessun update" e un vero valore numerico zero.

## 12. Test richiesti

Prima di documentare il backend come supportato, aggiungi test per:

- target con `terminated` e `truncated`;
- forma e limiti delle action;
- almeno un update che modifica i pesi previsti;
- checkpoint save/restore;
- resume dei contatori;
- multi-agent transition count;
- async policy version e chiusura worker;
- import e dispatch da `train.py`;
- caricamento da `run.py`.

Esegui:

```bash
python/.venv/bin/python -m unittest discover -s python/tests
python/.venv/bin/python -m compileall -q python
```

Infine fai uno smoke test su uno scenario minimo e non usare lo scenario complesso come
prima verifica dell'algoritmo.
