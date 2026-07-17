# Architettura del framework

Il progetto separa la simulazione dal training:

- Godot possiede mondo, fisica, agenti, observation, reward ed eventi terminali;
- Python possiede ambiente Gymnasium, raccolta delle transizioni, replay o rollout,
  learner TensorFlow/Keras, checkpoint e valutazione;
- il bridge TCP scambia messaggi JSON e mantiene il passo della simulazione esplicito.

Questa separazione permette di cambiare scenario senza scrivere un nuovo trainer e di
aggiungere algoritmi senza incorporare TensorFlow nel progetto Godot.

## Flusso runtime

```text
python/train.py
    |
    +-- selezione algoritmo
    +-- GodotProcessManager avvia N processi
    +-- ScenarioGymEnv apre una connessione per processo
            |
            +-- BridgeServer
                    |
                    +-- ScenarioController
                            +-- Agent[]
                            +-- ScenarioEventSystem
                            +-- ScenarioRewardSystem
                            +-- ProgressProvider
```

Per ogni episodio Python invia `reset`. Godot randomizza lo scenario, azzera componenti
e restituisce le observation iniziali. A ogni `step`, Python invia una o piu' azioni,
Godot le applica, avanza `physics_frames_per_step` frame fisici e restituisce:

- observation successiva;
- reward totale e termini diagnostici;
- `terminated`, quando il compito termina realmente;
- `truncated`, quando termina soltanto per un limite esterno;
- informazioni per agente, eventi, progress e causa terminale.

Il learner deve usare `terminated` per interrompere il bootstrap. Una truncation non
equivale alla fine naturale del mondo e conserva il valore dello stato successivo.

## Contratto dello scenario

Il `BridgeServer` deve avere `controller_path` impostato sullo `ScenarioController`.
Il controller espone al bridge almeno:

- `get_spec()` o le informazioni equivalenti sugli agenti;
- `configure(config)`;
- `reset_episode_with_request(request)`;
- `step(actions)`.

Il controller generico implementa gia' questo contratto. Lo scenario dovrebbe
specializzare composizione, reset ed eventi, non duplicare il protocollo TCP.

Ogni corpo controllato deve essere registrato in `controlled_agents` e fornire i metodi
richiamati dal controller, normalmente delegando al figlio `Agent`:

- `get_observations()`;
- `get_action_space()`;
- `apply_action(action)`;
- `get_reward(context)`;
- reset dello stato fisico e dell'agente.

## Spazi di azione

`ActionSpace` aggrega i figli e classifica automaticamente lo spazio:

- un solo `DiscreteActionSet`: `discrete`;
- uno o piu' `ContinuousAction`: `continuous`;
- componenti discrete e continue, oppure piu' componenti discrete: `hybrid`.

Lo spazio dichiarato da Godot e' la fonte di verita'. Python non deve contenere nomi
come `accelerate`, `steer` o `shoot`: legge dimensioni, limiti e componenti dalla spec.

## Observation

`ObservationSystem` chiede a ogni `ObservationSource` di registrare valori nel nodo
`Agent`. Una observation puo' essere scalare, booleana, `Vector2`, `Vector3` o array;
`Agent.get_observation_vector()` la appiattisce mantenendo l'ordine di registrazione.

La dimensione e l'ordine devono rimanere stabili per tutta la vita del modello. Cambiare
una observation, il suo ordine o il suo significato rende normalmente incompatibili i
checkpoint precedenti.

## Reward ed eventi

Le reward sono divise in due livelli:

- `RewardSystem`, figlio del singolo `Agent`, valuta movimento, input, sensori e stato
  locale del corpo;
- `ScenarioRewardSystem`, figlio dello scenario, valuta progress, goal, vittoria,
  collisioni globali e condizioni che coinvolgono piu' oggetti.

Il totale di un agente e' personale: una scena multi-agent non somma automaticamente le
reward di tutti. Il parameter sharing condivide i pesi, non il ritorno dell'episodio.

`ScenarioEventSystem` converte collisioni, aree o condizioni osservabili in eventi con
nomi stabili. Reward e terminalita' possono quindi dipendere dagli eventi senza legarsi
direttamente alla scena concreta.

## Single-agent e multi-agent

In single-agent `step` accetta una sola azione e restituisce il primo canale. In
multi-agent usa una mappa di azioni e restituisce un canale per agente. Gli agenti con
observation/action space compatibili possono usare la stessa policy; ogni loro
transizione entra comunque separatamente nel replay o nel rollout.

Piu' agenti non significano piu' modelli. Il comportamento predefinito e' parameter
sharing. Policy separate richiedono un'estensione esplicita del trainer e una chiara
assegnazione `agent_id -> policy_id`.

## Collector sync e async

Il collector sincrono aspetta insieme gli environment e rende il flusso piu' semplice da
riprodurre. Il collector asincrono lascia avanzare ogni environment indipendentemente e
pubblica periodicamente snapshot della policy ai worker.

L'async non mescola le transizioni: ogni evento conserva worker, episodio e policy
version. Puo' pero' introdurre policy lag, perche' un collector termina un episodio con
una snapshot leggermente meno recente del learner. I parametri di sincronizzazione e la
coda controllano questo compromesso.

## Checkpoint, best policy e replay

I checkpoint TensorFlow conservano modello, target network, optimizer e contatori che
l'algoritmo registra. Per algoritmi off-policy il replay viene salvato separatamente in
`replay-<episodio>.npz`. Un resume completo dovrebbe ripristinare entrambi.

La best policy viene valutata in un processo separato usando `python/run.py`. La
directory `best/` contiene checkpoint promossi soltanto dopo una valutazione congelata,
quindi non va confusa con l'ultimo checkpoint cronologico.

## File runtime

I processi Godot scrivono i log in `.runtime/godot_logs/`. La directory e' ignorata da
Git e puo' essere cancellata a processi fermi. Modelli, checkpoint e dataset demo hanno
invece valore persistente e vanno conservati nelle directory configurate dall'utente.
