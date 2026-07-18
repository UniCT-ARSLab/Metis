# Metis in Godot

Il lato Godot di Metis descrive agenti e scenari attraverso nodi componibili configurati
dall'Inspector. Godot resta la fonte di verita' per fisica, observation, action, reward
ed eventi terminali.

## Albero consigliato

```text
Scenario
├── ScenarioController
│   ├── ScenarioRewardSystem
│   ├── ProgressProvider
│   └── ScenarioEventSystem
├── Agents
│   └── AgentBody
│       └── Agent
│           ├── ActionSpace
│           ├── ObservationSystem
│           └── RewardSystem
└── BridgeServer
```

I nomi possono cambiare; i `NodePath` esportati devono puntare ai nodi corretti.

## `Agent`

`Agent` e' il contenitore RL del corpo fisico. Mantiene l'ordine delle observation,
delega le action ad `ActionSpace` e le reward a `RewardSystem`. Il padre resta
responsabile della dinamica concreta: velocita', animazioni, collisioni, proiettili e
controlli manuali.

Le API imperative `add_action()` e `add_observation()` restano disponibili. Per scene
riusabili e configurabili da Inspector sono preferibili i componenti figli.

## `ActionSpace`

Figli disponibili:

- `ContinuousAction`: nome, dimensione, limiti policy e limiti opzionali di esplorazione;
- `DiscreteActionSet`: componente discreta composta da figli `DiscreteAction`;
- `DiscreteAction`: nome, target, metodo e argomenti da invocare.

L'applicazione delle continue resta nel corpo: `apply_action()` decodifica il vettore e
lo assegna agli input fisici. Le discrete possono essere eseguite automaticamente dal
set oppure gestite dal corpo quando serve una logica particolare.

## `ObservationSystem`

Ogni `ObservationSource` registra una o piu' callable nell'Agent. Fonti esistenti:

- cinematica e velocita' del corpo;
- raycast normalizzati e clearance frontale;
- target e appartenenza a team;
- navigazione rispetto a un Path3D;
- chiamata di un metodo personalizzato con `MethodObservationSource`;
- lettura diretta di una property con `PropertyObservationSource`.

`MethodObservationSource` e `PropertyObservationSource` espongono un `source_path`
opzionale relativo al corpo agente. Se resta vuoto leggono direttamente il corpo.
Il plugin `Metis Inspector`, abilitato nel progetto, aggiunge al campo testuale un
menu `Select...` che elenca soltanto metodi o property compatibili. Il valore resta
comunque modificabile manualmente e viene serializzato nella scena come `StringName` o
`NodePath`: il runtime non dipende dal plugin editor.

`PropertyObservationSource` puo' leggere anche sottoproprieta' tramite un percorso come
`velocity:x`. Per valori numerici offre una trasformazione opzionale tra intervalli con
clamp, utile per normalizzare una property senza aggiungere un metodo al corpo.

Le observation devono essere numeriche, finite, normalizzate quando possibile e con
dimensione stabile. Lo scenario chiama reset/refresh delle source per evitare dati fisici
stantii dopo un teletrasporto.

## Reward locali e di scenario

`RewardSystem` valuta figli derivati da `RewardComponent`. Il context include almeno:

- `agent`: nodo `Agent`;
- `body`: corpo controllato;
- `observations`: dictionary corrente;
- `events`: eventi locali accumulati;
- dati aggiunti dal controller, come step e action applicata.

Ogni componente restituisce il valore gia' pesato e puo' implementare
`reset_reward(context)`. I valori sono esposti separatamente nei log come reward terms.

`ScenarioRewardSystem` valuta `ScenarioRewardComponent` per ogni agente. Il context
aggiunge `agent_id`, progress, eventi di scenario come chiavi dirette, terminalita' e truncation. I metodi
opzionali `is_agent_stalled()` e `get_terminal_reason()` possono terminare il singolo
canale senza terminare automaticamente gli altri agenti.

## Eventi

`ScenarioEventSystem` aggrega figli `ScenarioEventSource`. Una source mantiene stato per
agent id e restituisce un dictionary. Le source incluse coprono:

- ingresso in un'Area;
- soglia di una observation;
- eventi impostati manualmente dallo scenario.

Un evento descrive un fatto; una reward decide quanto quel fatto vale. Separare i due
consente di cambiare shaping senza riscrivere collisioni e trigger.

## Progress

`ProgressProvider` astrae una metrica ordinabile. Non implica necessariamente un
Path3D: puo' rappresentare distanza completata, blocchi distrutti, salute del boss o una
fase del compito. Implementazioni incluse:

- `Path3DProgressProvider`;
- `MethodProgressProvider`, che delega a un metodo della scena.

Il controller usa il progress per reward, diagnostica e reset curriculum. Evita di
inserirlo nelle observation quando vuoi che la policy resti indipendente dalla mappa.

## `ScenarioController`

Responsabilita' principali:

- registrazione e replica degli agenti;
- reset deterministico con seed globale e seed specifico per agente;
- applicazione delle action e avanzamento dei frame fisici;
- composizione dei canali di risposta;
- terminalita' per agente e cache degli agenti conclusi;
- curriculum di spawn tramite progress provider;
- disattivazione di camera e UI in headless;
- modalita' recording senza replica.

Lo scenario puo' estenderlo, ma dovrebbe usare gli hook e i componenti prima di
duplicare `step()` o il formato di risposta.

## `BridgeServer`

Il bridge accetta un client TCP, abilita `TCP_NODELAY`, interpreta messaggi JSON per
riga e inoltra `spec`, `configure`, `reset` e `step` al controller. In lockstep la scena
avanza soltanto durante una richiesta; in realtime continua a processare mantenendo
l'ultima action.

Il bridge non deve conoscere reward, action names o classi concrete degli agenti.

## Reset fisico corretto

Dopo un teletrasporto:

1. azzera velocita' lineare e angolare;
2. azzera gli input dell'action precedente;
3. ripristina transform di corpo e oggetti dinamici;
4. forza l'aggiornamento di raycast e sensori;
5. resetta observation source, reward, eventi e progress;
6. attendi il numero minimo di frame fisici necessario soltanto se la scena lo richiede.

Un reset incompleto produce observation del vecchio episodio e contamina il replay.
