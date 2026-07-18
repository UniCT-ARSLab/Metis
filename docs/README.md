# Documentazione di Metis

Questa documentazione e' pensata per essere letta a pezzi. Non serve conoscere tutto
il framework prima di creare il primo agente: scegli il percorso piu' vicino a quello
che vuoi costruire e torna alla reference quando incontri un nodo o un contratto che
vuoi capire meglio.

## Da dove cominciare

Se e' la prima volta che apri il progetto:

1. leggi [come sono divise le responsabilita'](reference/architecture.md) tra Godot,
   bridge e Python;
2. segui [Nuovo scenario e nuovo agente](tutorials/tutorial_nuovo_scenario_agente_rl.md)
   per vedere il contratto minimo funzionante;
3. scegli uno degli esempi completi qui sotto;
4. valida sempre la scena con `random_rollout.py` prima di avviare un training lungo.

## Voglio costruire uno scenario

| Obiettivo | Tutorial consigliato |
|---|---|
| Un primo agente con azioni discrete | [Breakout da zero](tutorials/tutorial_breakout_da_zero.md) |
| Due agenti identici che competono | [Pong multi-agent](tutorials/tutorial_pong_multi_agent.md) |
| Movimento continuo e azione discreta insieme | [Tanks 2v2 ibrido](tutorials/tutorial_tanks_hybrid_multi_agent.md) |
| Un veicolo che segue un percorso noto | [Guida autonoma con Path3D](tutorials/tutorial_guida_autonoma_path_vs_sensori.md) |
| Un veicolo che conosce soltanto i propri sensori | [Guida autonoma sensor-only](tutorials/tutorial_guida_autonoma_path_vs_sensori.md) |
| Squadre, palla fisica e controllo arcade 3D | [Soccer 3D](tutorials/tutorial_soccer_continuous_multi_agent.md) |

I tutorial non sono soltanto esempi di training. Mostrano come organizzare scene,
reset, sensori, reward, eventi, curriculum, multi-agent e comandi di esecuzione del
modello finale.

## Voglio usare una funzione del framework

- [Dimostrazioni manuali](guides/manual_demonstrations.md) spiega recorder, dataset,
  replay prefill, behavior cloning, DDPGfD e TD3+BC.
- [Esportare una policy](guides/esportare_policy.md) copre bundle Keras, vecchi file
  `.h5`, TFLite, ONNX e contratto di inferenza.
- [Aggiungere un algoritmo RL](guides/aggiungere_algoritmo_rl.md) segue il percorso da
  parser e modello fino a collector, checkpoint, runner e test.
- [Estendere Metis in Godot](guides/estendere_framework_godot.md) mostra come creare
  nuove action, observation source, reward, eventi e progress provider.

## Voglio capire come funziona

- [Architettura](reference/architecture.md): flusso di un episodio, contratto TCP,
  single/multi-agent, async, checkpoint e replay.
- [Lato Godot](reference/godot.md): `Agent`, `ActionSpace`, `ObservationSystem`, reward,
  eventi, progress, reset e `ScenarioController`.
- [Lato Python](reference/python.md): entrypoint pubblici, algoritmi, core, environment,
  strumenti e dipendenze interne.

## Cosa supporta Metis

In sintesi, il framework gestisce:

- action space discreti, continui e ibridi;
- DQN, PPO, DDPG, SAC, TD3 e varianti basate su dimostrazioni;
- uno o piu' environment, in raccolta sincrona o asincrona;
- single-agent, multi-agent con policy condivisa e self-play a squadre;
- observation e reward componibili dall'Inspector;
- eventi, progress provider, curriculum e reset randomizzati;
- registrazione manuale, checkpoint completi e valutazione della best policy;
- esecuzione realtime o lockstep ed export Keras, TFLite e ONNX;
- Linux con CPU/CUDA e macOS Apple Silicon con Metal.

Durante il training le finestre Godot possono usare il renderer del progetto, OpenGL
leggero, rendering software su CPU oppure Vulkan Forward+. La scelta avviene con
`--render-mode`; il default `light-gpu` lascia piu' risorse al learner. L'opzione conta
solo per le istanze visibili e puo' essere combinata con `--render-env-count` per
mostrare una sola preview anche quando gli environment sono molti. Su Linux, se
OpenGL ricade su `llvmpipe`, questa modalita' prova automaticamente la GPU discreta
indicata da `switcherooctl`. La sezione
[Uno o molti environment](../README.md#uno-o-molti-environment) contiene la tabella
completa e un esempio.

Per la matrice completa, gli esempi di comando e i limiti attuali consulta il
[README principale](../README.md).

## Convenzioni della documentazione

I comandi rivolti a chi usa Metis passano da quattro file:

```text
python/train.py
python/run.py
python/recorder.py
python/export.py
```

I moduli in `python/algorithms`, `python/core` e `python/envs` sono dettagli interni o
punti di estensione. Nei tutorial si configurano prima i nodi disponibili; si aggiunge
codice personalizzato soltanto quando la regola appartiene davvero allo scenario.

Quando comportamento e documentazione non coincidono, la scena e i test sono la fonte
da verificare per prima. Una modifica a observation, action space, reward o fisica deve
essere riportata anche nel tutorial interessato: sono parti dello stesso contratto di
training, non dettagli cosmetici.
