# Metis Documentation

**Modular Environment for Training Intelligent Systems**

Questa cartella e' il punto di ingresso alla documentazione di Metis. I file sono
divisi per scopo: i tutorial costruiscono scenari completi, le guide spiegano procedure
riusabili e la reference descrive contratti e responsabilita' del codice.

## Per iniziare

1. Leggi [Architettura di Metis](reference/architecture.md).
2. Segui [Nuovo scenario e nuovo agente](tutorials/tutorial_nuovo_scenario_agente_rl.md).
3. Scegli un tutorial completo vicino al tuo problema.
4. Usa le guide di estensione quando Metis non offre ancora il componente che ti serve.

## Tutorial

- [Nuovo scenario e nuovo agente](tutorials/tutorial_nuovo_scenario_agente_rl.md): contratto minimo Godot-Python, training e run.
- [Breakout da zero](tutorials/tutorial_breakout_da_zero.md): agente discreto single-agent.
- [Pong multi-agent](tutorials/tutorial_pong_multi_agent.md): parameter sharing, self-play e opponent pool.
- [Tanks 2v2 ibrido](tutorials/tutorial_tanks_hybrid_multi_agent.md): azioni continue e discrete nello stesso agente.
- [Guida autonoma](tutorials/tutorial_guida_autonoma_path_vs_sensori.md): osservazioni con Path3D oppure solo sensori.
- [Soccer 3D](tutorials/tutorial_soccer_continuous_multi_agent.md): squadre, palla fisica e policy condivise.

## Guide

- [Dimostrazioni manuali](guides/manual_demonstrations.md): registrazione, prefill e behavior cloning.
- [Esportare una policy](guides/esportare_policy.md): bundle Keras, TFLite, ONNX e contratto di inferenza.
- [Aggiungere un algoritmo RL a Metis](guides/aggiungere_algoritmo_rl.md): backend, async collector, checkpoint, runner e test.
- [Estendere Metis in Godot](guides/estendere_framework_godot.md): selettori Inspector e nuove observation, reward, event, action e progress provider.

## Reference

- [Architettura di Metis](reference/architecture.md): flusso completo tra Godot, Gymnasium e learner.
- [Struttura Python](reference/python.md): responsabilita' di CLI, algoritmi, core, env e tool.
- [Metis in Godot](reference/godot.md): albero dei nodi, contratti e contesti runtime.

## Regola di manutenzione

I comandi destinati all'utente devono passare solo da `python/train.py`, `python/run.py`,
`python/recorder.py` e `python/export.py`. I moduli in `python/algorithms`, `python/core` e `python/envs`
sono implementazione interna e possono essere importati nei test o nelle estensioni.
