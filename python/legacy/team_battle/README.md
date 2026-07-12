# TeamBattle Legacy Scripts

Questa cartella contiene la prima linea sperimentale del progetto, basata su uno scenario 2v2 TeamBattle/Tanks e trainer DQN dedicati.

Questi file sono stati archiviati perché il framework principale ora usa:

- `python/scenario_gym_env.py`
- `python/train_generic.py`
- `python/run_generic_policy.py`
- `python/random_scenario_rollout.py`

I file legacy possono ancora essere utili come riferimento storico per:

- DQN condiviso;
- self-play iniziale;
- vecchio wrapper `TeamBattleGymEnv`;
- vecchio runner DQN.

Se devi eseguirli, avviali dal repository root impostando `PYTHONPATH=python`, ad esempio:

```bash
PYTHONPATH=python python python/legacy/team_battle/train_shared_dqn.py --headless
```

Per nuovi scenari o nuovi agenti usa sempre gli script generici nella cartella `python/`.
