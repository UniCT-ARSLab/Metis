# Algorithm guide

Metis includes eight native TensorFlow/Keras trainers. Choosing among them is mostly
about the action space, whether useful demonstrations exist, and whether the task
benefits more from deterministic or stochastic exploration.

| Algorithm | Action space | Use it when |
|---|---|---|
| [DQN](dqn.md) | discrete | The agent chooses from a small, finite action set. |
| [PPO](ppo.md) | discrete, continuous, hybrid | You want on-policy updates, native hybrid actions, or protected residual learning. |
| [DDPG](ddpg.md) | continuous | You need a simple deterministic actor-critic baseline. |
| [DDPG+BC](ddpg-bc.md) | continuous | Demonstrations should keep guiding DDPG during online learning. |
| [DDPGfD](ddpgfd.md) | continuous | Demonstrations are scarce and must remain protected in prioritized replay. |
| [TD3](td3.md) | continuous | You want a more conservative deterministic baseline than DDPG. |
| [TD3+BC](td3-bc.md) | continuous | You have demonstrations and want TD3 to refine them online. |
| [SAC](sac.md) | continuous | Broad stochastic exploration is useful and sample efficiency matters. |

`--algorithm auto` selects DQN for discrete actions, DDPG for continuous actions, and
PPO for hybrid actions. That is a compatibility choice, not a claim that the selected
algorithm is always best for the task.

Every native trainer accepts the relevant entries in [Shared training options](common-options.md).
Each algorithm page lists the remaining flags owned by that learner, including their
defaults and interactions. The runtime remains the authoritative source:

```bash
PYTHONPATH=python python/.venv/bin/python -m algorithms.sac --help
```

The optional Stable-Baselines3 backend intentionally has a smaller capability surface.
See [Benchmarking training backends](../guides/benchmarking-backends.md) for its flags
and limitations.

## A practical decision order

1. Use DQN for genuinely discrete controls.
2. Use PPO when the action contract is hybrid or when on-policy behavior is important.
3. Start continuous control with SAC when exploration is difficult.
4. Prefer TD3 when deterministic deployment and low action noise matter more.
5. Add a BC or demonstration variant only when the demonstrations are valid for the
   exact observation, action, reward, and physics contract being trained.
6. Keep DDPG as a useful baseline. TD3 is usually the safer deterministic choice for a
   new task because its twin critics and delayed actor updates reduce overestimation.
