# Algorithm guide

Metis includes eight native TensorFlow/Keras trainers. Choosing among them is mostly about the action space, whether useful demonstrations exist, and whether the task benefits more from deterministic or stochastic exploration.

| Algorithm | Action space | Use it when |
|---|---|---|
| [DQN](dqn.md) | discrete | The agent chooses from a small, finite action set. |
| [PPO](ppo.md) | discrete, continuous, multi-discrete, hybrid | You want on-policy updates, native composed actions, optional expert imitation, or protected residual learning. |
| [DDPG](ddpg.md) | continuous | You need a simple deterministic actor-critic baseline. |
| [DDPG+BC](ddpg-bc.md) | continuous | Demonstrations should keep guiding DDPG during online learning. |
| [DDPGfD](ddpgfd.md) | continuous | Demonstrations are scarce and must remain protected in prioritized replay. |
| [TD3](td3.md) | continuous | You want a more conservative deterministic baseline than DDPG. |
| [TD3+BC](td3-bc.md) | continuous | You have demonstrations and want TD3 to refine them online. |
| [SAC](sac.md) | continuous | Broad stochastic exploration, replay reuse, and optional protected expert transitions are useful. |

`--algorithm auto` selects DQN for discrete actions, DDPG for continuous actions, and PPO for hybrid actions. That is a compatibility choice, not a claim that the selected algorithm is always best for the task.

Every native trainer accepts the relevant entries in [Shared training options](common-options.md). Each algorithm page lists the remaining flags owned by that learner, including their defaults and interactions. The runtime remains the authoritative source:

```bash
PYTHONPATH=python python/.venv/bin/python -m algorithms.sac --help
```

The optional Stable-Baselines3 backend intentionally has a smaller capability surface. See [Benchmarking training backends](../guides/benchmarking-backends.md) for its flags and limitations.

## Shared stability and demonstration boundaries

Native trainers share optional evaluation-driven learning rates and solved-task certification. The certification path requires single-agent, single-policy episodes and continues training when the pinned candidate fails its independent test. Single-policy SAC and the DDPG/TD3 family additionally support an EMA actor for evaluation and export; PPO and DQN do not. See [Training stability controls](../training-stability.md) for defaults, statistical assumptions and checkpoint requirements.

Demonstration retention and imitation are separate mechanisms. SAC's opt-in protection preserves complete expert transitions in uniform replay; DDPGfD already protects and prioritizes its expert prefix. Standard single-policy PPO keeps expert labels outside rollouts and adds optional BC or auxiliary imitation. PPO imitation is not combined with residual or independent multi-policy mode, and SAC demonstration routing is not implemented for independent policies. See [Manual demonstrations](../guides/manual-demonstrations.md) before choosing a workflow.

## A practical decision order

1. Use DQN for genuinely discrete controls.
2. Use PPO when the action contract is hybrid or when on-policy behavior is important.
3. Start continuous control with SAC when exploration is difficult.
4. Prefer TD3 when deterministic deployment and low action noise matter more.
5. Add expert pretraining, auxiliary imitation or a BC variant only when the demonstrations are valid for the exact observation, action, reward, and physics contract being trained.
6. Keep DDPG as a useful baseline. TD3 is usually the safer deterministic choice for a new task because its twin critics and delayed actor updates reduce overestimation.
