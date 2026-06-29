import numpy as np
from team_battle_gym_env import TeamBattleGymEnv


def main():
    env = TeamBattleGymEnv(port=5555, seed=7)
    obs, info = env.reset()
    total = np.zeros((2,), dtype=np.float32)

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += info["per_agent_rewards"]
        print(f"step={step:03d} action={action} mean_reward={reward:.4f} per_agent={info['per_agent_rewards']}")
        if terminated or truncated:
            break

    print("total per-agent reward:", total)
    print("final info:", info)
    env.close()


if __name__ == "__main__":
    main()
