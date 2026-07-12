import numpy as np
from team_battle_gym_env import TeamBattleGymEnv


def main():
    env = TeamBattleGymEnv(port=5555, seed=123)
    obs, info = env.reset()
    print("obs shape:", obs.shape)
    print("agent ids:", info["agent_ids"])
    print("alive:", info["alive_mask"])

    for i in range(3):
        action = np.array([0, 0], dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)
        print(i, reward, terminated, truncated, info["per_agent_rewards"])
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    main()
