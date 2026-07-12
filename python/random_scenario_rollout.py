import argparse
import os
import time

from godot_process_manager import GodotProcessManager
from scenario_gym_env import ScenarioGymEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Run random actions on any ScenarioGymEnv scenario.")
    parser.add_argument("--godot-bin", default=os.environ.get("GODOT_BIN"))
    parser.add_argument("--godot-project", default=None)
    parser.add_argument("--godot-scene", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6200)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--multi-agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--print-reward-terms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--connect-only", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    manager = None
    if not args.connect_only:
        manager = GodotProcessManager(
            godot_bin=args.godot_bin,
            project_dir=args.godot_project,
            scene_path=args.godot_scene,
        )
        manager.start_many([args.port], headless=args.headless)
        print(f"Started Godot on port {args.port}", flush=True)

    env = None
    try:
        env = ScenarioGymEnv(
            host=args.host,
            port=args.port,
            seed=args.seed,
            agent_id=args.agent_id,
            multi_agent=args.multi_agent,
        )
        obs, info = env.reset(seed=args.seed)
        print(
            f"Scenario action_type={env.action_type} action_space={env.action_space} "
            f"obs_shape={getattr(obs, 'shape', None)} agents={env.agent_ids}",
            flush=True,
        )

        total_reward = 0.0
        for step in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            action_value = action.tolist() if hasattr(action, "tolist") else action
            reward_terms = ""
            if args.print_reward_terms:
                if env.multi_agent:
                    agent_infos = info.get("per_agent_infos", [])
                    reward_terms = (
                        f" local_terms={[item.get('local_term_rewards', {}) for item in agent_infos]} "
                        f"scenario_rewards={[item.get('scenario_reward', 0.0) for item in agent_infos]} "
                        f"scenario_terms={[item.get('scenario_terms', {}) for item in agent_infos]}"
                    )
                else:
                    agent_info = info.get("agent_info", {})
                    reward_terms = (
                        f" terms={agent_info.get('local_term_rewards', {})} "
                        f"scenario_reward={agent_info.get('scenario_reward', 0.0)} "
                        f"scenario_terms={agent_info.get('scenario_terms', {})}"
                    )
            print(
                f"step={step:04d} action={action_value} reward={reward:.4f} "
                f"total={total_reward:.4f} terminated={terminated} truncated={truncated}{reward_terms}",
                flush=True,
            )
            if args.delay > 0:
                time.sleep(args.delay)
            if terminated or truncated:
                break
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if manager is not None:
            manager.stop_all()


if __name__ == "__main__":
    main()
