import argparse
import os
import time
from pathlib import Path

import numpy as np

from godot_process_manager import GodotProcessManager
from scenario_gym_env import ScenarioGymEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Record manual expert demonstrations from a Godot scenario.")
    parser.add_argument("--output", default="demos/tank_target_demo.npz")
    parser.add_argument("--append", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--godot-bin", default=os.environ.get("GODOT_BIN"))
    parser.add_argument("--godot-project", default=None)
    parser.add_argument("--godot-scene", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6200)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--multi-agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--manual-agent-id", default=None)
    parser.add_argument("--record-all-agents", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recording-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-replication", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--step-delay", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--connect-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-godot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-every", type=int, default=20)
    return parser.parse_args()


def godot_recording_args(args):
    user_args = []
    if args.recording_mode:
        user_args.append("--recording-mode")
    if args.disable_replication:
        user_args.append("--disable-agent-replication")
    if args.manual_agent_id:
        user_args.append(f"--recording-agent-id={args.manual_agent_id}")
    elif args.agent_id:
        user_args.append(f"--recording-agent-id={args.agent_id}")
    return user_args


def _append_or_create(existing, key, values):
    values = np.asarray(values)
    if existing is None or key not in existing:
        return values
    return np.concatenate([existing[key], values], axis=0)


def save_dataset(path, samples, env, append=False):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    existing = None
    if append and output.exists():
        with np.load(output) as existing_file:
            existing = {key: existing_file[key] for key in existing_file.files}

    action_dtype = np.float32 if env.action_type == "continuous" else np.int32
    arrays = {
        "obs": np.asarray(samples["obs"], dtype=np.float32),
        "actions": np.asarray(samples["actions"], dtype=action_dtype),
        "rewards": np.asarray(samples["rewards"], dtype=np.float32),
        "next_obs": np.asarray(samples["next_obs"], dtype=np.float32),
        "dones": np.asarray(samples["dones"], dtype=np.float32),
        "agent_ids": np.asarray(samples["agent_ids"], dtype=str),
        "episode_indices": np.asarray(samples["episode_indices"], dtype=np.int32),
        "step_indices": np.asarray(samples["step_indices"], dtype=np.int32),
    }

    if existing is not None:
        arrays = {
            key: _append_or_create(existing, key, value)
            for key, value in arrays.items()
        }

    arrays["action_names"] = np.asarray(env.action_names, dtype=str)
    arrays["action_type"] = np.asarray([env.action_type], dtype=str)
    arrays["obs_dim"] = np.asarray([env.obs_dim], dtype=np.int32)
    arrays["action_size"] = np.asarray([env.action_size], dtype=np.int32)
    arrays["num_actions"] = np.asarray([env.num_actions], dtype=np.int32)
    if env.action_type == "continuous":
        arrays["action_low"] = np.asarray(env.action_low, dtype=np.float32)
        arrays["action_high"] = np.asarray(env.action_high, dtype=np.float32)
    np.savez_compressed(output, **arrays)
    return output, len(arrays["actions"])


def manual_action_for_env(env, manual_agent_id):
    if env.multi_agent:
        return {
            agent_id: "manual" if agent_id == manual_agent_id else _idle_action_for_env(env)
            for agent_id in env.agent_ids
        }
    return "manual"


def _idle_action_for_env(env):
    if env.action_type == "continuous":
        return np.zeros((env.action_size,), dtype=np.float32).tolist()
    return 0


def _clean_applied_action(action, env):
    if env.action_type == "continuous":
        return np.asarray(action, dtype=np.float32).reshape(env.action_size)
    return int(action)


def record_single_transition(samples, obs, action, reward, next_obs, done, agent_id, episode, step):
    samples["obs"].append(obs)
    samples["actions"].append(action)
    samples["rewards"].append(reward)
    samples["next_obs"].append(next_obs)
    samples["dones"].append(done)
    samples["agent_ids"].append(agent_id)
    samples["episode_indices"].append(episode)
    samples["step_indices"].append(step)


def main():
    args = parse_args()

    manager = None
    if not args.connect_only:
        manager = GodotProcessManager(
            godot_bin=args.godot_bin,
            project_dir=args.godot_project,
            scene_path=args.godot_scene,
        )
        manager.start_many(
            [args.port],
            headless=args.headless,
            debug=args.debug_godot,
            user_args=godot_recording_args(args),
        )
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
        manual_agent_id = args.manual_agent_id or env.agent_id
        if manual_agent_id not in env.agent_ids:
            raise RuntimeError(f"manual_agent_id={manual_agent_id!r} not in scenario agents: {env.agent_ids}")
        if args.recording_mode:
            env.configure(recording_mode=True, recording_agent_id=manual_agent_id, manage_agent_cameras=True)

        print(
            f"Recording demos output={args.output} agent={manual_agent_id} multi_agent={args.multi_agent} "
            f"obs_dim={env.obs_dim} action_type={env.action_type} action_size={env.action_size} "
            f"actions={env.action_names}",
            flush=True,
        )
        print("Focus the Godot window and drive with the configured input actions.", flush=True)

        samples = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "next_obs": [],
            "dones": [],
            "agent_ids": [],
            "episode_indices": [],
            "step_indices": [],
        }

        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            episode_reward = 0.0

            for step in range(args.max_steps):
                action_payload = manual_action_for_env(env, manual_agent_id)
                next_obs, reward, terminated, truncated, info = env.step(action_payload)
                done = bool(terminated or truncated)

                if env.multi_agent:
                    per_agent_rewards = np.asarray(info["per_agent_rewards"], dtype=np.float32)
                    per_agent_done = np.asarray(info["per_agent_done"], dtype=np.bool_)
                    per_agent_infos = info.get("per_agent_infos", [{} for _ in env.agent_ids])
                    agent_indices = range(len(env.agent_ids)) if args.record_all_agents else [env.agent_ids.index(manual_agent_id)]

                    for agent_idx in agent_indices:
                        agent_info = per_agent_infos[agent_idx] if agent_idx < len(per_agent_infos) else {}
                        applied_action = _clean_applied_action(
                            agent_info.get("applied_action", _idle_action_for_env(env)),
                            env,
                        )
                        record_single_transition(
                            samples,
                            obs[agent_idx],
                            applied_action,
                            float(per_agent_rewards[agent_idx]),
                            next_obs[agent_idx],
                            bool(per_agent_done[agent_idx] or done),
                            env.agent_ids[agent_idx],
                            episode,
                            step,
                        )
                    episode_reward += float(np.mean(per_agent_rewards))
                else:
                    agent_info = info.get("agent_info", {})
                    applied_action = _clean_applied_action(
                        agent_info.get("applied_action", _idle_action_for_env(env)),
                        env,
                    )
                    record_single_transition(
                        samples,
                        obs,
                        applied_action,
                        float(reward),
                        next_obs,
                        done,
                        manual_agent_id,
                        episode,
                        step,
                    )
                    episode_reward += float(reward)

                if args.print_every > 0 and step % args.print_every == 0:
                    print(
                        f"episode={episode:04d} step={step:04d} reward={reward:.4f} "
                        f"samples={len(samples['actions'])} terminated={terminated} truncated={truncated}",
                        flush=True,
                    )

                obs = next_obs
                if args.step_delay > 0:
                    time.sleep(args.step_delay)
                if done:
                    break

            print(
                f"episode={episode:04d} recorded_steps={step + 1} episode_reward={episode_reward:.4f}",
                flush=True,
            )

        output, total = save_dataset(args.output, samples, env, append=args.append)
        print(f"Saved demonstrations: {output} transitions={total}", flush=True)
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
