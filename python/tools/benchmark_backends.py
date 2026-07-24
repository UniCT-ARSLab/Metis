"""Run reproducible Metis-vs-SB3 experiments and aggregate their evaluation results."""

import argparse
import ast
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare Metis and SB3 on the same Godot scenario and transition budget.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", default="benchmarks")
    parser.add_argument("--backends", nargs="+", choices=["metis", "sb3"], default=["metis", "sb3"])
    parser.add_argument("--algorithm", choices=["dqn", "ddpg", "td3", "sac", "ppo"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    parser.add_argument("--total-timesteps", type=int, required=True)
    parser.add_argument("--evaluation-episodes", type=int, default=30)
    parser.add_argument("--evaluation-seed", type=int, default=10_000)
    parser.add_argument("--evaluation-max-steps", type=int, default=1000)
    parser.add_argument("--success-threshold", type=float, default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--multi-agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--sb3-multi-agent-partial-done",
        choices=["error", "reset-all"],
        default="error",
    )
    parser.add_argument("--base-port", type=int, default=6200)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--godot-bin", default=os.environ.get("GODOT_BIN"))
    parser.add_argument("--godot-project", default=None)
    parser.add_argument("--godot-scene", required=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-mode", choices=["project", "cpu", "light-gpu", "gpu"], default="light-gpu")
    parser.add_argument(
        "--metis-python",
        default=sys.executable,
        help="Python interpreter containing the Metis/TensorFlow dependencies.",
    )
    parser.add_argument(
        "--sb3-python",
        default=sys.executable,
        help="Python interpreter containing the Stable-Baselines3 dependencies.",
    )
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "trainer_args",
        nargs=argparse.REMAINDER,
        help="Additional trainer arguments placed after '--'.",
    )
    args = parser.parse_args(argv)
    if args.trainer_args[:1] == ["--"]:
        args.trainer_args = args.trainer_args[1:]
    return args


def validate_args(args):
    if args.total_timesteps < 1:
        raise ValueError("--total-timesteps must be at least 1")
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1")
    if args.evaluation_episodes < 1:
        raise ValueError("--evaluation-episodes must be at least 1")
    if not args.godot_bin:
        raise ValueError("--godot-bin or GODOT_BIN is required")
    forbidden = {
        "--backend",
        "--algorithm",
        "--checkpoint-dir",
        "--checkpoint-every",
        "--metrics-jsonl",
        "--save-replay-buffer",
        "--no-save-replay-buffer",
        "--total-timesteps",
        "--env-seed-base",
        "--base-port",
        "--num-envs",
        "--multi-agent",
        "--no-multi-agent",
        "--sb3-multi-agent-partial-done",
    }
    for value in args.trainer_args:
        option = value.split("=", 1)[0]
        if option in forbidden:
            raise ValueError(
                f"{option} is controlled by the benchmark runner and cannot appear after '--'"
            )


def _git_metadata(repo_root):
    def run(*command):
        result = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    revision = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {"revision": revision, "dirty": bool(status)}


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _run_and_tee(command, log_path, cwd):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        with log_path.open("w", encoding="utf-8") as log:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
        return_code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    return return_code, time.monotonic() - started


def _common_train_command(args, repo_root, backend, seed, run_dir):
    python_executable = args.metis_python if backend == "metis" else args.sb3_python
    command = [
        python_executable,
        str(repo_root / "python" / "train.py"),
        "--backend",
        backend,
        "--algorithm",
        args.algorithm,
        "--godot-bin",
        args.godot_bin,
        "--godot-scene",
        args.godot_scene,
        "--num-envs",
        str(args.num_envs),
        "--base-port",
        str(args.base_port),
        "--num-episodes",
        str(2_147_483_647),
        "--total-timesteps",
        str(args.total_timesteps),
        "--max-steps-per-episode",
        str(args.max_steps_per_episode),
        "--env-seed-base",
        str(seed),
        "--checkpoint-dir",
        str(run_dir / "checkpoints"),
        "--checkpoint-every",
        "0",
        "--metrics-jsonl",
        str(run_dir / "metrics.jsonl"),
        "--collector-mode",
        "sync",
        "--render-mode",
        args.render_mode,
        "--headless" if args.headless else "--no-headless",
        "--multi-agent" if args.multi_agent else "--no-multi-agent",
    ]
    if args.godot_project:
        command.extend(["--godot-project", args.godot_project])
    if args.algorithm != "ppo":
        command.append("--no-save-replay-buffer")
    command.extend(args.trainer_args)
    if backend == "metis":
        command.append("--no-best-checkpoint")
    else:
        command.extend(
            [
                "--sb3-multi-agent-partial-done",
                args.sb3_multi_agent_partial_done,
                "--evaluation-episodes",
                str(args.evaluation_episodes),
                "--evaluation-seed",
                str(args.evaluation_seed),
                "--evaluation-max-steps",
                str(args.evaluation_max_steps),
                "--evaluation-json",
                str(run_dir / "evaluation.json"),
            ]
        )
    return command


def _native_evaluation_command(args, repo_root, run_dir):
    command = [
        args.metis_python,
        str(repo_root / "python" / "run.py"),
        "--algorithm",
        args.algorithm,
        "--load-from",
        "policy",
        "--policy-path",
        str(run_dir / "checkpoints"),
        "--godot-bin",
        args.godot_bin,
        "--godot-scene",
        args.godot_scene,
        "--episodes",
        str(args.evaluation_episodes),
        "--max-steps",
        str(args.evaluation_max_steps),
        "--seed",
        str(args.evaluation_seed),
        "--port",
        str(args.base_port),
        "--summary-json",
        str(run_dir / "evaluation.json"),
        "--print-every",
        "0",
        "--execution-mode",
        "lockstep",
        "--headless",
        "--multi-agent" if args.multi_agent else "--no-multi-agent",
    ]
    if args.godot_project:
        command.extend(["--godot-project", args.godot_project])
    return command


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path):
    rows = []
    path = Path(path)
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _first_number(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        numbers = [_first_number(item) for item in value]
        numbers = [item for item in numbers if item is not None]
        return statistics.fmean(numbers) if numbers else None
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None
    if parsed is not None and parsed != value:
        number = _first_number(parsed)
        if number is not None:
            return number
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    return float(match.group(0)) if match else None


def _success_rate(row):
    value = row.get("success_rate", row.get("success"))
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "")
    percent = re.search(r"\(([-+]?\d+(?:\.\d+)?)%\)", text)
    if percent:
        return float(percent.group(1)) / 100.0
    ratio = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if ratio and int(ratio.group(2)) > 0:
        return int(ratio.group(1)) / int(ratio.group(2))
    return None


def summarize_learning_curve(metrics, success_threshold=None):
    points = []
    first_time = None
    time_to_threshold = None
    for row in metrics:
        timestep = _first_number(row.get("total_timesteps"))
        reward = _first_number(row.get("reward_mean", row.get("reward", row.get("rewards"))))
        recorded_at = _first_number(row.get("recorded_at"))
        if recorded_at is not None and first_time is None:
            first_time = recorded_at
        if timestep is not None and reward is not None:
            points.append((timestep, reward))
        if success_threshold is not None and time_to_threshold is None:
            success_rate = _success_rate(row)
            if (
                success_rate is not None
                and success_rate >= success_threshold
                and recorded_at is not None
                and first_time is not None
            ):
                time_to_threshold = max(0.0, recorded_at - first_time)

    rewards_by_timestep = {}
    for timestep, reward in points:
        rewards_by_timestep.setdefault(timestep, []).append(reward)
    points = sorted(
        (timestep, statistics.fmean(rewards))
        for timestep, rewards in rewards_by_timestep.items()
    )
    auc = None
    if len(points) >= 2 and points[-1][0] > points[0][0]:
        area = sum(
            0.5 * (left[1] + right[1]) * (right[0] - left[0])
            for left, right in zip(points, points[1:])
        )
        auc = area / (points[-1][0] - points[0][0])
    return {
        "logged_points": len(points),
        "last_logged_timestep": int(points[-1][0]) if points else 0,
        "learning_reward_auc": auc,
        "time_to_success_threshold_seconds": time_to_threshold,
    }


def _aggregate(records):
    grouped = {}
    for record in records:
        if record.get("status") != "completed" or not record.get("evaluation"):
            continue
        grouped.setdefault(record["backend"], []).append(record)

    summary = []
    for backend, backend_records in sorted(grouped.items()):
        row = {
            "backend": backend,
            "algorithm": backend_records[0]["algorithm"],
            "runs": len(backend_records),
        }
        fields = {
            "wall_seconds": lambda item: item["wall_seconds"],
            "transitions_per_second": lambda item: item["transitions_per_second"],
            "reward_mean": lambda item: item["evaluation"]["reward_mean"],
            "success_rate": lambda item: item["evaluation"]["success_rate"],
            "steps_mean": lambda item: item["evaluation"]["steps_mean"],
            "learning_reward_auc": lambda item: item["learning_curve"].get("learning_reward_auc"),
            "time_to_success_threshold_seconds": lambda item: item["learning_curve"].get(
                "time_to_success_threshold_seconds"
            ),
        }
        for name, getter in fields.items():
            values = [getter(item) for item in backend_records]
            values = [float(value) for value in values if value is not None]
            row[f"{name}_mean"] = statistics.fmean(values) if values else None
            row[f"{name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
        summary.append(row)
    return summary


def _write_summary_csv(path, rows):
    fields = [
        "backend",
        "algorithm",
        "runs",
        "wall_seconds_mean",
        "wall_seconds_std",
        "transitions_per_second_mean",
        "transitions_per_second_std",
        "reward_mean_mean",
        "reward_mean_std",
        "success_rate_mean",
        "success_rate_std",
        "steps_mean_mean",
        "steps_mean_std",
        "learning_reward_auc_mean",
        "learning_reward_auc_std",
        "time_to_success_threshold_seconds_mean",
        "time_to_success_threshold_seconds_std",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    repo_root = Path(__file__).resolve().parents[2]
    experiment_dir = Path(args.output_dir) / args.name
    if experiment_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Benchmark directory already exists: {experiment_dir}. "
                "Choose another --name or pass --force."
            )
        shutil.rmtree(experiment_dir)
    experiment_dir.mkdir(parents=True)

    config = {
        "name": args.name,
        "algorithm": args.algorithm,
        "backends": args.backends,
        "seeds": args.seeds,
        "total_timesteps": args.total_timesteps,
        "evaluation_episodes": args.evaluation_episodes,
        "evaluation_seed": args.evaluation_seed,
        "evaluation_max_steps": args.evaluation_max_steps,
        "num_envs": args.num_envs,
        "multi_agent": args.multi_agent,
        "sb3_multi_agent_partial_done": args.sb3_multi_agent_partial_done,
        "max_steps_per_episode": args.max_steps_per_episode,
        "godot_scene": args.godot_scene,
        "metis_python": args.metis_python,
        "sb3_python": args.sb3_python,
        "trainer_args": args.trainer_args,
        "git": _git_metadata(repo_root),
        "created_at": time.time(),
    }
    _write_json(experiment_dir / "config.json", config)

    records = []
    for backend in args.backends:
        for seed in args.seeds:
            run_dir = experiment_dir / backend / f"seed-{seed}"
            run_dir.mkdir(parents=True)
            command = _common_train_command(args, repo_root, backend, seed, run_dir)
            _write_json(run_dir / "command.json", {"command": command})
            print(f"\n===== backend={backend} seed={seed} =====", flush=True)
            return_code, wall_seconds = _run_and_tee(
                command,
                run_dir / "train.log",
                repo_root,
            )

            evaluation_code = None
            if return_code == 0 and backend == "metis":
                evaluation_command = _native_evaluation_command(args, repo_root, run_dir)
                _write_json(run_dir / "evaluation_command.json", {"command": evaluation_command})
                evaluation_code, _evaluation_seconds = _run_and_tee(
                    evaluation_command,
                    run_dir / "evaluation.log",
                    repo_root,
                )

            evaluation = _read_json(run_dir / "evaluation.json")
            status = (
                "completed"
                if return_code == 0
                and (evaluation_code in {None, 0})
                and evaluation is not None
                else "failed"
            )
            record = {
                "backend": backend,
                "algorithm": args.algorithm,
                "seed": seed,
                "status": status,
                "return_code": return_code,
                "evaluation_return_code": evaluation_code,
                "wall_seconds": wall_seconds,
                "transitions_per_second": args.total_timesteps / max(wall_seconds, 1e-9),
                "evaluation": evaluation,
                "learning_curve": summarize_learning_curve(
                    _read_jsonl(run_dir / "metrics.jsonl"),
                    success_threshold=args.success_threshold,
                ),
                "run_dir": str(run_dir),
            }
            _write_json(run_dir / "run.json", record)
            _append_jsonl(experiment_dir / "runs.jsonl", record)
            records.append(record)
            if status != "completed" and args.fail_fast:
                raise RuntimeError(f"Benchmark run failed: backend={backend} seed={seed}")

    summary = _aggregate(records)
    report = {
        "config": config,
        "runs": records,
        "summary": summary,
        "complete": all(record["status"] == "completed" for record in records),
    }
    _write_json(experiment_dir / "report.json", report)
    _write_summary_csv(experiment_dir / "summary.csv", summary)
    print(f"\nBenchmark report: {experiment_dir / 'report.json'}", flush=True)
    print(f"Benchmark table:  {experiment_dir / 'summary.csv'}", flush=True)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
