import json
import re
from pathlib import Path


def checkpoint_number(path):
    match = re.search(r"ckpt-(\d+)\.zip$", Path(path).name)
    return int(match.group(1)) if match else -1


def state_path_for_model(model_path):
    return Path(model_path).with_suffix(".json")


def replay_path_for_model(model_path):
    path = Path(model_path)
    number = checkpoint_number(path)
    if number >= 0:
        return path.parent / f"replay-{number}.pkl"
    return path.parent / "final_replay_buffer.pkl"


def load_training_state(model_path):
    if model_path is None:
        return {}
    state_path = state_path_for_model(model_path)
    if not state_path.is_file():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def latest_checkpoint(checkpoint_dir):
    directory = Path(checkpoint_dir)
    final_model = directory / "final_model.zip"
    candidates = list(directory.glob("ckpt-*.zip"))
    if final_model.is_file():
        candidates.append(final_model)
    if not candidates:
        return None

    def progress(path):
        state = load_training_state(path)
        return (
            int(state.get("num_timesteps", 0)),
            int(state.get("completed_episodes", max(checkpoint_number(path), 0))),
            path.stat().st_mtime,
        )

    return max(candidates, key=progress)
