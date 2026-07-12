import json
import re
from pathlib import Path


def add_log_format_argument(parser):
    parser.add_argument(
        "--log-format",
        choices=["pretty", "compact"],
        default="pretty",
        help="Pretty prints one readable block per episode; compact keeps one machine-friendly line.",
    )


def print_episode_metrics(episode, sections, log_format="pretty"):
    normalized = [
        (name, [(str(key), str(value)) for key, value in metrics])
        for name, metrics in sections
        if metrics
    ]
    if log_format == "compact":
        fields = [f"episode={episode:04d}"]
        for _name, metrics in normalized:
            for key, value in metrics:
                compact_value = json.dumps(value) if any(char.isspace() for char in value) else value
                fields.append(f"{key}={compact_value}")
        print(" ".join(fields), flush=True)
        return

    print(f"episode={episode:04d}", flush=True)
    for name, metrics in normalized:
        values = "  ".join(f"{key}={value}" for key, value in metrics)
        print(f"  {name:<9} {values}", flush=True)


def normalize_checkpoint_path(path):
    path = str(path)
    if path.endswith(".index"):
        return path[:-len(".index")]
    if ".data-" in path:
        return path.split(".data-", 1)[0]
    return path


def resolve_resume_checkpoint(args, checkpoint_manager):
    requested_value = getattr(args, "resume_checkpoint", None)
    if requested_value:
        requested = Path(requested_value).expanduser()
        if requested.name.isdigit():
            requested = Path(args.checkpoint_dir) / f"ckpt-{requested.name}"
        elif requested.parent == Path("."):
            requested = Path(args.checkpoint_dir) / requested.name
        checkpoint_path = normalize_checkpoint_path(requested)
        if not Path(checkpoint_path + ".index").is_file():
            raise FileNotFoundError(
                f"Checkpoint {checkpoint_path!r} does not exist (missing {checkpoint_path + '.index'!r})"
            )
        return checkpoint_path

    if getattr(args, "resume", False):
        if not checkpoint_manager.latest_checkpoint:
            raise FileNotFoundError(f"No checkpoint found in --checkpoint-dir {args.checkpoint_dir!r}")
        return checkpoint_manager.latest_checkpoint
    return None


def checkpoint_number(checkpoint_path):
    match = re.search(r"ckpt-(\d+)$", str(checkpoint_path))
    return int(match.group(1)) if match else None


def replay_path_for_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    number = checkpoint_number(checkpoint_path)
    filename = f"replay-{number}.npz" if number is not None else checkpoint_path.name + ".replay.npz"
    return checkpoint_path.parent / filename


def save_replay_snapshot(checkpoint_path, checkpoint_manager, buffer):
    replay_path = replay_path_for_checkpoint(checkpoint_path)
    transitions = buffer.save(replay_path)
    print(f"Saved replay buffer: {replay_path} transitions={transitions}", flush=True)
    cleanup_stale_replay_buffers(checkpoint_manager)
    return replay_path


def cleanup_stale_replay_buffers(checkpoint_manager):
    retained_numbers = {
        checkpoint_number(path)
        for path in checkpoint_manager.checkpoints
        if checkpoint_number(path) is not None
    }
    for replay_path in Path(checkpoint_manager.directory).glob("replay-*.npz"):
        match = re.fullmatch(r"replay-(\d+)\.npz", replay_path.name)
        if match and int(match.group(1)) not in retained_numbers:
            replay_path.unlink()


def restore_replay_buffer(args, checkpoint_path, buffer):
    replay_path = replay_path_for_checkpoint(checkpoint_path)
    if replay_path.is_file():
        restored = buffer.load(replay_path)
        print(
            f"Restored replay buffer: {replay_path} transitions={restored}/{args.replay_capacity}",
            flush=True,
        )
        return restored
    if getattr(args, "require_replay_buffer", False):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path!r} has no replay buffer at {str(replay_path)!r}")
    print(
        f"WARNING: no replay buffer found for {checkpoint_path}; updates remain disabled until "
        f"replay_size reaches replay_warmup={args.replay_warmup}.",
        flush=True,
    )
    return 0
