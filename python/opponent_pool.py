import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MANIFEST_VERSION = 1


def add_opponent_pool_arguments(parser):
    group = parser.add_argument_group("opponent pool self-play")
    group.add_argument(
        "--opponent-pool",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train one team against frozen snapshots of older versions of the shared policy.",
    )
    group.add_argument(
        "--opponent-pool-dir",
        default=None,
        help="Snapshot directory; defaults to <checkpoint-dir>/opponents.",
    )
    group.add_argument("--opponent-snapshot-every", type=int, default=100)
    group.add_argument("--opponent-pool-size", type=int, default=10)
    group.add_argument(
        "--opponent-current-probability",
        type=float,
        default=0.2,
        help="Probability of using the current policy for both teams instead of a snapshot.",
    )
    group.add_argument(
        "--opponent-sampling",
        choices=("uniform", "latest"),
        default="uniform",
    )
    group.add_argument(
        "--learner-team",
        default=None,
        help="Fixed team_id controlled by the learner. By default it is randomized per environment and episode.",
    )


def parse_team_id(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def validate_team_layout(envs, enabled):
    if not enabled:
        return []
    if not envs or not envs[0].multi_agent:
        raise ValueError("--opponent-pool requires --multi-agent")

    reference_ids = list(envs[0].agent_ids)
    reference_teams = list(envs[0].agent_team_ids)
    if any(team_id is None for team_id in reference_teams):
        missing = [
            agent_id
            for agent_id, team_id in zip(reference_ids, reference_teams)
            if team_id is None
        ]
        raise ValueError(
            "--opponent-pool requires a team_id for every agent; missing for "
            + ", ".join(missing)
        )

    teams = sorted(set(reference_teams), key=str)
    if len(teams) != 2:
        raise ValueError(
            f"--opponent-pool currently requires exactly two teams, got {teams}"
        )

    for env in envs[1:]:
        if list(env.agent_ids) != reference_ids or list(env.agent_team_ids) != reference_teams:
            raise ValueError("All environments must expose the same agent IDs and team layout")
    return teams


@dataclass(frozen=True)
class OpponentMatch:
    model: object
    label: str
    use_current_policy: bool
    state: object = None


class OpponentPool:
    def __init__(self, args, algorithm, model_factory, metadata=None, state_getter=None):
        self.enabled = bool(args.opponent_pool)
        self.algorithm = str(algorithm)
        self.model_factory = model_factory
        self.max_size = int(args.opponent_pool_size)
        self.snapshot_every = int(args.opponent_snapshot_every)
        self.current_probability = float(args.opponent_current_probability)
        self.sampling = str(args.opponent_sampling)
        self.fixed_learner_team = parse_team_id(args.learner_team)
        self.seed_base = int(args.env_seed_base)
        self.directory = Path(args.opponent_pool_dir or Path(args.checkpoint_dir) / "opponents")
        self.manifest_path = self.directory / "manifest.json"
        self.metadata = dict(metadata or {})
        self.state_getter = state_getter
        self.entries = []
        self._opponent_model = None

        if not self.enabled:
            return
        if self.max_size < 1:
            raise ValueError("--opponent-pool-size must be at least 1")
        if self.snapshot_every < 1:
            raise ValueError("--opponent-snapshot-every must be at least 1")
        if not 0.0 <= self.current_probability <= 1.0:
            raise ValueError("--opponent-current-probability must be between 0 and 1")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._load_manifest()

    def _load_manifest(self):
        if not self.manifest_path.exists():
            return
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if int(payload.get("version", 0)) != MANIFEST_VERSION:
            raise ValueError(f"Unsupported opponent pool manifest: {self.manifest_path}")
        if payload.get("algorithm") != self.algorithm:
            raise ValueError(
                f"Opponent pool algorithm={payload.get('algorithm')!r} does not match {self.algorithm!r}"
            )
        stored_metadata = dict(payload.get("metadata", {}))
        if stored_metadata != self.metadata:
            raise ValueError(
                "Opponent pool model metadata does not match the current scenario: "
                f"stored={stored_metadata}, current={self.metadata}"
            )
        self.entries = [
            entry for entry in payload.get("snapshots", [])
            if (self.directory / entry["file"]).is_file()
        ]

    def _write_manifest(self):
        payload = {
            "version": MANIFEST_VERSION,
            "algorithm": self.algorithm,
            "metadata": self.metadata,
            "snapshots": self.entries,
        }
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)

    def snapshot(self, model, episode, force=False):
        if not self.enabled:
            return None
        episode = int(episode)
        if not force and episode % self.snapshot_every != 0:
            return None
        if any(int(entry["episode"]) == episode for entry in self.entries):
            return None

        filename = f"policy-{episode:08d}.weights.h5"
        model.save_weights(str(self.directory / filename))
        entry = {"episode": episode, "file": filename}
        if self.state_getter is not None:
            state = self.state_getter()
            entry["state"] = np.asarray(state).tolist()
        self.entries.append(entry)
        self.entries.sort(key=lambda entry: int(entry["episode"]))
        while len(self.entries) > self.max_size:
            removed = self.entries.pop(0)
            old_path = self.directory / removed["file"]
            if old_path.exists():
                old_path.unlink()
        self._write_manifest()
        return self.directory / filename

    def start_episode(self, current_model, episode):
        if not self.enabled:
            return OpponentMatch(None, "disabled", True)
        if not self.entries:
            self.snapshot(current_model, episode, force=True)
        eligible_entries = [
            entry for entry in self.entries if int(entry["episode"]) <= int(episode)
        ]
        if not eligible_entries:
            return OpponentMatch(None, "current:no_eligible_snapshot", True)

        rng = random.Random(self.seed_base + int(episode) * 104729)
        if rng.random() < self.current_probability:
            return OpponentMatch(None, "current", True)

        entry = eligible_entries[-1] if self.sampling == "latest" else rng.choice(eligible_entries)
        if self._opponent_model is None:
            self._opponent_model = self.model_factory()
        self._opponent_model.load_weights(str(self.directory / entry["file"]))
        return OpponentMatch(
            self._opponent_model,
            f"snapshot:{int(entry['episode'])}",
            False,
            entry.get("state"),
        )

    def learner_mask(self, agent_team_ids, teams, episode, env_index, use_current_policy=False):
        if not self.enabled or use_current_policy:
            return np.ones((len(agent_team_ids),), dtype=np.bool_)

        if self.fixed_learner_team is not None:
            learner_team = self.fixed_learner_team
            if learner_team not in teams:
                raise ValueError(f"--learner-team={learner_team!r} is not present in scenario teams {teams}")
        else:
            rng = random.Random(self.seed_base + int(episode) * 130363 + int(env_index) * 433)
            learner_team = rng.choice(list(teams))
        return np.asarray([team_id == learner_team for team_id in agent_team_ids], dtype=np.bool_)

    @staticmethod
    def merge_actions(learner_actions, opponent_actions, learner_mask):
        result = np.asarray(learner_actions).copy()
        result[~np.asarray(learner_mask, dtype=np.bool_)] = np.asarray(opponent_actions)[
            ~np.asarray(learner_mask, dtype=np.bool_)
        ]
        return result
