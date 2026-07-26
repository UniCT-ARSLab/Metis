"""Policy assignment and persistence helpers for independent multi-policy training."""

import argparse
import copy
import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MANIFEST_FILENAME = "multi_policy.json"
MANIFEST_FORMAT = "metis-multi-policy"
MANIFEST_VERSION = 1
SHARED_POLICY_ID = "shared"


def add_multi_policy_arguments(parser, *, training=True):
    group = parser.add_argument_group("independent multi-policy")
    group.add_argument(
        "--multi-policy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train or run independent policies inside one multi-agent scenario. "
            "Without this flag, --multi-agent keeps parameter sharing."
        ),
    )
    group.add_argument(
        "--policy-assignment",
        choices=("auto", "policy_id", "team", "agent"),
        default="auto",
        help=(
            "How agents are assigned to policies. auto prefers distinct policy_id values "
            "from Godot, then team_id, and finally one policy per agent."
        ),
    )
    if training:
        group.add_argument(
            "--train-policy",
            action="append",
            default=[],
            help=(
                "Policy ID to update. Repeat to train a subset; by default every assigned "
                "policy learns."
            ),
        )


def safe_policy_key(policy_id):
    value = re.sub(r"[^A-Za-z0-9_]+", "_", str(policy_id)).strip("_")
    if not value:
        value = "policy"
    if value[0].isdigit():
        value = "policy_" + value
    return value


def _team_policy_id(team_id):
    return f"team_{team_id}"


@dataclass(frozen=True)
class PolicyAssignment:
    mode: str
    policy_ids: tuple
    agent_to_policy: dict
    trainable_policy_ids: tuple
    policy_keys: dict

    def policy_for_agent(self, agent_id):
        return self.agent_to_policy[str(agent_id)]

    def indices_for(self, agent_ids, policy_id):
        return [
            index
            for index, agent_id in enumerate(agent_ids)
            if self.policy_for_agent(agent_id) == policy_id
        ]

    def is_trainable(self, policy_id):
        return str(policy_id) in self.trainable_policy_ids

    def key_for(self, policy_id):
        return self.policy_keys[str(policy_id)]

    def as_dict(self, algorithm=None):
        payload = {
            "format": MANIFEST_FORMAT,
            "format_version": MANIFEST_VERSION,
            "assignment_mode": self.mode,
            "policy_ids": list(self.policy_ids),
            "trainable_policy_ids": list(self.trainable_policy_ids),
            "agent_to_policy": dict(self.agent_to_policy),
            "policy_keys": dict(self.policy_keys),
        }
        if algorithm is not None:
            payload["algorithm"] = str(algorithm)
        return payload


class MultiPolicySnapshot:
    """Atomically publish and synchronize a complete set of policy weights."""

    def __init__(self, weights_by_policy, state_by_policy=None):
        self._condition = threading.Condition()
        self._version = 0
        self._weights = self._copy_weight_map(weights_by_policy)
        self._state = copy.deepcopy(state_by_policy or {})

    @staticmethod
    def _copy_weights(weights):
        return tuple(np.array(weight, copy=True) for weight in weights)

    @classmethod
    def _copy_weight_map(cls, weights_by_policy):
        return {
            str(policy_id): cls._copy_weights(weights)
            for policy_id, weights in weights_by_policy.items()
        }

    @property
    def version(self):
        with self._condition:
            return self._version

    def publish(self, weights_by_policy, state_by_policy=None):
        copied_weights = self._copy_weight_map(weights_by_policy)
        copied_state = copy.deepcopy(state_by_policy or {})
        with self._condition:
            self._weights = copied_weights
            self._state = copied_state
            self._version += 1
            self._condition.notify_all()
            return self._version

    def sync_model(self, models_by_policy, current_version=-1):
        version, _state = self.sync_model_with_state(
            models_by_policy,
            current_version,
        )
        return version

    def sync_model_with_state(self, models_by_policy, current_version=-1):
        with self._condition:
            if current_version == self._version:
                return current_version, copy.deepcopy(self._state)
            version = self._version
            weights = self._copy_weight_map(self._weights)
            state = copy.deepcopy(self._state)

        expected = set(weights)
        current = {str(policy_id) for policy_id in models_by_policy}
        if current != expected:
            raise ValueError(
                "Collector policy models do not match the published snapshot: "
                f"expected={sorted(expected)}, got={sorted(current)}"
            )
        for policy_id, model in models_by_policy.items():
            model.set_weights(weights[str(policy_id)])
        return version, state

    def wait_for_newer(self, current_version, stop_event, timeout=0.2):
        with self._condition:
            if self._version > current_version or stop_event.is_set():
                return self._version > current_version
            timeout = max(0.0, float(timeout))
            if timeout == 0.0:
                return False
            while (
                self._version <= current_version
                and not stop_event.is_set()
            ):
                self._condition.wait(timeout=timeout)
            return self._version > current_version


def _resolve_assignment_mode(env, requested):
    requested = str(requested)
    if requested != "auto":
        return requested

    declared = [str(value or SHARED_POLICY_ID) for value in env.agent_policy_ids]
    if len(set(declared)) > 1:
        return "policy_id"

    teams = list(env.agent_team_ids)
    if teams and all(team is not None for team in teams) and len(set(teams)) > 1:
        return "team"
    return "agent"


def build_policy_assignment(envs, args):
    if not bool(getattr(args, "multi_policy", False)):
        return None
    if not bool(getattr(args, "multi_agent", False)):
        raise ValueError("--multi-policy requires --multi-agent")
    if not envs:
        raise ValueError("--multi-policy requires at least one environment")

    reference = envs[0]
    if len(reference.agent_ids) < 2:
        raise ValueError("--multi-policy requires at least two agents in the scenario")

    mode = _resolve_assignment_mode(reference, getattr(args, "policy_assignment", "auto"))
    if mode == "policy_id":
        values = [str(value or SHARED_POLICY_ID) for value in reference.agent_policy_ids]
    elif mode == "team":
        if any(team is None for team in reference.agent_team_ids):
            missing = [
                agent_id
                for agent_id, team in zip(reference.agent_ids, reference.agent_team_ids)
                if team is None
            ]
            raise ValueError(
                "--policy-assignment team requires team_id for every agent; missing for "
                + ", ".join(missing)
            )
        values = [_team_policy_id(team) for team in reference.agent_team_ids]
    elif mode == "agent":
        values = [str(agent_id) for agent_id in reference.agent_ids]
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(f"Unsupported policy assignment mode {mode!r}")

    agent_to_policy = OrderedDict(zip(reference.agent_ids, values))
    policy_ids = tuple(dict.fromkeys(values))
    if len(policy_ids) < 2:
        raise ValueError(
            f"--multi-policy resolved only one policy ({policy_ids[0]!r}). "
            "Assign distinct Agent.policy_id values, choose --policy-assignment team, "
            "or choose --policy-assignment agent."
        )

    requested_trainable = tuple(
        dict.fromkeys(str(value) for value in getattr(args, "train_policy", []) if str(value))
    )
    unknown = sorted(set(requested_trainable) - set(policy_ids))
    if unknown:
        raise ValueError(
            f"--train-policy contains unknown policies {unknown}; available={list(policy_ids)}"
        )
    trainable = requested_trainable or policy_ids

    policy_keys = {}
    used_keys = set()
    for policy_id in policy_ids:
        base = safe_policy_key(policy_id)
        key = base
        suffix = 2
        while key in used_keys:
            key = f"{base}_{suffix}"
            suffix += 1
        used_keys.add(key)
        policy_keys[policy_id] = key

    assignment = PolicyAssignment(
        mode=mode,
        policy_ids=policy_ids,
        agent_to_policy=dict(agent_to_policy),
        trainable_policy_ids=tuple(trainable),
        policy_keys=policy_keys,
    )
    validate_policy_assignment_across_envs(envs, assignment)
    return assignment


def validate_policy_assignment_across_envs(envs, assignment):
    reference_ids = list(envs[0].agent_ids)
    for env in envs:
        if list(env.agent_ids) != reference_ids:
            raise ValueError(
                "Independent multi-policy requires every environment to expose the same "
                "agent IDs in the same order"
            )
        if assignment.mode == "policy_id":
            resolved = [str(value or SHARED_POLICY_ID) for value in env.agent_policy_ids]
        elif assignment.mode == "team":
            resolved = [_team_policy_id(team) for team in env.agent_team_ids]
        else:
            resolved = list(env.agent_ids)
        current = dict(zip(env.agent_ids, resolved))
        if current != assignment.agent_to_policy:
            raise ValueError(
                "Policy assignment differs across environments: "
                f"expected={assignment.agent_to_policy}, got={current}"
            )


def validate_multi_policy_options(args, *, supports_async=False, supports_opponent_pool=False):
    if not bool(getattr(args, "multi_policy", False)):
        return
    if getattr(args, "collector_mode", "sync") == "async" and not supports_async:
        raise ValueError(
            "Independent --multi-policy training currently requires --collector-mode sync "
            "for this algorithm"
        )
    if bool(getattr(args, "opponent_pool", False)) and not supports_opponent_pool:
        raise ValueError(
            "--multi-policy cannot currently be combined with --opponent-pool. "
            "Each live policy already controls its assigned agents."
        )


def write_multi_policy_manifest(
    directory,
    assignment,
    algorithm,
    *,
    policy_metadata=None,
    episode=None,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = assignment.as_dict(algorithm)
    if episode is not None:
        payload["episode"] = int(episode)
    payload["policies"] = {}
    metadata = policy_metadata or {}
    for policy_id in assignment.policy_ids:
        item = {
            "key": assignment.key_for(policy_id),
            "directory": f"policies/{assignment.key_for(policy_id)}",
            "trainable": assignment.is_trainable(policy_id),
        }
        if policy_id in metadata:
            item["metadata"] = metadata[policy_id]
        payload["policies"][policy_id] = item

    path = directory / MANIFEST_FILENAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def load_multi_policy_manifest(path):
    path = Path(path)
    manifest_path = path / MANIFEST_FILENAME if path.is_dir() else path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Multi-policy manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"Unsupported multi-policy manifest format in {manifest_path}")
    if int(payload.get("format_version", 0)) != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported multi-policy manifest version={payload.get('format_version')}"
        )
    return payload


def assignment_from_manifest(payload, env):
    agent_to_policy = {
        str(agent_id): str(policy_id)
        for agent_id, policy_id in payload.get("agent_to_policy", {}).items()
    }
    missing = [agent_id for agent_id in env.agent_ids if agent_id not in agent_to_policy]
    extra = sorted(set(agent_to_policy) - set(env.agent_ids))
    if missing or extra:
        raise ValueError(
            "Multi-policy manifest does not match scenario agents: "
            f"missing={missing}, extra={extra}"
        )
    policy_ids = tuple(str(value) for value in payload.get("policy_ids", []))
    if not policy_ids:
        raise ValueError("Multi-policy manifest contains no policies")
    policy_keys = {
        str(policy_id): str(key)
        for policy_id, key in payload.get("policy_keys", {}).items()
    }
    for policy_id in policy_ids:
        policy_keys.setdefault(policy_id, safe_policy_key(policy_id))
    trainable = tuple(
        str(value)
        for value in payload.get("trainable_policy_ids", policy_ids)
    )
    return PolicyAssignment(
        mode=str(payload.get("assignment_mode", "manifest")),
        policy_ids=policy_ids,
        agent_to_policy=agent_to_policy,
        trainable_policy_ids=trainable,
        policy_keys=policy_keys,
    )


def replay_path_for_policy(checkpoint_path, assignment, policy_id):
    checkpoint_path = Path(checkpoint_path)
    match = re.search(r"ckpt-(\d+)$", checkpoint_path.name)
    suffix = match.group(1) if match else checkpoint_path.name
    return checkpoint_path.parent / (
        f"replay-{suffix}-{assignment.key_for(policy_id)}.npz"
    )


def save_policy_replays(checkpoint_path, checkpoint_manager, assignment, buffers):
    saved = {}
    for policy_id, buffer in buffers.items():
        if len(buffer) <= 0:
            continue
        path = replay_path_for_policy(checkpoint_path, assignment, policy_id)
        transitions = buffer.save(path)
        saved[policy_id] = str(path)
        print(
            f"Saved replay policy={policy_id}: {path} transitions={transitions}",
            flush=True,
        )

    retained = set()
    for checkpoint in checkpoint_manager.checkpoints:
        match = re.search(r"ckpt-(\d+)$", str(checkpoint))
        if match:
            retained.add(match.group(1))
    for path in Path(checkpoint_manager.directory).glob("replay-*-*.npz"):
        match = re.fullmatch(r"replay-(\d+)-.+\.npz", path.name)
        if match and match.group(1) not in retained:
            path.unlink()
    return saved


def restore_policy_replays(args, checkpoint_path, assignment, buffers):
    restored = {}
    for policy_id, buffer in buffers.items():
        path = replay_path_for_policy(checkpoint_path, assignment, policy_id)
        if path.is_file():
            count = buffer.load(path)
            restored[policy_id] = count
            print(
                f"Restored replay policy={policy_id}: {path} "
                f"transitions={count}/{args.replay_capacity}",
                flush=True,
            )
            continue
        restored[policy_id] = 0
        if bool(getattr(args, "require_replay_buffer", False)):
            raise FileNotFoundError(
                f"Checkpoint {checkpoint_path!r} has no replay for policy={policy_id!r} "
                f"at {path}"
            )
        print(
            f"WARNING: no replay found for policy={policy_id!r} at {path}; "
            "that policy waits for replay warmup before updating.",
            flush=True,
        )
    return restored
