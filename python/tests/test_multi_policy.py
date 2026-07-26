import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.multi_policy import (
    MultiPolicySnapshot,
    assignment_from_manifest,
    build_policy_assignment,
    load_multi_policy_manifest,
    safe_policy_key,
    write_multi_policy_manifest,
)

import numpy as np


class FakeEnv:
    def __init__(self, policy_ids, team_ids=None):
        self.agent_ids = [f"Agent{index}" for index in range(len(policy_ids))]
        self.agent_policy_ids = list(policy_ids)
        self.agent_team_ids = (
            list(team_ids)
            if team_ids is not None
            else [None for _ in policy_ids]
        )


def make_args(**overrides):
    values = {
        "multi_policy": True,
        "multi_agent": True,
        "policy_assignment": "auto",
        "train_policy": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class MultiPolicyAssignmentTests(unittest.TestCase):
    def test_auto_prefers_distinct_declared_policy_ids(self):
        env = FakeEnv(["red", "red", "blue", "blue"], [0, 0, 1, 1])

        assignment = build_policy_assignment([env], make_args())

        self.assertEqual(assignment.mode, "policy_id")
        self.assertEqual(assignment.policy_ids, ("red", "blue"))
        self.assertEqual(assignment.policy_for_agent("Agent2"), "blue")

    def test_auto_falls_back_to_team_for_default_shared_policy(self):
        env = FakeEnv(["shared", "shared"], ["left", "right"])

        assignment = build_policy_assignment([env], make_args())

        self.assertEqual(assignment.mode, "team")
        self.assertEqual(assignment.policy_ids, ("team_left", "team_right"))

    def test_auto_falls_back_to_agent_without_teams(self):
        env = FakeEnv(["shared", "shared"])

        assignment = build_policy_assignment([env], make_args())

        self.assertEqual(assignment.mode, "agent")
        self.assertEqual(assignment.agent_to_policy, {
            "Agent0": "Agent0",
            "Agent1": "Agent1",
        })

    def test_train_policy_selects_a_subset(self):
        env = FakeEnv(["red", "blue"])

        assignment = build_policy_assignment(
            [env],
            make_args(policy_assignment="policy_id", train_policy=["red"]),
        )

        self.assertEqual(assignment.trainable_policy_ids, ("red",))
        self.assertTrue(assignment.is_trainable("red"))
        self.assertFalse(assignment.is_trainable("blue"))

    def test_unknown_train_policy_is_rejected(self):
        env = FakeEnv(["red", "blue"])
        with self.assertRaisesRegex(ValueError, "unknown policies"):
            build_policy_assignment(
                [env],
                make_args(policy_assignment="policy_id", train_policy=["green"]),
            )

    def test_manifest_round_trip_preserves_assignment(self):
        env = FakeEnv(["red", "blue"])
        assignment = build_policy_assignment(
            [env],
            make_args(policy_assignment="policy_id"),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = write_multi_policy_manifest(temp_dir, assignment, "dqn")
            payload = load_multi_policy_manifest(manifest_path)
            restored = assignment_from_manifest(payload, env)

            self.assertEqual(restored.agent_to_policy, assignment.agent_to_policy)
            self.assertEqual(restored.policy_ids, assignment.policy_ids)
            self.assertEqual(payload["algorithm"], "dqn")
            self.assertTrue((Path(temp_dir) / "multi_policy.json").is_file())
            json.loads(manifest_path.read_text(encoding="utf-8"))

    def test_safe_policy_key_is_stable_and_trackable_friendly(self):
        self.assertEqual(safe_policy_key("Team Red/0"), "Team_Red_0")
        self.assertEqual(safe_policy_key("12"), "policy_12")

    def test_snapshot_synchronizes_all_policies_in_one_version(self):
        class Model:
            def __init__(self):
                self.weights = None

            def set_weights(self, weights):
                self.weights = [np.array(weight, copy=True) for weight in weights]

        snapshot = MultiPolicySnapshot({
            "red": [np.asarray([1.0], dtype=np.float32)],
            "blue": [np.asarray([-1.0], dtype=np.float32)],
        })
        models = {"red": Model(), "blue": Model()}

        version = snapshot.sync_model(models)
        self.assertEqual(version, 0)
        self.assertEqual(models["red"].weights[0].tolist(), [1.0])
        self.assertEqual(models["blue"].weights[0].tolist(), [-1.0])

        snapshot.publish({
            "red": [np.asarray([2.0], dtype=np.float32)],
            "blue": [np.asarray([-2.0], dtype=np.float32)],
        })
        version = snapshot.sync_model(models, version)
        self.assertEqual(version, 1)
        self.assertEqual(models["red"].weights[0].tolist(), [2.0])
        self.assertEqual(models["blue"].weights[0].tolist(), [-2.0])

        self.assertFalse(snapshot.wait_for_newer(version, threading.Event(), timeout=0.0))


if __name__ == "__main__":
    unittest.main()
