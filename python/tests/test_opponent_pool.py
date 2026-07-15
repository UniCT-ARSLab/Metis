import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from opponent_pool import OpponentPool, validate_team_layout


class FakeModel:
    def __init__(self):
        self.loaded = None

    def save_weights(self, path):
        Path(path).write_text("weights", encoding="utf-8")

    def load_weights(self, path):
        self.loaded = Path(path)


class FakeEnv:
    def __init__(self, teams, multi_agent=True):
        self.multi_agent = multi_agent
        self.agent_ids = [f"agent_{idx}" for idx in range(len(teams))]
        self.agent_team_ids = list(teams)


def pool_args(directory, **overrides):
    values = {
        "opponent_pool": True,
        "opponent_pool_dir": str(directory),
        "checkpoint_dir": str(directory.parent),
        "opponent_pool_size": 2,
        "opponent_snapshot_every": 10,
        "opponent_current_probability": 0.0,
        "opponent_sampling": "latest",
        "learner_team": None,
        "env_seed_base": 7,
    }
    values.update(overrides)
    return Namespace(**values)


class OpponentPoolTests(unittest.TestCase):
    def test_validates_two_team_layout(self):
        envs = [FakeEnv([0, 1]), FakeEnv([0, 1])]
        self.assertEqual(validate_team_layout(envs, True), [0, 1])

        with self.assertRaisesRegex(ValueError, "team_id"):
            validate_team_layout([FakeEnv([0, None])], True)
        with self.assertRaisesRegex(ValueError, "exactly two teams"):
            validate_team_layout([FakeEnv([0, 1, 2])], True)

    def test_snapshots_are_persisted_and_pruned(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir) / "opponents"
            pool = OpponentPool(
                pool_args(directory),
                "dqn",
                FakeModel,
                metadata={"obs_dim": 4},
            )
            model = FakeModel()
            pool.snapshot(model, 0, force=True)
            pool.snapshot(model, 10)
            pool.snapshot(model, 20)

            self.assertEqual([entry["episode"] for entry in pool.entries], [10, 20])
            self.assertFalse((directory / "policy-00000000.weights.h5").exists())

            restored = OpponentPool(
                pool_args(directory),
                "dqn",
                FakeModel,
                metadata={"obs_dim": 4},
            )
            self.assertEqual([entry["episode"] for entry in restored.entries], [10, 20])

    def test_match_uses_only_snapshots_not_newer_than_episode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir) / "opponents"
            pool = OpponentPool(pool_args(directory), "dqn", FakeModel)
            model = FakeModel()
            pool.snapshot(model, 10, force=True)
            pool.snapshot(model, 20, force=True)

            match = pool.start_episode(model, 15)
            self.assertEqual(match.label, "snapshot:10")
            self.assertEqual(match.model.loaded.name, "policy-00000010.weights.h5")

            older_match = pool.start_episode(model, 5)
            self.assertTrue(older_match.use_current_policy)
            self.assertEqual(older_match.label, "current:no_eligible_snapshot")

    def test_learner_mask_and_action_merge(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pool = OpponentPool(pool_args(Path(temp_dir) / "opponents"), "dqn", FakeModel)
            mask = pool.learner_mask([0, 0, 1, 1], [0, 1], episode=3, env_index=0)
            self.assertIn(mask.tolist(), ([True, True, False, False], [False, False, True, True]))
            merged = pool.merge_actions(
                np.asarray([1, 1, 1, 1]),
                np.asarray([2, 2, 2, 2]),
                mask,
            )
            np.testing.assert_array_equal(merged[mask], np.asarray([1, 1]))
            np.testing.assert_array_equal(merged[~mask], np.asarray([2, 2]))


if __name__ == "__main__":
    unittest.main()
