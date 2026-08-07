import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.common import summarize_episode_diagnostics, update_episode_diagnostics


def single_agent_state():
    return {
        "max_track_progress": 0.0,
        "last_track_progress": 0.0,
        "finish_reached": False,
        "collision_count": 0,
        "collision_seen": False,
        "stalled_count": 0,
        "stalled_seen": False,
    }


class EpisodeDiagnosticsTests(unittest.TestCase):
    def test_collision_is_read_from_scenario_component_values(self):
        state = single_agent_state()

        update_episode_diagnostics(
            state,
            {
                "scenario_terms": {
                    "component_values": {"collision": -20.0},
                },
            },
        )

        self.assertEqual(state["collision_count"], 1)

    def test_collision_is_read_without_reward_penalty(self):
        for info in (
            {"events": {"collision": 1.0}},
            {"terminal_reason": "collision"},
            {"collided": True},
        ):
            with self.subTest(info=info):
                state = single_agent_state()
                update_episode_diagnostics(state, info)
                self.assertEqual(state["collision_count"], 1)

    def test_multi_agent_collision_is_counted_only_once(self):
        state = {
            "max_track_progress": np.zeros((2,), dtype=np.float32),
            "last_track_progress": np.zeros((2,), dtype=np.float32),
            "finish_reached": np.zeros((2,), dtype=np.bool_),
            "collision_count": np.zeros((2,), dtype=np.int32),
            "collision_seen": np.zeros((2,), dtype=np.bool_),
            "stalled_count": np.zeros((2,), dtype=np.int32),
            "stalled_seen": np.zeros((2,), dtype=np.bool_),
        }
        info = {"terminal_reason": "collision"}

        update_episode_diagnostics(state, info, agent_idx=1)
        update_episode_diagnostics(state, info, agent_idx=1)

        self.assertEqual(state["collision_count"].tolist(), [0, 1])

    def test_collision_timing_distinguishes_before_at_and_after_success(self):
        before = single_agent_state()
        update_episode_diagnostics(
            before,
            {"terminal_reason": "collision", "collision_source": "environment"},
        )
        self.assertEqual(before["collision_before_success_count"], 1)

        simultaneous = single_agent_state()
        update_episode_diagnostics(
            simultaneous,
            {"finish_reached": True, "terminal_reason": "collision"},
        )
        self.assertEqual(simultaneous["collision_at_success_count"], 1)

        after = single_agent_state()
        update_episode_diagnostics(after, {"finish_reached": True})
        update_episode_diagnostics(
            after,
            {
                "terminal_reason": "collision",
                "collision_source": "self_body",
                "collision_details": {
                    "source": "self_body",
                    "checker_link": "wrist",
                    "body_link": "base",
                },
            },
        )
        summary = summarize_episode_diagnostics([after], multi_agent=False)
        self.assertEqual(summary["collisions_after_success"], 1)
        self.assertEqual(summary["collision_sources"], ["self_body"])
        self.assertEqual(
            summary["collision_pairs"],
            ["self_body:wrist->base"],
        )


if __name__ == "__main__":
    unittest.main()
