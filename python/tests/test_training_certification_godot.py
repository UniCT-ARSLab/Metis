import json
import os
from pathlib import Path
import socket
import tempfile
import unittest

import tensorflow as tf

from core.models import build_shared_q_network
from core.policy_artifact import PolicyArtifactSaver, build_policy_metadata
from core.training import BestCheckpointTracker
from envs.process_manager import GodotProcessManager
from envs.scenario import ScenarioGymEnv
from test_best_checkpoint_tracker import tracker_args, evaluation


def available_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@unittest.skipUnless(os.environ.get("GODOT_BIN") and Path(os.environ["GODOT_BIN"]).is_file(), "Set GODOT_BIN to run the real Godot certification integration")
class GodotCertificationIntegrationTests(unittest.TestCase):
    def test_real_evaluator_failure_is_recorded_and_training_remains_enabled(self):
        project = Path(__file__).resolve().parents[2] / "godot"
        scene = "res://scenarios/breakout/breakout_scenario.tscn"
        env = None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = GodotProcessManager(godot_bin=os.environ["GODOT_BIN"], project_dir=str(project), scene_path=scene, logs_dir=root / "fixture-logs", port_lease_dir=root / "leases")
            port = available_port()
            try:
                manager.start_many([port], headless=True)
                env = ScenarioGymEnv(port=port, timeout=15.)
                model = build_shared_q_network(env.obs_dim, env.num_actions, layers=[8, 8])
                policy = PolicyArtifactSaver(model, root / "fixture", build_policy_metadata("dqn", env)).save(100)
            finally:
                if env is not None:
                    env.close()
                manager.stop_all()
            args = tracker_args(root / "training", godot_bin=os.environ["GODOT_BIN"], godot_project=str(project), godot_scene=scene, best_evaluation_port=available_port(), best_evaluation_max_steps=2, best_evaluation_timeout=30., stop_when_solved=.95, solved_patience=1, solved_certification_episodes=3)
            tracker = BestCheckpointTracker(args, "dqn")
            qualified = evaluation(100, 1., 1.)
            tracker._ready = (qualified, policy, None)
            try:
                self.assertIsNone(tracker.process_control(qualified, True))
                self.assertFalse(tracker.stop_requested)
                self.assertTrue(tracker.should_evaluate(200))
                certificate = tracker.control.state_path.parent / "certification/attempt-1/report.json"
                self.assertTrue(certificate.exists())
                report = json.loads(certificate.read_text())
                self.assertFalse(report["passed"])
                self.assertEqual(report["evaluation"]["trials"], 3)
                self.assertEqual(report["evaluation"]["successes"], 0)
                self.assertEqual(report["evaluation"]["evaluation_phase"], "certification")
                self.assertIsNone(tracker._process)
            finally:
                tracker.close()


if __name__ == "__main__":
    unittest.main()
