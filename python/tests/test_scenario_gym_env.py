import unittest


from envs.scenario import ScenarioGymEnv


class ScenarioGymEnvExecutionModeTests(unittest.TestCase):
    def test_reset_can_request_current_state_initialization(self):
        env = ScenarioGymEnv.__new__(ScenarioGymEnv)
        env.seed_value = 0
        env.multi_agent = False
        env.agent_id = "RobotArm"
        env.agent_ids = ["RobotArm"]
        sent = []
        env._send = sent.append
        env._recv = lambda: {"obs": [0.25], "info": {"preserve_state": True}}

        obs, info = env.reset(seed=7, options={"preserve_state": True})

        self.assertEqual(sent, [{"cmd": "reset", "seed": 7, "preserve_state": True}])
        self.assertEqual(obs.tolist(), [0.25])
        self.assertTrue(info["preserve_state"])

    def test_execution_mode_request_updates_client_state(self):
        env = ScenarioGymEnv.__new__(ScenarioGymEnv)
        sent = []
        env._send = sent.append
        env._recv = lambda: {"ok": True, "mode": "realtime", "lockstep": False}

        reply = env.set_execution_mode("realtime")

        self.assertEqual(sent, [{
            "cmd": "execution_mode",
            "mode": "realtime",
            "simulation_fps": 60,
        }])
        self.assertEqual(reply["lockstep"], False)
        self.assertEqual(env.execution_mode, "realtime")

    def test_single_agent_reset_preserves_agent_reset_diagnostics(self):
        env = ScenarioGymEnv.__new__(ScenarioGymEnv)
        env.seed_value = 0
        env.multi_agent = False
        env.agent_id = "RobotArm"
        env.agent_ids = ["RobotArm"]
        env._send = lambda _request: None
        env._recv = lambda: {
            "agents": [{
                "id": "RobotArm",
                "obs": [0.25],
                "info": {
                    "reset": {
                        "task_reset_mode": "regular",
                        "regular_reset_fraction": 0.5,
                    }
                },
            }],
        }

        _obs, info = env.reset(seed=7)

        self.assertEqual(
            info["agent_info"]["reset"]["task_reset_mode"],
            "regular",
        )

    def test_invalid_execution_mode_is_rejected_locally(self):
        env = ScenarioGymEnv.__new__(ScenarioGymEnv)
        with self.assertRaisesRegex(ValueError, "lockstep.*realtime"):
            env.set_execution_mode("turbo")


if __name__ == "__main__":
    unittest.main()
