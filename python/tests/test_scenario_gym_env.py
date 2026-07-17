import unittest


from envs.scenario import ScenarioGymEnv


class ScenarioGymEnvExecutionModeTests(unittest.TestCase):
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

    def test_invalid_execution_mode_is_rejected_locally(self):
        env = ScenarioGymEnv.__new__(ScenarioGymEnv)
        with self.assertRaisesRegex(ValueError, "lockstep.*realtime"):
            env.set_execution_mode("turbo")


if __name__ == "__main__":
    unittest.main()
