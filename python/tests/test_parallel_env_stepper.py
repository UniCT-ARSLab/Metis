import threading
import unittest

from core.training import ParallelEnvStepper


class CoordinatedEnv:
    def __init__(self, name, coordinator):
        self.name = name
        self.coordinator = coordinator

    def step(self, action):
        with self.coordinator["lock"]:
            self.coordinator["started"] += 1
            if self.coordinator["started"] == self.coordinator["expected"]:
                self.coordinator["all_started"].set()
        ran_concurrently = self.coordinator["all_started"].wait(timeout=1.0)
        return self.name, action, ran_concurrently


class ParallelEnvStepperTests(unittest.TestCase):
    def test_steps_all_environments_concurrently_and_preserves_order(self):
        coordinator = {
            "lock": threading.Lock(),
            "started": 0,
            "expected": 4,
            "all_started": threading.Event(),
        }
        envs = [CoordinatedEnv(f"env-{idx}", coordinator) for idx in range(4)]
        states = [{"idx": idx} for idx in range(4)]
        stepper = ParallelEnvStepper(max_workers=4, enabled=True)
        try:
            results = stepper.step([
                (env, state, idx + 10)
                for idx, (env, state) in enumerate(zip(envs, states))
            ])
        finally:
            stepper.close()

        self.assertEqual([result[0].name for result in results], [f"env-{idx}" for idx in range(4)])
        self.assertEqual([result[1]["idx"] for result in results], list(range(4)))
        self.assertEqual([result[3][1] for result in results], [10, 11, 12, 13])
        self.assertTrue(all(result[3][2] for result in results))

    def test_single_environment_uses_sequential_path(self):
        class EchoEnv:
            def step(self, action):
                return action

        stepper = ParallelEnvStepper(max_workers=1, enabled=True)
        try:
            result = stepper.step([(EchoEnv(), {"state": 1}, "action")])
        finally:
            stepper.close()

        self.assertFalse(stepper.enabled)
        self.assertEqual(result[0][3], "action")


if __name__ == "__main__":
    unittest.main()
