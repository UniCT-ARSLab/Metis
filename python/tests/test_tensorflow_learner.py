import unittest

import numpy as np
import tensorflow as tf

from core.replay_buffer import ReplayBuffer
from algorithms.dqn import build_dqn_learner_step
from core.training import configure_tensorflow_devices


class TensorFlowLearnerTests(unittest.TestCase):
    @staticmethod
    def build_model():
        inputs = tf.keras.Input(shape=(3,))
        outputs = tf.keras.layers.Dense(2)(inputs)
        return tf.keras.Model(inputs, outputs)

    def test_compiled_dqn_batch_runs_one_optimizer_step_per_update(self):
        tf.keras.utils.set_random_seed(123)
        model = self.build_model()
        target = self.build_model()
        target.set_weights(model.get_weights())
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        buffer = ReplayBuffer(capacity=64)
        for value in range(32):
            obs = np.asarray([value, value % 3, 1.0], dtype=np.float32) / 32.0
            buffer.add(obs, value % 2, float(value % 5), obs + 0.01, value % 7 == 0)

        before = [weight.copy() for weight in model.get_weights()]
        learner_step = build_dqn_learner_step(
            model,
            target,
            optimizer,
            gamma=0.99,
            compiled=True,
        )
        losses = learner_step(buffer, batch_size=8, update_count=3)

        self.assertEqual(losses.shape, (3,))
        self.assertTrue(np.all(np.isfinite(losses)))
        self.assertEqual(int(optimizer.iterations.numpy()), 3)
        self.assertTrue(any(not np.array_equal(old, new) for old, new in zip(before, model.get_weights())))

    def test_memory_growth_is_only_applied_to_linux_gpu_devices(self):
        devices = [object(), object()]

        class Experimental:
            calls = []

            @classmethod
            def set_memory_growth(cls, device, enabled):
                cls.calls.append((device, enabled))

        class Config:
            experimental = Experimental

            @staticmethod
            def list_physical_devices(device_type):
                return devices if device_type == "GPU" else []

        class FakeTensorFlow:
            config = Config

        found, enabled = configure_tensorflow_devices(
            FakeTensorFlow,
            memory_growth=True,
            system_name="Linux",
        )

        self.assertEqual(found, devices)
        self.assertTrue(enabled)
        self.assertEqual(Experimental.calls, [(devices[0], True), (devices[1], True)])


if __name__ == "__main__":
    unittest.main()
