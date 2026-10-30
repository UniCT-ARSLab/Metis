from concurrent.futures import ThreadPoolExecutor
import gc
import unittest

import numpy as np
import tensorflow as tf

from algorithms.ppo import build_ppo_collector_sample_fns, build_sample_action_fn


class ZeroActorCritic:
    def __init__(self, discrete, continuous):
        self.discrete = discrete
        self.continuous = continuous

    def __call__(self, observations, training=False):
        batch = tf.shape(observations)[0]
        continuous = [tf.zeros((batch, self.continuous))] if self.continuous else []
        return [tf.zeros((batch, size)) for size in self.discrete] + continuous + [tf.zeros((batch, 1))]


class PpoCollectorRngTests(unittest.TestCase):
    def samplers(self, seed=71021, discrete=(3, 5), continuous=2, policies=False):
        meta = {"discrete": [{} for _ in discrete], "discrete_sizes": list(discrete), "continuous_size": continuous, "continuous_low": [-2.0] * continuous, "continuous_high": [2.0] * continuous}
        models = [{"p0": ZeroActorCritic(discrete, continuous), "p1": ZeroActorCritic(discrete, continuous)} for _ in range(4)] if policies else [ZeroActorCritic(discrete, continuous) for _ in range(4)]
        return build_ppo_collector_sample_fns(models, 3, meta, seed)

    def draw_one(self, function, count=32):
        continuous = function.input_signature[1].shape[0]
        return np.asarray([np.concatenate([part.numpy().ravel() for part in function(np.zeros((1, 3), dtype=np.float32), np.zeros(continuous, dtype=np.float32))]) for _ in range(count)])

    def draw(self, functions, reverse=False):
        values = {}
        for index in reversed(range(len(functions))) if reverse else range(len(functions)):
            values[index] = self.draw_one(functions[index])
        return np.asarray([values[index] for index in range(len(functions))])

    def test_discrete_continuous_and_hybrid_streams_are_distinct(self):
        for discrete, continuous in (((3, 5), 0), ((), 2), ((3, 5), 2)):
            for seed in (71021, 71023, 71025):
                with self.subTest(discrete=discrete, continuous=continuous, seed=seed):
                    values = self.draw(self.samplers(seed, discrete, continuous))
                    self.assertTrue(np.isfinite(values).all())
                    for index in range(4):
                        for other in range(index):
                            self.assertFalse(np.array_equal(values[index], values[other]))

    def test_streams_are_reproducible_despite_global_seed_and_call_order(self):
        tf.random.set_seed(100)
        expected = self.draw(self.samplers())
        tf.random.set_seed(999)
        np.testing.assert_array_equal(self.draw(self.samplers(), reverse=True), expected)
        actual = self.draw(self.samplers(seed=71023))
        self.assertTrue(all(not np.array_equal(first, second) for first, second in zip(expected, actual)))

    def test_concurrent_collectors_match_sequential_streams(self):
        functions = self.samplers()
        with ThreadPoolExecutor(max_workers=4) as executor:
            actual = np.asarray(list(executor.map(self.draw_one, functions)))
        np.testing.assert_array_equal(actual, self.draw(self.samplers()))
        self.assertTrue(all(function.experimental_get_tracing_count() == 1 for function in functions))

    def test_multi_policy_streams_are_independent_and_reproducible(self):
        flatten = lambda workers: [function for worker in workers for function in worker.values()]
        values = self.draw(flatten(self.samplers(policies=True)))
        for index in range(len(values)):
            for other in range(index):
                self.assertFalse(np.array_equal(values[index], values[other]))
        np.testing.assert_array_equal(values, self.draw(flatten(self.samplers(policies=True)), reverse=True))

    def test_generators_remain_alive_on_cpu_and_advance(self):
        functions = self.samplers()
        gc.collect()
        for function in functions:
            self.assertIs(function.rng_gen, function.categorical_rng_gen)
            self.assertIn("CPU:0", function.rng_gen.state.device)
            before = function.rng_gen.state.numpy().copy()
            self.draw_one(function, count=1)
            self.assertFalse(np.array_equal(before, function.rng_gen.state.numpy()))

    def test_hybrid_actions_and_log_probabilities_keep_the_policy_distribution(self):
        function = self.samplers()[0]
        discrete, continuous, log_prob, value = [part.numpy() for part in function(np.zeros((4096, 3), dtype=np.float32), np.zeros(2, dtype=np.float32))]
        for index, size in enumerate((3, 5)):
            self.assertTrue(((discrete[:, index] >= 0) & (discrete[:, index] < size)).all())
            frequencies = np.bincount(discrete[:, index], minlength=size) / len(discrete)
            np.testing.assert_allclose(frequencies, 1 / size, atol=0.035)
        self.assertAlmostEqual(float(continuous.mean()), 0.0, delta=0.05)
        self.assertAlmostEqual(float(continuous.std()), 1.0, delta=0.05)
        expected = -np.log(15) - np.log(2 * np.pi) - 0.5 * np.square(continuous).sum(axis=1)
        np.testing.assert_allclose(log_prob, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(value, np.zeros(4096))

    def test_existing_sync_and_residual_sampler_defaults_are_unchanged(self):
        meta = {"discrete": [{}], "discrete_sizes": [3], "continuous_size": 2, "continuous_low": [-2.0, -2.0], "continuous_high": [2.0, 2.0]}
        tf.random.set_seed(71021)
        functions = [build_sample_action_fn(ZeroActorCritic((3,), 2), 3, meta) for _ in range(2)]
        np.testing.assert_array_equal(self.draw_one(functions[0]), self.draw_one(functions[1]))
        residual = build_sample_action_fn(ZeroActorCritic((3,), 2), 3, meta, rng_gen=tf.random.Generator.from_seed(9))
        self.assertIsNone(residual.categorical_rng_gen)


if __name__ == "__main__":
    unittest.main()
