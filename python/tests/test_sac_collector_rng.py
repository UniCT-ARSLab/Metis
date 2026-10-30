from concurrent.futures import ThreadPoolExecutor
import gc
import unittest

import numpy as np
import tensorflow as tf

from algorithms.sac import build_sac_collector_sample_fns, build_sac_sample_fn


class ZeroActor:
    def __call__(self, observations, training=False):
        values = tf.zeros((tf.shape(observations)[0], 2), dtype=tf.float32)
        return values, values


class SacCollectorRngTests(unittest.TestCase):
    def samplers(self, seed=71021, policies=False):
        models = [{"p0": ZeroActor(), "p1": ZeroActor()} for _ in range(4)] if policies else [ZeroActor() for _ in range(4)]
        return build_sac_collector_sample_fns(models, 3, [-2.0, 0.0], [2.0, 4.0], -20.0, 2.0, seed)

    def draw(self, samplers, reverse=False, count=16):
        observation = np.zeros((1, 3), dtype=np.float32)
        result = [[] for _ in samplers]
        order = list(range(len(samplers)))
        if reverse:
            order.reverse()
        for _ in range(count):
            for worker in order:
                result[worker].append(samplers[worker](observation).numpy()[0])
        return np.asarray(result)

    def test_collector_streams_are_distinct_for_all_study_seeds(self):
        for seed in (71021, 71023, 71025):
            with self.subTest(seed=seed):
                values = self.draw(self.samplers(seed))
                for first in range(4):
                    for second in range(first):
                        self.assertFalse(np.array_equal(values[first], values[second]))
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue((values >= [-2.0, 0.0]).all())
                self.assertTrue((values <= [2.0, 4.0]).all())

    def test_seeded_streams_do_not_depend_on_global_seed_or_worker_call_order(self):
        tf.random.set_seed(100)
        expected = self.draw(self.samplers())
        tf.random.set_seed(999)
        actual = self.draw(self.samplers(), reverse=True)
        np.testing.assert_array_equal(actual, expected)

    def test_different_training_seeds_change_each_collector_stream(self):
        first, second = self.draw(self.samplers(71021)), self.draw(self.samplers(71023))
        self.assertTrue(all(not np.array_equal(a, b) for a, b in zip(first, second)))

    def test_multi_policy_streams_are_independent_and_reproducible(self):
        workers = self.samplers(policies=True)
        functions = [function for worker in workers for function in worker.values()]
        values = self.draw(functions)
        for first in range(len(values)):
            for second in range(first):
                self.assertFalse(np.array_equal(values[first], values[second]))
        fresh = [function for worker in self.samplers(policies=True) for function in worker.values()]
        np.testing.assert_array_equal(values, self.draw(fresh, reverse=True))

    def test_concurrent_worker_calls_match_their_sequential_streams(self):
        samplers = self.samplers()
        observation = np.zeros((1, 3), dtype=np.float32)

        def collect(function):
            return np.asarray([function(observation).numpy()[0] for _ in range(80)])

        with ThreadPoolExecutor(max_workers=4) as executor:
            actual = np.asarray(list(executor.map(collect, samplers)))
        np.testing.assert_array_equal(actual, self.draw(self.samplers(), count=80))
        self.assertTrue(all(function.experimental_get_tracing_count() == 1 for function in samplers))

    def test_generators_stay_alive_and_on_cpu_after_factory_returns(self):
        functions = self.samplers()
        gc.collect()
        for function in functions:
            self.assertIn("CPU:0", function.rng_gen.state.device)
            before = function.rng_gen.state.numpy().copy()
            function(np.zeros((1, 3), dtype=np.float32))
            self.assertFalse(np.array_equal(before, function.rng_gen.state.numpy()))

    def test_deterministic_sampler_does_not_consume_generator_state(self):
        generator = tf.random.Generator.from_seed(100)
        function = build_sac_sample_fn(ZeroActor(), 3, [-2.0, 0.0], [2.0, 4.0], -20.0, 2.0, deterministic=True, rng_gen=generator)
        before = generator.state.numpy().copy()
        for _ in range(3):
            np.testing.assert_array_equal(function(np.zeros((4, 3), dtype=np.float32)).numpy(), [[0.0, 2.0]] * 4)
        np.testing.assert_array_equal(before, generator.state.numpy())

    def test_explicit_generator_preserves_gaussian_tanh_scaling(self):
        generator = tf.random.Generator.from_seed(100)
        reference = tf.random.Generator.from_state(generator.state.numpy(), generator.algorithm)
        function = build_sac_sample_fn(ZeroActor(), 3, [-2.0, 0.0], [2.0, 4.0], -20.0, 2.0, rng_gen=generator)
        expected = [-2.0, 0.0] + 0.5 * (tf.tanh(reference.normal((3, 2))).numpy() + 1.0) * [4.0, 4.0]
        np.testing.assert_allclose(function(np.zeros((3, 3), dtype=np.float32)).numpy(), expected, rtol=0, atol=1e-7)

    def test_default_sampler_keeps_sync_global_rng_path(self):
        tf.random.set_seed(71021)
        functions = [build_sac_sample_fn(ZeroActor(), 3, [-2.0, 0.0], [2.0, 4.0], -20.0, 2.0) for _ in range(4)]
        self.assertTrue(all(function.rng_gen is None for function in functions))
        values = self.draw(functions)
        self.assertTrue(all(np.array_equal(values[0], value) for value in values[1:]))


if __name__ == "__main__":
    unittest.main()
