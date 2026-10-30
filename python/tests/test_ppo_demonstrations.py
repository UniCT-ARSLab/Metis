import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from algorithms import ppo
from core import argspec, ppo_demonstrations as demos
from core.models import build_hybrid_actor_critic, network_layers, set_network_layers


def options(**changes):
    parser = argspec._capture_parser(ppo.parse_args)
    args = parser.parse_args([])
    for key, value in changes.items():
        setattr(args, key, value)
    return args


def metadata(kind="continuous"):
    spec = {}
    if kind in ("discrete", "hybrid"):
        spec["command"] = {"action_type": "discrete", "size": 3}
    if kind in ("continuous", "hybrid"):
        spec["force"] = {"action_type": "continuous", "size": 1, "low": -1., "high": 1.}
    return ppo.build_action_metadata(spec)


def dataset(kind="continuous", count=8):
    meta = metadata(kind)
    return {"obs": np.zeros((count, 3), np.float32), "discrete_actions": np.full((count, len(meta["discrete"])), 2, np.int32), "continuous_actions": np.full((count, meta["continuous_size"]), .7, np.float32)}


def model_and_batch(meta):
    model = build_hybrid_actor_critic(3, meta["discrete_sizes"], meta["continuous_size"], layers=[8, 8])
    log_std = tf.Variable(np.full(meta["continuous_size"], -.5, np.float32))
    data = dataset("hybrid" if meta["discrete"] and meta["continuous_size"] else "continuous" if meta["continuous_size"] else "discrete")
    log_probs, _, values = ppo.evaluate_actions(model, log_std, data["obs"], data["discrete_actions"], data["continuous_actions"], meta)
    batch = {**data, "log_probs": log_probs.numpy(), "returns": values.numpy(), "advantages": np.zeros(8, np.float32)}
    return model, log_std, batch


class DemoDatasetTests(unittest.TestCase):
    def test_cli_defaults_off_and_godot_schema_exposes_options(self):
        args = options()
        self.assertEqual(args.demo_bc_epochs, 0)
        self.assertEqual(args.ppo_demo_coef, 0.)
        demos.validate_options(args)
        fields = {arg["dest"]: arg for arg in argspec.spec_for_algorithm("ppo")["arguments"]}
        self.assertEqual(fields["ppo_demo_coef"]["kind"], "float")
        self.assertEqual(fields["stop_after_bc"]["kind"], "bool")
        self.assertNotIn("demo_protect", fields)

    def test_loads_recorder_continuous_data_and_ignores_transition_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "expert.npz"
            np.savez(path, obs=np.zeros((8, 3)), actions=np.full((8, 1), .5), action_type=["continuous"], rewards=np.ones(8), next_obs=np.ones((8, 3)), dones=np.zeros(8))
            data = demos.load_demonstrations([path], 3, metadata())
            self.assertEqual(set(data), {"obs", "discrete_actions", "continuous_actions"})
            self.assertEqual(data["continuous_actions"].shape, (8, 1))
            self.assertFalse(data["obs"].flags.writeable)

    def test_loads_scalar_discrete_and_explicit_hybrid_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "expert.npz"
            np.savez(path, obs=np.zeros((8, 3)), actions=np.full(8, 2))
            self.assertEqual(demos.load_demonstrations([path], 3, metadata("discrete"))["discrete_actions"].shape, (8, 1))
            np.savez(path, **dataset("hybrid"))
            data = demos.load_demonstrations([path], 3, metadata("hybrid"))
            np.testing.assert_array_equal(data["discrete_actions"], 2)
            np.testing.assert_allclose(data["continuous_actions"], .7)

    def test_global_cap_combines_sources_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"{index}.npz" for index in range(2)]
            for index, path in enumerate(paths):
                np.savez(path, obs=np.full((3, 3), index), actions=np.zeros((3, 1)))
            data = demos.load_demonstrations(paths, 3, metadata(), max_transitions=4)
            np.testing.assert_array_equal(data["obs"][:, 0], [0, 0, 0, 1])

    def test_rejects_malformed_continuous_data(self):
        cases = [{"obs": np.zeros((2, 4)), "actions": np.zeros((2, 1))}, {"obs": np.zeros((2, 3)), "actions": np.zeros((1, 1))}, {"obs": np.full((2, 3), np.nan), "actions": np.zeros((2, 1))}, {"obs": np.zeros((2, 3)), "actions": np.full((2, 1), np.inf)}, {"obs": np.zeros((2, 3)), "actions": np.full((2, 1), 1.1)}, {"obs": np.empty((0, 3)), "actions": np.empty((0, 1))}, {"obs": np.zeros((2, 3))}, {"obs": np.zeros((2, 3)), "actions": np.zeros((2, 1)), "action_type": ["discrete"]}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npz"
            for index, arrays in enumerate(cases):
                with self.subTest(index=index):
                    np.savez(path, **arrays)
                    with self.assertRaises(ValueError):
                        demos.load_demonstrations([path], 3, metadata())

    def test_rejects_fractional_discrete_labels_and_ambiguous_hybrid_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npz"
            for label in (-1, .5, 3):
                np.savez(path, obs=np.zeros((2, 3)), actions=np.full(2, label))
                with self.subTest(label=label), self.assertRaises(ValueError):
                    demos.load_demonstrations([path], 3, metadata("discrete"))
            np.savez(path, obs=np.zeros((2, 3)), actions=np.zeros((2, 2)))
            with self.assertRaisesRegex(ValueError, "hybrid"):
                demos.load_demonstrations([path], 3, metadata("hybrid"))

    def test_rejects_unsupported_modes_unused_datasets_and_invalid_options(self):
        cases = [{"demo_bc_epochs": 1}, {"ppo_demo_coef": .1}, {"demo_path": ["expert.npz"]}, {"demo_path": ["expert.npz"], "demo_bc_epochs": 1, "multi_policy": True}, {"demo_path": ["expert.npz"], "demo_bc_epochs": 1, "policy_mode": "residual"}, {"demo_bc_epochs": -1}, {"demo_bc_batch_size": 0}, {"demo_bc_learning_rate": float("nan")}, {"ppo_demo_coef": float("inf")}, {"ppo_demo_coef": -.1}, {"demo_max_transitions": -1}, {"ppo_demo_decay_updates": -1}, {"ppo_demo_final_coef": .1}, {"demo_validation_path": ["heldout.npz"]}, {"stop_after_bc": True}]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                demos.validate_options(options(**changes))

    def test_multiple_discrete_heads_follow_contract_order(self):
        meta = ppo.build_action_metadata({"move": {"action_type": "discrete", "size": 3}, "grip": {"action_type": "discrete", "size": 2}})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "expert.npz"
            np.savez(path, obs=np.zeros((2, 3)), actions=[[2, 1], [0, 0]])
            data = demos.load_demonstrations([path], 3, meta)
            np.testing.assert_array_equal(data["discrete_actions"], [[2, 1], [0, 0]])
            model = build_hybrid_actor_critic(3, meta["discrete_sizes"], 0, layers=[8])
            self.assertTrue(np.isfinite(demos.evaluate_imitation(model, data, meta, 2)))

    def test_invalid_demo_modes_fail_before_starting_godot(self):
        for extra in (["--multi-policy"], ["--policy-mode", "residual"]):
            with self.subTest(extra=extra), mock.patch("sys.argv", ["ppo", "--demo-path", "expert.npz", "--demo-bc-epochs", "1", *extra]), mock.patch.object(ppo, "GodotProcessManager") as manager:
                with self.assertRaisesRegex(ValueError, "single-policy"):
                    ppo.main()
                manager.assert_not_called()


class DemoLearningTests(unittest.TestCase):
    def test_bc_learns_each_action_contract_without_training_value_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ("discrete", "continuous", "hybrid"):
                with self.subTest(kind=kind):
                    meta = metadata(kind)
                    model, log_std, _ = model_and_batch(meta)
                    data = dataset(kind)
                    value_weights = [weight.copy() for weight in model.layers[-1].get_weights()]
                    initial_std = log_std.numpy().copy()
                    before = demos.evaluate_imitation(model, data, meta, 8)
                    args = options(demo_bc_epochs=25, demo_bc_batch_size=8, demo_bc_learning_rate=.05, weights_path=str(Path(tmp) / f"{kind}.weights.h5"))
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = demos.pretrain(model, data, meta, args)
                    self.assertLess(demos.evaluate_imitation(model, data, meta, 8), before * .2)
                    self.assertTrue(Path(result["weights_path"]).exists())
                    for before_value, after_value in zip(value_weights, model.layers[-1].get_weights()):
                        np.testing.assert_array_equal(before_value, after_value)
                    np.testing.assert_array_equal(log_std.numpy(), initial_std)

    def test_validation_restores_best_epoch_and_saved_weights(self):
        meta = metadata()
        model, _, _ = model_and_batch(meta)
        seen = []

        def measure(*unused):
            seen.append(model.get_weights())
            return [.9, .5, .8, .2, .7, .4][len(seen) - 1]

        with tempfile.TemporaryDirectory() as tmp:
            args = options(demo_bc_epochs=3, demo_bc_batch_size=8, weights_path=str(Path(tmp) / "model.weights.h5"))
            with mock.patch.object(demos, "evaluate_imitation", side_effect=measure), contextlib.redirect_stdout(io.StringIO()):
                result = demos.pretrain(model, dataset(), meta, args, dataset())
            self.assertEqual(result["best_epoch"], 2)
            restored, _, _ = model_and_batch(meta)
            restored.load_weights(result["weights_path"])
            for expected, current, saved in zip(seen[3], model.get_weights(), restored.get_weights()):
                np.testing.assert_array_equal(expected, current)
                np.testing.assert_array_equal(expected, saved)

    def test_expert_sampling_is_counter_based_and_does_not_consume_global_rng(self):
        data = dataset(count=32)
        data["obs"][:, 0] = np.arange(32)
        np.random.seed(71)
        before = np.random.get_state()
        first = demos.sample_batch(data, 16, 2, 17)
        again = demos.sample_batch(data, 16, 2, 17)
        other = demos.sample_batch(data, 16, 3, 17)
        np.testing.assert_array_equal(first[0].numpy(), again[0].numpy())
        self.assertFalse(np.array_equal(first[0].numpy(), other[0].numpy()))
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_auxiliary_imitation_learns_with_zero_on_policy_advantages(self):
        for compiled in (False, True):
            meta = metadata("hybrid")
            model, log_std, batch = model_and_batch(meta)
            args = options(ppo_epochs=1, batch_size=8, entropy_coef=0., value_loss_coef=0., ppo_demo_coef=1., ppo_demonstrations=dataset("hybrid"), demo_bc_batch_size=8, tf_compile_learner=compiled)
            optimizer = tf.keras.optimizers.Adam(.03)
            before = demos.evaluate_imitation(model, args.ppo_demonstrations, meta, 8)
            original_keys = set(batch)
            for _ in range(24):
                metrics = ppo.ppo_update(model, log_std, optimizer, batch, meta, args)
            with self.subTest(compiled=compiled):
                self.assertLess(demos.evaluate_imitation(model, args.ppo_demonstrations, meta, 8), before * .5)
                self.assertEqual(metrics["demo_samples"], 8)
                self.assertEqual(metrics["n_total"], 8)
                self.assertEqual(set(batch), original_keys)
                self.assertEqual(metrics["demo_coefficient"], 1.)

    def test_disabled_imitation_preserves_update_metrics_weights_and_rng(self):
        meta = metadata()
        first, first_std, batch = model_and_batch(meta)
        second, second_std, _ = model_and_batch(meta)
        second.set_weights(first.get_weights())
        args = options(ppo_epochs=1, batch_size=4, tf_compile_learner=False)
        legacy = options(ppo_epochs=1, batch_size=4, tf_compile_learner=False)
        for name in list(vars(legacy)):
            if name.startswith("demo_") or name.startswith("ppo_demo_"):
                delattr(legacy, name)
        np.random.seed(83)
        expected = ppo.ppo_update(first, first_std, tf.keras.optimizers.Adam(.001), batch, meta, legacy)
        expected_rng = np.random.get_state()
        np.random.seed(83)
        actual = ppo.ppo_update(second, second_std, tf.keras.optimizers.Adam(.001), batch, meta, args)
        self.assertEqual(expected, actual)
        self.assertNotIn("demo_loss", actual)
        for a, b in zip(first.get_weights(), second.get_weights()):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(expected_rng[1], np.random.get_state()[1])

    def test_decay_and_sampling_follow_restored_optimizer_counter(self):
        args = options(ppo_demo_coef=1., ppo_demo_final_coef=.2, ppo_demo_decay_updates=10)
        for update, expected in ((0, 1.), (5, .6), (10, .2), (20, .2)):
            self.assertAlmostEqual(float(demos.coefficient(args, update).numpy()), expected, places=6)
        with tempfile.TemporaryDirectory() as tmp:
            optimizer = tf.keras.optimizers.Adam()
            optimizer.iterations.assign(7)
            path = tf.train.Checkpoint(optimizer=optimizer).save(str(Path(tmp) / "state"))
            restored = tf.keras.optimizers.Adam()
            tf.train.Checkpoint(optimizer=restored).restore(path).assert_existing_objects_matched()
            self.assertEqual(int(restored.iterations.numpy()), 7)
            self.assertAlmostEqual(float(demos.coefficient(args, restored.iterations).numpy()), .44, places=6)
            a = demos.sample_batch(dataset(), 4, int(optimizer.iterations.numpy()), 13)
            b = demos.sample_batch(dataset(), 4, int(restored.iterations.numpy()), 13)
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x.numpy(), y.numpy())

    def test_auxiliary_checkpoint_contract_rejects_silent_data_or_schedule_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "expert.npz"
            np.savez(path, obs=np.zeros((8, 3)), actions=np.zeros((8, 1)))
            args = options(demo_path=[str(path)], ppo_demo_coef=.1)
            model, _, _ = model_and_batch(metadata())
            ppo.prepare_demonstrations(args, model, metadata(), 3, None)
            ckpt = str(Path(tmp) / "ckpt-1")
            Path(ckpt + ".ppo_demos.json").write_text(json.dumps(args.ppo_demo_contract))
            ppo.prepare_demonstrations(args, model, metadata(), 3, ckpt)
            for changed in (options(), options(demo_path=[str(path)], ppo_demo_coef=.2)):
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    ppo.prepare_demonstrations(changed, model, metadata(), 3, ckpt)
            np.savez(path, obs=np.ones((8, 3)), actions=np.zeros((8, 1)))
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                ppo.prepare_demonstrations(args, model, metadata(), 3, ckpt)

    def test_resume_pretraining_requires_explicit_request_and_bc_only_cannot_silently_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "expert.npz"
            np.savez(path, obs=np.zeros((8, 3)), actions=np.zeros((8, 1)))
            args = options(demo_path=[str(path)], demo_bc_epochs=2)
            model, _, _ = model_and_batch(metadata())
            with mock.patch.object(demos, "pretrain") as pretrain:
                self.assertFalse(ppo.prepare_demonstrations(args, model, metadata(), 3, "legacy-1"))
                pretrain.assert_not_called()
                args.stop_after_bc = True
                with self.assertRaisesRegex(ValueError, "demo-bc-on-resume"):
                    ppo.prepare_demonstrations(args, model, metadata(), 3, "legacy-1")
                args.demo_bc_on_resume = True
                self.assertTrue(ppo.prepare_demonstrations(args, model, metadata(), 3, "legacy-1"))
                pretrain.assert_called_once()

    def test_copied_checkpoint_retains_embedded_contract_without_sidecar(self):
        from core.training import copy_checkpoint_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "expert.npz"
            np.savez(path, obs=np.zeros((8, 3)), actions=np.zeros((8, 1)))
            args = options(demo_path=[str(path)], ppo_demo_coef=.1)
            model, _, _ = model_and_batch(metadata())
            checkpoint = tf.train.Checkpoint(demo_contract=tf.Variable("", dtype=tf.string, trainable=False))
            ppo.prepare_demonstrations(args, model, metadata(), 3, None, checkpoint)
            source = checkpoint.save(str(root / "source"))
            destination = str(root / "best" / "ckpt-1")
            copy_checkpoint_files(source, destination)
            self.assertFalse(Path(destination + ".ppo_demos.json").exists())
            restored = tf.train.Checkpoint(demo_contract=tf.Variable("", dtype=tf.string, trainable=False))
            restored.restore(destination).assert_existing_objects_matched()
            ppo.prepare_demonstrations(args, model, metadata(), 3, destination, restored)
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                ppo.prepare_demonstrations(options(), model, metadata(), 3, destination, restored)


class TrainingEnv:
    instances = []
    obs_dim = 3
    action_type = "continuous"
    action_size = 1
    action_names = ["force"]
    action_low = [-1.]
    action_high = [1.]
    action_space_spec = {"force": {"action_type": "continuous", "size": 1, "low": -1., "high": 1.}}
    agent_id = "Agent"
    agent_ids = ["Agent"]
    agent_team_ids = [0]

    def __init__(self, **unused):
        self.steps = 0
        self.closed = False
        self.instances.append(self)

    def configure(self, **unused):
        pass

    def reset(self, seed=None):
        self.steps = 0
        return np.zeros(3, np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(3, np.float32), 1., self.steps >= 2, False, {}

    def close(self):
        self.closed = True

    def agent_summary(self):
        return "single agent"

    def team_summary(self):
        return "single team"

    def _spec_for_agent(self, agent_id):
        return {"observation_names": ["x", "y", "z"], "action_space": self.action_space_spec}


class PPOExpertTrainingTests(unittest.TestCase):
    def run_training(self, mode, stop=False, resume=False, kind="continuous", auxiliary_only=False):
        old_layers = network_layers()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                demo = root / "expert.npz"
                np.savez(demo, **dataset(kind))
                spec = {}
                if kind in ("discrete", "hybrid"):
                    spec["command"] = {"action_type": "discrete", "size": 3}
                if kind in ("continuous", "hybrid"):
                    spec["force"] = {"action_type": "continuous", "size": 1, "low": -1., "high": 1.}
                env_class = type("ExpertTrainingEnv", (TrainingEnv,), {"action_type": kind, "action_space_spec": spec})
                argv = ["ppo", "--collector-mode", mode, "--num-envs", "2", "--no-parallel-env-steps", "--num-episodes", "2", "--max-steps-per-episode", "2", "--network-layers", "8", "8", "--ppo-epochs", "1", "--batch-size", "4", "--demo-path", str(demo), "--demo-bc-epochs", "2", "--demo-bc-batch-size", "4", "--ppo-demo-coef", ".1", "--ppo-demo-decay-updates", "20", "--no-dashboard", "--no-best-checkpoint", "--no-health-monitor", "--no-auto-recovery", "--checkpoint-dir", str(root / "checkpoints"), "--weights-path", str(root / "ppo.weights.h5")]
                if stop:
                    argv.append("--stop-after-bc")
                if auxiliary_only:
                    argv.extend(["--demo-bc-epochs", "0", "--ppo-rollout-steps", "2", "--total-timesteps", "8"])
                TrainingEnv.instances = []
                manager = mock.Mock()
                output = io.StringIO()
                with mock.patch("sys.argv", argv), mock.patch.object(ppo, "GodotProcessManager", return_value=manager), mock.patch.object(ppo, "ScenarioGymEnv", env_class), contextlib.redirect_stdout(output):
                    ppo.main()
                checkpoint = tf.train.latest_checkpoint(str(root / "checkpoints"))
                self.assertIsNotNone(checkpoint)
                self.assertTrue(Path(checkpoint + ".ppo_demos.json").exists())
                self.assertEqual((root / "ppo_bc.weights.h5").exists(), not auxiliary_only)
                self.assertTrue((root / "checkpoints" / "policy.keras").exists())
                self.assertTrue(all(env.closed for env in TrainingEnv.instances))
                manager.stop_all.assert_called_once()
                if stop:
                    self.assertTrue(all(env.steps == 0 for env in TrainingEnv.instances))
                    self.assertIn("no on-policy rollout", output.getvalue())
                else:
                    self.assertIn("demo_samples=", output.getvalue())
                    self.assertTrue(all(env.steps > 0 for env in TrainingEnv.instances))
                    self.assertIn("total_timesteps=8", output.getvalue())
                if resume:
                    argv.extend(["--resume", "--num-episodes", "3"])
                    manager.reset_mock()
                    with mock.patch("sys.argv", argv), mock.patch.object(ppo, "GodotProcessManager", return_value=manager), mock.patch.object(ppo, "ScenarioGymEnv", env_class), mock.patch.object(demos, "pretrain") as pretrain, contextlib.redirect_stdout(output):
                        ppo.main()
                    pretrain.assert_not_called()
                    self.assertIn("Skipped PPO behavior cloning on resume", output.getvalue())
                    manager.stop_all.assert_called_once()
        finally:
            set_network_layers(old_layers)

    def test_sync_bc_then_ppo_training(self):
        self.run_training("sync")

    def test_async_bc_then_ppo_training(self):
        self.run_training("async")

    def test_stop_after_bc_saves_without_collecting_rollouts(self):
        self.run_training("sync", stop=True)

    def test_resume_retains_auxiliary_contract_and_skips_pretraining(self):
        self.run_training("sync", resume=True)

    def test_hybrid_async_training_uses_separate_expert_labels(self):
        self.run_training("async", kind="hybrid")

    def test_discrete_sync_training_uses_separate_expert_labels(self):
        self.run_training("sync", kind="discrete")

    def test_auxiliary_only_fixed_rollouts_do_not_count_expert_samples_in_budget(self):
        self.run_training("async", auxiliary_only=True)


if __name__ == "__main__":
    unittest.main()
