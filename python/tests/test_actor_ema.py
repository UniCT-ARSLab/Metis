import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import tensorflow as tf

from algorithms import common, sac
from core.actor_ema import ActorEMA, EMAPolicyArtifactSaver, validate_actor_ema_options, validate_actor_ema_checkpoint
from core.models import build_continuous_actor, build_continuous_critic, build_sac_actor, network_layers, set_network_layers
from core.replay_buffer import ReplayBuffer
from core.training import BestCheckpointTracker, apply_ready_best_checkpoint
from core.training_health import RecoveryRequest, TensorFlowCheckpointRecovery
from test_best_checkpoint_tracker import tracker_args, evaluation


def actor_model():
    inputs = tf.keras.Input((3,))
    outputs = tf.keras.layers.Dense(1, kernel_initializer="zeros", bias_initializer="zeros")(inputs)
    return tf.keras.Model(inputs, outputs)


class ActorEMATests(unittest.TestCase):
    def test_exact_averaging_and_raw_actor_is_not_changed(self):
        actor = actor_model()
        ema = ActorEMA(actor, .9)
        ema.prepare()
        actor.set_weights([np.ones_like(weight) for weight in actor.get_weights()])
        ema.update()
        for weight in ema.shadow:
            np.testing.assert_allclose(weight.numpy(), .1, atol=1e-7)
        for weight in actor.get_weights():
            np.testing.assert_array_equal(weight, 1.)
        tf.function(ema.update)()
        for weight in ema.shadow:
            np.testing.assert_allclose(weight.numpy(), .19, atol=1e-7)
        self.assertEqual(int(ema.updates.numpy()), 2)

    def test_saved_ema_and_counter_restore_exactly_and_reject_changed_decay(self):
        with tempfile.TemporaryDirectory() as tmp:
            actor = actor_model()
            ema = ActorEMA(actor, .9)
            ema.prepare()
            actor.set_weights([np.ones_like(weight) for weight in actor.get_weights()])
            ema.update()
            prefix = tf.train.Checkpoint(actor_ema=ema).save(str(Path(tmp) / "state"))
            restored = ActorEMA(actor_model(), .9)
            tf.train.Checkpoint(actor_ema=restored).restore(prefix).assert_existing_objects_matched()
            restored.prepare()
            self.assertEqual(int(restored.updates.numpy()), 1)
            for first, second in zip(ema.shadow, restored.shadow):
                np.testing.assert_array_equal(first.numpy(), second.numpy())
            changed = ActorEMA(actor_model(), .8)
            tf.train.Checkpoint(actor_ema=changed).restore(prefix).assert_existing_objects_matched()
            with self.assertRaisesRegex(ValueError, "decay differs"):
                changed.prepare()

    def test_bc_reset_reinitializes_average_to_the_new_clone(self):
        actor = actor_model()
        ema = ActorEMA(actor, .9)
        ema.prepare()
        actor.set_weights([np.ones_like(weight) for weight in actor.get_weights()])
        ema.update()
        ema.prepare(reset=True)
        self.assertEqual(int(ema.updates.numpy()), 0)
        for weight in ema.shadow:
            np.testing.assert_array_equal(weight.numpy(), 1.)

    def test_export_is_an_ordinary_loadable_keras_policy_with_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            actor = actor_model()
            ema = ActorEMA(actor, .5)
            ema.prepare()
            actor.set_weights([np.ones_like(weight) for weight in actor.get_weights()])
            ema.update()
            saver = EMAPolicyArtifactSaver(ema, tmp, {"algorithm": "ddpg", "inference": {"decoder": "tanh_then_scale"}})
            destination = Path(tmp) / "policy.keras"
            saver.export(destination)
            loaded = tf.keras.models.load_model(destination, compile=False)
            np.testing.assert_allclose(loaded(np.zeros((1, 3))).numpy(), [[.5]])
            self.assertEqual(json.loads(destination.with_suffix(".json").read_text())["inference"]["policy_weights"], "ema")

    def test_invalid_and_unsupported_options_are_rejected(self):
        for value in (-.1, 1., float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_actor_ema_options(SimpleNamespace(actor_ema_decay=value, multi_policy=False))
        with self.assertRaisesRegex(ValueError, "single-policy"):
            validate_actor_ema_options(SimpleNamespace(actor_ema_decay=.9, multi_policy=True))

    def test_resume_cannot_silently_drop_saved_averaging(self):
        with tempfile.TemporaryDirectory() as tmp:
            ema = ActorEMA(actor_model(), .9)
            ema.prepare()
            path = tf.train.Checkpoint(actor_ema=ema).save(str(Path(tmp) / "checkpoint"))
            with self.assertRaisesRegex(ValueError, "requires its"):
                validate_actor_ema_checkpoint(SimpleNamespace(actor_ema_decay=0.), path)
            validate_actor_ema_checkpoint(SimpleNamespace(actor_ema_decay=.9), path)

    def test_legacy_recovery_reinitializes_average_to_restored_raw_actor(self):
        with tempfile.TemporaryDirectory() as tmp:
            actor = actor_model()
            actor.set_weights([np.ones_like(weight) for weight in actor.get_weights()])
            optimizer = tf.keras.optimizers.Adam(.001)
            optimizer.build(actor.trainable_variables)
            checkpoint = tf.train.Checkpoint(actor=actor, actor_optimizer=optimizer)
            legacy = checkpoint.save(str(Path(tmp) / "legacy"))
            checkpoint.actor_ema = ActorEMA(actor, .9)
            checkpoint.actor_ema.prepare()
            actor.set_weights([np.full_like(weight, 10.) for weight in actor.get_weights()])
            checkpoint.actor_ema.update()
            handler = TensorFlowCheckpointRecovery(checkpoint, tmp, [("actor", optimizer)])
            handler(RecoveryRequest(episode=1, checkpoint_path=legacy, reason="test", attempt=1, learning_rate_factor=.5, minimum_learning_rate_scale=.01))
            for weight in checkpoint.actor_ema.shadow:
                np.testing.assert_array_equal(weight.numpy(), 1.)
            self.assertEqual(int(checkpoint.actor_ema.updates.numpy()), 0)

    def test_sac_average_advances_only_on_actor_updates_in_eager_and_graph(self):
        old_layers = network_layers()
        set_network_layers([8, 8])
        try:
            for compiled in (False, True):
                actor = build_sac_actor(3, 1)
                critics = [build_continuous_critic(3, 1) for _ in range(4)]
                optimizers = [tf.keras.optimizers.Adam(.001) for _ in range(4)]
                ema = ActorEMA(actor, .9)
                ema.prepare()
                learner = sac.build_sac_learner_step(actor, *critics, *optimizers, tf.Variable(-1.), -1., .99, [-1.], [1.], -20., 2., compiled=compiled, actor_ema=ema)
                batch = (np.ones((4, 3), np.float32), np.zeros((4, 1), np.float32), np.ones((4, 1), np.float32), np.zeros((4, 3), np.float32), np.zeros((4, 1), np.float32))
                learner(*batch, update_policy=False)
                self.assertEqual(int(ema.updates.numpy()), 0)
                learner(*batch, update_policy=True)
                self.assertEqual(int(ema.updates.numpy()), 1)
        finally:
            set_network_layers(old_layers)

    def test_td3_average_respects_delayed_actor_updates(self):
        old_layers = network_layers()
        set_network_layers([8, 8])
        try:
            for compiled in (False, True):
                actor, target_actor = build_continuous_actor(3, 1), build_continuous_actor(3, 1)
                critics = [build_continuous_critic(3, 1) for _ in range(4)]
                ema = ActorEMA(actor, .9)
                ema.prepare()
                learner = common.build_deterministic_learner_step(actor, critics[0], target_actor, critics[1], tf.keras.optimizers.Adam(.001), tf.keras.optimizers.Adam(.001), .99, [-1.], [1.], variant="td3", critic2=critics[2], target_critic2=critics[3], critic2_optimizer=tf.keras.optimizers.Adam(.001), policy_delay=2, batch_size=2, compiled=compiled, actor_ema=ema)
                buffer = ReplayBuffer(8)
                for _ in range(4):
                    buffer.add(np.ones(3), np.zeros(1), 1., np.zeros(3), False)
                learner(buffer, update_actor=True, learner_update=1)
                self.assertEqual(int(ema.updates.numpy()), 0)
                learner(buffer, update_actor=True, learner_update=2)
                self.assertEqual(int(ema.updates.numpy()), 1)
        finally:
            set_network_layers(old_layers)


class EMABestCheckpointTests(unittest.TestCase):
    def test_best_keeps_evaluated_ema_artifact_and_training_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            actor = actor_model()
            ema = ActorEMA(actor, .5)
            ema.prepare()
            actor.set_weights([np.ones_like(weight) for weight in actor.get_weights()])
            ema.update()
            checkpoint = tf.train.Checkpoint(actor=actor, actor_ema=ema, episode=tf.Variable(100, dtype=tf.int64))
            source = checkpoint.save(str(Path(tmp) / "training"))
            tracker = BestCheckpointTracker(tracker_args(tmp), "ddpg")
            saver = EMAPolicyArtifactSaver(ema, tmp, {"algorithm": "ddpg"})
            tracker.set_policy_artifact_exporter(saver.export, retain_training_checkpoint=True)
            tracker.evaluate = lambda prefix, episode: evaluation(episode, 1., 1.)
            try:
                self.assertTrue(tracker.evaluate_async(source, 100))
                actor.set_weights([np.full_like(weight, 10.) for weight in actor.get_weights()])
                best = apply_ready_best_checkpoint(tracker, wait_timeout=10.)
                self.assertTrue(Path(str(best) + ".index").exists())
                averaged = tf.keras.models.load_model(Path(tmp) / "best/policy.keras", compile=False)
                np.testing.assert_allclose(averaged(np.zeros((1, 3))).numpy(), [[.5]])
                restored = ActorEMA(actor_model(), .5)
                tf.train.Checkpoint(actor_ema=restored).restore(str(best)).expect_partial()
                restored.prepare()
                self.assertEqual(int(restored.updates.numpy()), 1)
                for weight in restored.shadow:
                    np.testing.assert_array_equal(weight.numpy(), .5)
            finally:
                tracker.close()

    def test_raw_and_ema_best_comparison_baselines_do_not_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = BestCheckpointTracker(tracker_args(tmp), "sac")
            raw.record_best(evaluation(100, 1., 1.), "ckpt-100")
            raw.close()
            averaged = BestCheckpointTracker(tracker_args(tmp, actor_ema_decay=.9), "sac")
            try:
                self.assertIsNone(averaged.best_summary)
                self.assertTrue(averaged.is_improvement(evaluation(200, .5, .5)))
            finally:
                averaged.close()

    def test_full_native_checkpoints_restore_the_exact_inference_model(self):
        from core.models import build_hybrid_actor_critic, build_shared_q_network
        from run import build_policy_checkpoint

        old_layers = network_layers()
        set_network_layers([8, 8])
        try:
            factories = {"sac": lambda: build_sac_actor(3, 1), "ddpg": lambda: build_continuous_actor(3, 1), "td3": lambda: build_continuous_actor(3, 1), "dqn": lambda: build_shared_q_network(3, 3), "ppo": lambda: build_hybrid_actor_critic(3, [3], 1)}
            with tempfile.TemporaryDirectory() as tmp:
                for algorithm, factory in factories.items():
                    with self.subTest(algorithm=algorithm):
                        model = factory()
                        model.set_weights([np.full_like(weight, .123) for weight in model.get_weights()])
                        optimizer = tf.keras.optimizers.Adam(.001)
                        optimizer.build(model.trainable_variables)
                        key = "model" if algorithm in ("dqn", "ppo") else "actor"
                        checkpoint = tf.train.Checkpoint(**{key: model, "optimizer": optimizer})
                        path = checkpoint.save(str(Path(tmp) / algorithm))
                        restored = factory()
                        restored.set_weights([np.zeros_like(weight) for weight in restored.get_weights()])
                        build_policy_checkpoint(restored, algorithm).restore(path).expect_partial()
                        for actual, expected in zip(restored.get_weights(), model.get_weights()):
                            np.testing.assert_array_equal(actual, expected)
        finally:
            set_network_layers(old_layers)


if __name__ == "__main__":
    unittest.main()
