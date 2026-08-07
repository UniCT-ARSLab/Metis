"""The residual-PPO effective policy is a STANDARD deployable/evaluable artifact.

Covers: (a) the fused export policy (base + gated bounded residual) decoded exactly as run.py would decode
a continuous policy equals the runtime ResidualActionAdapter effective action; (b) it round-trips through
tf.keras.models.load_model with NO custom_objects (BoundedGateCombine is a registered serializable); (c) the
generic BestCheckpointTracker policy-artifact hook stages + promotes a policy.keras through the SAME machinery
the standard checkpoint path uses -- so residual PPO uses standard evaluation/best-checkpoint infrastructure,
not an M8-specific detour."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PYDIR = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, PYDIR)

import tensorflow as tf  # noqa: E402
import keras  # noqa: E402

import core.policy_artifact  # noqa: E402,F401  -- registers BoundedGateCombine for load_model
from core.composite_policy import build_residual_export_policy  # noqa: E402
from core.models import build_hybrid_actor_critic  # noqa: E402
from core.policy_action_adapter import ResidualActionAdapter  # noqa: E402
from core.training import BestCheckpointTracker, PolicyEvaluationResult  # noqa: E402
from algorithms.ppo import split_model_outputs  # noqa: E402
from tests.test_best_checkpoint_tracker import evaluation, tracker_args  # noqa: E402

OBS, ACT = 27, 7
GATE = dict(delta_max=0.005, gate_outer=0.11, gate_inner=0.04, err_start=14, err_size=3)
SYM_LOW, SYM_HIGH = np.full(ACT, -1, np.float32), np.full(ACT, 1, np.float32)
# Deliberately asymmetric and != [-1, 1] to catch any hardcoded [-1,1] clip / scale.
ASYM_LOW = np.array([-2.0, -0.5, -1.0, -3.0, 0.0, -1.5, -0.2], np.float32)
ASYM_HIGH = np.array([1.0, 2.0, 0.5, 3.0, 4.0, 0.5, 1.8], np.float32)


def _am(low, high):
    return {"discrete": [], "discrete_sizes": [], "continuous_size": ACT, "continuous": [{"slice": slice(0, ACT)}],
            "single_continuous": True, "single_discrete": False, "continuous_low": low, "continuous_high": high}


def _stack(seed=1):
    keras.utils.set_random_seed(seed)
    base = tf.keras.Sequential([tf.keras.layers.Input((OBS,)), tf.keras.layers.Dense(16, activation="relu"),
                                tf.keras.layers.Dense(ACT, activation="tanh")])
    base(np.zeros((1, OBS), np.float32))
    actor = build_hybrid_actor_critic(OBS, [], ACT, continuous_activation="linear", separate_value_tower=True)
    actor(np.zeros((1, OBS), np.float32))
    ml = actor.get_layer("continuous_mean")
    ml.set_weights([np.random.randn(*w.shape).astype(np.float32) * 0.8 for w in ml.get_weights()])
    return base, actor


def _adapter(base, low, high, delta_max):
    def base_fn(o):
        return np.clip(base(np.asarray(o, np.float32)[None, :], training=False).numpy()[0], low, high)
    return ResidualActionAdapter(base_fn, low=low, high=high, err_start=GATE["err_start"], err_size=GATE["err_size"],
                                 gate_outer=GATE["gate_outer"], gate_inner=GATE["gate_inner"],
                                 delta_max=delta_max, update_mask="gate", base_sha=None)


def _runpy_ppo_decode(out, low, high):
    # run.py select_continuous_action PPO branch: a list/tuple output -> take output[-2] (mean), clip to bounds.
    mean = out[-2] if isinstance(out, (list, tuple)) else out
    return np.clip(mean.numpy()[0], low, high).astype(np.float32)


class FusionParityTests(unittest.TestCase):
    def _check(self, low, high, delta_max):
        base, actor = _stack()
        adapter = _adapter(base, low, high, delta_max)
        am = _am(low, high)
        fused = build_residual_export_policy(base, actor, **{**GATE, "delta_max": delta_max},
                                             action_low=low, action_high=high)
        rng = np.random.default_rng(0)
        worst, gate_active = 0.0, 0
        for _ in range(300):
            obs = (rng.standard_normal(OBS).astype(np.float32)) * 0.03   # small err -> gate sometimes open
            _, cmean, _ = split_model_outputs(actor(obs[None, :], training=False), am)
            eff_adapter, _raw, diag = adapter.transform(obs, cmean.numpy()[0])
            gate_active += int(diag["gate"] > 0.0)
            decoded = _runpy_ppo_decode(fused(obs[None, :], training=False), low, high)
            worst = max(worst, float(np.max(np.abs(np.asarray(eff_adapter) - decoded))))
        self.assertGreater(gate_active, 0, "test obs never opened the gate; parity would be trivial")
        self.assertLess(worst, 1e-5, f"fused effective policy diverged from the adapter by {worst}")

    def test_parity_symmetric_unit_bounds(self):
        self._check(SYM_LOW, SYM_HIGH, GATE["delta_max"])

    def test_parity_asymmetric_nonunit_bounds(self):
        # The critical case: hardcoded [-1,1] clip OR run.py's scale_action_numpy remap would break here.
        self._check(ASYM_LOW, ASYM_HIGH, 0.02)


class ReloadTests(unittest.TestCase):
    def test_roundtrip_no_custom_objects_preserves_bounds(self):
        base, actor = _stack(seed=2)
        fused = build_residual_export_policy(base, actor, **GATE, action_low=ASYM_LOW, action_high=ASYM_HIGH)
        obs = (np.random.default_rng(3).standard_normal((5, OBS)).astype(np.float32)) * 0.03
        before = fused(obs, training=False)[0].numpy()           # action head
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "policy.keras"
            fused.save(str(path))
            reloaded = tf.keras.models.load_model(str(path))     # no custom_objects: layer is registered
        after = reloaded(obs, training=False)[0].numpy()
        self.assertTrue(np.allclose(before, after, atol=1e-6))   # bounds serialized (get_config round-trip)
        self.assertTrue(np.all(after <= ASYM_HIGH + 1e-6) and np.all(after >= ASYM_LOW - 1e-6))

    def test_reflects_live_actor_weights(self):
        base, actor = _stack(seed=4)
        fused = build_residual_export_policy(base, actor, **GATE)
        obs = (np.random.default_rng(5).standard_normal((4, OBS)).astype(np.float32)) * 0.03
        a0 = fused(obs, training=False)[0].numpy()
        ml = actor.get_layer("continuous_mean")
        ml.set_weights([w + 0.5 for w in ml.get_weights()])       # a training step on the shared actor
        self.assertGreater(float(np.max(np.abs(fused(obs, training=False)[0].numpy() - a0))), 1e-6)


class WeightsOnlyBaseTests(unittest.TestCase):
    def test_weights_only_base_is_refused(self):
        from types import SimpleNamespace
        import algorithms.ppo as ppo
        am = _am(SYM_LOW, SYM_HIGH)
        model = build_hybrid_actor_critic(OBS, [], ACT, continuous_activation="linear", separate_value_tower=True)
        model(np.zeros((1, OBS), np.float32))
        with tempfile.TemporaryDirectory() as d:
            weights = Path(d) / "base.weights.h5"                # weights-only: no architecture
            tf.keras.Sequential([tf.keras.layers.Input((OBS,)), tf.keras.layers.Dense(ACT, activation="tanh")]).save_weights(str(weights))
            args = SimpleNamespace(policy_mode="residual", collector_mode="sync", base_policy=str(weights),
                                   residual_gate_config="0.11,0.04,14,3", residual_delta_max=0.005,
                                   residual_update_mask="gate")
            with self.assertRaises(RuntimeError) as ctx:
                ppo.build_action_adapter(args, model, am, zero_init_head=False)
            self.assertIn("weights-only", str(ctx.exception).lower())


class TrackerArtifactHookTests(unittest.TestCase):
    @staticmethod
    def _tiny(dest):
        # A deployable BUNDLE: policy.keras + a coherent sibling policy.json manifest (mirrors ppo.py's exporter).
        m = tf.keras.Sequential([tf.keras.layers.Input((OBS,)), tf.keras.layers.Dense(ACT, activation="tanh")])
        m(np.zeros((1, OBS), np.float32)); m.save(str(dest))
        Path(dest).with_suffix(".json").write_text(json.dumps({"algorithm": "ppo", "model_file": "policy.keras"}))

    def test_evaluation_command_uses_policy_load_for_keras(self):
        with tempfile.TemporaryDirectory() as d:
            tracker = BestCheckpointTracker(tracker_args(d), "ppo")
            summary = Path(d) / "summary.json"
            keras_cmd = tracker._evaluation_command(str(Path(d) / "policy-5.keras"), summary, 5)
            self.assertIn("--load-from", keras_cmd)
            self.assertEqual(keras_cmd[keras_cmd.index("--load-from") + 1], "policy")
            self.assertIn("--policy-path", keras_cmd)
            self.assertNotIn("--checkpoint-path", keras_cmd)
            ckpt_cmd = tracker._evaluation_command(str(Path(d) / "ckpt-5"), summary, 5)   # unchanged path
            self.assertEqual(ckpt_cmd[ckpt_cmd.index("--load-from") + 1], "checkpoint")
            self.assertIn("--checkpoint-path", ckpt_cmd)

    def test_stage_and_promote_policy_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            tracker = BestCheckpointTracker(tracker_args(d), "ppo")
            tracker.set_policy_artifact_exporter(self._tiny)
            tracker.evaluate = lambda staged, episode: evaluation(episode, 0.6, 1.0)   # stub the Godot subprocess
            self.assertTrue(tracker.evaluate_async(str(Path(d) / "ckpt-7"), 7))
            staged = tracker._pending[2]
            self.assertTrue(str(staged).endswith(".keras") and Path(staged).is_file())  # exporter ran, not copy_checkpoint_files
            result = tracker.poll_ready(timeout=15.0)
            self.assertIsNotNone(result)
            best = tracker.promote_ready(result)
            self.assertEqual(Path(best).name, "policy.keras")
            self.assertTrue((tracker.directory / "policy.keras").is_file())
            self.assertTrue((tracker.directory / "policy.json").is_file())                 # bundle: manifest too
            self.assertEqual(json.loads((tracker.directory / "policy.json").read_text())["algorithm"], "ppo")
            self.assertFalse(Path(staged).exists())                                        # staged .keras consumed
            self.assertFalse(Path(staged).with_suffix(".json").exists())                   # staged .json consumed
            meta = json.loads((tracker.directory / "best_metrics.json").read_text())
            self.assertTrue(str(meta["checkpoint"]).endswith("policy.keras"))
            tracker.close()

    def test_clear_staging_removes_orphan_policy_bundles(self):
        # A killed run can strand policy-N.keras (+ .json) in staging; _clear_staging (run on init) must drop them.
        with tempfile.TemporaryDirectory() as d:
            tracker = BestCheckpointTracker(tracker_args(d), "ppo")
            self._tiny(tracker.staging_directory / "policy-9.keras")
            self.assertTrue((tracker.staging_directory / "policy-9.keras").is_file())
            self.assertTrue((tracker.staging_directory / "policy-9.json").is_file())
            tracker._clear_staging()
            self.assertFalse((tracker.staging_directory / "policy-9.keras").exists())
            self.assertFalse((tracker.staging_directory / "policy-9.json").exists())
            tracker.close()


if __name__ == "__main__":
    unittest.main()
