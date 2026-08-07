"""Q-term schedule, separated Q/BC gradient telemetry, critic-warmup Q-ranking audit, and the
lexicographic best metric -- the infrastructure added after the TD3+BC v4 actor-collapse.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf  # noqa: E402

from algorithms.common import (  # noqa: E402
    build_deterministic_learner_step,
    scheduled_q_weight,
)
from core.critic_audit import q_ranking_audit  # noqa: E402
from core.replay_buffer import ReplayBuffer  # noqa: E402
from core.training import BestCheckpointTracker  # noqa: E402

OBS_DIM = 4
ACT_DIM = 2


class ScheduledQWeightTests(unittest.TestCase):
    def test_ramp_from_zero_and_clamp(self):
        # Off at the first policy update so the unfrozen actor is BC-only.
        self.assertEqual(scheduled_q_weight(0, 0.0, 1.0, 50000), 0.0)
        self.assertAlmostEqual(scheduled_q_weight(25000, 0.0, 1.0, 50000), 0.5, places=5)
        self.assertAlmostEqual(scheduled_q_weight(50000, 0.0, 1.0, 50000), 1.0, places=5)
        self.assertAlmostEqual(scheduled_q_weight(999999, 0.0, 1.0, 50000), 1.0, places=5)


class QRankingAuditTests(unittest.TestCase):
    def _actor_zero(self, obs):
        # BC clone maps everything to the mid of the action range (raw tanh 0 -> scaled midpoint).
        return np.zeros((len(obs), ACT_DIM), np.float32)

    def test_gate_passes_when_critic_prefers_good_actions(self):
        low, high = [-1.0, -1.0], [1.0, 1.0]
        obs = np.zeros((32, OBS_DIM), np.float32)
        expert = np.zeros((32, ACT_DIM), np.float32)  # near the clone (both mid-range)

        # Critic peaks at the midpoint (0,0): expert/clone high, random/saturated low -> gate PASS.
        def critic(oa):
            act = np.asarray(oa[1], dtype=np.float64)
            return (-np.sum(act ** 2, axis=1, keepdims=True)).astype(np.float32)

        rep = q_ranking_audit(critic, self._actor_zero, obs, expert, low, high, seed=0)
        self.assertTrue(rep["gate_passed"], rep["q_mean"])
        self.assertEqual(rep["pairs"]["expert_vs_random"]["win_rate"], 1.0)
        self.assertGreater(rep["pairs"]["clone_vs_saturated_pos"]["mean"], 0.0)
        # Every good-vs-bad pair reports quantiles.
        for p in rep["pairs"].values():
            for k in ("win_rate", "mean", "median", "p10", "p90"):
                self.assertIn(k, p)

    def test_gate_fails_when_critic_prefers_saturated(self):
        low, high = [-1.0, -1.0], [1.0, 1.0]
        obs = np.zeros((32, OBS_DIM), np.float32)
        expert = np.zeros((32, ACT_DIM), np.float32)

        # Critic REWARDS magnitude: saturated actions score highest, expert/clone lowest -> FAIL.
        def bad_critic(oa):
            act = np.asarray(oa[1], dtype=np.float64)
            return (np.sum(act ** 2, axis=1, keepdims=True)).astype(np.float32)

        rep = q_ranking_audit(bad_critic, self._actor_zero, obs, expert, low, high, seed=0)
        self.assertFalse(rep["gate_passed"])
        self.assertLess(rep["pairs"]["expert_vs_saturated_pos"]["win_rate"], 0.9)

    def test_per_cell_negative_margin_fails_gate_even_if_win_rate_ok(self):
        low, high = [-1.0, -1.0], [1.0, 1.0]
        # Two cells. Critic is good globally but INVERTED on cell 1: there it scores the midpoint
        # (expert/clone) BELOW everything. A mean/win-rate-only view could pass; the per-cell gate
        # must catch the bad region.
        obs = np.zeros((40, OBS_DIM), np.float32)
        obs[20:, 0] = 1.0  # mark cell-1 samples in obs so the critic can tell them apart
        expert = np.zeros((40, ACT_DIM), np.float32)
        cells = np.array([0] * 20 + [1] * 20)

        def critic(oa):
            o = np.asarray(oa[0], dtype=np.float64)
            act = np.asarray(oa[1], dtype=np.float64)
            base = -np.sum(act ** 2, axis=1)      # midpoint best
            flip = np.where(o[:, 0] > 0.5, -1.0, 1.0)  # cell 1 inverts the preference
            return (base * flip).reshape(-1, 1).astype(np.float32)

        rep = q_ranking_audit(critic, self._actor_zero, obs, expert, low, high,
                              group_ids=cells, seed=0)
        self.assertFalse(rep["gate_passed"])
        self.assertFalse(rep["gate"]["no_negative_cell"])
        self.assertLess(rep["pairs"]["expert_vs_random"]["worst_cell_margin"], 0.0)

    def test_perturbed_is_diagnostic_not_gated(self):
        low, high = [-1.0, -1.0], [1.0, 1.0]
        obs = np.zeros((32, OBS_DIM), np.float32)
        expert = np.zeros((32, ACT_DIM), np.float32)

        # Critic strongly prefers the exact midpoint; a perturbed action scores a bit lower than the
        # clone, but perturbed must NOT count as a "bad" family, so the gate still PASSES.
        def critic(oa):
            act = np.asarray(oa[1], dtype=np.float64)
            return (-np.sum(act ** 2, axis=1, keepdims=True)).astype(np.float32)

        rep = q_ranking_audit(critic, self._actor_zero, obs, expert, low, high, seed=0)
        self.assertTrue(rep["gate_passed"])
        self.assertIn("perturbed_vs_random", rep["diagnostic"])
        self.assertNotIn("expert_vs_perturbed", rep["pairs"])

    def test_twin_clipping_uses_min(self):
        low, high = [-1.0, -1.0], [1.0, 1.0]
        obs = np.zeros((8, OBS_DIM), np.float32)
        expert = np.zeros((8, ACT_DIM), np.float32)
        c1 = lambda oa: np.full((len(oa[0]), 1), 5.0, np.float32)
        c2 = lambda oa: np.full((len(oa[0]), 1), 1.0, np.float32)
        rep = q_ranking_audit(c1, self._actor_zero, obs, expert, low, high, critic2=c2, seed=0)
        # min(5, 1) = 1 for every family.
        self.assertAlmostEqual(rep["q_mean"]["expert"], 1.0, places=5)


def _td3_bc_learner():
    rng = np.random.default_rng(0)
    actor = tf.keras.Sequential([
        tf.keras.layers.Input((OBS_DIM,)),
        tf.keras.layers.Dense(8, activation="tanh"),
        tf.keras.layers.Dense(ACT_DIM, activation="tanh"),
    ])

    def _critic():
        oi = tf.keras.layers.Input((OBS_DIM,))
        ai = tf.keras.layers.Input((ACT_DIM,))
        x = tf.keras.layers.Concatenate()([oi, ai])
        x = tf.keras.layers.Dense(8, activation="relu")(x)
        return tf.keras.Model([oi, ai], tf.keras.layers.Dense(1)(x))

    def _clone(m):
        c = tf.keras.models.clone_model(m)
        c.set_weights(m.get_weights())
        return c

    critic, critic2 = _critic(), _critic()
    demo = {
        "obs": rng.standard_normal((16, OBS_DIM)).astype(np.float32),
        "actions": np.clip(rng.standard_normal((16, ACT_DIM)), -1, 1).astype(np.float32),
    }
    learner = build_deterministic_learner_step(
        actor, critic, _clone(actor), _clone(critic),
        tf.keras.optimizers.Adam(1e-4), tf.keras.optimizers.Adam(1e-4),
        gamma=0.99, action_low=[-1.0] * ACT_DIM, action_high=[1.0] * ACT_DIM,
        variant="td3_bc", critic2=critic2, target_critic2=_clone(critic2),
        critic2_optimizer=tf.keras.optimizers.Adam(1e-4),
        policy_delay=1, demo_data=demo, demo_batch_size=8, demo_q_filter=False,
        batch_size=8, compiled=False,
    )
    buffer = ReplayBuffer(capacity=1000)
    for _ in range(64):
        buffer.add(rng.standard_normal(OBS_DIM).astype(np.float32),
                   np.clip(rng.standard_normal(ACT_DIM), -1, 1).astype(np.float32),
                   float(rng.standard_normal()),
                   rng.standard_normal(OBS_DIM).astype(np.float32), False)
    return learner, actor, _clone(actor), buffer


class GradientTelemetryTests(unittest.TestCase):
    def test_q_weight_zero_means_no_q_gradient_bc_anchors(self):
        learner, _actor, ref, buf = _td3_bc_learner()
        r = learner(buf, update_actor=True, learner_update=1, bc_weight=1.0,
                    q_filter_active=False, q_weight=0.0, gradient_telemetry=True,
                    bc_reference_actor=ref)
        # Q term is off -> zero Q-gradient; BC anchor is the only pull -> positive BC gradient.
        self.assertAlmostEqual(r["q_grad_norm"], 0.0, places=6)
        self.assertGreater(r["bc_grad_norm"], 0.0)
        self.assertEqual(r["q_weight"], 0.0)
        for key in ("raw_q_loss", "td3_bc_q_scale", "scaled_q_loss", "grad_cosine",
                    "actor_bc_deviation_pre_update", "actor_bc_deviation_post_update"):
            self.assertIn(key, r)
        # scaled_q_loss = q_weight * scale * q_loss = 0 when q_weight is 0.
        self.assertAlmostEqual(r["scaled_q_loss"], 0.0, places=6)

    def test_q_weight_one_produces_q_gradient(self):
        learner, _actor, ref, buf = _td3_bc_learner()
        r = learner(buf, update_actor=True, learner_update=1, bc_weight=1.0,
                    q_filter_active=False, q_weight=1.0, gradient_telemetry=True,
                    bc_reference_actor=ref)
        self.assertGreater(r["q_grad_norm"], 0.0)
        self.assertIsNotNone(r["td3_bc_q_scale"])

    def test_deviation_pre_and_post_update(self):
        learner, _actor, ref, buf = _td3_bc_learner()
        # ref is a clone of the actor's INITIAL weights -> pre-update deviation is ~0. After one BC
        # update the actor has moved, so post-update deviation is strictly larger than pre.
        r = learner(buf, update_actor=True, learner_update=1, bc_weight=1.0,
                    q_filter_active=False, q_weight=0.0, gradient_telemetry=True,
                    bc_reference_actor=ref)
        self.assertIsNotNone(r["actor_bc_deviation_pre_update"])
        self.assertIsNotNone(r["actor_bc_deviation_post_update"])
        self.assertAlmostEqual(r["actor_bc_deviation_pre_update"], 0.0, places=5)
        self.assertGreater(r["actor_bc_deviation_post_update"],
                           r["actor_bc_deviation_pre_update"])


class _FakeTracker:
    # Bind the real ranking logic to a light stand-in (the real __init__ spins up Godot machinery).
    comparison_key = BestCheckpointTracker.comparison_key
    is_improvement = BestCheckpointTracker.is_improvement

    def __init__(self, best_metric, best_key=None):
        self.args = SimpleNamespace(best_metric=best_metric)
        self.best_key = best_key


class _Result:
    def __init__(self, summary):
        self.summary = summary


class LexicographicMetricTests(unittest.TestCase):
    def test_collapsed_policy_never_becomes_best(self):
        t = _FakeTracker("lexicographic", best_key=None)
        collapsed = _Result({"success_rate": 0.0, "collision_rate": 1.0,
                             "progress_mean": 0.65, "reward_mean": -34.0})
        # 0 success + 100% collision: refused as best even though progress is high and best is None.
        self.assertFalse(t.is_improvement(collapsed))

    def test_fewer_collisions_ranks_above_more(self):
        t = _FakeTracker("lexicographic")
        safer = t.comparison_key({"success_rate": 0.0, "collision_rate": 0.3,
                                  "progress_mean": 0.4, "reward_mean": -5.0})
        riskier = t.comparison_key({"success_rate": 0.0, "collision_rate": 0.9,
                                    "progress_mean": 0.8, "reward_mean": -5.0})
        # Higher progress cannot outweigh more collisions.
        self.assertGreater(safer, riskier)

    def test_success_dominates_collision(self):
        t = _FakeTracker("lexicographic")
        succeeds = t.comparison_key({"success_rate": 0.5, "collision_rate": 0.5,
                                     "progress_mean": 0.4, "reward_mean": 0.0})
        collides_less = t.comparison_key({"success_rate": 0.2, "collision_rate": 0.0,
                                          "progress_mean": 0.9, "reward_mean": 0.0})
        self.assertGreater(succeeds, collides_less)


class StopAfterWarmupFlagTests(unittest.TestCase):
    def test_flag_and_q_weight_args_default(self):
        from unittest import mock
        import algorithms.common as common
        with mock.patch("sys.argv", ["td3_bc"]):
            args = common.parse_args("td3_bc")
        self.assertFalse(args.stop_after_critic_warmup)
        self.assertEqual(args.demo_q_weight_start, 0.0)
        self.assertEqual(args.demo_q_weight_end, 1.0)
        self.assertEqual(args.gradient_telemetry_every, 0)


class ParserGuardTests(unittest.TestCase):
    """Multi-policy is not (yet) supported for the audit / gradient telemetry -> the parser must
    FAIL loudly rather than silently ignore the flag."""

    def _parse(self, argv):
        from unittest import mock
        import algorithms.common as common
        with mock.patch("sys.argv", argv):
            return common.parse_args("td3_bc")

    def test_multi_policy_plus_stop_after_warmup_errors(self):
        with self.assertRaises(SystemExit):
            self._parse(["td3_bc", "--multi-policy", "--stop-after-critic-warmup"])

    def test_multi_policy_plus_gradient_telemetry_errors(self):
        with self.assertRaises(SystemExit):
            self._parse(["td3_bc", "--multi-policy", "--gradient-telemetry-every", "10"])

    def test_single_policy_allows_both(self):
        args = self._parse(["td3_bc", "--stop-after-critic-warmup",
                            "--gradient-telemetry-every", "10"])
        self.assertTrue(args.stop_after_critic_warmup)
        self.assertEqual(args.gradient_telemetry_every, 10)


class LoopContractIntegrationTests(unittest.TestCase):
    """The per-update contract shared by all four deterministic loops, exercised with the REAL
    learner + REAL warmup-gating runtime (no Godot): stop-after-warmup lands before the first actor
    update; q_weight is 0 on that first update; the counter advances only on policy_updated."""

    def _runtime(self, warmup):
        from core.training_health import OffPolicyRecoveryRuntime
        args = SimpleNamespace(_training_health_monitor=None)
        return OffPolicyRecoveryRuntime(
            args, ReplayBuffer(capacity=10),
            initial_critic_warmup=warmup, normal_policy_update_every=1)

    def test_stop_lands_before_any_actor_update(self):
        rt = self._runtime(warmup=5)
        actor_ever_updated = False
        fired_with_actor_frozen = None
        for _ in range(5):
            update_actor = rt.should_update_policy()   # loop order: gate BEFORE the update
            actor_ever_updated = actor_ever_updated or update_actor
            if rt.record_critic_update():              # warmup just completed on this step
                fired_with_actor_frozen = not actor_ever_updated
                break
        self.assertTrue(fired_with_actor_frozen)

    def test_first_actor_update_uses_q_weight_zero_and_counter_tracks_policy_updated(self):
        learner, _actor, _ref, buf = _td3_bc_learner()
        counter = tf.Variable(0, dtype=tf.int64, trainable=False)  # checkpointed counter
        rt = self._runtime(warmup=4)
        first_actor_q_weight = None
        for _ in range(10):
            update_actor = rt.should_update_policy()
            psw = int(counter.numpy())
            q_weight = scheduled_q_weight(psw, 0.0, 1.0, 8)
            r = learner(buf, update_actor=update_actor, learner_update=1, bc_weight=1.0,
                        q_filter_active=False, q_weight=q_weight)
            rt.record_critic_update()
            if r["policy_updated"]:
                if first_actor_q_weight is None:
                    first_actor_q_weight = q_weight
                counter.assign_add(1)  # advance ONLY on a real policy update
        # First post-warmup actor update had q_weight 0 -> BC-only, no Q push (anti-collapse), and
        # the counter only moved on the updates that actually happened.
        self.assertEqual(first_actor_q_weight, 0.0)
        self.assertGreater(int(counter.numpy()), 0)


if __name__ == "__main__":
    unittest.main()
