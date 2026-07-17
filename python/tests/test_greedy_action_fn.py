"""The traced collector policy must be the same policy as the eager one.

build_greedy_action_fn is a speed optimisation: if it ever disagrees with the eager
call the collectors would silently be running a different policy than the learner
trains, and no throughput number would reveal it.
"""

import os
import sys
import sysconfig
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Match the trainers' CUDA setup before importing TF, so this exercises the same
# device placement the collectors actually run under.
_purelib = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
if _purelib.exists():
    _libs = [str(d / "lib") for d in _purelib.iterdir() if (d / "lib").is_dir()]
    _cur = [x for x in os.environ.get("LD_LIBRARY_PATH", "").split(":") if x]
    _missing = [x for x in _libs if x not in _cur]
    if _missing and os.environ.get("GODOT_GYM_TF_LD_READY") != "1":
        os.environ["LD_LIBRARY_PATH"] = ":".join(_missing + _cur)
        os.environ["GODOT_GYM_TF_LD_READY"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from core.models import build_greedy_action_fn, build_shared_q_network  # noqa: E402

OBS_DIM = 5
NUM_ACTIONS = 3


def eager_action(model, obs):
    q_values = model(np.expand_dims(obs, axis=0), training=False).numpy()[0]
    return int(np.argmax(q_values))


class GreedyActionFnTests(unittest.TestCase):
    def setUp(self):
        with tf.device("/CPU:0"):
            self.model = build_shared_q_network(obs_dim=OBS_DIM, num_actions=NUM_ACTIONS)
        self.greedy = build_greedy_action_fn(self.model, OBS_DIM)
        self.rng = np.random.default_rng(7)

    def traced_action(self, obs):
        batch = np.expand_dims(obs, axis=0).astype(np.float32)
        return int(self.greedy(batch).numpy()[0])

    def test_agrees_with_eager_on_random_observations(self):
        for _ in range(200):
            obs = self.rng.normal(size=(OBS_DIM,)).astype(np.float32)
            self.assertEqual(self.traced_action(obs), eager_action(self.model, obs))

    def test_still_agrees_after_set_weights(self):
        # sync_model publishes new learner weights via set_weights. The graph closes
        # over the variables, so it must pick the change up without re-tracing.
        for _ in range(5):
            new_weights = [
                self.rng.normal(size=w.shape).astype(np.float32)
                for w in self.model.get_weights()
            ]
            self.model.set_weights(new_weights)
            for _ in range(20):
                obs = self.rng.normal(size=(OBS_DIM,)).astype(np.float32)
                self.assertEqual(self.traced_action(obs), eager_action(self.model, obs))

    def test_traces_once_across_batch_sizes(self):
        # The dynamic batch dimension means single- and multi-agent collectors share
        # one graph. Retracing per batch size would defeat the fixed signature.
        for batch in (1, 3, 1, 7):
            self.greedy(self.rng.normal(size=(batch, OBS_DIM)).astype(np.float32))
        concrete_fns = self.greedy._list_all_concrete_functions_for_serialization()
        self.assertEqual(len(concrete_fns), 1)

    def test_batched_call_agrees_with_eager_per_row(self):
        # The multi-agent collector calls this with (n_agents, obs_dim). Each row must
        # match the eager argmax for that observation.
        obs_batch = self.rng.normal(size=(4, OBS_DIM)).astype(np.float32)
        traced = self.greedy(obs_batch).numpy()
        self.assertEqual(traced.shape, (4,))
        for idx in range(4):
            self.assertEqual(int(traced[idx]), eager_action(self.model, obs_batch[idx]))

    def test_concurrent_batched_calls_stay_well_formed(self):
        # Why the trace exists: eager inference from several collector threads corrupts
        # output. This is the multi-agent shape the DQN async branch feeds it.
        import threading

        errors = []

        def hammer():
            try:
                for _ in range(200):
                    out = self.greedy(np.zeros((4, OBS_DIM), dtype=np.float32)).numpy()
                    assert out.shape == (4,), out.shape
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30.0)
        self.assertEqual(errors, [])


class DqnAsyncUsesTracedInferenceTests(unittest.TestCase):
    """The multi-agent async branch was the one that stayed eager after single-agent was
    traced. This guards against it silently regressing to a raw model(obs) call.
    """

    def test_run_async_dqn_makes_no_eager_inference_call(self):
        source = (Path(__file__).resolve().parents[1] / "algorithms" / "dqn.py").read_text()
        body = source.split("def run_async_dqn", 1)
        self.assertEqual(len(body), 2, "run_async_dqn not found")
        run_async = body[1]
        for eager in ("local_model(observations", "opponent_model(observations"):
            self.assertNotIn(
                eager, run_async,
                msg=f"run_async_dqn does eager inference `{eager}...` -- must use the traced fn",
            )


if __name__ == "__main__":
    unittest.main()
