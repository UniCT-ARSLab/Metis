"""Resume tests for the SHARED residual-PPO checkpoint path (python/algorithms/ppo.py). Covers the
save/load round-trip, fail-closed manifest validation, partial-checkpoint ignore, the async guard, and
a REAL TWO-PROCESS interruption/resume equivalence (a subprocess script, not an in-process save/load):
running N generations continuously must equal N-1 then exit then resume 1 — same weights, optimizer
iterations, next stochastic action and NumPy shuffle."""
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

PYDIR = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, PYDIR)

import tensorflow as tf  # noqa: E402
import keras  # noqa: E402

import algorithms.ppo as ppo  # noqa: E402
from core.models import (  # noqa: E402
    build_hybrid_actor_critic,
    default_network_layers,
    set_network_layers,
)


def _manifest(**over):
    m = {"policy_mode": "residual", "base_sha": "abc", "obs_dim": 6, "action_size": 3, "residual_delta_max": 0.005,
         "gate_outer": 0.11, "gate_inner": 0.04, "err_start": 0, "err_size": 2, "update_mask": "gate"}
    m.update(over)
    return m


def _residual_stack():
    m = build_hybrid_actor_critic(6, [], 3, continuous_activation="linear", separate_value_tower=True)
    m(np.zeros((1, 6), np.float32))
    ls = tf.Variable(np.full(3, -1.5, np.float32), trainable=True)
    opt = tf.keras.optimizers.Adam(1e-3); vopt = tf.keras.optimizers.Adam(3e-3)
    tf_gen = tf.random.Generator.from_seed(0)
    gv = tf.Variable(0, dtype=tf.int64); uv = tf.Variable(0, dtype=tf.int64)
    ck = tf.train.Checkpoint(model=m, log_std=ls, optimizer=opt, value_optimizer=vopt, tf_gen=tf_gen, generation=gv, policy_updates=uv)
    return m, ls, opt, vopt, tf_gen, gv, uv, ck


class ManifestTests(unittest.TestCase):
    def test_fail_closed_on_mismatch(self):
        ppo.validate_residual_manifest(_manifest(), _manifest())                      # identical: OK
        for bad in [{"base_sha": "zzz"}, {"gate_outer": 0.2}, {"obs_dim": 7}, {"residual_delta_max": 0.01}, {"update_mask": "all"}]:
            with self.assertRaises(RuntimeError):
                ppo.validate_residual_manifest(_manifest(**bad), _manifest())


class RoundTripTests(unittest.TestCase):
    def test_save_load_restores_everything(self):
        m, ls, opt, vopt, tf_gen, gv, uv, ck = _residual_stack()
        av, vv = ppo._split_actor_value_vars(m, ls, 3)
        opt.apply_gradients(zip([tf.ones_like(v) for v in av], av))          # move actor + iterations
        vopt.apply_gradients(zip([tf.ones_like(v) for v in vv], vv))
        gv.assign(4); uv.assign(9); rng = np.random.default_rng(7); _ = rng.random(3)
        probe = np.random.rand(4, 6).astype(np.float32)
        det0 = m(probe)[-2].numpy().copy()
        with tempfile.TemporaryDirectory() as d:
            mgr = tf.train.CheckpointManager(ck, directory=str(Path(d) / "ppo_ckpt"), max_to_keep=2)
            ppo.save_residual_checkpoint(d, mgr, _manifest(), np_rng=rng, counters={"generation": 4, "policy_updates": 9}, model=m)
            g_orig = tf_gen.normal([3]).numpy()                                      # original stream continues post-save
            m2, ls2, opt2, vopt2, tf_gen2, gv2, uv2, ck2 = _residual_stack()
            ppo.materialize_optimizer_slots(m2, ls2, opt2, vopt2, 6, 3, None)
            rng2 = np.random.default_rng(999)
            counters, saved_m, _ = ppo.load_residual_checkpoint(d, ck2, _manifest(), rng2, model=m2)
            g_res = tf_gen2.normal([3]).numpy()                                      # restored stream
        self.assertEqual(counters, {"generation": 4, "policy_updates": 9})
        self.assertTrue(np.allclose(det0, m2(probe)[-2].numpy(), atol=1e-6))          # weights restored
        self.assertEqual(int(opt2.iterations.numpy()), int(opt.iterations.numpy()))   # actor optimizer iters
        self.assertEqual(int(vopt2.iterations.numpy()), int(vopt.iterations.numpy()))  # value optimizer iters
        av2, _ = ppo._split_actor_value_vars(m2, ls2, 3)
        self.assertTrue(any(np.any(sv.numpy() != 0) for v in av2 for sv in [opt2.variables[1]]))  # optimizer slots restored non-zero
        self.assertTrue(np.allclose(g_orig, g_res, atol=1e-6))                        # tf RNG restored -> same next draw
        self.assertTrue(np.allclose(rng2.random(3), np.random.default_rng(7).random(6)[3:]))  # np RNG restored


class PartialIgnoreTests(unittest.TestCase):
    def test_partial_save_is_ignored(self):
        m, ls, opt, vopt, tf_gen, gv, uv, ck = _residual_stack()
        with tempfile.TemporaryDirectory() as d:
            mgr = tf.train.CheckpointManager(ck, directory=str(Path(d) / "ppo_ckpt"), max_to_keep=5)
            gv.assign(1); ppo.save_residual_checkpoint(d, mgr, _manifest(), np_rng=np.random.default_rng(0), counters={"generation": 1, "policy_updates": 1}, model=m)
            # simulate a crash AFTER a further tf save but BEFORE sidecar/latest_complete were published:
            gv.assign(2); mgr.save()                                                    # ckpt-2 exists, NO sidecar-2, latest_complete still -> 1
            m2, ls2, opt2, vopt2, tf_gen2, gv2, uv2, ck2 = _residual_stack()
            ppo.materialize_optimizer_slots(m2, ls2, opt2, vopt2, 6, 3, None)
            counters, _, _ = ppo.load_residual_checkpoint(d, ck2, _manifest(), np.random.default_rng(0), model=m2)
        self.assertEqual(counters["generation"], 1)                                    # the last COMPLETE one, not the partial


class AsyncGuardTests(unittest.TestCase):
    def test_residual_async_refused(self):
        from types import SimpleNamespace
        m, *_ = _residual_stack()
        am = {"continuous_size": 3}
        args = SimpleNamespace(policy_mode="residual", collector_mode="async", base_policy="x", residual_gate_config="0.11,0.04,0,2",
                               residual_delta_max=0.005, residual_update_mask="gate")
        with self.assertRaises(RuntimeError):
            ppo.build_action_adapter(args, m, am)


_SCRIPT = textwrap.dedent('''
    import sys, json, numpy as np, tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
    sys.path.insert(0, sys.argv[4])
    import keras
    import algorithms.ppo as ppo
    from core.models import build_hybrid_actor_critic
    MODE, CKDIR, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
    tf.random.set_seed(0); keras.utils.set_random_seed(0)   # Keras-3 initializers use keras' own RNG
    OD, AC = 6, 3
    am = {"discrete": [], "discrete_sizes": [], "continuous_size": AC, "continuous": [{"slice": slice(0, AC)}],
          "single_discrete": False, "single_continuous": True,
          "continuous_low": np.full(AC, -1, np.float32), "continuous_high": np.full(AC, 1, np.float32)}
    m = build_hybrid_actor_critic(OD, [], AC, continuous_activation="linear", separate_value_tower=True); m(np.zeros((1, OD), np.float32))
    ml = m.get_layer("continuous_mean"); ml.set_weights([np.zeros_like(w) for w in ml.get_weights()])
    ls = tf.Variable(np.full(AC, -1.5, np.float32), trainable=True)
    opt = tf.keras.optimizers.Adam(1e-3); vopt = tf.keras.optimizers.Adam(3e-3)
    tf_gen = tf.random.Generator.from_seed(0)
    gv = tf.Variable(0, dtype=tf.int64); uv = tf.Variable(0, dtype=tf.int64)
    ck = tf.train.Checkpoint(model=m, log_std=ls, optimizer=opt, value_optimizer=vopt, tf_gen=tf_gen, generation=gv, policy_updates=uv)
    mgr = tf.train.CheckpointManager(ck, directory=CKDIR + "/ppo_ckpt", max_to_keep=5)
    rng = np.random.default_rng(0)
    man = {"policy_mode": "residual", "base_sha": "x", "obs_dim": OD, "action_size": AC, "residual_delta_max": 0.005,
           "gate_outer": 0.11, "gate_inner": 0.04, "err_start": 0, "err_size": 2, "update_mask": "gate"}
    sample_fn = ppo.build_sample_action_fn(m, OD, am, rng_gen=tf_gen)
    class A:
        clip_ratio = 0.1; value_loss_coef = 0.5; entropy_coef = 0.002; ppo_epochs = 2; batch_size = 8
        ppo_log_std_min = -5.0; ppo_log_std_max = -0.5; ppo_grad_clip = 1.0; ppo_target_kl = 0.0
        tf_compile_learner = False; tf_xla = False
    start = 0
    if MODE == "resume":
        ppo.materialize_optimizer_slots(m, ls, opt, vopt, OD, AC, am)
        ml.set_weights([np.full_like(w, 0.7) for w in ml.get_weights()])  # perturb; restore must overwrite
        counters, _, _ = ppo.load_residual_checkpoint(CKDIR, ck, man, rng, model=m)
        start = int(counters["generation"]) + 1
    def one_gen():
        O = []; U = []; LP = []
        for t in range(8):
            obs = np.full(OD, 0.1 * t - 0.4, np.float32)   # deterministic per-step obs (no env, dims fixed)
            sel = ppo.select_action(sample_fn, ls.numpy(), obs, am)
            O.append(obs.copy()); U.append(sel["continuous_action"]); LP.append(sel["log_prob"])
        b = {"obs": np.asarray(O, np.float32), "discrete_actions": np.zeros((8, 0), np.int32),
             "continuous_actions": np.asarray(U, np.float32), "log_probs": np.asarray(LP, np.float32),
             "returns": np.arange(8, dtype=np.float32), "advantages": np.linspace(-1, 1, 8).astype(np.float32),
             "train_masks": np.ones(8, np.float32)}
        ppo.ppo_update(m, ls, opt, b, am, A(), vopt, shuffle_rng=rng)
    for g in range(start, N):
        one_gen(); gv.assign(g); uv.assign(g + 1)
        ppo.save_residual_checkpoint(CKDIR, mgr, man, np_rng=rng, counters={"generation": int(g), "policy_updates": int(g + 1)}, model=m)
    probe = np.linspace(0, 1, OD * 4).reshape(4, OD).astype(np.float32)
    det = m(probe)[-2].numpy(); nxt = ppo.select_action(sample_fn, ls.numpy(), probe[0], am)["continuous_action"]
    idx = np.arange(12); rng.shuffle(idx)
    print("FINGERPRINT " + json.dumps({"det": round(float(det.sum()), 6), "opt": int(opt.iterations.numpy()),
          "vopt": int(vopt.iterations.numpy()), "nxt": [round(float(x), 6) for x in nxt], "shuffle": idx.tolist()}))
''')


class TwoProcessResumeEquivalenceTests(unittest.TestCase):
    def _run(self, script, mode, ckdir, n):
        import os
        # CPU + deterministic + oneDNN OFF (oneDNN reorders float reductions -> non-bit-reproducible runs).
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TF_DETERMINISTIC_OPS="1", TF_ENABLE_ONEDNN_OPTS="0", TF_CPP_MIN_LOG_LEVEL="3")
        out = subprocess.run([sys.executable, script, mode, ckdir, str(n), PYDIR], capture_output=True, text=True, timeout=300, env=env)
        line = [l for l in out.stdout.splitlines() if l.startswith("FINGERPRINT ")]
        self.assertTrue(line, f"no fingerprint. stderr tail:\n{out.stderr[-800:]}")
        return json.loads(line[0][len("FINGERPRINT "):])

    def test_continuous_equals_interrupt_resume(self):
        with tempfile.TemporaryDirectory() as dc, tempfile.TemporaryDirectory() as dr:
            script = Path(dc) / "s.py"; script.write_text(_SCRIPT)
            cont = self._run(str(script), "run", dc, 3)                 # 3 generations in ONE process
            _ = self._run(str(script), "run", dr, 2)                    # 2 generations, then EXIT
            res = self._run(str(script), "resume", dr, 3)               # NEW process resumes to 3
        self.assertEqual(cont["opt"], res["opt"]); self.assertEqual(cont["vopt"], res["vopt"])  # optimizer iters
        self.assertAlmostEqual(cont["det"], res["det"], places=4)       # same trained weights
        self.assertTrue(np.allclose(cont["nxt"], res["nxt"], atol=1e-4))  # same next stochastic action (tf RNG)
        self.assertEqual(cont["shuffle"], res["shuffle"])              # same NumPy shuffle order


# ---------------------------------------------------------------------------
# REAL entry-point two-process resume: this drives train.py --algorithm ppo --policy-mode residual end to
# end (arg parsing -> ppo.main -> sync single-policy loop -> _save_ckpt/resume) with a Godot-less fake env,
# proving the resume helpers are wired into the OFFICIAL path (not just called by the tests/M8 orchestrator).
# ---------------------------------------------------------------------------
_ENTRY_SCRIPT = textwrap.dedent('''
    import os, sys
    os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["OMP_NUM_THREADS"] = "1"; os.environ.setdefault("GODOT_GYM_TF_LD_READY", "1")
    MODE, CKDIR, BASE, N, PYDIR = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    sys.path.insert(0, PYDIR)
    import numpy as np, tensorflow as tf, keras
    tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
    keras.utils.set_random_seed(0)   # deterministic model init across processes (Keras-3 initializers)
    OBS, ACT = 27, 7
    SPEC = {"arm": {"action_type": "continuous", "size": ACT, "low": -1.0, "high": 1.0}}
    class FakeEnv:
        def __init__(self, port=None, seed=None, timeout=None, agent_id=None, multi_agent=False):
            self.obs_dim = OBS; self.action_type = "continuous"; self.action_size = ACT
            self.action_names = ["arm%d" % i for i in range(ACT)]
            self.action_low = [-1.0] * ACT; self.action_high = [1.0] * ACT
            self.action_space_spec = SPEC; self.agent_id = "Agent"; self.agent_ids = ["Agent"]
            self.agent_team_ids = [0]; self.multi_agent = False; self._s = 0
        def configure(self, **kw): pass
        def _obs(self): return np.full((OBS,), 0.01 + 0.001 * self._s, dtype=np.float32)
        def reset(self, seed=None): self._s = 0; return self._obs(), {"seed": seed}
        def step(self, a): self._s += 1; return self._obs(), 1.0, self._s >= 2, False, {"agent_info": {}}
        def close(self): pass
        def agent_summary(self): return "fake"
        def team_summary(self): return "team=1"
        def _spec_for_agent(self, aid): return {"action_space": SPEC}
        @property
        def agent_specs(self): return [self._spec_for_agent(self.agent_id)]
    class FakeManager:
        def __init__(self, *a, **k): pass
        def start_many(self, *a, **k): pass
        def close(self, *a, **k): pass
        def stop_all(self, *a, **k): pass
    import algorithms.ppo as ppo
    ppo.GodotProcessManager = FakeManager; ppo.ScenarioGymEnv = FakeEnv
    ppo.maybe_start_dashboard = lambda *a, **k: None
    import train
    argv = ["train.py", "--algorithm", "ppo", "--backend", "metis",
            "--collector-mode", "sync", "--no-parallel-env-steps", "--num-envs", "1", "--no-multi-agent",
            "--policy-mode", "residual", "--base-policy", BASE, "--residual-gate-config", "0.11,0.04,14,3",
            "--residual-delta-max", "0.005", "--num-episodes", N, "--max-steps-per-episode", "2",
            "--checkpoint-every", "100", "--keep-checkpoints", "10",
            "--weights-path", os.path.join(CKDIR, "w.weights.h5"), "--checkpoint-dir", CKDIR,
            "--env-seed-base", "100", "--learning-rate", "3e-4", "--value-learning-rate", "3e-4", "--headless"]
    if MODE == "resume":
        argv.append("--resume")
    sys.argv = argv
    train.main()
''')


def _entry_fingerprint(ckdir, base):
    # The subprocess ran the real PPO entry point, which builds its networks from PPO's default
    # widths. Rebuild the same architecture here or the checkpoint will not load.
    set_network_layers(default_network_layers("ppo"))
    """Reconstruct the FINAL residual state written by a real train.py run and derive the user-visible
    signals: trained weights, both optimizers' iterations, the next stochastic action (restored tf RNG) and
    the next NumPy shuffle (restored np RNG). Same architecture/args as ppo.main's residual branch."""
    import hashlib
    from types import SimpleNamespace
    OBS, ACT, SEED = 27, 7, 100
    spec = {"arm": {"action_type": "continuous", "size": ACT, "low": -1.0, "high": 1.0}}
    am = ppo.build_action_metadata(spec)
    args = SimpleNamespace(policy_mode="residual", base_policy=base, residual_gate_config="0.11,0.04,14,3",
                           residual_delta_max=0.005, residual_update_mask="gate", collector_mode="sync",
                           value_learning_rate=3e-4, learning_rate=3e-4)
    model = build_hybrid_actor_critic(OBS, [], ACT, continuous_activation="linear", separate_value_tower=True)
    model(np.zeros((1, OBS), np.float32), training=False)
    log_std = tf.Variable(np.full((ACT,), -0.5, np.float32), name="continuous_log_std")
    opt = tf.keras.optimizers.Adam(3e-4); vopt = tf.keras.optimizers.Adam(3e-4)
    ppo.materialize_optimizer_slots(model, log_std, opt, vopt, OBS, ACT, am)
    epv = tf.Variable(0, dtype=tf.int64); gnv = tf.Variable(0, dtype=tf.int64); puv = tf.Variable(0, dtype=tf.int64)
    tfg = tf.random.Generator.from_seed(SEED)
    ck = tf.train.Checkpoint(model=model, log_std=log_std, optimizer=opt, value_optimizer=vopt,
                             tf_gen=tfg, generation=gnv, policy_updates=puv, episode=epv)
    nprng = np.random.default_rng(SEED)
    counters, _saved, _p = ppo.load_residual_checkpoint(ckdir, ck, ppo.residual_manifest(args, None, OBS, ACT, None), nprng, model=model)
    adapter = ppo.build_action_adapter(args, model, am, zero_init_head=False)
    sample_fn = ppo.build_sample_action_fn(model, OBS, am, rng_gen=tfg)
    sel = ppo.select_action(sample_fn, log_std, np.full((OBS,), 0.02, np.float32), am, adapter=adapter)
    w = model.get_weights()
    return {
        "counters": counters,
        "wsha": hashlib.sha256(b"".join(np.ascontiguousarray(x).tobytes() for x in w)).hexdigest()[:16],
        "log_std": [round(float(x), 7) for x in log_std.numpy()],
        "opt": int(opt.iterations.numpy()), "vopt": int(vopt.iterations.numpy()),
        "next_action": [round(float(x), 7) for x in np.asarray(sel["env_action"], np.float32)],
        "shuffle": nprng.permutation(16).tolist(),
    }


class RealTrainEntryResumeTests(unittest.TestCase):
    def _train(self, ckdir, base, n, resume=False):
        import os
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TF_DETERMINISTIC_OPS="1", TF_ENABLE_ONEDNN_OPTS="0", TF_CPP_MIN_LOG_LEVEL="3")
        out = subprocess.run([sys.executable, self._script, "resume" if resume else "run", ckdir, base, str(n), PYDIR],
                             capture_output=True, text=True, timeout=600, env=env)
        self.assertIn("Saved weights", out.stdout, f"train.py run failed. stderr tail:\n{out.stderr[-1200:]}")
        return out

    def test_real_entry_point_resume_equivalence(self):
        with tempfile.TemporaryDirectory() as work:
            base = str(Path(work) / "base.keras")
            keras.utils.set_random_seed(3)
            bm = tf.keras.Sequential([tf.keras.layers.Input((27,)), tf.keras.layers.Dense(32, activation="relu"),
                                      tf.keras.layers.Dense(7, activation="tanh")])
            bm(np.zeros((1, 27), np.float32)); bm.save(base)
            self._script = str(Path(work) / "entry.py"); Path(self._script).write_text(_ENTRY_SCRIPT)
            cont = str(Path(work) / "cont"); split = str(Path(work) / "split")
            self._train(cont, base, 4)                 # 4 episodes in ONE process
            self._train(split, base, 2)                # 2 episodes, then EXIT
            self._train(split, base, 4, resume=True)   # NEW process --resume to 4
            a = _entry_fingerprint(cont, base)
            b = _entry_fingerprint(split, base)
        self.assertEqual(a["counters"], b["counters"])         # same episode/generation/policy_updates
        self.assertEqual(a["opt"], b["opt"]); self.assertEqual(a["vopt"], b["vopt"])  # both optimizer iters
        self.assertEqual(a["wsha"], b["wsha"])                 # bit-identical trained weights
        self.assertEqual(a["log_std"], b["log_std"])           # restored policy log_std
        self.assertEqual(a["next_action"], b["next_action"])   # same next stochastic action (restored tf RNG)
        self.assertEqual(a["shuffle"], b["shuffle"])           # same next NumPy shuffle (restored np RNG)
        self.assertEqual(a["counters"]["episode"], 4)


class GenericResidualExportE2ETests(unittest.TestCase):
    """End-to-end generic residual PPO on a SYNTHETIC task with NO cells / hold / target_reached: a plain
    continuous fake env (reward per step, fixed-length episodes). Proves train.py --policy-mode residual is a
    complete generic path that produces a STANDARD deployable policy.keras (the fused effective policy) usable
    with no residual/M8 awareness."""
    def test_residual_run_exports_runnable_effective_policy(self):
        with tempfile.TemporaryDirectory() as work:
            base = str(Path(work) / "base.keras")
            keras.utils.set_random_seed(3)
            bm = tf.keras.Sequential([tf.keras.layers.Input((27,)), tf.keras.layers.Dense(16, activation="relu"),
                                      tf.keras.layers.Dense(7, activation="tanh")])
            bm(np.zeros((1, 27), np.float32)); bm.save(base)
            script = str(Path(work) / "entry.py"); Path(script).write_text(_ENTRY_SCRIPT)
            ckdir = str(Path(work) / "run")
            import os
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TF_DETERMINISTIC_OPS="1", TF_ENABLE_ONEDNN_OPTS="0", TF_CPP_MIN_LOG_LEVEL="3")
            out = subprocess.run([sys.executable, script, "run", ckdir, base, "3", PYDIR],
                                 capture_output=True, text=True, timeout=600, env=env)
            self.assertIn("Saved weights", out.stdout, f"residual run failed:\n{out.stderr[-1200:]}")
            policy = Path(ckdir) / "policy.keras"
            self.assertTrue(policy.is_file(), "residual run did not export a deployable policy.keras")
            # loads with NO custom_objects (BoundedGateCombine registered via core.policy_artifact) and runs;
            # multi-head [action, value] output (run.py's PPO continuous decode reads output[-2]).
            model = tf.keras.models.load_model(str(policy))
            out = model(np.full((3, 27), 0.02, np.float32), training=False)
            act = (out[-2] if isinstance(out, (list, tuple)) else out).numpy()
            self.assertEqual(act.shape, (3, 7))
            self.assertTrue(np.all(np.abs(act) <= 1.0 + 1e-6), "effective policy action out of [-1,1]")
            # a coherent deployable bundle: policy.json manifest beside policy.keras
            self.assertTrue((Path(ckdir) / "policy.json").is_file(), "missing policy.json manifest")


class ResidualModeGuardTests(unittest.TestCase):
    """Fail-closed entry guard: residual PPO is sync single-policy only. multi-policy/multi-agent/async
    must raise EXPLICITLY, never silently train unprotected standard actions."""
    def test_validate_refuses_unsupported_combinations(self):
        from types import SimpleNamespace
        base = dict(policy_mode="residual", multi_policy=False, multi_agent=False, collector_mode="sync",
                    auto_recovery=False, freeze_base_policy=True)
        # each unsupported combination raises fail-closed (incl. auto-recovery + a non-frozen base)
        for over in [dict(multi_policy=True), dict(multi_agent=True), dict(collector_mode="async"),
                     dict(auto_recovery=True), dict(freeze_base_policy=False)]:
            with self.assertRaises(RuntimeError):
                ppo.validate_residual_mode(SimpleNamespace(**{**base, **over}))
        ppo.validate_residual_mode(SimpleNamespace(**base))                                   # supported: no raise
        ppo.validate_residual_mode(SimpleNamespace(policy_mode="standard", multi_policy=True,  # standard: never guarded
                                                   multi_agent=True, collector_mode="async",
                                                   auto_recovery=True, freeze_base_policy=False))

    def test_entrypoint_raises_before_touching_godot(self):
        # ppo.main() calls validate_residual_mode right after parse_args, so an unsupported combination
        # raises at the ENTRY (before any Godot/env), proving no silent fallback in the real dispatch.
        old = sys.argv
        try:
            for flag in ("--multi-agent", "--multi-policy"):
                sys.argv = ["ppo", "--policy-mode", "residual", flag]
                with self.assertRaises(RuntimeError):
                    ppo.main()
        finally:
            sys.argv = old


class ResumeFromCkpt0Tests(unittest.TestCase):
    """A valid resume from ckpt-0 (start_episode==0) must NOT re-zero a trained residual head. The zero-init
    decision uses an EXPLICIT is_resuming flag, never start_episode."""
    def test_episode0_complete_checkpoint_keeps_nonzero_residual(self):
        self.assertTrue(ppo.residual_should_zero_init(False, None))       # fresh -> zero
        self.assertFalse(ppo.residual_should_zero_init(True, None))       # resume from ckpt-0 -> keep
        m, ls, opt, vopt, tf_gen, gv, uv, ck = _residual_stack()
        mean = m.get_layer("continuous_mean")
        mean.set_weights([np.full_like(w, 0.31) for w in mean.get_weights()])   # trained (non-zero) residual head
        nonzero = [w.copy() for w in mean.get_weights()]
        with tempfile.TemporaryDirectory() as d:
            mgr = tf.train.CheckpointManager(ck, directory=str(Path(d) / "ppo_ckpt"), max_to_keep=2)
            gv.assign(0); uv.assign(0)
            ppo.save_residual_checkpoint(d, mgr, _manifest(), np_rng=np.random.default_rng(0),
                                         counters={"episode": 0, "generation": 0, "policy_updates": 0}, model=m)
            m2, ls2, opt2, vopt2, tf_gen2, gv2, uv2, ck2 = _residual_stack()
            m2.get_layer("continuous_mean").set_weights([np.zeros_like(w) for w in nonzero])   # start zeroed
            ppo.materialize_optimizer_slots(m2, ls2, opt2, vopt2, 6, 3, None)
            counters, _, _ = ppo.load_residual_checkpoint(d, ck2, _manifest(), np.random.default_rng(0), model=m2)
        self.assertEqual(int(counters["episode"]), 0)                         # a genuine episode-0 checkpoint
        restored = m2.get_layer("continuous_mean").get_weights()
        for a, b in zip(restored, nonzero):
            self.assertTrue(np.allclose(a, b))                               # restored the trained head
            self.assertTrue(np.any(a != 0))                                  # and it is NOT zero (would be, if re-zeroed)


if __name__ == "__main__":
    unittest.main()
