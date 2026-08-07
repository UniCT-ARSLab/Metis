"""Stratified offline minibatch sampler for the M6.1 critic-only experiment.

Mixes five sources with fixed quotas and structured balancing (M6.1 spec):
  50%   M5 v2                         (the ONLY positive BC/demo transitions)
  25%   candidate paired_bad=True     balanced by (cell, phase, family)
  10%   candidate non-bad / better    balanced by (cell, phase, family)
  7.5%  ood_burst safety_negative     branch-balanced: pick a branch, then a transition of it
                                       (never sample the 885 rows as independent events)
  7.5%  baseline / recovery / non-negative burst

HARD guard: only M5 v2 + the counterexample TRAIN split may enter the replay. selection / final_test
must NEVER be loaded here (they are audit-only). dones = terminated (Metis): truncated bootstraps.
"""
import hashlib

import numpy as np

DEFAULT_QUOTAS = {
    "m5v2": 0.50,
    "cand_bad": 0.25,
    "cand_nonbad": 0.10,
    "safety": 0.075,
    "neutral": 0.075,
}


def _npz_pose_ids(path):
    import numpy as _np
    with _np.load(path, allow_pickle=True) as z:
        return set(map(str, z["pose_id"].tolist())) if "pose_id" in z.files else set()


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CounterexampleSampler:
    def __init__(self, m5v2_path, counterexample_train_path, *, eval_paths=(),
                 quotas=None, seed=0, expected_train_sha256=None):
        # ---- guard: the counterexample replay must NOT be an eval split ----
        replay_poses = _npz_pose_ids(counterexample_train_path)
        for ep in eval_paths:
            if replay_poses & _npz_pose_ids(ep):
                raise ValueError(
                    f"replay {counterexample_train_path} shares pose_id with eval split {ep}: "
                    "selection/final_test must NEVER enter the replay")
        if expected_train_sha256 and _file_sha256(counterexample_train_path) != expected_train_sha256:
            raise ValueError("counterexample train checksum mismatch (dataset is FROZEN)")

        self.rng = np.random.default_rng(seed)
        self._eval_paths = tuple(eval_paths)

        m5 = np.load(m5v2_path, allow_pickle=True)
        self._m5 = dict(obs=np.asarray(m5["obs"], np.float32),
                        actions=np.asarray(m5["actions"], np.float32),
                        rewards=np.asarray(m5["rewards"], np.float32),
                        next_obs=np.asarray(m5["next_obs"], np.float32),
                        dones=np.asarray(m5["terminated"], np.float32))
        ce = np.load(counterexample_train_path, allow_pickle=True)
        self._ce = dict(obs=np.asarray(ce["obs"], np.float32),
                        actions=np.asarray(ce["actions"], np.float32),
                        rewards=np.asarray(ce["rewards"], np.float32),
                        next_obs=np.asarray(ce["next_obs"], np.float32),
                        dones=np.asarray(ce["dones"], np.float32))  # dones == terminated already
        src = ce["source"]
        pbad = ce["paired_bad"].astype(bool)
        sneg = ce["safety_negative"].astype(bool)
        cand = src == "candidate"
        burst = src == "ood_burst"
        base_m = src == "baseline"
        recov_m = src == "recovery"
        nonneg_burst_m = burst & ~sneg
        # category index arrays (into the counterexample table)
        self._idx = {
            "cand_bad": np.where(cand & pbad)[0],
            "cand_nonbad": np.where(cand & ~pbad)[0],
            "safety": np.where(burst & sneg)[0],
            # neutral = union of three sub-sources, each balanced separately (NOT flat-sampled).
            "neutral": np.where(base_m | recov_m | nonneg_burst_m)[0],
        }
        cell = ce["cell"].astype(np.int64)
        phase = np.asarray(ce["phase"])
        fam = np.asarray(ce["family"])
        bid = ce["branch_id"].astype(np.int64)
        cellphfam = np.array([f"{c}|{p}|{f}" for c, p, f in zip(cell.tolist(), phase.tolist(), fam.tolist())])
        cellph = np.array([f"{c}|{p}" for c, p in zip(cell.tolist(), phase.tolist())])
        self._groups_bad = self._group(self._idx["cand_bad"], cellphfam)
        self._groups_nonbad = self._group(self._idx["cand_nonbad"], cellphfam)
        # safety: branch-balanced (pick a branch_id, then a transition of that branch)
        self._safety_branches = self._group(np.where(burst & sneg)[0], bid)
        self._safety_branch_keys = list(self._safety_branches)
        # neutral sub-sources: baseline balanced by (cell,phase); recovery + nonneg-burst branch-first
        self._neutral = {
            "baseline": ("grouped", self._group(np.where(base_m)[0], cellph)),
            "recovery": ("branch", self._group(np.where(recov_m)[0], bid)),
            "nonneg_burst": ("branch", self._group(np.where(nonneg_burst_m)[0], bid)),
        }

        self.quotas = dict(quotas or DEFAULT_QUOTAS)
        s = sum(self.quotas.values())
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"quotas must sum to 1.0 (got {s})")

    @staticmethod
    def _group(indices, keys):
        groups = {}
        for i in indices:
            groups.setdefault(keys[i], []).append(int(i))
        return {k: np.asarray(v, np.int64) for k, v in groups.items()}

    def _counts(self, batch_size):
        # deterministic rounding that sums exactly to batch_size (largest remainder)
        raw = {k: self.quotas[k] * batch_size for k in self.quotas}
        base = {k: int(np.floor(v)) for k, v in raw.items()}
        rem = batch_size - sum(base.values())
        order = sorted(self.quotas, key=lambda k: raw[k] - base[k], reverse=True)
        for k in order[:rem]:
            base[k] += 1
        return base

    def _sample_grouped(self, groups, n):
        if n <= 0 or not groups:
            return np.empty(0, np.int64)
        keys = list(groups)
        out = np.empty(n, np.int64)
        for j in range(n):
            g = keys[self.rng.integers(len(keys))]           # uniform over (cell,phase,family)
            pool = groups[g]
            out[j] = pool[self.rng.integers(len(pool))]      # then uniform within the group
        return out

    def _sample_safety(self, n):
        if n <= 0 or not self._safety_branch_keys:
            return np.empty(0, np.int64)
        out = np.empty(n, np.int64)
        for j in range(n):
            b = self._safety_branch_keys[self.rng.integers(len(self._safety_branch_keys))]  # branch first
            pool = self._safety_branches[b]
            out[j] = pool[self.rng.integers(len(pool))]      # then a transition of that branch
        return out

    def _sample_branch_first(self, branches, n):
        keys = list(branches)
        if n <= 0 or not keys:
            return np.empty(0, np.int64)
        out = np.empty(n, np.int64)
        for j in range(n):
            b = keys[self.rng.integers(len(keys))]
            pool = branches[b]
            out[j] = pool[self.rng.integers(len(pool))]
        return out

    def _sample_neutral(self, n):
        """Pick a sub-source uniformly (baseline / recovery / nonneg_burst), then balance within:
        baseline by (cell,phase), recovery + nonneg_burst branch-first. Prevents the ~132 baseline
        rows from being drowned by the ~18k recovery/burst rows."""
        avail = [k for k, (_, g) in self._neutral.items() if g]
        if n <= 0 or not avail:
            return np.empty(0, np.int64)
        out = np.empty(n, np.int64)
        for j in range(n):
            k = avail[self.rng.integers(len(avail))]
            kind, groups = self._neutral[k]
            if kind == "grouped":
                out[j] = self._sample_grouped(groups, 1)[0]
            else:
                out[j] = self._sample_branch_first(groups, 1)[0]
        return out

    def sample(self, batch_size):
        c = self._counts(batch_size)
        m5_i = self.rng.integers(len(self._m5["obs"]), size=c["m5v2"])
        ce_parts = {
            "cand_bad": self._sample_grouped(self._groups_bad, c["cand_bad"]),
            "cand_nonbad": self._sample_grouped(self._groups_nonbad, c["cand_nonbad"]),
            "safety": self._sample_safety(c["safety"]),
            "neutral": self._sample_neutral(c["neutral"]),
        }
        obs, act, rew, nobs, done, comp = [], [], [], [], [], {}
        if len(m5_i):
            obs.append(self._m5["obs"][m5_i]); act.append(self._m5["actions"][m5_i])
            rew.append(self._m5["rewards"][m5_i]); nobs.append(self._m5["next_obs"][m5_i])
            done.append(self._m5["dones"][m5_i])
        comp["m5v2"] = int(len(m5_i))
        for name, idx in ce_parts.items():
            comp[name] = int(len(idx))
            if len(idx):
                obs.append(self._ce["obs"][idx]); act.append(self._ce["actions"][idx])
                rew.append(self._ce["rewards"][idx]); nobs.append(self._ce["next_obs"][idx])
                done.append(self._ce["dones"][idx])
        obs = np.concatenate(obs); act = np.concatenate(act); rew = np.concatenate(rew)
        nobs = np.concatenate(nobs); done = np.concatenate(done)
        perm = self.rng.permutation(len(obs))          # shuffle the concatenated quotas
        batch = dict(obs=obs[perm], actions=act[perm], rewards=rew[perm],
                     next_obs=nobs[perm], dones=done[perm])
        return batch, comp

    def category_sizes(self):
        return {"m5v2": int(len(self._m5["obs"]))} | {k: int(len(v)) for k, v in self._idx.items()}
