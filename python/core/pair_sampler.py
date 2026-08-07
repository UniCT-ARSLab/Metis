"""Pairwise supervision sampler for the M6.1 critic-only fine-tuning (v3/v4/v5).

Builds causal comparison pairs EXCLUSIVELY from the counterexample TRAIN split. For every candidate
in the chosen population we pair it with the BASELINE action taken at the same anchor (same pose_id +
phase, same reset s0):

    delta_Q(s) = Q(s, a_candidate) - Q(s, a_baseline)     (evaluated at the candidate's obs s)

Two populations:
- "significant" (v3/v4): source==candidate & label_usable & |return_delta_episode| >= margin. Sampled
  50/50 by return_delta sign, then balanced by (cell, phase, family) within each sign.
- "paired_bad" (v5 SAFETY critic): additionally requires paired_bad==True. These are the empirically
  WORSE / OOD actions (all return_delta <= -margin). NO sign balancing (all one sign); instead a
  hard-negative draw balanced by (cell, phase, family), with a guaranteed `attack_fraction` reserved
  for the pga + collapsed_v4 attack families (balanced 50/50 between them) so the rare pga family is
  never drowned. candidate_better (positive) pairs are DELIBERATELY excluded here -- they stay in the
  TD replay and are reported only as telemetry, never turned into negatives.

selection / final_test are AUDIT-ONLY and must never be loaded here; a hard pose_id-overlap guard
enforces it. The TRAIN split is FROZEN (optional checksum).
"""
import hashlib

import numpy as np

ATTACK_FAMILIES = ("pga", "collapsed_v4")


def _npz_pose_ids(path):
    with np.load(path, allow_pickle=True) as z:
        return set(map(str, z["pose_id"].tolist())) if "pose_id" in z.files else set()


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class PairSampler:
    def __init__(self, counterexample_train_path, *, eval_paths=(), seed=0,
                 expected_train_sha256=None, margin=None, population="significant",
                 attack_fraction=0.5):
        # ---- hard guard: the pair source is the TRAIN replay; it must never overlap an eval split.
        train_poses = _npz_pose_ids(counterexample_train_path)
        for ep in eval_paths:
            if train_poses & _npz_pose_ids(ep):
                raise ValueError(
                    f"pair source {counterexample_train_path} shares pose_id with eval split {ep}: "
                    "selection/final_test must NEVER enter the PairSampler")
        if expected_train_sha256 and _file_sha256(counterexample_train_path) != expected_train_sha256:
            raise ValueError("counterexample train checksum mismatch (dataset is FROZEN)")
        if population not in ("significant", "paired_bad"):
            raise ValueError(f"unknown population {population!r}")

        self.rng = np.random.default_rng(seed)
        self.population = population
        self.attack_fraction = float(attack_fraction)
        z = np.load(counterexample_train_path, allow_pickle=True)
        src = z["source"]
        usable = z["label_usable"].astype(bool)
        ret = z["return_delta_episode"].astype(np.float64)
        pose = z["pose_id"]
        phase = np.asarray(z["phase"])
        cell = z["cell"].astype(np.int64)
        fam = np.asarray(z["family"])
        obs = np.asarray(z["obs"], np.float32)
        act = np.asarray(z["actions"], np.float32)
        pbad = z["paired_bad"].astype(bool) if "paired_bad" in z.files else np.zeros(len(src), bool)

        self.margin = float(z["margin"][0]) if margin is None else float(margin)

        base_m = src == "baseline"
        base_action = {}
        for i in np.where(base_m)[0]:
            base_action[(str(pose[i]), str(phase[i]))] = act[i]

        cand = src == "candidate"
        keep = cand & usable & (np.abs(ret) >= self.margin)
        if population == "paired_bad":
            keep = keep & pbad
        ci = np.where(keep)[0]

        p_obs, p_cand, p_base, p_ret, p_key, p_pos = [], [], [], [], [], []
        p_fam, p_cellph = [], []
        n_no_anchor = 0
        for i in ci:
            k = (str(pose[i]), str(phase[i]))
            if k not in base_action:                 # defensive: skip candidates without an anchor
                n_no_anchor += 1
                continue
            p_obs.append(obs[i]); p_cand.append(act[i]); p_base.append(base_action[k])
            p_ret.append(ret[i]); p_fam.append(str(fam[i]))
            p_key.append(f"{int(cell[i])}|{str(phase[i])}|{str(fam[i])}")
            p_cellph.append(f"{int(cell[i])}|{str(phase[i])}")
            p_pos.append(ret[i] > 0.0)

        self._obs = np.asarray(p_obs, np.float32)
        self._act_cand = np.asarray(p_cand, np.float32)
        self._act_base = np.asarray(p_base, np.float32)
        self._ret = np.asarray(p_ret, np.float64)
        self._family = np.asarray(p_fam)
        self._cellphase = np.asarray(p_cellph)
        keys = np.asarray(p_key)
        pos = np.asarray(p_pos, bool)
        fam_p = self._family
        self.n_no_anchor = int(n_no_anchor)
        if len(self._obs) == 0:
            raise ValueError("PairSampler: no pairs in the chosen population")

        if population == "significant":
            self._pos_groups = self._group(np.where(pos)[0], keys)
            self._neg_groups = self._group(np.where(~pos)[0], keys)
            if not self._pos_groups or not self._neg_groups:
                raise ValueError(
                    f"'significant' needs both signs (pos={len(self._pos_groups)}, "
                    f"neg={len(self._neg_groups)}); cannot balance 50/50")
        else:  # paired_bad: attack families (pga/collapsed) vs the rest, each balanced by key.
            is_attack = np.isin(fam_p, ATTACK_FAMILIES)
            self._attack_by_family = {
                f: self._group(np.where(fam_p == f)[0], keys)
                for f in ATTACK_FAMILIES if (fam_p == f).any()}
            self._other_groups = self._group(np.where(~is_attack)[0], keys)
            if not self._attack_by_family and not self._other_groups:
                raise ValueError("paired_bad population empty after grouping")

    @staticmethod
    def _group(indices, keys):
        groups = {}
        for i in indices:
            groups.setdefault(keys[i], []).append(int(i))
        return {k: np.asarray(v, np.int64) for k, v in groups.items()}

    def all_pairs(self):
        """The FULL deduplicated pair set (no sampling), for stable train-side sign telemetry.
        Returns obs, act_cand, act_base, return_delta, family, cellphase."""
        return {"obs": self._obs, "act_cand": self._act_cand, "act_base": self._act_base,
                "return_delta": self._ret, "family": self._family, "cellphase": self._cellphase}

    def n_pairs(self):
        return int(len(self._obs))

    def n_pos(self):
        return int(np.sum(self._ret > 0))

    def n_neg(self):
        return int(np.sum(self._ret < 0))

    def _draw(self, groups, n):
        keys = list(groups)
        out = np.empty(n, np.int64)
        for j in range(n):
            g = keys[self.rng.integers(len(keys))]           # uniform over (cell,phase,family)
            pool = groups[g]
            out[j] = pool[self.rng.integers(len(pool))]      # then uniform within the group
        return out

    def _sample_significant(self, batch_size):
        n_pos = batch_size // 2                               # exact 50/50 (neg takes the odd slot)
        n_neg = batch_size - n_pos
        idx = np.concatenate([self._draw(self._pos_groups, n_pos),
                              self._draw(self._neg_groups, n_neg)])
        return idx, {"n_pos": int(n_pos), "n_neg": int(n_neg), "batch": int(batch_size)}

    def _sample_paired_bad(self, batch_size):
        fams = list(self._attack_by_family)
        n_attack = int(round(self.attack_fraction * batch_size)) if fams else 0
        n_other = batch_size - n_attack
        if not self._other_groups:                            # no non-attack families -> all attack
            n_attack, n_other = batch_size, 0
        parts, comp = [], {"batch": int(batch_size), "n_attack": int(n_attack), "n_other": int(n_other)}
        if n_attack:
            # split the attack quota EQUALLY across present attack families (pga, collapsed_v4)
            per = [n_attack // len(fams)] * len(fams)
            for j in range(n_attack - sum(per)):
                per[j] += 1
            for f, npf in zip(fams, per):
                parts.append(self._draw(self._attack_by_family[f], npf))
                comp[f"n_{f}"] = int(npf)
        if n_other:
            parts.append(self._draw(self._other_groups, n_other))
        idx = np.concatenate(parts) if parts else np.empty(0, np.int64)
        return idx, comp

    def sample(self, batch_size):
        if self.population == "significant":
            idx, comp = self._sample_significant(batch_size)
        else:
            idx, comp = self._sample_paired_bad(batch_size)
        idx = idx[self.rng.permutation(len(idx))]
        batch = dict(obs=self._obs[idx], act_cand=self._act_cand[idx],
                     act_base=self._act_base[idx], return_delta=self._ret[idx])
        return batch, comp
