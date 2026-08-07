"""Consolidated report over the three counterexample splits (train/selection/final-test): SHA-256,
counts, paired_bad / safety_negative breakdowns, outcomes, collisions, clamps, margins, and the
cross-split validator. Read-only. Fail-closed exit if the split validator rejects."""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.counterexample import FAMILIES, validate_split_files  # noqa: E402


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _by(a, mask, keys):
    """Count rows (under mask) grouped by the tuple of `keys` columns."""
    idx = np.where(mask)[0]
    c = collections.Counter()
    for i in idx:
        c[tuple(str(a[k][i]) for k in keys)] += 1
    return {"|".join(k): v for k, v in sorted(c.items())}


def _split_report(path):
    a = dict(np.load(path, allow_pickle=True))
    meta = json.loads(str(a["meta"][0]))
    src = a["source"]
    cand = src == "candidate"
    burst = src == "ood_burst"
    pbad = a["paired_bad"].astype(bool)
    sneg = a["safety_negative"].astype(bool)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "transitions": int(len(a["obs"])),
        "counts_by_source": dict(collections.Counter(src.tolist())),
        "candidate_outcomes": dict(collections.Counter(a["outcome"][cand].tolist())),
        "collisions": int(a["collision"].sum()),
        "self_collisions": int(a["self_collision"].sum()),
        "godot_clamps": int(meta.get("godot_clamps", -1)),
        "invalid_branches": int(meta.get("invalid_anchor_branches", -1)),
        "fidelity_violations": int(meta.get("fidelity_violations", -1)),
        "global_margin": float(meta.get("global_margin", float("nan"))),
        "margin_by_bucket": meta.get("margin_by_bucket", {}),
        "baseline_outcomes": meta.get("baseline_outcomes", {}),
        "candidate_matrix": meta.get("candidate_matrix", {}),
        "paired_bad_by_rule": meta.get("paired_bad_by_rule", {}),
        "discarded_baseline_failures": int(meta.get("discarded_baseline_failures", -1)),
        "candidate_better": int(meta.get("candidate_better", -1)),
        "paired_bad_total": int(pbad[cand].sum()),
        "paired_bad_by_family": _by(a, cand & pbad, ["family"]),
        "paired_bad_by_cell_phase": _by(a, cand & pbad, ["cell", "phase"]),
        "paired_bad_outside_candidate": int(pbad[~cand].sum()),
        "safety_negative_total": int(sneg[burst].sum()),
        "safety_negative_by_family": _by(a, burst & sneg, ["family"]),
        "safety_negative_by_cell_phase": _by(a, burst & sneg, ["cell", "phase"]),
        "unique_pose_id": len(set(a["pose_id"].tolist())),
        "anchor_states": len(set(zip(a["pose_id"].tolist(), a["phase"].tolist()))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    splits = {"train": args.train, "selection": args.selection, "final_test": args.final}
    validator = validate_split_files(splits)

    report = {"splits": {name: _split_report(p) for name, p in splits.items()},
              "validator": validator}
    report["total_transitions"] = sum(report["splits"][n]["transitions"] for n in splits)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(report, fh, indent=2)

    print("=" * 78)
    print("COUNTEREXAMPLE CONSOLIDATED REPORT")
    print("=" * 78)
    for name in ("train", "selection", "final_test"):
        s = report["splits"][name]
        print(f"\n[{name}] {s['path']}")
        print(f"  sha256={s['sha256']}")
        print(f"  transitions={s['transitions']}  counts_by_source={s['counts_by_source']}")
        print(f"  anchor_states={s['anchor_states']}  unique_pose_id={s['unique_pose_id']}")
        print(f"  candidate_outcomes={s['candidate_outcomes']}")
        print(f"  baseline_outcomes={s['baseline_outcomes']}  discarded_baseline_failures={s['discarded_baseline_failures']}")
        print(f"  candidate_matrix(base->cand)={s['candidate_matrix']}")
        print(f"  paired_bad(candidate)={s['paired_bad_total']}  outside_candidate={s['paired_bad_outside_candidate']}  candidate_better={s['candidate_better']}")
        print(f"    by_rule={s['paired_bad_by_rule']}")
        print(f"    by_family={s['paired_bad_by_family']}")
        print(f"    by_cell_phase={s['paired_bad_by_cell_phase']}")
        print(f"  safety_negative(burst)={s['safety_negative_total']}")
        print(f"    by_family={s['safety_negative_by_family']}")
        print(f"  collisions={s['collisions']} self_collisions={s['self_collisions']} "
              f"godot_clamps={s['godot_clamps']} invalid_branches={s['invalid_branches']} "
              f"fidelity_violations={s['fidelity_violations']}")
        print(f"  global_margin={s['global_margin']:.4f}  buckets_with_margin={len(s['margin_by_bucket'])}")
    print(f"\nTOTAL transitions={report['total_transitions']}")
    print(f"VALIDATOR: {json.dumps({k: (v if k=='disjoint' else v) for k, v in validator.items() if k=='disjoint'})}")
    for name in ("train", "selection", "final_test"):
        v = validator[name]
        print(f"  {name}: rows={v['rows']} unique_pose_id={v['unique_pose_id']} "
              f"unique_seed={v['unique_seed']} anchor_states={v['anchor_states']}")
    print(f"\nWrote {args.out_json}")
    if not validator.get("disjoint"):
        raise SystemExit("VALIDATOR FAILED: splits not disjoint")


if __name__ == "__main__":
    main()
