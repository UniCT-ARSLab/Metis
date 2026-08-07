"""Generic demonstration-dataset invariants. Task-agnostic: no cell/region/phase concept.

A feed-forward policy maps observation -> action. A dataset that pairs the SAME observation with
DIFFERENT actions is unlearnable, and a leading no-op whose next_obs equals obs poisons imitation.
These checks catch that class of bug (e.g. a plan-follower that advances a hidden waypoint index
AFTER emitting a zero action toward the already-reached home waypoint).
"""
import numpy as np


def check_demo_invariants(obs, actions, episode_indices=None, step_indices=None,
                          next_obs=None, applied_actions=None,
                          obs_tol=1e-5, action_tol=1e-3):
    """Return a dict of invariant -> list of offending indices (empty lists = all pass).

    Invariants:
      consecutive_conflict : obs_t == obs_{t+1} (<=obs_tol) yet action_t != action_{t+1} (>action_tol)
                             within one episode -- the identical-state / different-action contradiction.
      duplicate_obs_conflict: any two transitions whose obs match (<=obs_tol) but actions differ
                             (>action_tol) -- the same feed-forward contradiction, globally.
      first_action_zero    : an episode's first action is ~all-zero (a no-op start on a task that
                             begins far from the target).
      recorded_ne_applied  : recorded action != the applied action (when applied_actions given).
    """
    obs = np.asarray(obs, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    n = len(actions)
    ep = (np.asarray(episode_indices) if episode_indices is not None
          else np.zeros(n, dtype=np.int64))
    out = {"consecutive_conflict": [], "duplicate_obs_conflict": [],
           "first_action_zero": [], "recorded_ne_applied": []}

    # consecutive within-episode conflict
    for t in range(n - 1):
        if ep[t] != ep[t + 1]:
            continue
        if np.max(np.abs(obs[t] - obs[t + 1])) <= obs_tol and \
                np.linalg.norm(actions[t] - actions[t + 1]) > action_tol:
            out["consecutive_conflict"].append(t)

    # first action of each episode ~zero
    order = np.argsort(ep, kind="stable")
    seen = set()
    for idx in order:
        e = int(ep[idx])
        if e in seen:
            continue
        seen.add(e)
        if np.linalg.norm(actions[idx]) <= action_tol:
            out["first_action_zero"].append(int(idx))

    # global duplicate-obs conflict via rounding key (bounded work)
    keys = {}
    for t in range(n):
        k = tuple(np.round(obs[t] / max(obs_tol, 1e-9)).astype(np.int64))
        if k in keys:
            t0 = keys[k]
            if np.linalg.norm(actions[t] - actions[t0]) > action_tol:
                out["duplicate_obs_conflict"].append((t0, t))
        else:
            keys[k] = t

    if applied_actions is not None:
        applied = np.asarray(applied_actions, dtype=np.float64)
        for t in range(min(n, len(applied))):
            if np.linalg.norm(actions[t] - applied[t]) > action_tol:
                out["recorded_ne_applied"].append(t)

    return out


def summarize(violations):
    return {k: len(v) for k, v in violations.items()}


def is_clean(violations):
    return all(len(v) == 0 for v in violations.values())
