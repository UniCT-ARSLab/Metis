"""Opt-in evaluation-driven learning rates and simulator certification."""

import argparse
import json
import math
from pathlib import Path


def add_control_arguments(parser):
    parser.add_argument("--stop-when-solved", type=float, default=None, help="Success threshold for solved qualification and independent simulator certification; disabled by default. Must be strictly between 0 and 1.")
    parser.add_argument("--solved-patience", type=int, default=3, help="Consecutive qualifying frozen evaluations before certifying a pinned candidate.")
    parser.add_argument("--solved-certification-episodes", type=int, default=300, help="Fresh single-agent episodes per certification attempt.")
    parser.add_argument("--solved-certification-seed", type=int, default=1000000000, help="First certification seed; every attempt reserves a new disjoint episode deck.")
    parser.add_argument("--solved-certification-confidence", type=float, default=.95, help="Run-scoped confidence budget for exact one-sided binomial success bounds; attempts and observed regions share the error budget.")
    parser.add_argument("--lr-decay-on-plateau", type=float, default=None, help="Multiply optimizer learning rates after a frozen-evaluation plateau; disabled by default.")
    parser.add_argument("--lr-plateau-patience", type=int, default=6, help="Evaluations without improvement before a plateau LR reduction; counter resets after a reduction.")
    parser.add_argument("--lr-decay-on-best", type=float, default=None, help="Opt-in alternative: multiply rates on every new best, instead of plateau decay.")
    parser.add_argument("--lr-min-scale", type=float, default=.01, help="Minimum scheduler LR relative to each optimizer's first controlled rate.")
    parser.add_argument("--recovery-lr-cumulative", action=argparse.BooleanOptionalAction, default=False, help="Accumulate recovery LR reductions without raising rates when restoring an older checkpoint. Off by default.")


def _counts(summary):
    successes, trials = summary.get("successes"), summary.get("trials")
    if isinstance(successes, bool) or isinstance(trials, bool):
        raise ValueError("success/trial counts must be integers")
    if not isinstance(successes, int) or not isinstance(trials, int) or not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid success/trial counts")
    rate = float(summary.get("success_rate", successes / trials))
    if not math.isfinite(rate) or abs(rate - successes / trials) > 1e-6:
        raise ValueError("inconsistent success rate")
    return successes, trials


def _binomial_upper_tail(successes, trials, probability):
    if probability <= 0:
        return 1.0 if successes == 0 else 0.0
    if probability >= 1:
        return 1.0
    logs = [math.lgamma(trials + 1) - math.lgamma(k + 1) - math.lgamma(trials - k + 1) + k * math.log(probability) + (trials - k) * math.log1p(-probability) for k in range(successes, trials + 1)]
    largest = max(logs)
    return min(1.0, math.exp(largest) * math.fsum(math.exp(value - largest) for value in logs))


def success_lower_bound(successes, trials, alpha):
    """Exact one-sided Clopper-Pearson bound for independent Bernoulli episodes."""
    if not 0 <= successes <= trials or trials <= 0 or not 0 < alpha < 1:
        raise ValueError("invalid binomial confidence parameters")
    if successes == 0:
        return 0.0
    if successes == trials:
        return math.exp(math.log(alpha) / trials)
    low, high = 0.0, 1.0
    for _ in range(60):
        midpoint = (low + high) / 2
        if _binomial_upper_tail(successes, trials, midpoint) < alpha:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


class TrainingControl:
    def __init__(self, args, algorithm):
        self.args = args
        self.algorithm = algorithm
        self.threshold = getattr(args, "stop_when_solved", None)
        self.plateau_factor = getattr(args, "lr_decay_on_plateau", None)
        self.best_factor = getattr(args, "lr_decay_on_best", None)
        self.cumulative_recovery = bool(getattr(args, "recovery_lr_cumulative", False))
        self.scheduled = self.plateau_factor is not None or self.best_factor is not None
        self.enabled = self.threshold is not None or self.scheduled or self.cumulative_recovery
        self.stop_requested = False
        self.optimizers = {}
        self.state_path = Path(args.checkpoint_dir) / "training_control.json"
        self.events_path = self.state_path.with_suffix(".jsonl")
        self.state = {"version": 1, "algorithm": algorithm, "streak": 0, "plateau": 0, "attempts": 0, "last_episode": None, "context": None, "rates": {}, "certified": None}
        self._validate()
        if not self.enabled:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        contract = self._contract()
        if self.state_path.exists():
            saved = json.loads(self.state_path.read_text())
            if saved.get("contract") != contract:
                raise ValueError("Training-control configuration changed; use a new checkpoint directory for a different certification or LR contract")
            self.state.update(saved)
            # A resumed learner must qualify again; spent decks and LR ceilings never rewind.
            self.state.update(streak=0, plateau=0, last_episode=None, context=None)
            self.stop_requested = bool(self.state.get("certified"))
        self.state["contract"] = contract
        self._save()

    def _validate(self):
        if self.enabled and not getattr(self.args, "best_checkpoint", False):
            raise ValueError("Training controls require --best-checkpoint")
        if self.threshold is not None:
            if not math.isfinite(self.threshold) or not 0 < self.threshold < 1:
                raise ValueError("--stop-when-solved must be strictly between 0 and 1; finite evaluations cannot certify probability 1")
            if getattr(self.args, "multi_agent", False) or getattr(self.args, "multi_policy", False):
                raise ValueError("Solved certification currently requires single-agent, single-policy episodes")
            for name, default in (("solved_patience", 3), ("solved_certification_episodes", 300)):
                if int(getattr(self.args, name, default)) < 1:
                    raise ValueError(f"--{name.replace('_', '-')} must be positive")
            confidence = float(getattr(self.args, "solved_certification_confidence", .95))
            if not math.isfinite(confidence) or not 0 < confidence < 1:
                raise ValueError("--solved-certification-confidence must be in (0, 1)")
            seed = int(getattr(self.args, "solved_certification_seed", 1000000000))
            count = int(getattr(self.args, "solved_certification_episodes", 300))
            self._validate_deck(seed, count)
        if self.plateau_factor is not None and self.best_factor is not None:
            raise ValueError("Choose --lr-decay-on-plateau or --lr-decay-on-best, not both")
        for value in (self.plateau_factor, self.best_factor):
            if value is not None and (not math.isfinite(value) or not 0 < value < 1):
                raise ValueError("LR decay factors must be strictly between 0 and 1")
        if int(getattr(self.args, "lr_plateau_patience", 6)) < 1:
            raise ValueError("--lr-plateau-patience must be positive")
        minimum = float(getattr(self.args, "lr_min_scale", .01))
        if not math.isfinite(minimum) or not 0 < minimum <= 1:
            raise ValueError("--lr-min-scale must be in (0, 1]")

    def _validate_deck(self, seed, count):
        selection_seed = int(getattr(self.args, "best_evaluation_seed", 10000))
        selection_count = int(getattr(self.args, "best_evaluation_episodes", 20))
        if seed < 0 or seed + count > 2**32 or (seed < selection_seed + selection_count and selection_seed < seed + count):
            raise ValueError("Certification seeds must fit uint32 and be disjoint from the selection deck")
        training_end = int(getattr(self.args, "episode_seed_multiplier", 1000)) * (int(getattr(self.args, "num_episodes", 0)) + int(getattr(self.args, "num_envs", 1))) + int(getattr(self.args, "env_seed_base", 100))
        if seed <= training_end:
            raise ValueError("Choose certification seeds beyond the declared training reset-seed range")

    def _contract(self):
        names = ("stop_when_solved", "solved_patience", "solved_certification_episodes", "solved_certification_seed", "solved_certification_confidence", "lr_decay_on_plateau", "lr_plateau_patience", "lr_decay_on_best", "lr_min_scale", "recovery_lr_cumulative", "best_evaluation_seed", "best_evaluation_episodes", "best_evaluation_training_episode", "best_evaluation_max_steps", "best_metric", "godot_project", "godot_scene", "physics_frames_per_step", "actor_ema_decay", "network_layers", "agent_id", "policy_mode")
        fixed_episode = getattr(self.args, "best_evaluation_training_episode", None)
        if fixed_episode is None and (self.threshold is not None or self.scheduled):
            fixed_episode = int(getattr(self.args, "num_episodes", 0))
        return {"algorithm": self.algorithm, "fixed_evaluation_episode": fixed_episode, **{name: getattr(self.args, name, None) for name in names}}

    def _save(self, event=None):
        if not self.enabled:
            return
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.state_path)
        if event:
            with self.events_path.open("a") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")

    @staticmethod
    def read_rate(optimizer):
        value = optimizer.learning_rate
        return float(value.numpy() if hasattr(value, "numpy") else value)

    def bind_optimizers(self, optimizers):
        if not self.enabled:
            return
        for name, optimizer in optimizers:
            if optimizer is None:
                continue
            name = str(name)
            current = self.read_rate(optimizer)
            saved = self.state["rates"].setdefault(name, {"base": current, "ceiling": current})
            optimizer.learning_rate.assign(min(current, saved["ceiling"]))
            self.optimizers[name] = optimizer
        self.remember_rates()

    def remember_rates(self):
        for name, optimizer in self.optimizers.items():
            self.state["rates"][name]["ceiling"] = min(self.state["rates"][name]["ceiling"], self.read_rate(optimizer))
        self._save()

    def recovery_rate(self, name, current, factor, minimum_scale, original):
        saved = self.state["rates"].get(name, {"base": original, "ceiling": current})
        if self.cumulative_recovery:
            ceiling = min(current, saved["ceiling"])
            return min(ceiling, max(saved["base"] * minimum_scale, ceiling * factor))
        target = (saved["base"] if self.scheduled else original) * max(minimum_scale, factor)
        return min(target, saved["ceiling"]) if self.scheduled else target

    def _decay(self, factor, episode, reason):
        changed = {}
        for name, optimizer in self.optimizers.items():
            current = self.read_rate(optimizer)
            floor = self.state["rates"][name]["base"] * float(getattr(self.args, "lr_min_scale", .01))
            target = min(current, max(floor, current * factor))
            optimizer.learning_rate.assign(target)
            changed[name] = target
        self.remember_rates()
        self._save({"event": "lr_reduced", "episode": episode, "reason": reason, "learning_rates": changed})
        print(f"Training control: {reason} learning_rates={changed}", flush=True)

    def reset_quality(self):
        self.state.update(streak=0, plateau=0, last_episode=None, context=None)
        self._save()

    def evaluation_failed(self):
        self.state["streak"] = 0
        self._save()

    def observe(self, result, improved, eligible=True):
        if not self.enabled or self.stop_requested:
            return None
        episode = int(result.episode)
        previous = self.state["last_episode"]
        if previous is not None and episode <= previous:
            return None
        context = result.summary.get("curriculum_level")
        if self.state["context"] != context or not eligible:
            self.state.update(streak=0, plateau=0)
        self.state.update(last_episode=episode, context=context)
        if not eligible:
            self._save()
            return None
        if improved:
            self.state["plateau"] = 0
            if self.best_factor is not None:
                self._decay(self.best_factor, episode, "new best")
        else:
            self.state["plateau"] += 1
            if self.plateau_factor is not None and self.state["plateau"] >= int(getattr(self.args, "lr_plateau_patience", 6)):
                self._decay(self.plateau_factor, episode, "plateau")
                self.state["plateau"] = 0
        if self.threshold is None:
            self._save()
            return None
        rate = float(result.summary.get("selection_success_rate", result.summary.get("success_rate", float("nan"))))
        self.state["streak"] = self.state["streak"] + 1 if math.isfinite(rate) and self.threshold <= rate <= 1 else 0
        if self.state["streak"] < int(getattr(self.args, "solved_patience", 3)):
            self._save()
            return None
        if self.state.get("certification_exhausted", False):
            self.state["streak"] = 0
            self._save()
            return None
        attempt = self.state["attempts"] + 1
        count = int(getattr(self.args, "solved_certification_episodes", 300))
        seed = self.state.get("next_seed", int(getattr(self.args, "solved_certification_seed", 1000000000)) + (attempt - 1) * count)
        selection_seed = int(getattr(self.args, "best_evaluation_seed", 10000))
        selection_end = selection_seed + int(getattr(self.args, "best_evaluation_episodes", 20))
        if seed < selection_end and selection_seed < seed + count:
            seed = selection_end
        try:
            self._validate_deck(seed, count)
        except ValueError as error:
            self.state.update(certification_exhausted=True, streak=0)
            self._save({"event": "certification_unavailable", "reason": str(error)})
            print(f"Certification unavailable: {error}; continuing training without declaring solved", flush=True)
            return None
        # Reserve before launching a child: interruptions cannot cause a held-out deck to be reused.
        self.state.update(attempts=attempt, streak=0, next_seed=seed + count)
        request = {"attempt": attempt, "episodes": count, "seed": seed, "candidate_episode": episode, "regions": sorted(result.summary.get("target_regions", {})), "alpha": (1 - float(getattr(self.args, "solved_certification_confidence", .95))) / (attempt * (attempt + 1))}
        self._save({"event": "certification_started", **request})
        return request

    def assess(self, summary, request):
        report = {**request, "passed": False, "threshold": self.threshold, "confidence": float(getattr(self.args, "solved_certification_confidence", .95)), "scope": "Independent single-agent simulator episodes; not physical-device safety certification", "checks": {}}
        try:
            if not isinstance(summary, dict):
                raise ValueError("certification evaluator failed or timed out")
            successes, trials = _counts(summary)
            if trials != request["episodes"] or summary.get("episodes") != request["episodes"]:
                raise ValueError("certification did not complete the requested independent episodes")
            modes = summary.get("reset_modes", {})
            if modes and ("regular" not in modes or _counts(modes["regular"])[1] != trials):
                raise ValueError("certification includes seeded/non-regular resets")
            groups = {"overall": summary}
            regions = summary.get("target_regions", {})
            if not set(request["regions"]).issubset(regions):
                raise ValueError("certification is missing a required target region")
            groups.update({"region:" + name: values for name, values in regions.items()})
            per_check_alpha = request["alpha"] / len(groups)
            for name, values in groups.items():
                k, n = _counts(values)
                bound = success_lower_bound(k, n, per_check_alpha)
                report["checks"][name] = {"successes": k, "trials": n, "lower_bound": bound, "alpha": per_check_alpha, "passed": bound >= self.threshold}
            selection = float(summary.get("selection_success_rate", successes / trials))
            report["passed"] = math.isfinite(selection) and selection >= self.threshold and all(check["passed"] for check in report["checks"].values())
            report["reason"] = "all certification bounds passed" if report["passed"] else "insufficient success margin or confidence; continue training"
        except (ValueError, TypeError, KeyError) as error:
            report["reason"] = str(error)
        return report

    def record_certification(self, report, policy_path=None):
        report = {**report, "policy": None if policy_path is None else str(policy_path)}
        if report["passed"]:
            self.stop_requested = True
            self.state["certified"] = report
        self._save({"event": "certification_passed" if report["passed"] else "certification_failed", **report})
        print(f"Simulator certification attempt={report['attempt']} passed={report['passed']}: {report['reason']}", flush=True)
