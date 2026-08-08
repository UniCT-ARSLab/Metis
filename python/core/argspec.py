"""Dump a training algorithm's full argparse specification as JSON.

Used by the Godot editor add-on to auto-generate the "All options" section of the Train wizard, so
every flag an algorithm accepts is exposed and stays in sync with the CLI. Each algorithm module
builds its parser inside ``parse_args()`` (which then calls ``parser.parse_args()``); we monkeypatch
``parse_args`` to capture the fully-built parser WITHOUT actually parsing or training. The CUDA
re-exec that the training entry points perform on import is suppressed via the env guard below so
this stays a quick, side-effect-free introspection.
"""

from __future__ import annotations

import os

# Must be set before importing any algorithm module: it skips ensure_nvidia_pip_libs_on_path()'s
# os.execv re-exec, which would otherwise restart this process.
os.environ.setdefault("GODOT_GYM_TF_LD_READY", "1")

import argparse
import importlib
import json
import sys

# Make the training packages importable no matter how this file is invoked (python -m core.argspec
# from python/, or python python/core/argspec.py from the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# algorithm -> module that exposes parse_args() (mirrors train.py BACKENDS).
ALGORITHM_MODULES = {
	"sac": "algorithms.sac",
	"ppo": "algorithms.ppo",
	"dqn": "algorithms.dqn",
	"ddpg": "algorithms.ddpg",
	"td3": "algorithms.td3",
}

# The non-training entry points, dumped into the same file under their own keys so the add-on's Run
# and Record windows build their forms the same way the Train wizard does -- from the CLI, instead of
# from defaults copied into GDScript that then drift.
COMMAND_MODULES = {
	"run": "run",
	"record": "recorder",
}


class _ParserCaptured(Exception):
	pass


def _capture_parser(build_and_parse):
	captured = {}
	original = argparse.ArgumentParser.parse_args

	def _intercept(self, *args, **kwargs):
		captured["parser"] = self
		raise _ParserCaptured()

	argparse.ArgumentParser.parse_args = _intercept
	try:
		build_and_parse()
	except _ParserCaptured:
		pass
	finally:
		argparse.ArgumentParser.parse_args = original
	return captured.get("parser")


def _kind(action):
	if isinstance(action, argparse.BooleanOptionalAction):
		return "bool"
	if isinstance(action, argparse._StoreTrueAction):
		return "flag_true"
	if isinstance(action, argparse._StoreFalseAction):
		return "flag_false"
	if action.choices:
		return "choice"
	if action.type is int:
		return "int"
	if action.type is float:
		return "float"
	return "str"


def _jsonable(value):
	if isinstance(value, (str, int, float, bool)) or value is None:
		return value
	return str(value)


def _group_titles(parser):
	dest_to_group = {}
	for group in parser._action_groups:
		title = group.title or ""
		if title in ("positional arguments", "options", "optional arguments"):
			title = ""
		for action in group._group_actions:
			dest_to_group[action.dest] = title
	return dest_to_group


def spec_for_algorithm(algorithm):
	if algorithm in COMMAND_MODULES:
		return spec_for_command(algorithm)
	module_name = ALGORITHM_MODULES.get(algorithm)
	if module_name is None:
		raise ValueError(f"Unknown algorithm: {algorithm!r}")
	module = importlib.import_module(module_name)
	if hasattr(module, "parse_args"):
		parser = _capture_parser(module.parse_args)
	else:
		# Deterministic variants (ddpg, td3, …) share algorithms.common.parse_args(trainer_variant).
		common = importlib.import_module("algorithms.common")
		parser = _capture_parser(lambda: common.parse_args(algorithm))
	if parser is None:
		raise RuntimeError(f"Could not capture the parser for {algorithm}")

	return {"algorithm": algorithm, "arguments": _describe(parser)}


def _describe(parser):
	"""Every user-facing option of a built parser, as plain JSON-able dictionaries."""
	groups = _group_titles(parser)
	arguments = []
	for action in parser._actions:
		if not action.option_strings:
			continue
		if action.help == argparse.SUPPRESS:
			continue
		if "-h" in action.option_strings or "--help" in action.option_strings:
			continue
		arguments.append({
			"flags": list(action.option_strings),
			"dest": action.dest,
			"kind": _kind(action),
			# Exported so the editor knows a flag takes several values (--network-layers 256 256):
			# without it the add-on would send "256 256" as ONE argv token and argparse would reject
			# it with `invalid int value: '256 256'`.
			"nargs": _jsonable(action.nargs),
			"default": _jsonable(action.default),
			"choices": [str(c) for c in action.choices] if action.choices else None,
			"help": (action.help or "").strip(),
			"group": groups.get(action.dest, ""),
		})
	return arguments


def spec_for_command(command):
	"""Capture run.py / recorder.py the same way, so their forms come from the CLI too."""
	module_name = COMMAND_MODULES.get(command)
	if module_name is None:
		raise ValueError(f"Unknown command: {command!r}")
	module = importlib.import_module(module_name)
	parser = _capture_parser(module.parse_args)
	if parser is None:
		raise RuntimeError(f"Could not capture the parser for {command}")
	return {"algorithm": command, "arguments": _describe(parser)}


def main(argv=None):
	parser = argparse.ArgumentParser(description="Dump a Metis entry point's argparse spec as JSON.")
	parser.add_argument("--algorithm", choices=sorted(ALGORITHM_MODULES) + sorted(COMMAND_MODULES))
	parser.add_argument("--all", action="store_true",
		help="Dump every algorithm plus the run and record commands.")
	parser.add_argument("--output", help="Write JSON to this file instead of stdout.")
	args = parser.parse_args(argv)

	if args.all:
		names = sorted(ALGORITHM_MODULES) + sorted(COMMAND_MODULES)
		result = {name: spec_for_algorithm(name)["arguments"] for name in names}
	elif args.algorithm:
		result = {args.algorithm: spec_for_algorithm(args.algorithm)["arguments"]}
	else:
		parser.error("pass --algorithm <name> or --all")

	text = json.dumps(result, indent=2)
	if args.output:
		with open(args.output, "w", encoding="utf-8") as handle:
			handle.write(text + "\n")
	else:
		sys.stdout.write(text + "\n")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
