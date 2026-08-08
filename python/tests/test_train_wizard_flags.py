"""Keep the Godot Train wizard's curated option list in sync with the Python CLI.

The add-on's per-algorithm page (``ALGO_OPTIONS`` in ``metis_train_dialog.gd``) references arguments
by their argparse ``dest``. Nothing at runtime notices a mismatch: the wizard skips a dest it cannot
find in ``argspec.json``, so renaming a flag in Python makes the corresponding field silently
disappear from the editor instead of raising. This test parses the GDScript table and checks every
entry against the real parsers, then replays the exact argv the wizard would build to confirm the
CLI accepts it and lands the value on the right attribute.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
import unittest
from pathlib import Path

os.environ.setdefault("GODOT_GYM_TF_LD_READY", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import argspec, training  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DIALOG = REPO_ROOT / "godot/addons/metis/editor/run/metis_train_dialog.gd"

# The Run and Record windows use the same dest-plus-label table, under the name CURATED, against the
# run.py / recorder.py specs rather than an algorithm's.
COMMAND_DIALOGS = {
    "run": REPO_ROOT / "godot/addons/metis/editor/run/metis_run_dialog.gd",
    "record": REPO_ROOT / "godot/addons/metis/editor/run/metis_record_dialog.gd",
}

# Groups declared outside core.training, because importing the constants there would close an import
# cycle (curriculum) or belongs to the module that owns the feature.
EXTERNAL_GROUPS = {"independent multi-policy", "opponent pool self-play"}


def parse_algo_options(source):
    """Extract ``{algorithm: [dest, ...]}`` from the GDScript ``ALGO_OPTIONS`` constant."""
    block = source.split("const ALGO_OPTIONS := {")[1].split("\n}")[0]
    table = {}
    algorithm = None
    for line in block.splitlines():
        opened = re.match(r'\t"(\w+)": \[', line)
        if opened:
            algorithm = opened.group(1)
            table[algorithm] = []
            continue
        entry = re.match(r'\t\t\["(\w+)", ".*"\],', line)
        if entry and algorithm is not None:
            table[algorithm].append(entry.group(1))
    return table


def parse_curated(source):
    """Extract the dest list from a `const CURATED := [...]` table in a single-page dialog."""
    block = source.split("const CURATED := [")[1].split("\n]")[0]
    return [m.group(1) for m in re.finditer(r'\t\["(\w+)", ".*"\],', block)]


def build_parser(algorithm):
    module = importlib.import_module(argspec.ALGORITHM_MODULES[algorithm])
    if hasattr(module, "parse_args"):
        return argspec._capture_parser(module.parse_args)
    common = importlib.import_module("algorithms.common")
    return argspec._capture_parser(lambda: common.parse_args(algorithm))


def wizard_tokens(arg, value):
    """Mirror ``_flags_from_controls()``: how the wizard turns one edited row into argv tokens."""
    flags = arg["flags"]
    if arg["kind"] == "bool":
        positive = next(flag for flag in flags if not flag.startswith("--no-"))
        negative = next((flag for flag in flags if flag.startswith("--no-")), None)
        return [positive if value else negative]
    if arg["kind"] == "choice":
        return [flags[0], str(value)]
    nargs = arg.get("nargs")
    if nargs in ("+", "*") or (isinstance(nargs, int) and nargs > 0):
        # The GDScript splits on whitespace so "--network-layers 256 256" reaches argparse as three
        # tokens; sending it as one would fail with `invalid int value: '256 256'`.
        return [flags[0]] + str(value).split()
    return [flags[0], str(value)]


def off_default(arg):
    """A value different from the argument's default, plus what argparse should end up storing."""
    if arg["kind"] == "bool":
        value = not bool(arg["default"])
        return value, value
    if arg["kind"] == "choice":
        alternatives = [c for c in arg["choices"] if c != str(arg["default"])]
        value = alternatives[0] if alternatives else arg["choices"][0]
        return value, value
    if arg.get("nargs") in ("+", "*"):
        return "128 128", [128, 128]
    if arg["kind"] == "int":
        value = int(arg["default"] or 0) + 3
        return value, value
    if arg["kind"] == "str":
        # argparse stores these verbatim; sending a number here would compare a str against a float.
        value = "metis-test-value"
        return value, value
    return 0.123, 0.123


class ArgumentGroupTests(unittest.TestCase):
    """The add-on renders one heading per argparse group, so an ungrouped or misspelt argument shows
    up in the editor as a stray "General" section instead of failing anywhere."""

    def declared_titles(self):
        return {value for name, value in vars(training).items()
                if name.startswith("GROUP_") and isinstance(value, str)} | EXTERNAL_GROUPS

    def test_every_argument_belongs_to_a_group(self):
        for algorithm in argspec.ALGORITHM_MODULES:
            with self.subTest(algorithm=algorithm):
                ungrouped = [a["flags"][0] for a in argspec.spec_for_algorithm(algorithm)["arguments"]
                             if not a.get("group")]
                self.assertEqual(ungrouped, [])

    def test_group_titles_are_the_declared_ones(self):
        # Catches a title typed by hand at a new call site: it would render as its own near-duplicate
        # heading right next to the real one.
        declared = self.declared_titles()
        for algorithm in argspec.ALGORITHM_MODULES:
            with self.subTest(algorithm=algorithm):
                used = {a["group"] for a in argspec.spec_for_algorithm(algorithm)["arguments"]}
                self.assertEqual(sorted(used - declared), [])

    def test_each_title_appears_once_per_parser(self):
        # argparse creates a second identical section when add_argument_group is called twice with
        # the same title; argument_group() exists to prevent that, and this is the check.
        for algorithm in argspec.ALGORITHM_MODULES:
            with self.subTest(algorithm=algorithm):
                titles = [group.title for group in build_parser(algorithm)._action_groups
                          if group.title]
                self.assertEqual(sorted(titles), sorted(set(titles)))


class CommandDialogFlagTests(unittest.TestCase):
    """Same contract as the Train wizard, for the Run and Record windows: a dest they name and the
    CLI no longer has disappears from the form silently."""

    def test_every_curated_dest_exists(self):
        for command, path in COMMAND_DIALOGS.items():
            with self.subTest(command=command):
                dests = parse_curated(path.read_text())
                self.assertTrue(dests, f"{command} dialog has no curated fields")
                known = {a["dest"] for a in argspec.spec_for_command(command)["arguments"]}
                self.assertEqual([d for d in dests if d not in known], [])
                self.assertEqual(len(dests), len(set(dests)))

    def test_the_dialog_command_is_accepted_by_the_real_parser(self):
        for command, path in COMMAND_DIALOGS.items():
            with self.subTest(command=command):
                spec = {a["dest"]: a for a in argspec.spec_for_command(command)["arguments"]}
                module = importlib.import_module(argspec.COMMAND_MODULES[command])
                parser = argspec._capture_parser(module.parse_args)
                argv, expected = [], {}
                for dest in parse_curated(path.read_text()):
                    value, stored = off_default(spec[dest])
                    expected[dest] = stored
                    argv.extend(t for t in wizard_tokens(spec[dest], value) if t is not None)
                parsed = parser.parse_args(argv)
                for dest, stored in expected.items():
                    self.assertEqual(getattr(parsed, dest), stored, f"{command} --{dest}")

    def test_every_argument_belongs_to_a_group(self):
        for command in argspec.COMMAND_MODULES:
            with self.subTest(command=command):
                ungrouped = [a["flags"][0] for a in argspec.spec_for_command(command)["arguments"]
                             if not a.get("group")]
                self.assertEqual(ungrouped, [])


class TrainWizardFlagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = parse_algo_options(DIALOG.read_text())

    def test_the_gdscript_table_was_parsed(self):
        # Guards the test itself: a formatting change to ALGO_OPTIONS must not turn every check below
        # into a vacuous pass over an empty table.
        self.assertEqual(sorted(self.table), sorted(argspec.ALGORITHM_MODULES))
        for algorithm, dests in self.table.items():
            self.assertTrue(dests, f"{algorithm} has no curated options")

    def test_every_curated_dest_exists_and_is_listed_once(self):
        for algorithm, dests in self.table.items():
            with self.subTest(algorithm=algorithm):
                known = {arg["dest"] for arg in argspec.spec_for_algorithm(algorithm)["arguments"]}
                self.assertEqual([d for d in dests if d not in known], [])
                self.assertEqual(len(dests), len(set(dests)))

    def test_the_wizard_command_is_accepted_by_the_real_parser(self):
        for algorithm, dests in self.table.items():
            with self.subTest(algorithm=algorithm):
                spec = {a["dest"]: a for a in argspec.spec_for_algorithm(algorithm)["arguments"]}
                parser = build_parser(algorithm)
                argv = []
                expected = {}
                for dest in dests:
                    value, stored = off_default(spec[dest])
                    expected[dest] = stored
                    argv.extend(t for t in wizard_tokens(spec[dest], value) if t is not None)
                parsed = parser.parse_args(argv)
                for dest, stored in expected.items():
                    self.assertEqual(getattr(parsed, dest), stored, f"{algorithm} --{dest}")


if __name__ == "__main__":
    unittest.main()
