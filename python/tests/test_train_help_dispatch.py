"""`train.py --algorithm X --help` must show X's options, and must never start Godot to find out.

The dispatcher answers --help before argparse does, because argparse's automatic help action fires
inside parse_known_args() and exits the process. That interception has two sharp edges worth pinning
down: resolving --algorithm auto means launching a Godot instance to probe the scenario's action
space, which no help request may do; and the frontend-only options have to be stripped before the
argv is handed to a backend that does not accept them.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train  # noqa: E402


class OptionValueTests(unittest.TestCase):
    def test_reads_both_spellings(self):
        self.assertEqual(train.option_value(["--algorithm", "sac"], "--algorithm"), "sac")
        self.assertEqual(train.option_value(["--algorithm=sac"], "--algorithm"), "sac")
        self.assertEqual(train.option_value(["--num-envs", "4"], "--algorithm"), None)

    def test_a_trailing_option_with_no_value_is_not_a_match(self):
        # "--algorithm" as the last token would otherwise index past the end of argv.
        self.assertEqual(train.option_value(["--help", "--algorithm"], "--algorithm"), None)

    def test_the_first_occurrence_wins_like_argparse_last_does_not_matter_here(self):
        self.assertEqual(train.option_value(["--algorithm", "sac", "--num-envs", "2"],
                                            "--algorithm"), "sac")


class BackendModuleTests(unittest.TestCase):
    def test_metis_maps_every_algorithm(self):
        for algorithm, module in train.BACKENDS.items():
            self.assertEqual(train.backend_module_for(algorithm, "metis"), module)

    def test_sb3_rejects_algorithms_it_cannot_run(self):
        self.assertEqual(train.backend_module_for("sac", "sb3"), train.SB3_BACKEND)
        with self.assertRaises(RuntimeError) as caught:
            train.backend_module_for("ddpgfd", "sb3")
        self.assertIn("ddpgfd", str(caught.exception))


class DelegateHelpTests(unittest.TestCase):
    """Only the decline paths are exercised directly: the accept path imports a trainer (and with it
    TensorFlow), so it is checked through a stubbed module instead."""

    def test_declines_without_a_concrete_algorithm(self):
        for argv in ([], ["--help"], ["--algorithm", "auto", "--help"],
                     ["--algorithm", "bogus", "--help"]):
            with self.subTest(argv=argv):
                self.assertFalse(train.delegate_help(argv))

    def test_auto_never_probes_godot(self):
        # The probe is what launches Godot; reaching it from a help request would be the bug.
        with patch.object(train, "probe_action_type", side_effect=AssertionError("probed Godot")):
            self.assertFalse(train.delegate_help(["--algorithm", "auto", "--help"]))

    def test_delegates_with_frontend_options_stripped(self):
        seen = {}

        class FakeBackend:
            @staticmethod
            def main():
                seen["argv"] = list(sys.argv)

        original = list(sys.argv)
        with patch.object(train.importlib, "import_module", return_value=FakeBackend) as imported:
            self.assertTrue(train.delegate_help(
                ["--algorithm", "sac", "--num-envs", "4", "--help"]))
        sys.argv = original
        imported.assert_called_once_with("algorithms.sac")
        # --algorithm is a train.py-only option: the trainer's parser would reject it.
        self.assertNotIn("--algorithm", seen["argv"])
        self.assertIn("--help", seen["argv"])
        self.assertIn("--num-envs", seen["argv"])

    def test_sb3_receives_the_resolved_algorithm(self):
        seen = {}

        class FakeBackend:
            @staticmethod
            def main():
                seen["argv"] = list(sys.argv)

        original = list(sys.argv)
        with patch.object(train.importlib, "import_module", return_value=FakeBackend) as imported:
            self.assertTrue(train.delegate_help(
                ["--backend", "sb3", "--algorithm", "ppo", "--help"]))
        sys.argv = original
        imported.assert_called_once_with(train.SB3_BACKEND)
        # The shared SB3 backend needs the algorithm back, since one module serves all of them.
        self.assertEqual(seen["argv"][-2:], ["--algorithm", "ppo"])

    def test_an_unsupported_sb3_algorithm_exits_instead_of_falling_through(self):
        # Falling through would print the dispatcher's help and imply the combination is valid.
        with self.assertRaises(SystemExit) as caught:
            train.delegate_help(["--backend", "sb3", "--algorithm", "ddpgfd", "--help"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
