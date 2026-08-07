import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import metis_cli
from core import doctor


class MetisCliTests(unittest.TestCase):
    def test_help_does_not_import_training_backends(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = metis_cli.main(["--help"])
        self.assertEqual(result, 0)
        self.assertIn("metis <command>", output.getvalue())

    def test_dispatch_preserves_command_arguments(self):
        fake_module = mock.Mock()
        fake_module.main.return_value = None
        with mock.patch.object(metis_cli.importlib, "import_module", return_value=fake_module):
            result = metis_cli.main(["train", "--algorithm", "dqn"])
        self.assertEqual(result, 0)
        self.assertEqual(metis_cli.sys.argv, ["metis train", "--algorithm", "dqn"])


class DoctorTests(unittest.TestCase):
    def test_core_report_is_machine_readable(self):
        report = doctor.build_report(["core"], accelerators=False)
        self.assertEqual(report["status"], "ok")
        self.assertIn("numpy", report["packages"])
        self.assertIn("gymnasium", report["packages"])
        self.assertIn("python", report)

    def test_report_rejects_an_unexpected_metis_version(self):
        with mock.patch.object(doctor, "_distribution_version", return_value="0.1.0"):
            report = doctor.build_report(
                ["core"],
                accelerators=False,
                expected_metis_version="0.2.0",
            )
        self.assertEqual(report["status"], "error")
        self.assertEqual(
            report["version_mismatch"],
            {"expected": "0.2.0", "installed": "0.1.0"},
        )

    def test_accelerated_profile_requires_a_visible_device(self):
        available_package = {
            "available": True,
            "version": "test",
            "module": "test",
        }
        with (
            mock.patch.object(
                doctor,
                "_module_status",
                return_value=available_package,
            ),
            mock.patch.object(
                doctor,
                "_distribution_version",
                return_value="0.1.0",
            ),
            mock.patch.object(
                doctor,
                "_accelerator_status",
                return_value={"tensorflow": []},
            ),
        ):
            report = doctor.build_report(
                ["native"],
                require_accelerator="tensorflow",
            )
        self.assertEqual(report["status"], "error")
        self.assertIn("accelerator_error", report)


if __name__ == "__main__":
    unittest.main()
