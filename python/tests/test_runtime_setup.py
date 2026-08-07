import hashlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SETUP_PATH = (
    ROOT
    / "godot"
    / "addons"
    / "metis"
    / "python_setup"
    / "setup_runtime.py"
)
SPEC = importlib.util.spec_from_file_location("metis_runtime_setup", SETUP_PATH)
runtime_setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_setup)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RuntimeSetupTests(unittest.TestCase):
    def test_bootstrap_rejects_old_python(self):
        with self.assertRaisesRegex(RuntimeError, "Python 3.11 or newer"):
            runtime_setup.validate_bootstrap(
                "cpu",
                version_info=(3, 10, 14),
                system="Linux",
                machine="x86_64",
            )

    def test_bootstrap_rejects_profiles_on_the_wrong_host(self):
        with self.assertRaisesRegex(RuntimeError, "only on Linux"):
            runtime_setup.validate_bootstrap(
                "linux_cuda",
                version_info=(3, 11, 0),
                system="Darwin",
                machine="arm64",
            )
        with self.assertRaisesRegex(RuntimeError, "Apple Silicon"):
            runtime_setup.validate_bootstrap(
                "macos_metal",
                version_info=(3, 11, 0),
                system="Darwin",
                machine="x86_64",
            )

    def test_payload_validation_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload_dir = Path(temporary)
            wheel = payload_dir / "metis_rl-0.1.0.whl"
            requirements = payload_dir / "requirements.txt"
            wheel.write_bytes(b"wheel")
            requirements.write_text("numpy\n", encoding="utf-8")
            manifest = {
                "plugin_version": "0.1.0",
                "python_version": "0.1.0",
                "wheel": wheel.name,
                "wheel_sha256": _digest(wheel),
                "profiles": {"cpu": requirements.name},
                "extras": {},
                "requirements": {
                    requirements.name: _digest(requirements),
                },
            }

            runtime_setup.validate_payload(payload_dir, manifest)
            requirements.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "checksum is invalid"):
                runtime_setup.validate_payload(payload_dir, manifest)

    def test_payload_profiles_must_reference_checked_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload_dir = Path(temporary)
            wheel = payload_dir / "metis_rl-0.1.0.whl"
            requirements = payload_dir / "requirements.txt"
            wheel.write_bytes(b"wheel")
            requirements.write_text("numpy\n", encoding="utf-8")
            manifest = {
                "plugin_version": "0.1.0",
                "python_version": "0.1.0",
                "wheel": wheel.name,
                "wheel_sha256": _digest(wheel),
                "profiles": {"cpu": "unchecked.txt"},
                "extras": {},
                "requirements": {
                    requirements.name: _digest(requirements),
                },
            }

            with self.assertRaisesRegex(RuntimeError, "unchecked requirement"):
                runtime_setup.validate_payload(payload_dir, manifest)

    def test_existing_runtime_must_match_the_addon_version(self):
        manifest = {"python_version": "0.1.0"}
        with mock.patch.object(
            runtime_setup.metadata,
            "version",
            return_value="0.2.0",
        ):
            with self.assertRaisesRegex(RuntimeError, "requires 0.1.0"):
                runtime_setup.validate_existing_metis(manifest)

    def test_sb3_is_validated_beside_the_native_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                project_dir=Path(temporary),
                status_path=Path(temporary) / "setup_status.json",
                profile="cpu",
                dashboard=False,
                export_tools=False,
                sb3=True,
                source_root=None,
                godot_bin=None,
                existing=False,
            )
            manifest = {
                "plugin_version": "0.1.0",
                "python_version": "0.1.0",
            }
            with mock.patch.object(runtime_setup, "run_logged") as run_logged:
                config = runtime_setup.validate_runtime(
                    args,
                    io.StringIO(),
                    Path(sys.executable),
                    manifest,
                )

            command = run_logged.call_args.args[0]
            requested_profiles = [
                command[index + 1]
                for index, value in enumerate(command)
                if value == "--profile"
            ]
            self.assertEqual(requested_profiles, ["core", "native", "sb3"])
            self.assertTrue(config["sb3"])


if __name__ == "__main__":
    unittest.main()
