import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from envs.process_manager import (
    GodotProcessManager,
    PortLeaseSet,
    _is_software_opengl,
    _parse_switcheroo_offload_environment,
)


class FakeProcess:
    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def wait(self, timeout=None):
        del timeout
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = -9


class GodotProcessManagerTests(unittest.TestCase):
    def test_detects_software_opengl_renderers(self):
        self.assertTrue(_is_software_opengl("OpenGL renderer string: llvmpipe"))
        self.assertTrue(_is_software_opengl("Accelerated: no"))
        self.assertFalse(_is_software_opengl("OpenGL renderer string: NVIDIA GeForce RTX"))

    def test_parses_non_default_switcheroo_environment(self):
        output = """Device: 0
  Name:        Intel Corporation Iris Xe
  Default:     yes
  Environment: DRI_PRIME=pci-0000_00_02_0

Device: 1
  Name:        NVIDIA Corporation RTX 4060
  Default:     no
  Environment: __GLX_VENDOR_LIBRARY_NAME=nvidia __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only
"""
        self.assertEqual(
            _parse_switcheroo_offload_environment(output),
            {
                "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
                "__NV_PRIME_RENDER_OFFLOAD": "1",
                "__VK_LAYER_NV_optimus": "NVIDIA_only",
            },
        )

    def test_default_runtime_paths_live_outside_the_python_package(self):
        manager = GodotProcessManager(godot_bin="godot")
        repository_root = Path(__file__).resolve().parents[2]
        self.assertEqual(manager.project_dir, repository_root / "godot")
        self.assertEqual(manager.logs_dir, repository_root / ".runtime" / "godot_logs")

    def start_and_capture(self, *, headless, render_env_count):
        commands = []

        def fake_popen(command, **_kwargs):
            commands.append(command)
            return FakeProcess()

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GodotProcessManager(
                godot_bin="godot",
                project_dir=temp_dir,
                logs_dir=temp_dir,
                port_lease_dir=Path(temp_dir) / "leases",
            )
            with (
                patch("envs.process_manager.subprocess.Popen", side_effect=fake_popen),
                patch("envs.process_manager.port_is_available", return_value=True),
                patch("envs.process_manager.time.sleep"),
                patch.object(manager, "_wait_for_log_ready", return_value=True),
            ):
                manager.start_many(
                    [6200, 6201, 6202],
                    headless=headless,
                    render_env_count=render_env_count,
                )
                manager.stop_all()
        return commands

    def test_omitted_render_count_renders_every_environment(self):
        commands = self.start_and_capture(headless=False, render_env_count=None)
        self.assertEqual(["--headless" in command for command in commands], [False, False, False])

    def test_render_count_limits_visible_environments(self):
        commands = self.start_and_capture(headless=False, render_env_count=1)
        self.assertEqual(["--headless" in command for command in commands], [False, True, True])

    def test_headless_flag_overrides_render_count(self):
        commands = self.start_and_capture(headless=True, render_env_count=None)
        self.assertEqual(["--headless" in command for command in commands], [True, True, True])

    def test_each_environment_uses_a_distinct_engine_log(self):
        commands = self.start_and_capture(headless=True, render_env_count=None)
        log_paths = [
            Path(command[command.index("--log-file") + 1]).name
            for command in commands
        ]
        self.assertEqual(
            log_paths,
            ["godot_engine_6200.log", "godot_engine_6201.log", "godot_engine_6202.log"],
        )

    def test_light_gpu_passes_detected_offload_environment_to_godot(self):
        popen_calls = []

        def fake_popen(command, **kwargs):
            popen_calls.append((command, kwargs))
            return FakeProcess()

        offload_env = {
            "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
            "__NV_PRIME_RENDER_OFFLOAD": "1",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GodotProcessManager(
                godot_bin="godot",
                project_dir=temp_dir,
                logs_dir=temp_dir,
                port_lease_dir=Path(temp_dir) / "leases",
            )
            with (
                patch("envs.process_manager.subprocess.Popen", side_effect=fake_popen),
                patch("envs.process_manager.port_is_available", return_value=True),
                patch("envs.process_manager.time.sleep"),
                patch.object(manager, "_wait_for_log_ready", return_value=True),
                patch(
                    "envs.process_manager._light_gpu_fallback_environment",
                    return_value=offload_env,
                ),
            ):
                manager.start_many(
                    [6200, 6201],
                    headless=False,
                    render_env_count=1,
                    render_mode="light-gpu",
                )
                manager.stop_all()

        rendered_command, rendered_kwargs = popen_calls[0]
        self.assertIn("opengl3", rendered_command)
        self.assertEqual(rendered_kwargs["env"]["__GLX_VENDOR_LIBRARY_NAME"], "nvidia")
        self.assertEqual(rendered_kwargs["env"]["__NV_PRIME_RENDER_OFFLOAD"], "1")
        self.assertIn("--headless", popen_calls[1][0])
        self.assertIsNone(popen_calls[1][1]["env"])

    def test_port_lease_refuses_a_live_owner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("envs.process_manager.port_is_available", return_value=True):
                first = PortLeaseSet([6210], temp_dir).acquire()
                self.addCleanup(first.release)
                with self.assertRaisesRegex(RuntimeError, "live process"):
                    PortLeaseSet([6210], temp_dir).acquire()

    def test_stale_port_lease_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "port-6211.json"
            path.write_text(
                json.dumps({"host": "127.0.0.1", "port": 6211, "pid": 99999999, "token": "old"}),
                encoding="utf-8",
            )
            with patch("envs.process_manager.port_is_available", return_value=True):
                lease = PortLeaseSet([6211], temp_dir).acquire()
            self.assertNotEqual(json.loads(path.read_text())["token"], "old")
            lease.release()
            self.assertFalse(path.exists())

    def test_occupied_port_is_rejected_before_popen(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GodotProcessManager(
                godot_bin="godot",
                project_dir=temp_dir,
                logs_dir=temp_dir,
                port_lease_dir=Path(temp_dir) / "leases",
            )
            with (
                patch("envs.process_manager.port_is_available", return_value=False),
                patch("envs.process_manager.subprocess.Popen") as popen,
            ):
                with self.assertRaisesRegex(RuntimeError, "occupied"):
                    manager.start_many([6219])
            popen.assert_not_called()
            self.assertEqual(manager.processes, [])

    def test_partial_readiness_failure_rolls_back_every_process(self):
        processes = [FakeProcess(), FakeProcess()]
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GodotProcessManager(
                godot_bin="godot",
                project_dir=temp_dir,
                logs_dir=temp_dir,
                port_lease_dir=Path(temp_dir) / "leases",
            )
            with (
                patch("envs.process_manager.subprocess.Popen", side_effect=processes),
                patch("envs.process_manager.port_is_available", return_value=True),
                patch("envs.process_manager.time.sleep"),
                patch.object(manager, "_wait_for_log_ready", side_effect=[True, False]),
            ):
                with self.assertRaisesRegex(RuntimeError, "did not become ready"):
                    manager.start_many([6212, 6213])

            self.assertEqual([proc.returncode for proc in processes], [0, 0])
            self.assertEqual(manager.processes, [])
            self.assertTrue(manager.last_stop_report["clean"])
            self.assertEqual(list((Path(temp_dir) / "leases").glob("*.json")), [])

    def test_processes_start_in_a_new_posix_session(self):
        if os.name != "posix":
            self.skipTest("POSIX-only process-group contract")
        calls = []

        def fake_popen(command, **kwargs):
            calls.append((command, kwargs))
            return FakeProcess()

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GodotProcessManager(
                godot_bin="godot",
                project_dir=temp_dir,
                logs_dir=temp_dir,
                port_lease_dir=Path(temp_dir) / "leases",
            )
            with (
                patch("envs.process_manager.subprocess.Popen", side_effect=fake_popen),
                patch("envs.process_manager.port_is_available", return_value=True),
                patch("envs.process_manager.time.sleep"),
                patch.object(manager, "_wait_for_log_ready", return_value=True),
            ):
                manager.start_many([6214])
                manager.stop_all()
        self.assertTrue(calls[0][1]["start_new_session"])

    def test_spawn_failure_rolls_back_started_process_and_every_lease(self):
        first_process = FakeProcess()
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GodotProcessManager(
                godot_bin="godot",
                project_dir=temp_dir,
                logs_dir=temp_dir,
                port_lease_dir=Path(temp_dir) / "leases",
            )
            with (
                patch(
                    "envs.process_manager.subprocess.Popen",
                    side_effect=[first_process, OSError("spawn failed")],
                ),
                patch("envs.process_manager.port_is_available", return_value=True),
            ):
                with self.assertRaisesRegex(OSError, "spawn failed"):
                    manager.start_many([6216, 6217])

            self.assertEqual(first_process.returncode, 0)
            self.assertEqual(manager.processes, [])
            self.assertTrue(manager.last_stop_report["clean"])
            self.assertEqual(manager.last_stop_report["ports"], [6216, 6217])
            self.assertEqual(list((Path(temp_dir) / "leases").glob("*.json")), [])

    def test_malformed_lease_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "port-6218.json"
            path.write_text("not-json\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Invalid port lease"):
                PortLeaseSet([6218], temp_dir).acquire()
            self.assertTrue(path.exists())

    def test_owned_lease_tampering_makes_cleanup_fail_closed(self):
        process = FakeProcess()
        with tempfile.TemporaryDirectory() as temp_dir:
            lease_dir = Path(temp_dir) / "leases"
            manager = GodotProcessManager(
                godot_bin="godot",
                project_dir=temp_dir,
                logs_dir=temp_dir,
                port_lease_dir=lease_dir,
            )
            with (
                patch("envs.process_manager.subprocess.Popen", return_value=process),
                patch("envs.process_manager.port_is_available", return_value=True),
                patch("envs.process_manager.time.sleep"),
                patch.object(manager, "_wait_for_log_ready", return_value=True),
            ):
                manager.start_many([6220])
                lease_path = lease_dir / "port-6220.json"
                payload = json.loads(lease_path.read_text(encoding="utf-8"))
                payload["token"] = "tampered"
                lease_path.write_text(json.dumps(payload), encoding="utf-8")
                report = manager.stop_all()

            self.assertFalse(report["clean"])
            self.assertIn("port lease files remain", report["errors"][0])
            self.assertTrue(lease_path.exists())

    def test_stop_is_idempotent_and_keeps_the_original_report(self):
        manager = GodotProcessManager(godot_bin="godot")
        manager.last_stop_report = {"clean": False, "errors": ["original failure"]}
        self.assertEqual(manager.stop_all(), manager.last_stop_report)


class StubbornProcess(FakeProcess):
    def __init__(self):
        super().__init__()
        self.killed = False

    def wait(self, timeout=None):
        if not self.killed:
            raise subprocess.TimeoutExpired("godot", timeout)
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


class GodotForcedCleanupTests(unittest.TestCase):
    def test_stop_escalates_to_kill_after_grace_period(self):
        process = StubbornProcess()
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GodotProcessManager(
                godot_bin="godot",
                project_dir=temp_dir,
                logs_dir=temp_dir,
                port_lease_dir=Path(temp_dir) / "leases",
            )
            with (
                patch("envs.process_manager.subprocess.Popen", return_value=process),
                patch("envs.process_manager.port_is_available", return_value=True),
                patch("envs.process_manager.time.sleep"),
                patch.object(manager, "_wait_for_log_ready", return_value=True),
            ):
                manager.start_many([6215])
                report = manager.stop_all()
        self.assertTrue(process.killed)
        self.assertTrue(report["clean"])


if __name__ == "__main__":
    unittest.main()
